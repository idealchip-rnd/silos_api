# ==============================================
# Flask API for Silo Temperature Monitoring
# ==============================================
from sqlalchemy import func
from flask import Flask, request, Response
from flask_cors import CORS
from datetime import datetime, timedelta
from models import (
    db, Silo, Cable, Sensor, Reading, Product, StatusColor,
    SiloProductAssignment, Alert
)

# --- add to imports near the top ---
from models import User
from werkzeug.security import check_password_hash, generate_password_hash
from sqlalchemy.exc import IntegrityError

# ReadingRaw may not define relationships, so we handle joins in-code
try:
    from models import ReadingRaw  # expects: id, sensor_id, value_c, polled_at, poll_run_id (optional)
except Exception:  # pragma: no cover
    ReadingRaw = None

from sqlalchemy.orm import selectinload
from sqlalchemy import and_
from datetime import datetime
from collections import OrderedDict, defaultdict
import os
import json
import math
import string  # <-- for hex normalization

DISCONNECT_SENTINELS = {-127.0}  # add more if you use others

# ------------------------------------------------
# App & Config
# ------------------------------------------------
app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URL',
    'mysql+pymysql://silos_user:Idealchip123%40@localhost/silos'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# ------------------------------------------------
# Helpers
# ------------------------------------------------
def _is_disconnect_temp(temp) -> bool:
    if temp is None:
        return True
    try:
        # treat NaN/inf and your sentinel(s) as disconnect
        if isinstance(temp, (int, float)):
            if math.isnan(temp) or math.isinf(temp):
                return True
            if float(temp) in DISCONNECT_SENTINELS:
                return True
    except Exception:
        pass
    return False

def json_response(payload, status=200):
    return Response(
        json.dumps(payload, ensure_ascii=False, sort_keys=False),
        status=status,
        mimetype='application/json'
    )

def _parse_dt(s: str | None):
    if not s:
        return None
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        return datetime.fromisoformat(s)
    except Exception:
        return None

def _day_key(dt: datetime) -> str:
    return dt.date().isoformat()

def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)

# Timestamp columns (readings vs raw)
def _ts_reading(row=None):
    """
    Return the datetime column/field for Reading or a Reading row.
    Priority: hour_start -> start_hour -> sampled_at -> timestamp.
    """
    candidates = ("hour_start", "start_hour", "sampled_at", "timestamp")
    if row is None:
        # Return the first matching column object
        for attr in candidates:
            col = getattr(Reading, attr, None)
            if col is not None:
                return col
        raise RuntimeError(f"Reading has no time column (checked {candidates}).")
    # For a row instance
    for attr in candidates + ("polled_at",):
        val = getattr(row, attr, None)
        if val is not None:
            return val
    return None

def _ts_raw():
    # Priority: readings_raw.polled_at if raw table exists; else reuse readings ts
    if ReadingRaw is not None:
        return getattr(ReadingRaw, "polled_at", None)
    return _ts_reading()

READ_TS_COL = _ts_reading()
RAW_TS_COL  = _ts_raw()

def _timestamp_iso_from_any(row):
    t = _ts_reading(row) or getattr(row, "polled_at", None)
    return t.isoformat() if t else None

def _temperature_from_any(row):
    # readings_raw has value_c, readings has temperature
    if hasattr(row, "value_c"):
        return float(row.value_c) if row.value_c is not None else None
    return row.temperature if hasattr(row, "temperature") else None

from datetime import datetime

def _normalize_ts_for_flatten(ts: str | None) -> str:
    """Normalize an ISO timestamp string to second precision for grouping."""
    if not ts:
        return ""
    try:
        # Handles both "YYYY-MM-DDTHH:MM:SS[.ffffff][+TZ]" variants
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        dt = dt.replace(microsecond=0)
        # keep timezone if present in input
        if ts.endswith("Z"):
            return dt.isoformat().replace("+00:00", "Z")
        return dt.isoformat()
    except Exception:
        # Fallback: strip microseconds if present as plain text
        if "." in ts:
            base, rest = ts.split(".", 1)
            # keep timezone suffix if any after microseconds
            if "+" in rest:
                tz = "+" + rest.split("+", 1)[1]
                return base + tz
            if "Z" in rest:
                return base + "Z"
            return base
        return ts

# ------------------------------------------------
# Status / Color Helpers
# ------------------------------------------------
_STATUS_COLOR_CACHE = None

def _normalize_hex(color: str | None) -> str | None:
    """Ensure hex color has a leading '#' if it looks like bare hex."""
    if not color:
        return None
    c = color.strip()
    hexchars = set(string.hexdigits)
    if not c.startswith("#") and all(ch in hexchars for ch in c) and len(c) in (3, 6):
        return "#" + c
    return c

def _load_status_colors():
    global _STATUS_COLOR_CACHE
    if _STATUS_COLOR_CACHE is None:
        _STATUS_COLOR_CACHE = {}
        for row in StatusColor.query.all():
            _STATUS_COLOR_CACHE[row.status] = _normalize_hex(getattr(row, "color_hex", None))

def get_status_color(temp, product):
    # Honor disconnect sentinels
    if _is_disconnect_temp(temp):
        return "disconnect", _color_for_state("disconnect")

    if temp is None or product is None:
        return None, None

    warn = product.temp_warn or 35
    crit = product.temp_critical or 40

    if temp >= crit:
        state = "critical"
    elif temp >= warn:
        state = "warn"
    else:
        state = "normal"

    _load_status_colors()
    return state, _normalize_hex(_STATUS_COLOR_CACHE.get(state, "#ffffff"))

def worst_state_color(states):
    # rank disconnect as severest
    priority = {"disconnect": 4, "critical": 3, "warn": 2, "normal": 1, None: 0}
    worst = None
    for s in (states or []):
        if priority.get(s, 0) > priority.get(worst, 0):
            worst = s

    _load_status_colors()
    if worst:
        return worst, (_STATUS_COLOR_CACHE.get(worst) or _color_for_state(worst) or "#ffffff")
    return None, "#ffffff"

def _color_for_state(state: str | None) -> str | None:
    """Map severity -> hex using DB StatusColor first, then sane fallbacks."""
    if not state:
        return None
    _load_status_colors()
    if _STATUS_COLOR_CACHE and state in _STATUS_COLOR_CACHE:
        return _normalize_hex(_STATUS_COLOR_CACHE[state])
    # Fallbacks if StatusColor table is empty/incomplete
    return _normalize_hex({"critical": "#d14141", "warn": "#c7c150", "normal": "#46d446", "disconnect": "#808080"}.get(state))

# ------------------------------------------------
# DB Query Helpers
#   - Base (readings):        /by-*, /avg/by-*, /max/*
#   - Base (readings_raw):    /latest/*, /avg/latest/*
# ------------------------------------------------
def _base_readings_query(sensor_ids, start, end):
    q = Reading.query.filter(Reading.sensor_id.in_(sensor_ids))
    if start:
        q = q.filter(READ_TS_COL >= start)
    if end:
        q = q.filter(READ_TS_COL <= end)
    return q.options(
        selectinload(Reading.sensor)
        .selectinload(Sensor.cable)
        .selectinload(Cable.silo)
        .selectinload(Silo.group)
    )

def _preload_sensors(sensor_ids):
    """Load sensor -> cable -> silo -> group graph for given IDs."""
    if not sensor_ids:
        return {}
    sensors = (Sensor.query
               .filter(Sensor.id.in_(sensor_ids))
               .options(
                   selectinload(Sensor.cable)
                     .selectinload(Cable.silo)
                     .selectinload(Silo.group)
               ).all())
    return {s.id: s for s in sensors}

def _preload_products_for_silo_ids(silo_ids):
    if not silo_ids:
        return {}
    assigns = (SiloProductAssignment.query
               .filter(SiloProductAssignment.silo_id.in_(list(silo_ids)))
               .options(selectinload(SiloProductAssignment.product))
               .all())
    return {a.silo_id: a.product for a in assigns}

def _preload_products_from_reading_rows(rows):
    silo_ids = {r.sensor.cable.silo_id for r in rows}
    return _preload_products_for_silo_ids(silo_ids)

def _sensors_for_cables(cable_ids):
    return (
        Sensor.query
        .filter(Sensor.cable_id.in_(cable_ids))
        .options(
            selectinload(Sensor.cable)
            .selectinload(Cable.silo)
            .selectinload(Silo.group)
        )
        .all()
    )

def _sensors_for_silos(silo_ids):
    return (
        Sensor.query
        .join(Cable).join(Silo)
        .filter(Silo.id.in_(silo_ids))
        .options(
            selectinload(Sensor.cable)
            .selectinload(Cable.silo)
            .selectinload(Silo.group)
        )
        .all()
    )

def _silo_number_to_ids(numbers):
    if not numbers:
        return []
    return [s.id for s in Silo.query.filter(Silo.silo_number.in_(numbers)).all()]

def _sensor_rows_for_cables_window_from_readings(cable_ids, start, end):
    sensors = _sensors_for_cables(cable_ids)
    if not sensors:
        return [], {}
    sensor_ids = [s.id for s in sensors]
    q = _base_readings_query(sensor_ids, start, end)
    rows = q.order_by(Reading.id.asc(), READ_TS_COL.asc(), Reading.sensor_id.asc()).all()
    return rows, _preload_products_from_reading_rows(rows)

def _sensor_rows_for_silos_window_from_readings(silo_ids, start, end):
    sensors = _sensors_for_silos(silo_ids)
    if not sensors:
        return [], {}
    sensor_ids = [s.id for s in sensors]
    q = _base_readings_query(sensor_ids, start, end)
    rows = q.order_by(Reading.id.asc(), READ_TS_COL.asc(), Reading.sensor_id.asc()).all()
    return rows, _preload_products_from_reading_rows(rows)

def _raw_rows_for_silo_ids(silo_ids, start=None, end=None):
    """Fetch raw rows for the silo set (via sensors). No relationships needed on ReadingRaw."""
    if ReadingRaw is None:
        # fallback to readings if raw table not available
        return _sensor_rows_for_silos_window_from_readings(silo_ids, start, end)

    sensors = _sensors_for_silos(silo_ids)
    if not sensors:
        return [], {}, {}
    sensor_ids = [s.id for s in sensors]
    sensor_by_id = {s.id: s for s in sensors}

    q = ReadingRaw.query.filter(ReadingRaw.sensor_id.in_(sensor_ids))
    if start:
        q = q.filter(ReadingRaw.polled_at >= start)
    if end:
        q = q.filter(ReadingRaw.polled_at <= end)

    rows = (q.order_by(ReadingRaw.id.asc(), ReadingRaw.polled_at.asc(), ReadingRaw.sensor_id.asc()).all())

    # products by silo
    silo_ids_set = {sensor_by_id[r.sensor_id].cable.silo_id for r in rows} if rows else set()
    products = _preload_products_for_silo_ids(silo_ids_set)
    return rows, sensor_by_id, products

# -------- Convenience: fetch all silo IDs / numbers --------
def _all_silo_ids():
    return [sid for (sid,) in db.session.query(Silo.id).all()]

def _all_silo_numbers():
    return [num for (num,) in db.session.query(Silo.silo_number).all()]

# ------------------------------------------------
# Format Helpers
# ------------------------------------------------
def format_levels_row(silo, cable_number, timestamp_iso, level_values, product):
    row = OrderedDict()
    row["silo_group"] = silo.group.name if silo.group else None
    row["silo_number"] = silo.silo_number
    row["cable_number"] = cable_number
    states = []

    for lvl in range(8):
        temp = level_values.get(lvl)

        if temp is None:
            # treat missing reading as a DISCONNECT state
            state = "disconnect"
            color = _color_for_state("disconnect")
        else:
            # normal threshold-based coloring
            state, color = get_status_color(temp, product) if product else (None, None)

        row[f"level_{lvl}"] = (round(temp, 2) if temp is not None else None)
        row[f"color_{lvl}"] = _normalize_hex(color)  # <-- normalize before emitting
        states.append(state)

    # ensure 'disconnect' wins over warn/critical/normal
    _, silo_color = worst_state_color(states)
    row["silo_color"] = _normalize_hex(silo_color) or _color_for_state("disconnect") or "#ffffff"
    row["timestamp"] = timestamp_iso
    return row

def format_sensor_row_from_reading(r, product_by_silo):
    sensor = r.sensor
    cable = sensor.cable
    silo = cable.silo
    temp = _temperature_from_any(r)
    product = product_by_silo.get(silo.id)
    state, color = get_status_color(temp, product) if product else (None, None)
    return OrderedDict([
        ("sensor_id", r.sensor_id),
        ("group_id", silo.group.id if silo.group else None),
        ("silo_number", silo.silo_number),
        ("cable_index", cable.cable_index),
        ("level_index", sensor.sensor_index),
        ("state", state),
        ("color", _normalize_hex(color)),  # <-- normalize
        ("temperature", round(temp, 2) if temp is not None else None),
        ("timestamp", _timestamp_iso_from_any(r)),
    ])

def format_sensor_row_from_raw(raw_row, sensor, product_by_silo):
    """raw_row has sensor_id, value_c, polled_at; sensor is a Sensor with relationships."""
    cable = sensor.cable
    silo = cable.silo
    temp = _temperature_from_any(raw_row)
    product = product_by_silo.get(silo.id)
    state, color = get_status_color(temp, product) if product else (None, None)
    return OrderedDict([
        ("sensor_id", raw_row.sensor_id),
        ("group_id", silo.group.id if silo.group else None),
        ("silo_number", silo.silo_number),
        ("cable_index", cable.cable_index),
        ("level_index", sensor.sensor_index),
        ("state", state),
        ("color", _normalize_hex(color)),  # <-- normalize
        ("temperature", round(temp, 2) if temp is not None else None),
        ("timestamp", raw_row.polled_at.isoformat()),
    ])

# ----- Flatten per-cable rows -> one row per silo -----
def _worst_color_from_row(row: dict) -> str:
    _load_status_colors()
    status_to_color = _STATUS_COLOR_CACHE or {}
    color_to_status = {c: s for s, c in status_to_color.items() if c}

    status_rank   = {"disconnect": 4, "critical": 3, "warn": 2, "normal": 1}
    fallback_rank = {"#9e9e9e": 4, "#808080": 4, "#d14141": 3, "#c7c150": 2, "#46d446": 1}

    best_color, best_rank = None, -1
    for k, v in row.items():
        if "_color_" not in k or not isinstance(v, str) or not v:
            continue
        st  = color_to_status.get(_normalize_hex(v))
        rnk = status_rank.get(st, fallback_rank.get(_normalize_hex(v), 0))
        if rnk > best_rank:
            best_rank, best_color = rnk, _normalize_hex(v)

    return best_color or "#ffffff"

def _flatten_rows_per_silo(per_cable_rows):
    """
    Take per-cable rows (one per cable per timestamp) and produce
    one row PER (silo_group, silo_number, timestamp [second precision]).
    Also set silo_color to the worst color present in that row's body.
    """
    _load_status_colors()
    status_to_color = _STATUS_COLOR_CACHE or {}
    color_to_status = {c: s for s, c in status_to_color.items() if c}

    grouped = {}  # (silo_group, silo_number, timestamp_norm) -> row dict (flat)

    for r in per_cable_rows:
        sg = r.get("silo_group")
        sn = r.get("silo_number")
        ts_raw = r.get("timestamp") or ""
        ts = _normalize_ts_for_flatten(ts_raw)   # normalize to second precision
        cn = r.get("cable_number")
        key = (sg, sn, ts)

        g = grouped.get(key)
        if g is None:
            g = {
                "silo_group": sg,
                "silo_number": sn,
                "cable_count": 0,
                "timestamp": ts,   # normalized timestamp
            }
            grouped[key] = g

        # track max cable index seen (0-based)
        if isinstance(cn, int):
            g["cable_count"] = max(g["cable_count"], cn + 1)

        # copy levels/colors from this cable row into the flat row
        for lvl in range(8):
            lv_key = f"level_{lvl}"
            cl_key = f"color_{lvl}"
            g[f"cable_{cn}_{lv_key}"] = r.get(lv_key)
            g[f"cable_{cn}_{cl_key}"] = _normalize_hex(r.get(cl_key))

    # finalize: compute worst color in each flat row
    out = []
    for g in grouped.values():
        worst_color = _worst_color_from_row(g)
        g["silo_color"] = worst_color
        out.append(g)

    # stable sort: by silo_number then timestamp (oldest->newest)
    out.sort(key=lambda d: (d["silo_number"], d["timestamp"] or ""))
    return out

# ------------------------------------------------
# Cable-row helpers (for /readings/by-cable*)
# ------------------------------------------------
STATUS_RANK = {"normal": 0, "warn": 1, "critical": 2}

def _init_cable_row(cable: Cable, silo: Silo, ts_iso: str) -> OrderedDict:
    row = OrderedDict([
        ("silo_group",  silo.group.name if silo.group else None),
        ("silo_number", silo.silo_number),
        ("cable_number", cable.cable_index),
        ("silo_color",  "#ffffff"),
        ("timestamp",   ts_iso),
    ])
    row["_worst_rank"] = -1
    row["_levels"] = {}  # level_index -> (temp, color)
    return row

def _finalize_cable_row(row: OrderedDict) -> OrderedDict:
    out = OrderedDict([
        ("silo_group",  row["silo_group"]),
        ("silo_number", row["silo_number"]),
        ("cable_number", row["cable_number"]),
    ])
    for lvl in sorted(row["_levels"].keys()):
        temp, color = row["_levels"][lvl]
        out[f"level_{lvl}"] = temp
        out[f"color_{lvl}"] = _normalize_hex(color)  # normalize on emit
    out["silo_color"] = _normalize_hex(row["silo_color"])
    out["timestamp"]  = row["timestamp"]
    return out

# ------------------------------------------------
# Math Helpers
# ------------------------------------------------
def avg_ignore_none(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None

def level_lists_to_avg(level_lists):
    return {lvl: avg_ignore_none(lst) for lvl, lst in level_lists.items()}

def level_pick_max(level_lists):
    return {lvl: (round(max(v), 2) if (v := [x for x in lst if x is not None]) else None)
            for lvl, lst in level_lists.items()}

def _iso_day_anchor(day_str: str) -> str:
    return f"{day_str}T00:00:00"

def _levels_from_mask_or_index(level_mask, level_index):
    """Return list of level indices (0..7) from mask, else single index, else []."""
    if level_mask:
        out = []
        for i in range(64):  # allow big masks, but we'll clamp later
            if (level_mask >> i) & 1:
                out.append(i)
        return out
    if level_index is not None:
        return [level_index]
    return []

def _status_color_for_limit(limit_type: str) -> str:
    """
    Map alert.limit_type to a hex color using StatusColor table when possible.
    Fallbacks: critical=#d14141, warn=#c7c150, disconnect=#9e9e9e, else #ffffff.
    """
    _load_status_colors()  # builds mapping like {"normal":"#..","warn":"#..","critical":"#.."}
    table = _STATUS_COLOR_CACHE or {}
    if limit_type == "critical":
        return table.get("critical") or "#d14141"
    if limit_type == "warn":
        return table.get("warn") or "#c7c150"
    if limit_type == "disconnect":
        # could also add a row in StatusColor named "disconnect"; until then:
        return table.get("disconnect") or "#9e9e9e"
    return "#ffffff"

# ============================
# Fill Level Estimation (k=2)
# ============================

def _is_valid_temp_for_avg(t):
    """Valid sample for averaging (exclude disconnects/sentinels/None)."""
    if t is None:
        return False
    try:
        if isinstance(t, (int, float)):
            if math.isnan(t) or math.isinf(t):
                return False
            if float(t) in DISCONNECT_SENTINELS:
                return False
            return True
    except Exception:
        pass
    return False

def _kmeans2_1d(values, max_iter=50):
    """
    Tiny 1-D k-means for k=2.
    values: list[float] (already filtered)
    Returns: labels (0/1 for each value), centroids (c0, c1)
    """
    if not values or len(values) == 1:
        # trivial case: everything same cluster
        c = values[0] if values else None
        return [0] * len(values), (c, c)

    vmin, vmax = min(values), max(values)
    c0, c1 = vmin, vmax  # init at extremes

    for _ in range(max_iter):
        # assign
        grp0, grp1 = [], []
        for v in values:
            if abs(v - c0) <= abs(v - c1):
                grp0.append(v)
            else:
                grp1.append(v)

        # recompute
        new_c0 = sum(grp0) / len(grp0) if grp0 else c0
        new_c1 = sum(grp1) / len(grp1) if grp1 else c1

        # convergence
        if abs(new_c0 - c0) < 1e-9 and abs(new_c1 - c1) < 1e-9:
            break
        c0, c1 = new_c0, new_c1

    # final assignment to produce labels
    labels = []
    for v in values:
        labels.append(0 if abs(v - c0) <= abs(v - c1) else 1)
    return labels, (c0, c1)

def _latest_whole_silo_profile(silo_ids, start=None, end=None):
    """
    Build one temperature profile per silo (levels 0..7) using the latest RAW snapshot:
      - choose the most recent polled_at per silo (like /avg/latest/by-silo-id)
      - average across all cables for each level (ignore disconnect/Nones)
    Returns: list of dicts [{"silo": Silo, "timestamp": iso, "levels": {lvl->avg or None}, "product": Product}, ...]
    """
    rows, sensor_by_id, products = _raw_rows_for_silo_ids(silo_ids, start, end)
    if not rows:
        return []

    # latest timestamp per silo
    latest_ts = {}
    for r in rows:
        silo_id = sensor_by_id[r.sensor_id].cable.silo_id
        ts = r.polled_at
        if ts and ts > latest_ts.get(silo_id, datetime.min):
            latest_ts[silo_id] = ts

    # collect values for each level at that ts (average across all cables)
    level_lists_by_silo = {}    # sid -> {lvl: [temps]}
    for r in rows:
        s = sensor_by_id[r.sensor_id]
        sid = s.cable.silo_id
        ts = r.polled_at
        if ts != latest_ts.get(sid):
            continue
        lvl = s.sensor_index
        t = _temperature_from_any(r)
        if _is_valid_temp_for_avg(t):
            level_lists_by_silo.setdefault(sid, defaultdict(list))
            level_lists_by_silo[sid][lvl].append(float(t))

    # load Silo objects (labels)
    silos = (Silo.query.filter(Silo.id.in_(list(latest_ts.keys())))
             .options(selectinload(Silo.group)).all())
    silo_by_id = {s.id: s for s in silos}

    out = []
    for sid, ts in latest_ts.items():
        # build 0..7 with average of lists
        lv_lists = level_lists_by_silo.get(sid, {})
        levels_avg = {}
        for lvl in range(8):
            lst = lv_lists.get(lvl, [])
            levels_avg[lvl] = (round(sum(lst) / len(lst), 2) if lst else None)
        out.append({
            "silo": silo_by_id.get(sid),
            "timestamp": ts.isoformat(),
            "levels": levels_avg,
            "product": products.get(sid)
        })
    # stable order
    out.sort(key=lambda d: d["silo"].silo_number if d["silo"] else 0)
    return out

def _estimate_fill_from_profile(levels_dict):
    """
    Given {0..7 -> temp or None}, run k-means (k=2) on valid temps.
    Decide which cluster is 'material' (higher centroid).
    Compute the topmost level that belongs to 'material'.

    Returns:
      {
        "fill_index_float": float in [0,8]  (0 bottom empty, 8 full to top),
        "fill_percent": 0..100,
        "cluster_means": {"air": c_air, "material": c_mat},
        "assignments": {level_index: "air"/"material"/None},
        "valid_count": int
      }
    """
    # Build ordered list by level index 0..7
    ordered = [(lvl, levels_dict.get(lvl)) for lvl in range(8)]
    valid = [(lvl, float(t)) for (lvl, t) in ordered if _is_valid_temp_for_avg(t)]
    if len(valid) < 2:
        return {
            "fill_index_float": None,
            "fill_percent": None,
            "cluster_means": {"air": None, "material": None},
            "assignments": {lvl: (None if not _is_valid_temp_for_avg(levels_dict.get(lvl)) else "material") for lvl, _ in ordered},
            "valid_count": len(valid)
        }

    vals = [t for _, t in valid]
    labels, cents = _kmeans2_1d(vals)
    c0, c1 = cents
    # Higher centroid = material
    if c0 >= c1:
        mat_label, air_label = 0, 1
        c_mat, c_air = c0, c1
    else:
        mat_label, air_label = 1, 0
        c_mat, c_air = c1, c0

    # Map assignments back to level indices
    assign = {}
    for (i, (lvl, _)) in enumerate(valid):
        assign[lvl] = "material" if labels[i] == mat_label else "air"
    # mark invalid as None
    for lvl, t in ordered:
        if lvl not in assign:
            assign[lvl] = None

    # Determine topmost "material" level.
    # Levels are 0(bottom) .. 7(top). We want the highest index that is material,
    # then add 1 for a float boundary (so if material up to 4, fill ~5/8).
    mat_levels = [lvl for lvl, tag in assign.items() if tag == "material"]
    if not mat_levels:
        fill_idx_f = 0.0
    else:
        top_mat = max(mat_levels)
        # Optional micro interpolation with neighbor if it flips at boundary:
        # Find closest "air" above it to smooth boundary a bit.
        next_air = None
        for lv in range(top_mat + 1, 8):
            if assign.get(lv) == "air":
                next_air = lv
                break
        # Using simple 0.5 interpolation if there is an immediate boundary flip:
        if next_air == top_mat + 1:
            # small blend toward the next level boundary
            fill_idx_f = top_mat + 0.5
        else:
            fill_idx_f = top_mat + 1.0  # filled through that whole level

    fill_idx_f = max(0.0, min(8.0, float(fill_idx_f)))
    fill_percent = round((fill_idx_f / 8.0) * 100.0, 1)

    return {
        "fill_index_float": round(fill_idx_f, 2),
        "fill_percent": fill_percent,
        "cluster_means": {"air": round(c_air, 2), "material": round(c_mat, 2)},
        "assignments": assign,
        "valid_count": len(valid)
    }

# ======================================================
#                      SENSORS
# ======================================================

# --- add somewhere after your other routes (before the __main__ block) ---

@app.post("/login")
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        return json_response({"error": "Username and password required"}, 400)

    user = db.session.query(User).filter_by(username=username).first()
    if not user:
        # do not reveal if user exists
        return json_response({"error": "Invalid credentials"}, 401)

    # users.password_hash exists in your models; verify it
    if not check_password_hash(user.password_hash, password):
        return json_response({"error": "Invalid credentials"}, 401)

    return json_response({
        "message": "Login successful",
        "user": {
            "id": user.id,
            "username": user.username,
            "role": user.role
        }
    })

# -------- ALL (readings) --------
@app.get('/readings/by-sensor')
def readings_by_sensor_all():
    sensor_ids = request.args.getlist('sensor_id', type=int)
    if not sensor_ids:
        return json_response([])
    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))

    q = _base_readings_query(sensor_ids, start, end)
    rows = q.order_by(Reading.id.asc(), READ_TS_COL.asc(), Reading.sensor_id.asc()).all()

    product_by_silo = _preload_products_from_reading_rows(rows)
    out = [format_sensor_row_from_reading(r, product_by_silo) for r in rows]
    return json_response(out)

# -------- LATEST (readings_raw) --------
@app.get('/readings/latest/by-sensor')
def readings_by_sensor_latest():
    sensor_ids = request.args.getlist('sensor_id', type=int)
    if not sensor_ids:
        return json_response([])
    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))

    # Use readings_raw
    if ReadingRaw is None:
        # fallback to readings table latest
        q = _base_readings_query(sensor_ids, start, end)
        rows = q.order_by(READ_TS_COL.desc(), Reading.sensor_id.asc(), Reading.id.desc()).all()
        latest = {}
        for r in rows:
            if r.sensor_id not in latest:
                latest[r.sensor_id] = r
        product_by_silo = _preload_products_from_reading_rows(list(latest.values()))
        out = [format_sensor_row_from_reading(r, product_by_silo) for r in sorted(latest.values(), key=lambda x: x.sensor_id)]
        return json_response(out)

    q = ReadingRaw.query.filter(ReadingRaw.sensor_id.in_(sensor_ids))
    if start:
        q = q.filter(ReadingRaw.polled_at >= start)
    if end:
        q = q.filter(ReadingRaw.polled_at <= end)
    rows = q.order_by(ReadingRaw.polled_at.desc(), ReadingRaw.sensor_id.asc(), ReadingRaw.id.desc()).all()

    latest = {}
    for r in rows:
        if r.sensor_id not in latest:
            latest[r.sensor_id] = r

    sensor_by_id = _preload_sensors(list(latest.keys()))
    silo_ids = {sensor_by_id[sid].cable.silo_id for sid in latest.keys()}
    products = _preload_products_for_silo_ids(silo_ids)

    out = []
    for sid in sorted(latest.keys()):
        raw = latest[sid]
        sensor = sensor_by_id.get(sid)
        if not sensor:
            continue
        out.append(format_sensor_row_from_raw(raw, sensor, products))
    return json_response(out)

# -------- MAX (readings) --------
@app.get('/readings/max/by-sensor')
def readings_by_sensor_max():
    sensor_ids = request.args.getlist('sensor_id', type=int)
    if not sensor_ids:
        return json_response([])
    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))

    q = _base_readings_query(sensor_ids, start, end)
    rows = q.order_by(READ_TS_COL.desc(), Reading.sensor_id.asc()).all()

    best_row_per_day = {}
    best_temp_per_day = {}
    for r in rows:
        ts = _ts_reading(r)
        k = (r.sensor_id, _day_key(ts))
        cur = best_temp_per_day.get(k)
        temp = _temperature_from_any(r)
        if cur is None or (temp is not None and temp > cur):
            best_temp_per_day[k] = temp
            best_row_per_day[k] = r

    chosen = list(best_row_per_day.values())
    chosen.sort(key=lambda r: _day_key(_ts_reading(r)), reverse=True)
    product_by_silo = _preload_products_from_reading_rows(chosen)
    out = [format_sensor_row_from_reading(r, product_by_silo) for r in chosen]
    return json_response(out)

# ======================================================
#                      CABLES
# ======================================================

# -------- ALL (readings) --------
@app.get('/readings/by-cable')
def readings_by_cable_all():
    cable_ids = request.args.getlist('cable_id', type=int)
    if not cable_ids:
        return json_response([])
    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))

    readings, product_by_silo = _sensor_rows_for_cables_window_from_readings(cable_ids, start, end)
    if not readings:
        return json_response([])

    grouped = {}
    for r in readings:
        s = r.sensor; c = s.cable; silo = c.silo
        ts_iso = _timestamp_iso_from_any(r)
        key = (c.id, ts_iso)
        row = grouped.get(key)
        if row is None:
            row = _init_cable_row(c, silo, ts_iso=ts_iso)
            grouped[key] = row

        product = product_by_silo.get(silo.id)
        temp = _temperature_from_any(r)
        state, color = get_status_color(temp, product) if product else (None, "#ffffff")
        rank = STATUS_RANK.get(state, -1)
        if rank > row["_worst_rank"]:
            row["_worst_rank"] = rank
            row["silo_color"] = color
        row["_levels"][s.sensor_index] = (round(temp, 2) if temp is not None else None, color)

    items = sorted(grouped.items(), key=lambda kv: (kv[1]["cable_number"], kv[1]["timestamp"]))
    out = [_finalize_cable_row(row) for _, row in items]
    return json_response(out)

# -------- LATEST (readings_raw) --------
@app.get('/readings/latest/by-cable')
def readings_by_cable_latest():
    cable_ids = request.args.getlist('cable_id', type=int)
    if not cable_ids:
        return json_response([])
    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))

    # get sensors for those cables
    sensors = _sensors_for_cables(cable_ids)
    if not sensors:
        return json_response([])
    sensor_ids = [s.id for s in sensors]
    sensor_by_id = {s.id: s for s in sensors}
    silo_ids = {s.cable.silo_id for s in sensors}

    if ReadingRaw is None:
        # fallback to readings
        rows, products = _sensor_rows_for_cables_window_from_readings(cable_ids, start, end)
        if not rows:
            return json_response([])
        latest_ts = {}
        for r in rows:
            c = r.sensor.cable
            ts = _ts_reading(r)
            if ts and ts > latest_ts.get(c.id, datetime.min):
                latest_ts[c.id] = ts
        per_cable_levels, meta = {}, {}
        products = _preload_products_from_reading_rows(rows)
        for r in rows:
            s = r.sensor; c = s.cable; silo = c.silo
            ts = _ts_reading(r)
            if ts != latest_ts.get(c.id):
                continue
            if c.id not in per_cable_levels:
                per_cable_levels[c.id] = {}
                meta[c.id] = (silo, c.cable_index, ts, products.get(silo.id))
            if s.sensor_index not in per_cable_levels[c.id]:
                per_cable_levels[c.id][s.sensor_index] = _temperature_from_any(r)
        out = []
        for cid, levels in per_cable_levels.items():
            silo, cable_number, ts, product = meta[cid]
            ts_norm = _normalize_ts_for_flatten(ts.isoformat())  # normalize timestamp
            out.append(format_levels_row(silo, cable_number, ts_norm, levels, product))
        out.sort(key=lambda d: (d["silo_number"], d["cable_number"]))
        return json_response(out)

    # raw path
    q = ReadingRaw.query.filter(ReadingRaw.sensor_id.in_(sensor_ids))
    if start:
        q = q.filter(ReadingRaw.polled_at >= start)
    if end:
        q = q.filter(ReadingRaw.polled_at <= end)
    rows = q.order_by(ReadingRaw.polled_at.desc(), ReadingRaw.sensor_id.asc(), ReadingRaw.id.desc()).all()
    if not rows:
        return json_response([])

    latest_ts = {}  # cable_id -> ts
    for r in rows:
        c = sensor_by_id[r.sensor_id].cable
        if r.polled_at > latest_ts.get(c.id, datetime.min):
            latest_ts[c.id] = r.polled_at

    product_by_silo = _preload_products_for_silo_ids(silo_ids)
    per_cable_levels = {}
    per_cable_meta = {}

    for r in rows:
        s = sensor_by_id[r.sensor_id]; c = s.cable; silo = c.silo
        ts = r.polled_at
        if ts != latest_ts.get(c.id):
            continue
        if c.id not in per_cable_levels:
            per_cable_levels[c.id] = {}
            per_cable_meta[c.id] = (silo, c.cable_index, ts, product_by_silo.get(silo.id))
        if s.sensor_index not in per_cable_levels[c.id]:
            per_cable_levels[c.id][s.sensor_index] = _temperature_from_any(r)

    out = []
    for cid, levels in per_cable_levels.items():
        silo, cable_number, ts, product = per_cable_meta[cid]
        ts_norm = _normalize_ts_for_flatten(ts.isoformat())  # normalize timestamp
        out.append(format_levels_row(silo, cable_number, ts_norm, levels, product))
    out.sort(key=lambda d: (d["silo_number"], d["cable_number"]))
    return json_response(out)

# -------- MAX (readings) --------
@app.get('/readings/max/by-cable')
def readings_by_cable_max():
    cable_ids = request.args.getlist('cable_id', type=int)
    if not cable_ids:
        return json_response([])
    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))

    readings, products = _sensor_rows_for_cables_window_from_readings(cable_ids, start, end)
    if not readings:
        return json_response([])

    grouped = {}
    meta = {}
    for r in readings:
        s = r.sensor; c = s.cable; silo = c.silo
        tsv = _ts_reading(r)
        d = _day_key(tsv)
        key = (c.id, d)
        if key not in grouped:
            grouped[key] = defaultdict(list)
            meta[key] = (silo, c.cable_index, products.get(silo.id))
        grouped[key][s.sensor_index].append(_temperature_from_any(r))

    out = []
    for key, level_lists in grouped.items():
        silo, cable_number, product = meta[key]
        levels_max = level_pick_max(level_lists)
        out.append(format_levels_row(silo, cable_number, _iso_day_anchor(key[1]), levels_max, product))

    out.sort(key=lambda d: (_parse_iso(d["timestamp"]), d["silo_number"], d["cable_number"]), reverse=True)
    out.sort(key=lambda d: (d["silo_number"], d["cable_number"]))
    out.sort(key=lambda d: _parse_iso(d["timestamp"]), reverse=True)
    return json_response(out)

# ======================================================
#                      SILOS
# ======================================================

def _sensor_rows_for_silos_window(silo_ids, start, end):
    return _sensor_rows_for_silos_window_from_readings(silo_ids, start, end)

def _avg_rows_for_silo_ids(silo_ids, start, end):
    return _sensor_rows_for_silos_window_from_readings(silo_ids, start, end)

# -------- ALL (readings) --------
@app.get('/readings/by-silo-id')
def readings_by_silo_id_all():
    silo_ids = request.args.getlist('silo_id', type=int)
    if not silo_ids:
        silo_ids = _all_silo_ids()
    if not silo_ids:
        return json_response([])

    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))
    readings, products = _sensor_rows_for_silos_window_from_readings(silo_ids, start, end)
    if not readings:
        return json_response([])

    grouped = {}
    meta = {}
    for r in readings:
        s = r.sensor; c = s.cable; silo = c.silo
        ts_iso = _timestamp_iso_from_any(r)
        key = (silo.id, c.id, ts_iso)
        if key not in grouped:
            grouped[key] = {}
            meta[key] = (silo, c.cable_index, products.get(silo.id))
        grouped[key][s.sensor_index] = _temperature_from_any(r)

    out = []
    for key, levels in grouped.items():
        silo, cable_number, product = meta[key]
        out.append(format_levels_row(silo, cable_number, key[2], levels, product))

    out.sort(key=lambda d: (_parse_iso(d["timestamp"]), d["silo_number"], d["cable_number"]))
    return json_response(_flatten_rows_per_silo(out))

# -------- LATEST (readings_raw) --------
@app.get('/readings/latest/by-silo-id')
def readings_by_silo_id_latest():
    silo_ids = request.args.getlist('silo_id', type=int)
    if not silo_ids:
        silo_ids = _all_silo_ids()
    if not silo_ids:
        return json_response([])

    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))
    rows, sensor_by_id, products = _raw_rows_for_silo_ids(silo_ids, start, end)
    if not rows:
        return json_response([])

    # 1) Track the latest timestamp per (silo_id, cable_id)
    latest_ts = {}  # (silo_id, cable_id) -> datetime
    for r in rows:
        s = sensor_by_id[r.sensor_id]; c = s.cable; silo = c.silo
        ts = r.polled_at
        key = (silo.id, c.id)
        if ts and ts > latest_ts.get(key, datetime.min):
            latest_ts[key] = ts

    # 2) Build levels for those latest cable timestamps
    per_key_levels = {}
    meta = {}
    for r in rows:
        s = sensor_by_id[r.sensor_id]; c = s.cable; silo = c.silo
        ts = r.polled_at
        key = (silo.id, c.id)
        if ts != latest_ts.get(key):
            continue
        if key not in per_key_levels:
            per_key_levels[key] = {}
            meta[key] = (silo, c.cable_index, ts, products.get(silo.id))
        if s.sensor_index not in per_key_levels[key]:
            per_key_levels[key][s.sensor_index] = _temperature_from_any(r)

    if not per_key_levels:
        return json_response([])

    # 3) NEW: coalesce all cable rows in a silo to the same "latest second" bucket
    #    This ensures flattening merges cable_0 and cable_1 into one combined row.
    latest_sec_per_silo = {}
    for (silo_id, _cable_id), ts in latest_ts.items():
        # bucket to second precision (strip microseconds)
        sec = ts.replace(microsecond=0)
        if (silo_id not in latest_sec_per_silo) or (sec > latest_sec_per_silo[silo_id]):
            latest_sec_per_silo[silo_id] = sec

    # 4) Emit one per-cable row but **with the silo's latest bucketed timestamp**
    out = []
    for (silo_id, _cable_id), levels in per_key_levels.items():
        silo, cable_number, ts_actual, product = meta[(silo_id, _cable_id)]
        shared_ts = latest_sec_per_silo.get(silo_id, ts_actual.replace(microsecond=0))
        out.append(
            format_levels_row(silo, cable_number, shared_ts.isoformat(), levels, product)
        )

    # 5) Flatten to one row per silo+timestamp (they now share the same ts)
    out.sort(key=lambda d: (d["silo_number"], d["cable_number"]))
    return json_response(_flatten_rows_per_silo(out))

# -------- MAX (readings) --------
@app.get('/readings/max/by-silo-id')
def readings_by_silo_id_max():
    silo_ids = request.args.getlist('silo_id', type=int)
    if not silo_ids:
        silo_ids = _all_silo_ids()
    if not silo_ids:
        return json_response([])

    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))
    readings, products = _sensor_rows_for_silos_window_from_readings(silo_ids, start, end)
    if not readings:
        return json_response([])

    grouped = {}
    meta = {}
    for r in readings:
        s = r.sensor; c = s.cable; silo = c.silo
        tsv = _ts_reading(r)
        d = _day_key(tsv)
        key = (silo.id, c.id, d)
        if key not in grouped:
            grouped[key] = defaultdict(list)
            meta[key] = (silo, c.cable_index, products.get(silo.id))
        grouped[key][s.sensor_index].append(_temperature_from_any(r))

    out = []
    for key, level_lists in grouped.items():
        silo, cable_number, product = meta[key]
        levels_max = level_pick_max(level_lists)
        out.append(format_levels_row(silo, cable_number, _iso_day_anchor(key[2]), levels_max, product))

    out.sort(key=lambda d: (_parse_iso(d["timestamp"]), d["silo_number"], d["cable_number"]), reverse=True)
    out.sort(key=lambda d: (d["silo_number"], d["cable_number"]))
    out.sort(key=lambda d: _parse_iso(d["timestamp"]), reverse=True)
    return json_response(_flatten_rows_per_silo(out))

# -------- by SILO NUMBER wrappers --------
@app.get('/readings/by-silo-number')
def readings_by_silo_number_all():
    numbers = request.args.getlist('silo_number', type=int)
    if not numbers:
        numbers = _all_silo_numbers()
    if not numbers:
        return json_response([])
    silo_ids = _silo_number_to_ids(numbers)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] +
                                  [(k, v) for k, v in request.args.items() if k != 'silo_number']):
        return readings_by_silo_id_all()

@app.get('/readings/latest/by-silo-number')
def readings_by_silo_number_latest():
    numbers = request.args.getlist('silo_number', type=int)
    if not numbers:
        numbers = _all_silo_numbers()
    if not numbers:
        return json_response([])
    silo_ids = _silo_number_to_ids(numbers)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] +
                                  [(k, v) for k, v in request.args.items() if k != 'silo_number']):
        return readings_by_silo_id_latest()

@app.get('/readings/max/by-silo-number')
def readings_by_silo_number_max():
    numbers = request.args.getlist('silo_number', type=int)
    if not numbers:
        numbers = _all_silo_numbers()
    if not numbers:
        return json_response([])
    silo_ids = _silo_number_to_ids(numbers)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] +
                                  [(k, v) for k, v in request.args.items() if k != 'silo_number']):
        return readings_by_silo_id_max()

# ======================================================
#           SILO-AVERAGED (across all cables)
# ======================================================

# -------- ALL (readings) --------
@app.get('/readings/avg/by-silo-id')
def readings_by_silo_id_avg_all():
    silo_ids = request.args.getlist('silo_id', type=int)
    if not silo_ids:
        silo_ids = _all_silo_ids()
    if not silo_ids:
        return json_response([])

    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))

    readings, products = _avg_rows_for_silo_ids(silo_ids, start, end)
    if not readings:
        return json_response([])

    grouped = {}
    for r in readings:
        s = r.sensor; silo = s.cable.silo
        ts_iso = _timestamp_iso_from_any(r)
        key = (silo.id, ts_iso)
        if key not in grouped:
            grouped[key] = defaultdict(list)
        grouped[key][s.sensor_index].append(_temperature_from_any(r))

    out = []
    # reusable silo cache
    silo_by_id = {s.id: s for s in {rr.sensor.cable.silo for rr in readings}}
    for (sid, ts), level_lists in grouped.items():
        silo = silo_by_id[sid]
        product = products.get(sid)
        levels_avg = level_lists_to_avg(level_lists)
        out.append(format_levels_row(silo, None, ts, levels_avg, product))

    out.sort(key=lambda d: (d["silo_number"], _parse_iso(d["timestamp"])))
    return json_response(out)

# -------- LATEST (readings_raw) --------
@app.get('/readings/avg/latest/by-silo-id')
def readings_by_silo_id_avg_latest():
    silo_ids = request.args.getlist('silo_id', type=int)
    if not silo_ids:
        silo_ids = _all_silo_ids()
    if not silo_ids:
        return json_response([])

    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))

    # NEW: let caller choose how to color levels
    color_from = (request.args.get('color_from') or 'avg').lower()
    color_from_max = (color_from == 'max')

    rows, sensor_by_id, products = _raw_rows_for_silo_ids(silo_ids, start, end)
    if not rows:
        return json_response([])

    def second_bucket(dt): return dt.replace(microsecond=0) if dt else None

    latest_second = {}
    for r in rows:
        s = sensor_by_id.get(r.sensor_id)
        if not s or not r.polled_at:
            continue
        sid = s.cable.silo_id
        sec = second_bucket(r.polled_at)
        cur = latest_second.get(sid)
        if (cur is None) or (sec > cur):
            latest_second[sid] = sec

    if not latest_second:
        return json_response([])

    chosen_per_sensor = {}
    for r in rows:
        s = sensor_by_id.get(r.sensor_id)
        if not s or not r.polled_at:
            continue
        sid = s.cable.silo_id
        if second_bucket(r.polled_at) != latest_second.get(sid):
            continue
        key = (sid, r.sensor_id)
        prev = chosen_per_sensor.get(key)
        if (prev is None) or (getattr(r, "id", 0) > getattr(prev, "id", 0)):
            chosen_per_sensor[key] = r

    per_silo_vals = {}  # sid -> {lvl: [temps]}
    for (sid, sensor_id), r in chosen_per_sensor.items():
        s = sensor_by_id[sensor_id]
        lvl = s.sensor_index
        t   = _temperature_from_any(r)
        if _is_disconnect_temp(t):
            continue
        per_silo_vals.setdefault(sid, defaultdict(list))
        per_silo_vals[sid][lvl].append(float(t))

    silos = (Silo.query.filter(Silo.id.in_(list(per_silo_vals.keys())))
             .options(selectinload(Silo.group)).all())
    silo_by_id = {s.id: s for s in silos}

    out = []
    for sid, level_lists in per_silo_vals.items():
        silo    = silo_by_id.get(sid)
        product = products.get(sid)
        ts_iso  = latest_second[sid].isoformat()

        # 1) temperatures to return (averages)
        levels_avg = {}
        # 2) optional set used only to compute colors when color_from=max
        levels_max = {}

        for lvl in range(8):
            vals = level_lists.get(lvl, [])
            levels_avg[lvl] = round(sum(vals)/len(vals), 2) if vals else None
            levels_max[lvl] = (max(vals) if vals else None)

        # Build row from averages first
        row = format_levels_row(silo, None, ts_iso, levels_avg, product)

        if color_from_max:
            # Override per-level colors using the worst contributing cable’s temp
            states = []
            for lvl in range(8):
                tmax = levels_max.get(lvl)
                if tmax is None:
                    # keep whatever disconnect/None the builder put
                    continue
                st, col = get_status_color(tmax, product) if product else (None, None)
                row[f"color_{lvl}"] = col
                states.append(st)
            # Recompute silo_color from updated colors
            row["silo_color"] = _worst_color_from_row(row)

        out.append(row)

    out.sort(key=lambda d: d["silo_number"])
    return json_response(out)

# -------- MAX (readings) --------
@app.get('/readings/avg/max/by-silo-id')
def readings_by_silo_id_avg_max():
    silo_ids = request.args.getlist('silo_id', type=int)
    if not silo_ids:
        silo_ids = _all_silo_ids()
    if not silo_ids:
        return json_response([])

    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))

    readings, products = _avg_rows_for_silo_ids(silo_ids, start, end)
    if not readings:
        return json_response([])

    per_ts_levels = {}
    for r in readings:
        s = r.sensor; silo = s.cable.silo
        ts_iso = _timestamp_iso_from_any(r)
        key = (silo.id, ts_iso)
        if key not in per_ts_levels:
            per_ts_levels[key] = defaultdict(list)
        per_ts_levels[key][s.sensor_index].append(_temperature_from_any(r))

    per_ts_avgs = {}
    for key, level_lists in per_ts_levels.items():
        per_ts_avgs[key] = level_lists_to_avg(level_lists)

    per_day_max_of_avg = {}
    for (sid, ts), avgs in per_ts_avgs.items():
        day = _day_key(_parse_iso(ts))
        dkey = (sid, day)
        if dkey not in per_day_max_of_avg:
            per_day_max_of_avg[dkey] = {}
        for lvl, avg_val in avgs.items():
            if avg_val is None:
                continue
            cur = per_day_max_of_avg[dkey].get(lvl)
            if cur is None or avg_val > cur:
                per_day_max_of_avg[dkey][lvl] = avg_val

    # load silos
    silo_ids_used = {sid for (sid, _) in per_day_max_of_avg.keys()}
    silos = (Silo.query.filter(Silo.id.in_(list(silo_ids_used)))
             .options(selectinload(Silo.group)).all())
    silo_by_id = {s.id: s for s in silos}

    out = []
    for (sid, day), lvl_max_avg in per_day_max_of_avg.items():
        silo = silo_by_id.get(sid)
        product = products.get(sid)
        complete_levels = {lvl: lvl_max_avg.get(lvl) for lvl in range(8)}
        out.append(format_levels_row(silo, None, _iso_day_anchor(day), complete_levels, product))

    out.sort(key=lambda d: (_parse_iso(d["timestamp"]), d["silo_number"]), reverse=True)
    out.sort(key=lambda d: d["silo_number"])
    out.sort(key=lambda d: _parse_iso(d["timestamp"]), reverse=True)
    return json_response(out)

# -------- SILO NUMBER wrappers for AVG --------
@app.get('/readings/avg/by-silo-number')
def readings_by_silo_number_avg_all():
    numbers = request.args.getlist('silo_number', type=int)
    if not numbers:
        numbers = _all_silo_numbers()
    if not numbers:
        return json_response([])
    silo_ids = _silo_number_to_ids(numbers)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] +
                                  [(k, v) for k, v in request.args.items() if k != 'silo_number']):
        return readings_by_silo_id_avg_all()

@app.get('/readings/avg/latest/by-silo-number')
def readings_by_silo_number_avg_latest():
    numbers = request.args.getlist('silo_number', type=int)
    if not numbers:
        numbers = _all_silo_numbers()
    if not numbers:
        return json_response([])
    silo_ids = _silo_number_to_ids(numbers)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] +
                                  [(k, v) for k, v in request.args.items() if k != 'silo_number']):
        return readings_by_silo_id_avg_latest()

@app.get('/readings/avg/max/by-silo-number')
def readings_by_silo_number_avg_max():
    numbers = request.args.getlist('silo_number', type=int)
    if not numbers:
        numbers = _all_silo_numbers()
    if not numbers:
        return json_response([])
    silo_ids = _silo_number_to_ids(numbers)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] +
                                  [(k, v) for k, v in request.args.items() if k != 'silo_number']):
        return readings_by_silo_id_avg_max()

# -------- by SILO GROUP ID (maps to silo IDs) --------
def _silo_group_to_ids(group_ids):
    if not group_ids:
        return []
    silos = Silo.query.filter(Silo.silo_group_id.in_(group_ids)).all()
    return [s.id for s in silos]

@app.get('/readings/by-silo-group-id')
def readings_by_silo_group_id_all():
    group_ids = request.args.getlist('silo_group_id', type=int)
    if not group_ids:
        return json_response([])
    silo_ids = _silo_group_to_ids(group_ids)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] +
                                  [(k, v) for k, v in request.args.items() if k != 'silo_group_id']):
        return readings_by_silo_id_all()

@app.get('/readings/latest/by-silo-group-id')
def readings_by_silo_group_id_latest():
    group_ids = request.args.getlist('silo_group_id', type=int)
    if not group_ids:
        return json_response([])
    silo_ids = _silo_group_to_ids(group_ids)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] +
                                  [(k, v) for k, v in request.args.items() if k != 'silo_group_id']):
        return readings_by_silo_id_latest()

@app.get('/readings/max/by-silo-group-id')
def readings_by_silo_group_id_max():
    group_ids = request.args.getlist('silo_group_id', type=int)
    if not group_ids:
        return json_response([])
    silo_ids = _silo_group_to_ids(group_ids)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] +
                                  [(k, v) for k, v in request.args.items() if k != 'silo_group_id']):
        return readings_by_silo_id_max()

@app.get('/readings/avg/by-silo-group-id')
def readings_by_silo_group_id_avg_all():
    group_ids = request.args.getlist('silo_group_id', type=int)
    if not group_ids:
        return json_response([])
    silo_ids = _silo_group_to_ids(group_ids)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] +
                                  [(k, v) for k, v in request.args.items() if k != 'silo_group_id']):
        return readings_by_silo_id_avg_all()

@app.get('/readings/avg/latest/by-silo-group-id')
def readings_by_silo_group_id_avg_latest():
    group_ids = request.args.getlist('silo_group_id', type=int)
    if not group_ids:
        return json_response([])
    silo_ids = _silo_group_to_ids(group_ids)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] +
                                  [(k, v) for k, v in request.args.items() if k != 'silo_group_id']):
        return readings_by_silo_id_avg_latest()

@app.get('/readings/avg/max/by-silo-group-id')
def readings_by_silo_group_id_avg_max():
    group_ids = request.args.getlist('silo_group_id', type=int)
    if not group_ids:
        return json_response([])
    silo_ids = _silo_group_to_ids(group_ids)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] +
                                  [(k, v) for k, v in request.args.items() if k != 'silo_group_id']):
        return readings_by_silo_id_avg_max()

# -------- Alerts (leave as-is for now) --------

@app.get('/alerts/active')
def alerts_active():
    """
    Return one row per active alert as a SNAPSHOT of the affected silo
    at the alert timestamp. Each level's color is the true color derived
    from its temperature at/just before the alert time (no overrides).
    Silo color is the worst color among those levels.

    Query params (optional):
      window_hours: float/int hours for look-back (default 2)
    """

    def _snapshot_levels_at_ts(silo_id: int, ts_anchor: datetime, window: timedelta):
        """
        For the given silo and anchor time, pick the latest Reading per sensor
        with timestamp <= anchor and >= anchor - window. Then collapse to 8
        levels by taking MAX across cables for the same level index.
        Returns: (silo_obj, {level->temp or None}, product)
        """
        # sensors in silo
        sensors = _sensors_for_silos([silo_id])
        if not sensors:
            return None, {}, None

        silo = sensors[0].cable.silo
        sensor_ids = [s.id for s in sensors]

        # product for thresholds
        products = _preload_products_for_silo_ids({silo_id})
        product = products.get(silo_id)

        # bound window
        start = ts_anchor - window
        end = ts_anchor

        # latest row per sensor_id at/before ts_anchor
        q = _base_readings_query(sensor_ids, start, end).order_by(
            READ_TS_COL.desc(), Reading.sensor_id.asc(), Reading.id.desc()
        )
        rows = q.all()
        if not rows:
            # no data in window -> empty; format_levels_row will mark disconnects
            return silo, {}, product

        latest_per_sensor = {}
        for r in rows:
            sid = r.sensor_id
            # because ordered DESC by time then id, first seen is the latest <= anchor
            if sid not in latest_per_sensor:
                latest_per_sensor[sid] = r

        # collapse across cables by level index using MAX temperature
        level_max = {}  # level_index -> float
        for r in latest_per_sensor.values():
            s = r.sensor
            lvl = s.sensor_index
            temp = _temperature_from_any(r)
            if temp is None:
                continue
            t = float(round(temp, 2))
            cur = level_max.get(lvl)
            if cur is None or t > cur:
                level_max[lvl] = t

        # make sure we have explicit keys for 0..7 (missing -> None)
        complete = {lvl: level_max.get(lvl) for lvl in range(8)}
        return silo, complete, product

    # --- fetch active alerts newest-first by "coalesced" timestamp ---
    window_param = request.args.get("window_hours")
    try:
        window_hours = float(window_param) if window_param is not None else 2.0
    except ValueError:
        window_hours = 2.0
    lookback = timedelta(hours=window_hours)

    coalesced_ts = func.coalesce(Alert.last_seen_at, Alert.first_seen_at)
    alerts = (Alert.query
              .filter(Alert.status == 'active')
              .order_by(coalesced_ts.desc())
              .all())
    if not alerts:
        return json_response([])

    # prefetch silos for labels
    silo_ids = {a.silo_id for a in alerts if a.silo_id is not None}
    silos = (Silo.query
             .filter(Silo.id.in_(list(silo_ids)))
             .options(selectinload(Silo.group))
             .all())
    silo_by_id = {s.id: s for s in silos}

    out = []
    for a in alerts:
        ts_anchor = a.last_seen_at or a.first_seen_at or datetime.utcnow()

        # build snapshot at the alert time
        silo_obj, levels_by_idx, product = _snapshot_levels_at_ts(a.silo_id, ts_anchor, lookback)

        # fall back to preloaded silo (for names) if needed
        silo_for_names = silo_by_id.get(a.silo_id) or silo_obj
        if not silo_for_names:
            # no silo -> skip record (or emit minimal)
            continue

        # use the common formatter so level coloring + disconnect handling is consistent
        row = format_levels_row(
            silo_for_names,
            cable_number=None,
            timestamp_iso=ts_anchor.isoformat(timespec="seconds"),
            level_values=levels_by_idx,
            product=product
        )

        # add alert metadata (do NOT alter colors)
        affected_levels = None
        if a.level_index is not None:
            affected_levels = [int(a.level_index)]
        elif a.level_mask:
            mask = int(a.level_mask)
            affected_levels = [i for i in range(8) if (mask & (1 << i))] or None

        row["alert_type"] = a.limit_type
        row["affected_levels"] = affected_levels
        row["active_since"] = (a.first_seen_at.isoformat(timespec="seconds")
                               if a.first_seen_at else None)

        out.append(row)

    return json_response(out)

@app.get('/silos/level-estimate')
def silos_level_estimate():
    """
    Estimate silo fill level using k-means (k=2) on whole-silo temperature profile.
    - Build a single 8-level profile per silo by averaging all cables per level
      at the latest available RAW timestamp for that silo.
    - Cluster the 8 temps into 'air' vs 'material' (higher-mean cluster = material).
    - Compute a fractional fill index (0..8) and percent.

    Query:
      silo_id (repeatable)  -> specific silos; default = all silos
      window_start / window_end (ISO, optional) -> bound the raw scan window (usually not needed)
      debug=1               -> include cluster assignments and inputs

    Returns: list of rows (one per silo).
    """
    silo_ids = request.args.getlist('silo_id', type=int)
    if not silo_ids:
        silo_ids = _all_silo_ids()
    if not silo_ids:
        return json_response([])

    start = _parse_dt(request.args.get('window_start'))
    end   = _parse_dt(request.args.get('window_end'))
    debug = request.args.get('debug') in ('1', 'true', 'yes')

    profiles = _latest_whole_silo_profile(silo_ids, start, end)
    out = []
    for p in profiles:
        silo = p["silo"]
        if not silo:
            continue
        levels = p["levels"]
        est = _estimate_fill_from_profile(levels)

        row = OrderedDict()
        row["silo_group"] = silo.group.name if silo.group else None
        row["silo_number"] = silo.silo_number
        row["timestamp"] = p["timestamp"]
        # echo the whole-silo averaged profile
        for lvl in range(8):
            row[f"level_{lvl}"] = levels.get(lvl)

        row["fill_index_float"] = est["fill_index_float"]
        row["fill_percent"] = est["fill_percent"]
        row["cluster_means"] = est["cluster_means"]
        if debug:
            row["assignments"] = est["assignments"]
            row["valid_count"] = est["valid_count"]

        out.append(row)

    # sort by silo_number
    out.sort(key=lambda d: d["silo_number"])
    return json_response(out)

# Optional convenience wrapper by silo_number
@app.get('/silos/level-estimate/by-number')
def silos_level_estimate_by_number():
    numbers = request.args.getlist('silo_number', type=int)
    if not numbers:
        numbers = _all_silo_numbers()
    if not numbers:
        return json_response([])
    silo_ids = _silo_number_to_ids(numbers)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] +
                                  [(k, v) for k, v in request.args.items() if k != 'silo_number']):
        return silos_level_estimate()

# ---------- Run ----------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
