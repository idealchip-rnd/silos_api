# ==============================================
# Flask API for Silo Temperature Monitoring
# ==============================================

from flask import Flask, request, Response
from flask_cors import CORS
from models import (
    db, Silo, Cable, Sensor, Reading, Product, StatusColor, SiloProductAssignment, Alert
)
from sqlalchemy.orm import selectinload
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
import os
import json

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

# ------------------------------------------------
# Status / Color Helpers
# ------------------------------------------------
_STATUS_COLOR_CACHE = None

def _load_status_colors():
    global _STATUS_COLOR_CACHE
    if _STATUS_COLOR_CACHE is None:
        _STATUS_COLOR_CACHE = {}
        for row in StatusColor.query.all():
            _STATUS_COLOR_CACHE[row.status] = getattr(row, "color_hex", None)

def get_status_color(temp, product):
    if temp is None or product is None:
        return None, None
    if temp <= product.temp_normal:
        state = "normal"
    elif temp <= product.temp_warn:
        state = "warn"
    else:
        state = "critical"
    _load_status_colors()
    return state, _STATUS_COLOR_CACHE.get(state, "#ffffff")

def worst_state_color(states):
    priority = {"critical": 3, "warn": 2, "normal": 1, None: 0}
    worst = None
    for s in states:
        if priority.get(s, 0) > priority.get(worst, 0):
            worst = s
    _load_status_colors()
    return worst, _STATUS_COLOR_CACHE.get(worst, "#ffffff") if worst else "#ffffff"

# ------------------------------------------------
# DB Query Helpers
# ------------------------------------------------
def _base_readings_query(sensor_ids, start, end):
    q = Reading.query.filter(Reading.sensor_id.in_(sensor_ids))
    if start:
        q = q.filter(Reading.timestamp >= start)
    if end:
        q = q.filter(Reading.timestamp <= end)
    return q.options(
        selectinload(Reading.sensor)
        .selectinload(Sensor.cable)
        .selectinload(Cable.silo)
        .selectinload(Silo.group)
    )

def _preload_products(rows):
    silo_ids = {r.sensor.cable.silo_id for r in rows}
    if not silo_ids:
        return {}
    assigns = (
        SiloProductAssignment.query
        .filter(SiloProductAssignment.silo_id.in_(silo_ids))
        .options(selectinload(SiloProductAssignment.product))
        .all()
    )
    return {a.silo_id: a.product for a in assigns}

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

def _sensor_rows_for_cables_window(cable_ids, start, end):
    sensors = _sensors_for_cables(cable_ids)
    if not sensors:
        return [], {}
    sensor_ids = [s.id for s in sensors]
    q = _base_readings_query(sensor_ids, start, end)
    rows = q.order_by(Reading.timestamp.desc(), Reading.sensor_id.asc()).all()
    return rows, _preload_products(rows)

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
        state, color = get_status_color(temp, product) if temp is not None else (None, None)
        row[f"level_{lvl}"] = temp
        row[f"color_{lvl}"] = color
        states.append(state)
    _, silo_color = worst_state_color(states)
    row["silo_color"] = silo_color
    row["timestamp"] = timestamp_iso
    return row

def format_sensor_row(r, product_by_silo):
    sensor = r.sensor
    cable = sensor.cable
    silo = cable.silo
    product = product_by_silo.get(silo.id)
    state, color = get_status_color(r.temperature, product) if product else (None, None)
    return OrderedDict([
        ("sensor_id", r.sensor_id),
        ("group_id", silo.group.id if silo.group else None),
        ("silo_number", silo.silo_number),
        ("cable_index", cable.cable_index),
        ("level_index", sensor.sensor_index),
        ("state", state),
        ("color", color),
        ("temperature", round(r.temperature, 2) if r.temperature is not None else None),
        ("timestamp", r.timestamp.isoformat()),
    ])

# ------------------------------------------------
# Cable-row helpers (for /readings/by-cable*)
# ------------------------------------------------
STATUS_RANK = {"normal": 0, "warn": 1, "critical": 2}

def _init_cable_row(cable: Cable, silo: Silo, ts_iso: str) -> OrderedDict:
    """
    Working row used while aggregating readings for a given (cable, timestamp).
    Keeps scratch fields (_worst_rank, _levels) until finalized.
    """
    row = OrderedDict([
        ("silo_group",  silo.group.name if silo.group else None),
        ("silo_number", silo.silo_number),
        ("cable_number", cable.cable_index),   # keep using cable_index like your output
        ("silo_color",  "#ffffff"),
        ("timestamp",   ts_iso),
    ])
    row["_worst_rank"] = -1
    row["_levels"] = {}  # level_index -> (temp, color)
    return row

def _finalize_cable_row(row: OrderedDict) -> OrderedDict:
    """
    Produce the response row with levels/colors in ascending level order.
    """
    out = OrderedDict([
        ("silo_group",  row["silo_group"]),
        ("silo_number", row["silo_number"]),
        ("cable_number", row["cable_number"]),
    ])
    # levels in ascending order
    for lvl in sorted(row["_levels"].keys()):
        temp, color = row["_levels"][lvl]
        out[f"level_{lvl}"] = temp
        out[f"color_{lvl}"] = color
    out["silo_color"] = row["silo_color"]
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

# ======================================================
#                      SENSORS
# ======================================================

@app.get('/readings/by-sensor')
def readings_by_sensor_all():
    sensor_ids = request.args.getlist('sensor_id', type=int)
    if not sensor_ids:
        return json_response([])

    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))

    q = _base_readings_query(sensor_ids, start, end)
    rows = q.order_by(Reading.timestamp.desc(), Reading.sensor_id.asc()).all()

    product_by_silo = _preload_products(rows)
    out = [format_sensor_row(r, product_by_silo) for r in rows]
    # sorter for this endpoint (unchanged behavior)
    out.sort(key=lambda d: ( _parse_iso(d["timestamp"]), d["sensor_id"]), reverse=True)
    return json_response(out)

@app.get('/readings/latest/by-sensor')
def readings_by_sensor_latest():
    sensor_ids = request.args.getlist('sensor_id', type=int)
    if not sensor_ids:
        return json_response([])

    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))

    q = _base_readings_query(sensor_ids, start, end)
    rows = q.order_by(Reading.timestamp.desc(), Reading.sensor_id.asc()).all()

    latest = {}
    for r in rows:
        if r.sensor_id not in latest:
            latest[r.sensor_id] = r

    chosen = [latest[sid] for sid in sorted(latest.keys())]  # sensor_id ASC
    product_by_silo = _preload_products(chosen)
    out = [format_sensor_row(r, product_by_silo) for r in chosen]
    return json_response(out)

@app.get('/readings/max/by-sensor')
def readings_by_sensor_max():
    sensor_ids = request.args.getlist('sensor_id', type=int)
    if not sensor_ids:
        return json_response([])

    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))

    q = _base_readings_query(sensor_ids, start, end)
    rows = q.order_by(Reading.timestamp.desc(), Reading.sensor_id.asc()).all()

    best_row_per_day = {}
    best_temp_per_day = {}
    for r in rows:
        k = (r.sensor_id, _day_key(r.timestamp))
        cur = best_temp_per_day.get(k)
        if cur is None or (r.temperature is not None and r.temperature > cur):
            best_temp_per_day[k] = r.temperature
            best_row_per_day[k] = r

    chosen = list(best_row_per_day.values())
    # keep original per-day-blocks idea: day DESC, then group/silo/cable/level/sensor ASC
    chosen.sort(key=lambda r: (
        (r.sensor.cable.silo.group.id if r.sensor.cable.silo.group else 0),
        r.sensor.cable.silo.silo_number,
        r.sensor.cable.cable_index,
        r.sensor.sensor_index,
        r.sensor_id
    ))
    chosen.sort(key=lambda r: _day_key(r.timestamp), reverse=True)

    product_by_silo = _preload_products(chosen)
    out = [format_sensor_row(r, product_by_silo) for r in chosen]
    return json_response(out)

# ======================================================
#                      CABLES  (per cable header)
# ======================================================

@app.get('/readings/by-cable')
def readings_by_cable_all():
    cable_ids = request.args.getlist('cable_id', type=int)
    if not cable_ids:
        return json_response([])

    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))

    # ✅ use the defined helper (fixes NameError)
    readings, product_by_silo = _sensor_rows_for_cables_window(cable_ids, start, end)
    if not readings:
        return json_response([])

    # (cable_id, epoch_seconds) -> working row
    grouped = {}
    for r in readings:
        s = r.sensor
        c = s.cable
        silo = c.silo

        epoch = r.timestamp.timestamp()
        key = (c.id, epoch)

        row = grouped.get(key)
        if row is None:
            row = _init_cable_row(c, silo, ts_iso=r.timestamp.isoformat())
            grouped[key] = row

        product = product_by_silo.get(silo.id)
        state, color = get_status_color(r.temperature, product) if product else (None, "#ffffff")
        rank = STATUS_RANK.get(state, -1)
        if rank > row["_worst_rank"]:
            row["_worst_rank"] = rank
            row["silo_color"] = color

        row["_levels"][s.sensor_index] = (
            round(r.temperature, 2) if r.temperature is not None else None,
            color
        )

    # ✅ EXACT order you asked for:
    # newest timestamp block first, and within each timestamp cable_number ASC (1,2,3…)
    items = sorted(
        grouped.items(),
        key=lambda kv: (-kv[0][1], kv[1]["cable_number"])
    )

    out = [_finalize_cable_row(row) for _, row in items]
    return json_response(out)

@app.get('/readings/latest/by-cable')
def readings_by_cable_latest():
    cable_ids = request.args.getlist('cable_id', type=int)
    if not cable_ids:
        return json_response([])

    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))

    readings, products = _sensor_rows_for_cables_window(cable_ids, start, end)
    if not readings:
        return json_response([])

    # keep latest timestamp per cable, then take the first value per level at that timestamp
    latest_ts = {}  # cable_id -> ts_iso (max)
    for r in readings:
        c = r.sensor.cable
        ts = r.timestamp.isoformat()
        if ts > latest_ts.get(c.id, ""):
            latest_ts[c.id] = ts

    per_cable_levels = {}
    per_cable_meta = {}
    for r in readings:
        s = r.sensor; c = s.cable; silo = c.silo
        ts = r.timestamp.isoformat()
        if ts != latest_ts.get(c.id):
            continue
        key = c.id
        if key not in per_cable_levels:
            per_cable_levels[key] = {}
            per_cable_meta[key] = (silo, c.cable_index, ts, products.get(silo.id))
        # keep first (they're all same timestamp anyway)
        if s.sensor_index not in per_cable_levels[key]:
            per_cable_levels[key][s.sensor_index] = round(r.temperature, 2) if r.temperature is not None else None

    out = []
    for cid, levels in per_cable_levels.items():
        silo, cable_number, ts, product = per_cable_meta[cid]
        out.append(
            format_levels_row(
                silo=silo,
                cable_number=cable_number,
                timestamp_iso=ts,
                level_values=levels,
                product=product
            )
        )

    out.sort(key=lambda d: (d["silo_number"], d["cable_number"]))  # asc
    return json_response(out)

@app.get('/readings/max/by-cable')
def readings_by_cable_max():
    cable_ids = request.args.getlist('cable_id', type=int)
    if not cable_ids:
        return json_response([])

    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))

    readings, products = _sensor_rows_for_cables_window(cable_ids, start, end)
    if not readings:
        return json_response([])

    # group per (cable, day) and pick max per level
    grouped = {}  # (cable_id, day) -> {level: [temps]}
    meta = {}
    for r in readings:
        s = r.sensor; c = s.cable; silo = c.silo
        d = _day_key(r.timestamp)
        key = (c.id, d)
        if key not in grouped:
            grouped[key] = defaultdict(list)
            meta[key] = (silo, c.cable_index, products.get(silo.id))
        grouped[key][s.sensor_index].append(round(r.temperature, 2) if r.temperature is not None else None)

    out = []
    for key, level_lists in grouped.items():
        silo, cable_number, product = meta[key]
        levels_max = level_pick_max(level_lists)
        out.append(
            format_levels_row(
                silo=silo,
                cable_number=cable_number,
                timestamp_iso=_iso_day_anchor(key[1]),
                level_values=levels_max,
                product=product
            )
        )

    # Sort: timestamp DESC, then silo_number, cable_number ASC
    out.sort(key=lambda d: (_parse_iso(d["timestamp"]), d["silo_number"], d["cable_number"]), reverse=True)
    out.sort(key=lambda d: (d["silo_number"], d["cable_number"]))  # secondary asc (stable)
    out.sort(key=lambda d: _parse_iso(d["timestamp"]), reverse=True)  # primary desc
    return json_response(out)

# ======================================================
#                      SILOS (per cable header)
# ======================================================

def _sensor_rows_for_silos_window(silo_ids, start, end):
    sensors = _sensors_for_silos(silo_ids)
    if not sensors:
        return [], {}
    sensor_ids = [s.id for s in sensors]
    q = _base_readings_query(sensor_ids, start, end)
    rows = q.order_by(Reading.timestamp.desc(), Reading.sensor_id.asc()).all()
    return rows, _preload_products(rows)

@app.get('/readings/by-silo-id')
def readings_by_silo_id_all():
    silo_ids = request.args.getlist('silo_id', type=int)
    if not silo_ids:
        silo_ids = _all_silo_ids()          # ← return all
    if not silo_ids:
        return json_response([])

    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))
    readings, products = _sensor_rows_for_silos_window(silo_ids, start, end)
    if not readings:
        return json_response([])

    # group: (silo_id, cable_id, timestamp) -> {level: temp}
    grouped = {}
    meta = {}
    for r in readings:
        s = r.sensor; c = s.cable; silo = c.silo
        ts = r.timestamp.isoformat()
        key = (silo.id, c.id, ts)
        if key not in grouped:
            grouped[key] = {}
            meta[key] = (silo, c.cable_index, products.get(silo.id))
        grouped[key][s.sensor_index] = round(r.temperature, 2) if r.temperature is not None else None

    out = []
    for key, levels in grouped.items():
        silo, cable_number, product = meta[key]
        out.append(
            format_levels_row(
                silo=silo,
                cable_number=cable_number,
                timestamp_iso=key[2],
                level_values=levels,
                product=product
            )
        )

    # Sort: timestamp DESC, silo_number ASC, cable_number ASC
    out.sort(key=lambda d: (_parse_iso(d["timestamp"]), d["silo_number"], d["cable_number"]), reverse=True)
    out.sort(key=lambda d: (d["silo_number"], d["cable_number"]))     # secondary asc (stable)
    out.sort(key=lambda d: _parse_iso(d["timestamp"]), reverse=True)  # primary desc
    return json_response(out)

@app.get('/readings/latest/by-silo-id')
def readings_by_silo_id_latest():
    silo_ids = request.args.getlist('silo_id', type=int)
    if not silo_ids:
        silo_ids = _all_silo_ids()          # ← return all
    if not silo_ids:
        return json_response([])

    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))
    readings, products = _sensor_rows_for_silos_window(silo_ids, start, end)
    if not readings:
        return json_response([])

    latest_ts_per = {}  # (silo_id, cable_id) -> iso
    for r in readings:
        s = r.sensor; c = s.cable; silo = c.silo
        ts = r.timestamp.isoformat()
        key = (silo.id, c.id)
        if ts > latest_ts_per.get(key, ""):
            latest_ts_per[key] = ts

    per_key_levels = {}
    meta = {}
    for r in readings:
        s = r.sensor; c = s.cable; silo = c.silo
        ts = r.timestamp.isoformat()
        key = (silo.id, c.id)
        if ts != latest_ts_per.get(key):
            continue
        if key not in per_key_levels:
            per_key_levels[key] = {}
            meta[key] = (silo, c.cable_index, ts, products.get(silo.id))
        if s.sensor_index not in per_key_levels[key]:
            per_key_levels[key][s.sensor_index] = round(r.temperature, 2) if r.temperature is not None else None

    out = []
    for key, levels in per_key_levels.items():
        silo, cable_number, ts, product = meta[key]
        out.append(
            format_levels_row(
                silo=silo,
                cable_number=cable_number,
                timestamp_iso=ts,
                level_values=levels,
                product=product
            )
        )

    out.sort(key=lambda d: (d["silo_number"], d["cable_number"]))  # asc
    return json_response(out)

@app.get('/readings/max/by-silo-id')
def readings_by_silo_id_max():
    silo_ids = request.args.getlist('silo_id', type=int)
    if not silo_ids:
        silo_ids = _all_silo_ids()          # ← return all
    if not silo_ids:
        return json_response([])

    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))
    readings, products = _sensor_rows_for_silos_window(silo_ids, start, end)
    if not readings:
        return json_response([])

    # ... (rest unchanged)


    # group per (silo, cable, day) -> {level: [temps]} then pick max per level
    grouped = {}
    meta = {}
    for r in readings:
        s = r.sensor; c = s.cable; silo = c.silo
        d = _day_key(r.timestamp)
        key = (silo.id, c.id, d)
        if key not in grouped:
            grouped[key] = defaultdict(list)
            meta[key] = (silo, c.cable_index, products.get(silo.id))
        grouped[key][s.sensor_index].append(round(r.temperature, 2) if r.temperature is not None else None)

    out = []
    for key, level_lists in grouped.items():
        silo, cable_number, product = meta[key]
        levels_max = level_pick_max(level_lists)
        out.append(
            format_levels_row(
                silo=silo,
                cable_number=cable_number,
                timestamp_iso=_iso_day_anchor(key[2]),
                level_values=levels_max,
                product=product
            )
        )

    # Day DESC, then silo_number/cable_number ASC
    out.sort(key=lambda d: (_parse_iso(d["timestamp"]), d["silo_number"], d["cable_number"]), reverse=True)
    out.sort(key=lambda d: (d["silo_number"], d["cable_number"]))
    out.sort(key=lambda d: _parse_iso(d["timestamp"]), reverse=True)
    return json_response(out)

# -------- by SILO NUMBER (maps to IDs) --------

@app.get('/readings/by-silo-number')
def readings_by_silo_number_all():
    numbers = request.args.getlist('silo_number', type=int)
    if not numbers:
        numbers = _all_silo_numbers()       # ← return all
    if not numbers:
        return json_response([])

    silo_ids = _silo_number_to_ids(numbers)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] + [
        (k, v) for k, v in request.args.items() if k != 'silo_number'
    ]):
        return readings_by_silo_id_all()


@app.get('/readings/latest/by-silo-number')
def readings_by_silo_number_latest():
    numbers = request.args.getlist('silo_number', type=int)
    if not numbers:
        numbers = _all_silo_numbers()       # ← return all
    if not numbers:
        return json_response([])

    silo_ids = _silo_number_to_ids(numbers)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] + [
        (k, v) for k, v in request.args.items() if k != 'silo_number'
    ]):
        return readings_by_silo_id_latest()


@app.get('/readings/max/by-silo-number')
def readings_by_silo_number_max():
    numbers = request.args.getlist('silo_number', type=int)
    if not numbers:
        numbers = _all_silo_numbers()       # ← return all
    if not numbers:
        return json_response([])

    silo_ids = _silo_number_to_ids(numbers)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] + [
        (k, v) for k, v in request.args.items() if k != 'silo_number'
    ]):
        return readings_by_silo_id_max()


# ======================================================
#           SILO-AVERAGED (across all cables)
# ======================================================

def _avg_rows_for_silo_ids(silo_ids, start, end):
    readings, products = _sensor_rows_for_silos_window(silo_ids, start, end)
    return readings, products

@app.get('/readings/avg/by-silo-id')
def readings_by_silo_id_avg_all():
    silo_ids = request.args.getlist('silo_id', type=int)
    if not silo_ids:
        silo_ids = _all_silo_ids()          # ← return all
    if not silo_ids:
        return json_response([])

    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))
    readings, products = _avg_rows_for_silo_ids(silo_ids, start, end)
    if not readings:
        return json_response([])

    # group per (silo, timestamp) -> {level: [temps]}
    grouped = {}
    for r in readings:
        s = r.sensor; silo = s.cable.silo
        ts = r.timestamp.isoformat()
        key = (silo.id, ts)
        if key not in grouped:
            grouped[key] = defaultdict(list)
        grouped[key][s.sensor_index].append(round(r.temperature, 2) if r.temperature is not None else None)

    out = []
    for key, level_lists in grouped.items():
        silo = next(s for s in {r.sensor.cable.silo for r in readings} if s.id == key[0])
        product = products.get(silo.id)
        levels_avg = level_lists_to_avg(level_lists)
        out.append(
            format_levels_row(
                silo=silo,
                cable_number=None,
                timestamp_iso=key[1],
                level_values=levels_avg,
                product=product
            )
        )

    # Sort: timestamp DESC, silo_number ASC
    out.sort(key=lambda d: ( _parse_iso(d["timestamp"]), d["silo_number"]), reverse=True)
    out.sort(key=lambda d: d["silo_number"])
    out.sort(key=lambda d: _parse_iso(d["timestamp"]), reverse=True)
    return json_response(out)

@app.get('/readings/avg/latest/by-silo-id')
def readings_by_silo_id_avg_latest():
    silo_ids = request.args.getlist('silo_id', type=int)
    if not silo_ids:
        silo_ids = _all_silo_ids()          # ← return all
    if not silo_ids:
        return json_response([])

    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))
    readings, products = _avg_rows_for_silo_ids(silo_ids, start, end)
    if not readings:
        return json_response([])

    latest_ts = {}  # silo_id -> iso
    for r in readings:
        silo = r.sensor.cable.silo
        ts = r.timestamp.isoformat()
        if ts > latest_ts.get(silo.id, ""):
            latest_ts[silo.id] = ts

    per_silo_levels = {}
    for r in readings:
        s = r.sensor; silo = s.cable.silo
        ts = r.timestamp.isoformat()
        if ts != latest_ts.get(silo.id):
            continue
        key = silo.id
        if key not in per_silo_levels:
            per_silo_levels[key] = defaultdict(list)
        per_silo_levels[key][s.sensor_index].append(round(r.temperature, 2) if r.temperature is not None else None)

    out = []
    # find silo objects once
    silo_by_id = {s.id: s for s in {r.sensor.cable.silo for r in readings}}
    for sid, level_lists in per_silo_levels.items():
        silo = silo_by_id[sid]
        product = products.get(sid)
        levels_avg = level_lists_to_avg(level_lists)
        out.append(
            format_levels_row(
                silo=silo,
                cable_number=None,
                timestamp_iso=latest_ts[sid],
                level_values=levels_avg,
                product=product
            )
        )

    out.sort(key=lambda d: d["silo_number"])  # asc
    return json_response(out)

@app.get('/readings/avg/max/by-silo-id')
def readings_by_silo_id_avg_max():
    silo_ids = request.args.getlist('silo_id', type=int)
    if not silo_ids:
        silo_ids = _all_silo_ids()          # ← return all
    if not silo_ids:
        return json_response([])

    start = _parse_dt(request.args.get('start'))
    end   = _parse_dt(request.args.get('end'))
    readings, products = _avg_rows_for_silo_ids(silo_ids, start, end)
    if not readings:
        return json_response([])

    # Step 1: per (silo, timestamp) collect per-level lists (across cables)
    per_ts_levels = {}  # (sid, ts_iso) -> {level: [temps]}
    for r in readings:
        s = r.sensor; silo = s.cable.silo
        ts = r.timestamp.isoformat()
        key = (silo.id, ts)
        if key not in per_ts_levels:
            per_ts_levels[key] = defaultdict(list)
        per_ts_levels[key][s.sensor_index].append(round(r.temperature, 2) if r.temperature is not None else None)

    # Step 2: compute per-(silo, ts) averages per level
    per_ts_avgs = {}  # (sid, ts_iso) -> {level: avg_or_None}
    for key, level_lists in per_ts_levels.items():
        per_ts_avgs[key] = level_lists_to_avg(level_lists)

    # Step 3: per (silo, day) pick MAX of those averages per level
    per_day_max_of_avg = {}  # (sid, day) -> {level: max_avg_or_None}
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

    # Build rows
    silo_by_id = {s.id: s for s in {r.sensor.cable.silo for r in readings}}
    out = []
    for (sid, day), lvl_max_avg in per_day_max_of_avg.items():
        silo = silo_by_id.get(sid)
        product = products.get(sid)
        # ensure all levels exist (missing -> None)
        complete_levels = {lvl: lvl_max_avg.get(lvl) for lvl in range(8)}
        out.append(
            format_levels_row(
                silo=silo,
                cable_number=None,
                timestamp_iso=_iso_day_anchor(day),
                level_values=complete_levels,
                product=product
            )
        )

    # Sort: timestamp DESC, then silo_number ASC
    out.sort(key=lambda d: (_parse_iso(d["timestamp"]), d["silo_number"]), reverse=True)
    out.sort(key=lambda d: d["silo_number"])
    out.sort(key=lambda d: _parse_iso(d["timestamp"]), reverse=True)
    return json_response(out)


# --------- SILO-NUMBER wrappers for AVG ---------

@app.get('/readings/avg/by-silo-number')
def readings_by_silo_number_avg_all():
    numbers = request.args.getlist('silo_number', type=int)
    if not numbers:
        numbers = _all_silo_numbers()       # ← return all
    if not numbers:
        return json_response([])

    silo_ids = _silo_number_to_ids(numbers)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] + [
        (k, v) for k, v in request.args.items() if k != 'silo_number'
    ]):
        return readings_by_silo_id_avg_all()


@app.get('/readings/avg/latest/by-silo-number')
def readings_by_silo_number_avg_latest():
    numbers = request.args.getlist('silo_number', type=int)
    if not numbers:
        numbers = _all_silo_numbers()       # ← return all
    if not numbers:
        return json_response([])

    silo_ids = _silo_number_to_ids(numbers)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] + [
        (k, v) for k, v in request.args.items() if k != 'silo_number'
    ]):
        return readings_by_silo_id_avg_latest()

@app.get('/readings/avg/max/by-silo-number')
def readings_by_silo_number_avg_max():
    numbers = request.args.getlist('silo_number', type=int)
    if not numbers:
        numbers = _all_silo_numbers()       # ← return all
    if not numbers:
        return json_response([])

    silo_ids = _silo_number_to_ids(numbers)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] + [
        (k, v) for k, v in request.args.items() if k != 'silo_number'
    ]):
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
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] + [
        (k, v) for k, v in request.args.items() if k != 'silo_group_id'
    ]):
        return readings_by_silo_id_all()

@app.get('/readings/latest/by-silo-group-id')
def readings_by_silo_group_id_latest():
    group_ids = request.args.getlist('silo_group_id', type=int)
    if not group_ids:
        return json_response([])
    silo_ids = _silo_group_to_ids(group_ids)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] + [
        (k, v) for k, v in request.args.items() if k != 'silo_group_id'
    ]):
        return readings_by_silo_id_latest()

@app.get('/readings/max/by-silo-group-id')
def readings_by_silo_group_id_max():
    group_ids = request.args.getlist('silo_group_id', type=int)
    if not group_ids:
        return json_response([])
    silo_ids = _silo_group_to_ids(group_ids)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] + [
        (k, v) for k, v in request.args.items() if k != 'silo_group_id'
    ]):
        return readings_by_silo_id_max()

# -------- AVG by SILO GROUP ID (maps to silo IDs) --------

@app.get('/readings/avg/by-silo-group-id')
def readings_by_silo_group_id_avg_all():
    group_ids = request.args.getlist('silo_group_id', type=int)
    if not group_ids:
        return json_response([])
    silo_ids = _silo_group_to_ids(group_ids)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] + [
        (k, v) for k, v in request.args.items() if k != 'silo_group_id'
    ]):
        return readings_by_silo_id_avg_all()

@app.get('/readings/avg/latest/by-silo-group-id')
def readings_by_silo_group_id_avg_latest():
    group_ids = request.args.getlist('silo_group_id', type=int)
    if not group_ids:
        return json_response([])
    silo_ids = _silo_group_to_ids(group_ids)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] + [
        (k, v) for k, v in request.args.items() if k != 'silo_group_id'
    ]):
        return readings_by_silo_id_avg_latest()

@app.get('/readings/avg/max/by-silo-group-id')
def readings_by_silo_group_id_avg_max():
    group_ids = request.args.getlist('silo_group_id', type=int)
    if not group_ids:
        return json_response([])
    silo_ids = _silo_group_to_ids(group_ids)
    with app.test_request_context(query_string=[('silo_id', sid) for sid in silo_ids] + [
        (k, v) for k, v in request.args.items() if k != 'silo_group_id'
    ]):
        return readings_by_silo_id_avg_max()

@app.get('/alerts/active')
def alerts_active():
    """
    FAKE version for frontend testing.
    Returns active alerts with fixed temperature values for each level.
    """
    alerts = (Alert.query
              .filter_by(resolved=False)
              .order_by(Alert.active_since.desc())
              .all())
    if not alerts:
        return json_response([])

    # Preload silo metadata
    silo_ids = list({a.silo_id for a in alerts})
    silos = (Silo.query
             .filter(Silo.id.in_(silo_ids))
             .options(selectinload(Silo.group))
             .all())
    silo_by_id = {s.id: s for s in silos}

    # Preload products for status colors
    assigns = (SiloProductAssignment.query
               .filter(SiloProductAssignment.silo_id.in_(silo_ids))
               .options(selectinload(SiloProductAssignment.product))
               .all())
    product_by_silo = {a.silo_id: a.product for a in assigns}

    # Fixed temperature pattern for testing
    fixed_levels = {
        0: 28.5,
        1: 29.0,
        2: 31.2,
        3: 33.8,
        4: 35.4,
        5: 37.0,
        6: 39.6,
        7: 41.3
    }

    out = []
    for a in alerts:
        silo = silo_by_id.get(a.silo_id)
        if not silo:
            continue

        row = format_levels_row(
            silo=silo,
            cable_number=None,
            timestamp_iso=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            level_values=fixed_levels,
            product=product_by_silo.get(a.silo_id)
        )
        row["alert_type"] = a.alert_type
        row["affected_level"] = a.affected_level
        row["active_since"] = a.active_since.strftime("%Y-%m-%dT%H:%M:%S")
        out.append(row)

    # Sort by severity then recency
    severity_rank = {"critical": 2, "warn": 1}
    out.sort(key=lambda d: (severity_rank.get(d["alert_type"], 0), d["active_since"]), reverse=True)
    return json_response(out)

# ---------- Run ----------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
