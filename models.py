from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.sql import func
from sqlalchemy import text, Index, ForeignKey
from sqlalchemy.dialects.mysql import BIGINT, INTEGER, DATETIME as MYSQL_DATETIME, DECIMAL as MYSQL_DECIMAL
from sqlalchemy.dialects.mysql import BIGINT as MyBigInt
from sqlalchemy import Enum as SAEnum
from sqlalchemy import String

db = SQLAlchemy()

# -----------------------------
# Core reference data
# -----------------------------
class SiloGroup(db.Model):
    __tablename__ = 'silo_groups'
    id   = db.Column(INTEGER(unsigned=False), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(50))

    silos = db.relationship('Silo', backref='group', lazy=True)


class Silo(db.Model):
    __tablename__ = 'silos'
    id               = db.Column(INTEGER, primary_key=True)
    silo_number      = db.Column(INTEGER, nullable=False)
    group_silo_index = db.Column(INTEGER)
    silo_group_id    = db.Column(INTEGER, db.ForeignKey('silo_groups.id'), nullable=False)
    cable_count      = db.Column(INTEGER)

    cables             = db.relationship('Cable', backref='silo', lazy=True)
    product_assignment = db.relationship('SiloProductAssignment', backref='silo', uselist=False)


class Slave(db.Model):
    __tablename__ = 'slaves'
    id            = db.Column(INTEGER, primary_key=True)
    slave_id      = db.Column(INTEGER, nullable=False, unique=True)
    channels_used = db.Column(INTEGER, nullable=False)

    cables = db.relationship('Cable', backref='slave', lazy=True)


class Cable(db.Model):
    __tablename__ = 'cables'
    id          = db.Column(INTEGER, primary_key=True)
    silo_id     = db.Column(INTEGER, db.ForeignKey('silos.id'), nullable=False)
    cable_index = db.Column(INTEGER)
    slave_id    = db.Column(INTEGER, db.ForeignKey('slaves.id'))
    channel     = db.Column(INTEGER)

    sensors = db.relationship('Sensor', backref='cable', lazy=True)


class Sensor(db.Model):
    __tablename__ = 'sensors'
    id           = db.Column(INTEGER, primary_key=True)
    cable_id     = db.Column(INTEGER, db.ForeignKey('cables.id'), nullable=False)
    sensor_index = db.Column(INTEGER)

    readings      = db.relationship('Reading', backref='sensor', lazy=True)
    readings_raw  = db.relationship('ReadingRaw', backref='sensor', lazy=True)


# -----------------------------
# Readings (processed & raw)
# -----------------------------
class Reading(db.Model):
    """
    Processed/aggregated readings table (matches your dump):
      - hour_start: start of the hour bucket (DATETIME)
      - sample_at: precise time the sample was taken (DATETIME(6))
      - value_c: temperature value in °C (DECIMAL)
    """
    __tablename__ = 'readings'

    id         = db.Column(BIGINT(unsigned=False), primary_key=True)
    sensor_id  = db.Column(INTEGER, db.ForeignKey('sensors.id'), nullable=False)
    hour_start = db.Column(MYSQL_DATETIME(fsp=0), nullable=False)     # DATETIME
    value_c    = db.Column(MYSQL_DECIMAL(6, 2), nullable=True)        # adjust precision/scale if needed
    sample_at  = db.Column(MYSQL_DATETIME(fsp=6), nullable=False)     # DATETIME(6)

    __table_args__ = (
        # helpful indexes for time-window queries
        Index('ix_readings_sensor_hour',  'sensor_id', 'hour_start'),
        Index('ix_readings_sensor_sample', 'sensor_id', 'sample_at'),
    )


class ReadingRaw(db.Model):
    """
    Raw polled readings table (matches your dump):
      - polled_at: raw poll timestamp (DATETIME(6))
      - value_c: temperature
      - poll_run_id: batch/run id (nullable)
    """
    __tablename__ = 'readings_raw'

    id          = db.Column(BIGINT(unsigned=False), primary_key=True)
    sensor_id   = db.Column(INTEGER, db.ForeignKey('sensors.id'), nullable=False)
    value_c     = db.Column(String(10), nullable=True)
    polled_at   = db.Column(MYSQL_DATETIME(fsp=6), nullable=False)    # DATETIME(6)
    poll_run_id = db.Column(BIGINT(unsigned=False), nullable=True)    # no FK if table unknown

    __table_args__ = (
        Index('ix_readings_raw_sensor_polled', 'sensor_id', 'polled_at'),
    )


# -----------------------------
# Products & thresholds
# -----------------------------
class Product(db.Model):
    __tablename__ = 'products'
    id            = db.Column(INTEGER, primary_key=True)
    name          = db.Column(db.String(64), nullable=False)
    temp_normal   = db.Column(db.Float)
    temp_warn     = db.Column(db.Float)
    temp_critical = db.Column(db.Float)

    assignments = db.relationship('SiloProductAssignment', backref='product', lazy=True)


class SiloProductAssignment(db.Model):
    __tablename__ = 'silo_product_assignments'
    silo_id     = db.Column(INTEGER, db.ForeignKey('silos.id'), primary_key=True)
    product_id  = db.Column(INTEGER, db.ForeignKey('products.id'))
    assigned_at = db.Column(db.DateTime, server_default=func.now())


class StatusColor(db.Model):
    __tablename__ = 'status_colors'
    id        = db.Column(INTEGER, primary_key=True)
    status    = db.Column(db.String(32), unique=True, nullable=False)  # 'normal','warn','critical'
    color_hex = db.Column(db.String(7), nullable=False)
    priority  = db.Column(INTEGER)


# -----------------------------
# Alerts & events
# -----------------------------
class Alert(db.Model):
    __tablename__ = 'alerts'

    id             = db.Column(db.Integer, primary_key=True)
    sensor_id      = db.Column(db.Integer)
    silo_id        = db.Column(db.Integer, db.ForeignKey('silos.id'), nullable=False)

    # either/or depending on how you mark affected sensors
    level_index    = db.Column(db.Integer)
    level_mask     = db.Column(MyBigInt(unsigned=False))   # matches BIGINT in MySQL

    # enums shown in your screenshot
    limit_type     = db.Column(db.Enum('warn', 'critical', 'disconnect', name='limit_type_enum'),
                               nullable=True)
    threshold_c    = db.Column(db.Numeric(5, 2), nullable=True)

    n_consecutive  = db.Column(db.Integer)
    m_consecutive  = db.Column(db.Integer)

    first_seen_at  = db.Column(db.DateTime)
    last_seen_at   = db.Column(db.DateTime)
    last_value_c   = db.Column(db.Numeric(6, 2))
    cleared_at     = db.Column(db.DateTime)
    ok_count       = db.Column(db.Integer)
    value_c        = db.Column(db.Numeric(5, 2))

    status         = db.Column(db.Enum('active', 'cleared', name='status_enum'))

    silo = db.relationship('Silo', backref=db.backref('alerts', lazy=True))

    # --------- Backward‑compat convenience aliases ----------
    @property
    def alert_type(self):
        # old name used in some code paths → map to new one
        return self.limit_type

    @property
    def affected_level(self):
        return self.level_index

    @property
    def active_since(self):
        # pick the earliest time we saw it; fall back if needed
        return self.first_seen_at or self.last_seen_at

    @property
    def resolved(self):
        # legacy boolean equivalent
        return (self.status == 'cleared') or (self.cleared_at is not None)

# -----------------------------
# Users
# -----------------------------
class User(db.Model):
    __tablename__ = 'users'

    id            = db.Column(INTEGER, primary_key=True)
    username      = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role          = db.Column(db.String(50), nullable=False, default='user')
    created_at    = db.Column(db.DateTime, server_default=func.now())

    VALID_ROLES = ['admin', 'technician', 'operator']

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def set_role(self, role: str):
        if role not in self.VALID_ROLES:
            raise ValueError("Invalid role")
        self.role = role
