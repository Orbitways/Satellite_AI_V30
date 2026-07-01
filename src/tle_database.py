"""
tle_database.py — SQLite store for TLE history and catalog metadata.

Main tables:
- tle_records: historical TLE records
- object_metadata: latest Space-Track/SATCAT object metadata by NORAD ID
- maneuvers / maneuver_predictions: experimental maneuver analysis tables
"""

import sqlite3, os, logging, math
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DB_PATH = os.path.join("data", "tle_database.sqlite")


def get_connection():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


import threading as _threading
_DB_INITIALIZED = False
_DB_INIT_LOCK = _threading.Lock()


def init_db():
    """Create or migrate tables. Thread-safe via lock."""
    global _DB_INITIALIZED
    if _DB_INITIALIZED:
        return
    with _DB_INIT_LOCK:
        if _DB_INITIALIZED:
            return
        conn = get_connection()
        try:
            conn.executescript("""
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS tle_records (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        norad_id    TEXT NOT NULL,
        name        TEXT NOT NULL,
        tle1        TEXT NOT NULL,
        tle2        TEXT NOT NULL,
        epoch       TEXT NOT NULL,
        orbit_class TEXT DEFAULT 'LEO',
        mm          REAL,
        inc         REAL,
        ecc         REAL,
        alt_km      REAL,
        source      TEXT DEFAULT 'spacetrack',
        ingested_at TEXT NOT NULL,
        UNIQUE(norad_id, epoch)
    );
CREATE INDEX IF NOT EXISTS idx_tle_norad ON tle_records(norad_id);
CREATE INDEX IF NOT EXISTS idx_tle_epoch ON tle_records(epoch);

CREATE TABLE IF NOT EXISTS object_metadata (
        norad_id        TEXT PRIMARY KEY,
        object_name     TEXT,
        object_type     TEXT,
        rcs_size        TEXT,
        country         TEXT,
        launch_date     TEXT,
        site            TEXT,
        decay_date      TEXT,
        ops_status_code TEXT,
        orbit_center    TEXT,
        orbit_type      TEXT,
        source          TEXT DEFAULT 'spacetrack_satcat',
        updated_at      TEXT NOT NULL
    );
CREATE INDEX IF NOT EXISTS idx_object_metadata_type ON object_metadata(object_type);
CREATE INDEX IF NOT EXISTS idx_object_metadata_status ON object_metadata(ops_status_code);

CREATE TABLE IF NOT EXISTS maneuvers (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        norad_id        TEXT NOT NULL,
        name            TEXT NOT NULL,
        detected_at     TEXT NOT NULL,
        epoch_before    TEXT,
        epoch_after     TEXT,
        delta_v_ms      REAL,
        residual_norm_m REAL,
        event_type      TEXT DEFAULT 'maneuver',
        severity        TEXT DEFAULT 'NORMAL',
        confirmed       INTEGER DEFAULT 0,
        notes           TEXT
    );
CREATE INDEX IF NOT EXISTS idx_man_norad ON maneuvers(norad_id);

CREATE TABLE IF NOT EXISTS maneuver_predictions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        norad_id        TEXT NOT NULL,
        name            TEXT NOT NULL,
        predicted_at    TEXT NOT NULL,
        predicted_epoch TEXT NOT NULL,
        delta_v_ms_pred REAL,
        confidence      REAL,
        model_version   TEXT DEFAULT 'v1',
        used_in_conj    INTEGER DEFAULT 0,
        notes           TEXT
    );
CREATE INDEX IF NOT EXISTS idx_pred_norad ON maneuver_predictions(norad_id);

CREATE TABLE IF NOT EXISTS db_meta (
        key   TEXT PRIMARY KEY,
        value TEXT
    );
""")
            conn.commit()
        finally:
            conn.close()
        _DB_INITIALIZED = True
        logger.debug(f"DB initialized → {DB_PATH}")


def _coalesce(d: dict, *keys):
    for key in keys:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return None


def _norm_str(value):
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _metadata_tuple(row: dict, now: str, source: str):
    norad = _coalesce(row, "NORAD_CAT_ID", "norad_id", "CATNR", "OBJECT_NUMBER")
    if norad is None:
        return None
    return (
        str(norad).strip(),
        _norm_str(_coalesce(row, "OBJECT_NAME", "SATNAME", "OBJECT_NAME_LONG", "name")),
        _norm_str(_coalesce(row, "OBJECT_TYPE", "object_type")),
        _norm_str(_coalesce(row, "RCS_SIZE", "RCS", "rcs_size")),
        _norm_str(_coalesce(row, "COUNTRY", "country")),
        _norm_str(_coalesce(row, "LAUNCH", "LAUNCH_DATE", "launch_date")),
        _norm_str(_coalesce(row, "SITE", "site")),
        _norm_str(_coalesce(row, "DECAY", "DECAY_DATE", "decay_date")),
        _norm_str(_coalesce(row, "OPS_STATUS_CODE", "ops_status_code")),
        _norm_str(_coalesce(row, "ORBIT_CENTER", "orbit_center")),
        _norm_str(_coalesce(row, "ORBIT_TYPE", "orbit_type")),
        source,
        now,
    )


def ingest_object_metadata(records: list[dict], source: str = "spacetrack_satcat", emit=None) -> dict:
    """Upsert Space-Track/SATCAT metadata by NORAD ID."""
    init_db()
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    upserted = 0
    skipped = 0
    errors = 0
    try:
        for i, row in enumerate(records or []):
            try:
                tup = _metadata_tuple(row, now, source)
                if tup is None:
                    skipped += 1
                    continue
                conn.execute("""
                    INSERT INTO object_metadata
                        (norad_id, object_name, object_type, rcs_size, country,
                         launch_date, site, decay_date, ops_status_code,
                         orbit_center, orbit_type, source, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(norad_id) DO UPDATE SET
                        object_name=excluded.object_name,
                        object_type=excluded.object_type,
                        rcs_size=excluded.rcs_size,
                        country=excluded.country,
                        launch_date=excluded.launch_date,
                        site=excluded.site,
                        decay_date=excluded.decay_date,
                        ops_status_code=excluded.ops_status_code,
                        orbit_center=excluded.orbit_center,
                        orbit_type=excluded.orbit_type,
                        source=excluded.source,
                        updated_at=excluded.updated_at
                """, tup)
                upserted += 1
            except Exception as e:
                errors += 1
                logger.debug(f"Metadata ingest error: {e}")

            if emit and (i % 500 == 0 or i == len(records) - 1):
                try:
                    emit(
                        f"Metadata ingestion {i + 1}/{len(records)}...",
                        78,
                        state="ingesting_metadata",
                        metadata_ingested=i + 1,
                        metadata_expected=len(records),
                    )
                except TypeError:
                    emit(f"Metadata ingestion {i + 1}/{len(records)}...", 78)

        conn.commit()
        conn.execute("INSERT OR REPLACE INTO db_meta VALUES ('last_metadata_ingest', ?)", (now,))
        conn.execute("INSERT OR REPLACE INTO db_meta VALUES ('metadata_records', ?)", (str(conn.execute("SELECT COUNT(*) FROM object_metadata").fetchone()[0]),))
        conn.commit()
    finally:
        conn.close()

    return {"upserted": upserted, "skipped": skipped, "errors": errors, "total": upserted + skipped, "timestamp": now}


def ingest_tles(tles: list, source: str = "spacetrack", emit=None) -> dict:
    """Ingest [(name, tle1, tle2)] into the TLE history table."""
    init_db()
    conn = get_connection()
    added = 0
    skipped = 0
    errors = 0
    now = datetime.now(timezone.utc).isoformat()
    try:
        for i, (name, tle1, tle2) in enumerate(tles):
            try:
                norad = tle1[2:7].strip()
                epoch = _parse_tle_epoch(tle1)
                mm = float(tle2[52:63]) if len(tle2) > 63 else 0
                inc = float(tle2[8:16]) if len(tle2) > 16 else 0
                ecc = float("0." + tle2[26:33]) if len(tle2) > 33 else 0
                alt = _alt_from_mm(mm) if mm > 0 else 0
                oc = "LEO" if alt < 2000 else "MEO" if alt < 20200 else "GEO"

                conn.execute("""
                    INSERT OR IGNORE INTO tle_records
                        (norad_id, name, tle1, tle2, epoch, orbit_class, mm, inc, ecc, alt_km, source, ingested_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, (norad, name.strip(), tle1, tle2, epoch, oc, mm, inc, ecc, round(alt, 1), source, now))

                if conn.execute("SELECT changes()").fetchone()[0] > 0:
                    added += 1
                else:
                    skipped += 1
            except Exception as e:
                errors += 1
                logger.debug(f"Ingest error {name}: {e}")

            if emit and (i % 500 == 0 or i == len(tles) - 1):
                pct = 80 + int(((i + 1) / max(len(tles), 1)) * 15)
                try:
                    emit(f"Ingestion {i + 1}/{len(tles)}...", pct, state="ingesting", ingested=i + 1, expected=len(tles))
                except TypeError:
                    emit(f"Ingestion {i + 1}/{len(tles)}...", pct)

        conn.commit()
        conn.execute("INSERT OR REPLACE INTO db_meta VALUES ('last_ingest', ?)", (now,))
        conn.execute("INSERT OR REPLACE INTO db_meta VALUES ('total_records', ?)", (str(conn.execute("SELECT COUNT(*) FROM tle_records").fetchone()[0]),))
        conn.commit()
    finally:
        conn.close()

    report = {"added": added, "skipped": skipped, "errors": errors, "total": added + skipped, "timestamp": now}
    if emit:
        try:
            emit(f"✓ {added} TLE added, {skipped} already present", 95, state="ingesting", ingested=added + skipped, expected=len(tles), added=added, skipped=skipped, errors=errors)
        except TypeError:
            emit(f"✓ {added} TLE added, {skipped} already present", 95)
    return report


_META_SELECT = """
       m.object_name AS meta_object_name,
       m.object_type AS object_type,
       m.rcs_size AS rcs_size,
       m.country AS country,
       m.launch_date AS launch_date,
       m.site AS site,
       m.decay_date AS decay_date,
       m.ops_status_code AS ops_status_code,
       m.orbit_center AS orbit_center,
       m.orbit_type AS orbit_type,
       m.source AS metadata_source,
       m.updated_at AS metadata_updated_at
"""


def get_stats() -> dict:
    init_db()
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM tle_records").fetchone()[0]
        unique = conn.execute("SELECT COUNT(DISTINCT norad_id) FROM tle_records").fetchone()[0]
        leo = conn.execute("SELECT COUNT(*) FROM tle_records WHERE orbit_class='LEO'").fetchone()[0]
        mans = conn.execute("SELECT COUNT(*) FROM maneuvers").fetchone()[0]
        preds = conn.execute("SELECT COUNT(*) FROM maneuver_predictions").fetchone()[0]
        meta = conn.execute("SELECT COUNT(*) FROM object_metadata").fetchone()[0]
        last = conn.execute("SELECT value FROM db_meta WHERE key='last_ingest'").fetchone()
        last_meta = conn.execute("SELECT value FROM db_meta WHERE key='last_metadata_ingest'").fetchone()
        return {
            "total_records": total,
            "unique_objects": unique,
            "leo_objects": leo,
            "metadata_objects": meta,
            "maneuvers_detected": mans,
            "maneuver_predictions": preds,
            "last_ingest": last[0] if last else None,
            "last_metadata_ingest": last_meta[0] if last_meta else None,
            "db_path": DB_PATH,
            "db_size_mb": round(os.path.getsize(DB_PATH) / 1e6, 2) if os.path.exists(DB_PATH) else 0,
        }
    finally:
        conn.close()


def get_history(norad_id: str, limit: int = 50) -> list:
    init_db()
    conn = get_connection()
    try:
        rows = conn.execute(f"""
            SELECT r.norad_id, r.name, r.tle1, r.tle2, r.epoch, r.alt_km, r.orbit_class, r.mm, r.inc, r.ecc, r.source, r.ingested_at,
                   {_META_SELECT}
            FROM tle_records r
            LEFT JOIN object_metadata m ON r.norad_id = m.norad_id
            WHERE r.norad_id=?
            ORDER BY r.epoch DESC LIMIT ?
        """, (norad_id, limit)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def lookup_tle(q: str, limit: int = 10) -> list:
    """Lookup latest TLE records by NORAD ID or satellite name. Joins metadata."""
    init_db()
    q = (q or "").strip()
    if not q:
        return []
    limit = max(1, min(int(limit), 50))
    conn = get_connection()
    try:
        if q.isdigit():
            rows = conn.execute(f"""
                SELECT r.norad_id, r.name, r.tle1, r.tle2, r.epoch, r.alt_km, r.orbit_class, r.mm, r.inc, r.ecc, r.source, r.ingested_at,
                       {_META_SELECT}
                FROM tle_records r
                LEFT JOIN object_metadata m ON r.norad_id = m.norad_id
                WHERE r.norad_id = ?
                ORDER BY r.epoch DESC
                LIMIT ?
            """, (q, limit)).fetchall()
        else:
            rows = conn.execute(f"""
                SELECT r.norad_id, r.name, r.tle1, r.tle2, r.epoch, r.alt_km, r.orbit_class, r.mm, r.inc, r.ecc, r.source, r.ingested_at,
                       {_META_SELECT}
                FROM tle_records r
                JOIN (
                    SELECT norad_id, MAX(epoch) AS latest_epoch
                    FROM tle_records
                    WHERE UPPER(name) LIKE UPPER(?)
                    GROUP BY norad_id
                    LIMIT ?
                ) latest ON r.norad_id = latest.norad_id AND r.epoch = latest.latest_epoch
                LEFT JOIN object_metadata m ON r.norad_id = m.norad_id
                ORDER BY r.name ASC
            """, (f"%{q}%", limit)).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_latest_tles(limit: int = 30000, orbit_class: str | None = "LEO") -> list:
    """Return latest TLE per NORAD object, joined with metadata."""
    init_db()
    limit = max(1, min(int(limit), 50000))
    conn = get_connection()
    try:
        if orbit_class:
            rows = conn.execute(f"""
                SELECT r.norad_id, r.name, r.tle1, r.tle2, r.epoch,
                       r.alt_km, r.orbit_class, r.mm, r.inc, r.ecc,
                       r.source, r.ingested_at,
                       {_META_SELECT}
                FROM tle_records r
                JOIN (
                    SELECT norad_id, MAX(epoch) AS latest_epoch
                    FROM tle_records
                    WHERE orbit_class = ?
                    GROUP BY norad_id
                    ORDER BY norad_id
                    LIMIT ?
                ) latest ON r.norad_id = latest.norad_id AND r.epoch = latest.latest_epoch
                LEFT JOIN object_metadata m ON r.norad_id = m.norad_id
                ORDER BY r.norad_id
            """, (orbit_class, limit)).fetchall()
        else:
            rows = conn.execute(f"""
                SELECT r.norad_id, r.name, r.tle1, r.tle2, r.epoch,
                       r.alt_km, r.orbit_class, r.mm, r.inc, r.ecc,
                       r.source, r.ingested_at,
                       {_META_SELECT}
                FROM tle_records r
                JOIN (
                    SELECT norad_id, MAX(epoch) AS latest_epoch
                    FROM tle_records
                    GROUP BY norad_id
                    ORDER BY norad_id
                    LIMIT ?
                ) latest ON r.norad_id = latest.norad_id AND r.epoch = latest.latest_epoch
                LEFT JOIN object_metadata m ON r.norad_id = m.norad_id
                ORDER BY r.norad_id
            """, (limit,)).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def store_maneuver(norad_id, name, delta_v_ms, residual_m, epoch_before=None, epoch_after=None, event_type="maneuver", severity="NORMAL"):
    init_db()
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO maneuvers
            (norad_id, name, detected_at, epoch_before, epoch_after,
             delta_v_ms, residual_norm_m, event_type, severity)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (norad_id, name, now, epoch_before, epoch_after, delta_v_ms, residual_m, event_type, severity))
    conn.commit(); conn.close()


def get_maneuvers(days: int = 7, min_dv: float = 0.0) -> list:
    init_db()
    conn = get_connection()
    try:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = conn.execute("""
            SELECT * FROM maneuvers
            WHERE detected_at >= ? AND delta_v_ms >= ?
            ORDER BY delta_v_ms DESC
        """, (cutoff, min_dv)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def store_prediction(norad_id, name, predicted_epoch, delta_v_ms_pred, confidence, model_version="v1", notes=""):
    init_db()
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO maneuver_predictions
            (norad_id, name, predicted_at, predicted_epoch,
             delta_v_ms_pred, confidence, model_version, notes)
        VALUES (?,?,?,?,?,?,?,?)
    """, (norad_id, name, now, predicted_epoch, delta_v_ms_pred, confidence, model_version, notes))
    conn.commit(); conn.close()


def get_predictions(min_confidence: float = 0.3) -> list:
    init_db()
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT * FROM maneuver_predictions
            WHERE confidence >= ?
            ORDER BY confidence DESC, predicted_epoch ASC
        """, (min_confidence,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _parse_tle_epoch(tle1: str) -> str:
    try:
        yr2 = int(tle1[18:20]); doy = float(tle1[20:32])
        year = 2000 + yr2 if yr2 < 57 else 1900 + yr2
        from datetime import timedelta
        dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
        return dt.isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def _alt_from_mm(mm: float) -> float:
    try:
        mu = 398600.4418
        T = 86400.0 / mm
        a = (mu * (T / (2 * math.pi)) ** 2) ** (1 / 3)
        return a - 6378.137
    except Exception:
        return 0.0
