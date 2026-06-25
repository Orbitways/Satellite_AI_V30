"""
tle_database.py — Base de données TLE SQLite pour tous les objets LEO.

Stocke l'historique TLE, permet la recherche et l'export pour l'entraînement IA.
Table principale : tle_records
Table des manœuvres détectées : maneuvers
Table des prédictions de manœuvres : maneuver_predictions
"""

import sqlite3, os, json, logging, math
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join("data", "tle_database.sqlite")


def get_connection():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH,
                           timeout=30,
                           check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


import threading as _threading
_DB_INITIALIZED = False
_DB_INIT_LOCK   = _threading.Lock()

def init_db():
    """Créer les tables si elles n'existent pas. Thread-safe via Lock."""
    global _DB_INITIALIZED
    if _DB_INITIALIZED:          # fast-path sans lock
        return
    with _DB_INIT_LOCK:          # double-checked locking
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
        mm          REAL,   -- mean motion rev/day
        inc         REAL,   -- inclination deg
        ecc         REAL,   -- eccentricity
        alt_km      REAL,   -- altitude approx km
        source      TEXT DEFAULT 'spacetrack',
        ingested_at TEXT NOT NULL,
        UNIQUE(norad_id, epoch)
    );
    CREATE INDEX IF NOT EXISTS idx_tle_norad ON tle_records(norad_id);
    CREATE INDEX IF NOT EXISTS idx_tle_epoch ON tle_records(epoch);

    CREATE TABLE IF NOT EXISTS maneuvers (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        norad_id        TEXT NOT NULL,
        name            TEXT NOT NULL,
        detected_at     TEXT NOT NULL,
        epoch_before    TEXT,
        epoch_after     TEXT,
        delta_v_ms      REAL,        -- ΔV estimé en m/s
        residual_norm_m REAL,        -- résidu différentiel en m
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
        predicted_epoch TEXT NOT NULL,  -- quand la manœuvre est prévue
        delta_v_ms_pred REAL,           -- ΔV prédit en m/s
        confidence      REAL,           -- score [0-1]
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
        logger.debug(f"DB initialisée → {DB_PATH}")


def ingest_tles(tles: list, source: str = "spacetrack", emit=None) -> dict:
    """
    Ingère une liste de TLE [(name, tle1, tle2)] dans la base.
    Ignore les doublons (norad_id + epoch déjà présents).
    Retourne un rapport.
    """
    init_db()
    conn = get_connection()
    added = 0; skipped = 0; errors = 0
    now = datetime.now(timezone.utc).isoformat()
    try:
        for i, (name, tle1, tle2) in enumerate(tles):
            try:
                norad  = tle1[2:7].strip()
                epoch  = _parse_tle_epoch(tle1)
                mm     = float(tle2[52:63]) if len(tle2) > 63 else 0
                inc    = float(tle2[8:16])  if len(tle2) > 16 else 0
                ecc    = float("0."+tle2[26:33]) if len(tle2) > 33 else 0
                alt    = _alt_from_mm(mm) if mm > 0 else 0
                oc     = "LEO" if alt < 2000 else "MEO" if alt < 20200 else "GEO"

                conn.execute("""
                    INSERT OR IGNORE INTO tle_records
                        (norad_id, name, tle1, tle2, epoch, orbit_class, mm, inc, ecc, alt_km, source, ingested_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, (norad, name.strip(), tle1, tle2, epoch, oc, mm, inc, ecc, round(alt,1), source, now))

                if conn.execute("SELECT changes()").fetchone()[0] > 0:
                    added += 1
                else:
                    skipped += 1

            except Exception as e:
                errors += 1
                logger.debug(f"Ingest erreur {name}: {e}")

            if emit and (i % 500 == 0 or i == len(tles) - 1):
                pct = 80 + int(((i + 1) / max(len(tles), 1)) * 15)
                msg = f"Ingestion {i + 1}/{len(tles)}..."

                try:
                    emit(
                        msg,
                        pct,
                        state="ingesting",
                        ingested=i + 1,
                        expected=len(tles),
                    )
                except TypeError:
                    emit(msg, pct)

        conn.commit()

        # Mettre à jour la meta
        conn.execute("INSERT OR REPLACE INTO db_meta VALUES ('last_ingest', ?)", (now,))
        conn.execute("INSERT OR REPLACE INTO db_meta VALUES ('total_records', ?)",
                     (str(conn.execute("SELECT COUNT(*) FROM tle_records").fetchone()[0]),))
        conn.commit()
    finally:
        conn.close()

    report = {
        "added": added, "skipped": skipped, "errors": errors,
        "total": added + skipped,
        "timestamp": now,
    }
    if emit:
        msg = f"✓ {added} TLE ajoutés, {skipped} déjà présents"

        try:
            emit(
                msg,
                95,
                state="ingesting",
                ingested=added + skipped,
                expected=len(tles),
                added=added,
                skipped=skipped,
                errors=errors,
            )
        except TypeError:
            emit(msg, 95)
    return report


def get_stats() -> dict:
    """Statistiques de la base."""
    init_db()
    conn = get_connection()
    try:
        total   = conn.execute("SELECT COUNT(*) FROM tle_records").fetchone()[0]
        unique  = conn.execute("SELECT COUNT(DISTINCT norad_id) FROM tle_records").fetchone()[0]
        leo     = conn.execute("SELECT COUNT(*) FROM tle_records WHERE orbit_class='LEO'").fetchone()[0]
        mans    = conn.execute("SELECT COUNT(*) FROM maneuvers").fetchone()[0]
        preds   = conn.execute("SELECT COUNT(*) FROM maneuver_predictions").fetchone()[0]
        last    = conn.execute("SELECT value FROM db_meta WHERE key='last_ingest'").fetchone()
        return {
            "total_records": total, "unique_objects": unique,
            "leo_objects": leo, "maneuvers_detected": mans,
            "maneuver_predictions": preds,
            "last_ingest": last[0] if last else None,
            "db_path": DB_PATH,
            "db_size_mb": round(os.path.getsize(DB_PATH)/1e6, 2) if os.path.exists(DB_PATH) else 0,
        }
    finally:
        conn.close()


def get_history(norad_id: str, limit: int = 50) -> list:
    """Retourne l'historique TLE d'un satellite."""
    init_db()
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT norad_id, name, tle1, tle2, epoch, alt_km, orbit_class
            FROM tle_records WHERE norad_id=?
            ORDER BY epoch DESC LIMIT ?
        """, (norad_id, limit)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

def lookup_tle(q: str, limit: int = 10) -> list:
    """
    Lookup latest TLE records by NORAD ID or satellite name.

    Returns one latest TLE per matched NORAD object.
    """
    init_db()

    q = (q or "").strip()
    if not q:
        return []

    limit = max(1, min(int(limit), 50))

    conn = get_connection()
    try:
        if q.isdigit():
            rows = conn.execute(
                """
                SELECT norad_id, name, tle1, tle2, epoch, alt_km, orbit_class
                FROM tle_records
                WHERE norad_id = ?
                ORDER BY epoch DESC
                LIMIT ?
                """,
                (q, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT r.norad_id, r.name, r.tle1, r.tle2, r.epoch, r.alt_km, r.orbit_class
                FROM tle_records r
                JOIN (
                    SELECT norad_id, MAX(epoch) AS latest_epoch
                    FROM tle_records
                    WHERE UPPER(name) LIKE UPPER(?)
                    GROUP BY norad_id
                    LIMIT ?
                ) latest
                ON r.norad_id = latest.norad_id
                AND r.epoch = latest.latest_epoch
                ORDER BY r.name ASC
                """,
                (f"%{q}%", limit),
            ).fetchall()

        return [dict(row) for row in rows]
    finally:
        conn.close()

def store_maneuver(norad_id, name, delta_v_ms, residual_m,
                   epoch_before=None, epoch_after=None,
                   event_type="maneuver", severity="NORMAL"):
    """Enregistre une manœuvre détectée."""
    init_db()
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO maneuvers
            (norad_id, name, detected_at, epoch_before, epoch_after,
             delta_v_ms, residual_norm_m, event_type, severity)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (norad_id, name, now, epoch_before, epoch_after,
          delta_v_ms, residual_m, event_type, severity))
    conn.commit(); conn.close()


def get_maneuvers(days: int = 7, min_dv: float = 0.0) -> list:
    """Retourne les manœuvres détectées récentes."""
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


def store_prediction(norad_id, name, predicted_epoch, delta_v_ms_pred,
                     confidence, model_version="v1", notes=""):
    """Enregistre une prédiction de manœuvre."""
    init_db()
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO maneuver_predictions
            (norad_id, name, predicted_at, predicted_epoch,
             delta_v_ms_pred, confidence, model_version, notes)
        VALUES (?,?,?,?,?,?,?,?)
    """, (norad_id, name, now, predicted_epoch,
          delta_v_ms_pred, confidence, model_version, notes))
    conn.commit(); conn.close()


def get_predictions(min_confidence: float = 0.3) -> list:
    """Retourne les prédictions de manœuvres."""
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
    """Extrait l'époque ISO depuis la ligne 1 TLE."""
    try:
        yr2 = int(tle1[18:20]); doy = float(tle1[20:32])
        year = 2000 + yr2 if yr2 < 57 else 1900 + yr2
        from datetime import timedelta
        dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
        return dt.isoformat()
    except:
        return datetime.now(timezone.utc).isoformat()


def _alt_from_mm(mm: float) -> float:
    """Altitude approximative depuis le mouvement moyen (rev/jour)."""
    try:
        mu = 398600.4418
        T  = 86400.0 / mm
        a  = (mu * (T / (2 * math.pi))**2) ** (1/3)
        return a - 6378.137
    except:
        return 0.0
