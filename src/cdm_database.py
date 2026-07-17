"""Storage and historical analysis for imported CDM records."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from tle_database import get_connection, init_db

OBJECT_CLASSES = ("debris", "inactive_satellite", "active_satellite", "unknown")
ACTIVE_OPS_STATUS = {"+", "P", "B", "S", "X"}
INACTIVE_OPS_STATUS = {"-", "D"}


def init_cdm_db():
    init_db()
    conn = get_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cdm_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cdm_id TEXT UNIQUE,
                target_norad TEXT NOT NULL,
                target_name TEXT,
                secondary_norad TEXT,
                secondary_name TEXT,
                creation_date TEXT,
                tca TEXT NOT NULL,
                miss_distance_km REAL,
                pc REAL,
                relative_speed_km_s REAL,
                object_type TEXT,
                object_class TEXT DEFAULT 'unknown',
                object_class_source TEXT,
                source TEXT,
                raw_json TEXT,
                imported_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cdm_target ON cdm_records(target_norad);
            CREATE INDEX IF NOT EXISTS idx_cdm_tca ON cdm_records(tca);
            CREATE INDEX IF NOT EXISTS idx_cdm_creation ON cdm_records(creation_date);
            """
        )
        conn.commit()
    finally:
        conn.close()


def _coalesce(row: dict[str, Any], *keys: str):
    lowered = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
        value = lowered.get(key.lower())
        if value not in (None, ""):
            return value
    return None


def _clean(value) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _as_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(str(value).strip().replace(",", ""))
    except Exception:
        return None


def _normalize_norad(value) -> Optional[str]:
    if value in (None, ""):
        return None
    value = str(value).strip()
    try:
        number = float(value)
        if number.is_integer():
            value = str(int(number))
    except Exception:
        pass
    return value.lstrip("0") or "0"


def _parse_dt(value) -> Optional[str]:
    if value in (None, ""):
        return None
    raw = str(value).strip()
    candidates = (raw, raw.replace("Z", "+00:00"))
    for candidate in candidates:
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            continue
    return None


def _parse_iso(value) -> Optional[datetime]:
    normalized = _parse_dt(value)
    if not normalized:
        return None
    return datetime.fromisoformat(normalized)


def _distance_km(row: dict[str, Any]) -> Optional[float]:
    value = _as_float(_coalesce(row, "miss_distance_km", "MISS_DISTANCE_KM", "MISS_DISTANCE", "MIN_RNG", "MINIMUM_RANGE"))
    if value is None:
        return None
    unit = str(_coalesce(row, "MISS_DISTANCE_UNIT", "DISTANCE_UNIT") or "km").lower()
    return value / 1000.0 if unit in {"m", "meter", "meters", "metre", "metres"} else value


def _speed_km_s(row: dict[str, Any]) -> Optional[float]:
    value = _as_float(_coalesce(row, "relative_speed_km_s", "RELATIVE_SPEED_KM_S", "RELATIVE_SPEED", "RELATIVE_VELOCITY"))
    if value is None:
        return None
    unit = str(_coalesce(row, "RELATIVE_SPEED_UNIT", "RELATIVE_VELOCITY_UNIT") or "km/s").lower()
    return value / 1000.0 if unit in {"m/s", "mps"} else value


def _lookup_metadata(conn, norad: Optional[str]) -> Optional[dict[str, Any]]:
    if not norad:
        return None
    try:
        row = conn.execute(
            "SELECT norad_id, object_name, object_type, decay_date, ops_status_code FROM object_metadata WHERE norad_id=?",
            (str(norad),),
        ).fetchone()
        return dict(row) if row else None
    except Exception:
        return None


def classify_secondary(row: dict[str, Any], metadata: Optional[dict[str, Any]] = None):
    object_type = _clean(_coalesce(row, "object_type", "OBJECT_TYPE", "secondary_object_type", "SECONDARY_OBJECT_TYPE"))
    if metadata:
        object_type = object_type or _clean(metadata.get("object_type"))
    value = (object_type or "").upper()
    if "DEBRIS" in value:
        return "debris", "object_type", object_type
    if "ROCKET" in value or "R/B" in value:
        return "inactive_satellite", "object_type", object_type
    if "PAYLOAD" in value:
        return "active_satellite", "object_type", object_type
    name = str(_coalesce(row, "secondary_name", "SECONDARY_NAME", "OBJECT_NAME") or "").upper()
    if "DEB" in name:
        return "debris", "name_heuristic", object_type
    if "R/B" in name or "ROCKET" in name:
        return "inactive_satellite", "name_heuristic", object_type
    if name:
        return "active_satellite", "name_heuristic", object_type
    return "unknown", "unknown", object_type


def _stable_cdm_id(row: dict[str, Any], normalized: dict[str, Any]) -> str:
    explicit = _clean(_coalesce(row, "cdm_id", "CDM_ID", "MESSAGE_ID", "CCSDS_CDM_ID", "id"))
    if explicit:
        return explicit
    base = "|".join(str(normalized.get(k) or "") for k in ("target_norad", "secondary_norad", "creation_date", "tca", "pc", "miss_distance_km"))
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]


def normalize_cdm_record(row: dict[str, Any], source: str, conn):
    target_norad = _normalize_norad(_coalesce(row, "target_norad", "TARGET_NORAD", "SAT_1_ID", "SAT1_ID", "PRIMARY_NORAD"))
    secondary_norad = _normalize_norad(_coalesce(row, "secondary_norad", "SECONDARY_NORAD", "SAT_2_ID", "SAT2_ID"))
    creation_date = _parse_dt(_coalesce(row, "creation_date", "CREATION_DATE", "CREATED", "MESSAGE_CREATION_DATE"))
    tca = _parse_dt(_coalesce(row, "tca", "TCA", "TIME_OF_CLOSEST_APPROACH"))
    if not target_norad:
        return None, "missing target NORAD"
    if not tca:
        return None, "missing TCA"
    metadata = _lookup_metadata(conn, secondary_norad)
    object_class, object_class_source, object_type = classify_secondary(row, metadata)
    normalized = {
        "target_norad": target_norad,
        "target_name": _clean(_coalesce(row, "target_name", "TARGET_NAME", "SAT_1_NAME")),
        "secondary_norad": secondary_norad,
        "secondary_name": _clean(_coalesce(row, "secondary_name", "SECONDARY_NAME", "SAT_2_NAME")) or (metadata or {}).get("object_name"),
        "creation_date": creation_date,
        "tca": tca,
        "miss_distance_km": _distance_km(row),
        "pc": _as_float(_coalesce(row, "pc", "PC", "COLLISION_PROBABILITY")),
        "relative_speed_km_s": _speed_km_s(row),
        "object_type": object_type,
        "object_class": object_class,
        "object_class_source": object_class_source,
        "source": source,
        "raw_json": json.dumps(row, ensure_ascii=False),
    }
    normalized["cdm_id"] = _stable_cdm_id(row, normalized)
    return normalized, None


def parse_cdm_csv(csv_text: str) -> list[dict[str, Any]]:
    if not csv_text or not csv_text.strip():
        return []
    return [dict(row) for row in csv.DictReader(io.StringIO(csv_text.strip()))]


def ingest_cdm_records(records: list[dict[str, Any]], source: str = "manual") -> dict[str, Any]:
    init_cdm_db()
    conn = get_connection()
    imported_at = datetime.now(timezone.utc).isoformat()
    upserted = skipped = errors = 0
    error_samples = []
    try:
        for row in records or []:
            normalized, error = normalize_cdm_record(row, source, conn)
            if error or not normalized:
                skipped += 1
                if len(error_samples) < 10:
                    error_samples.append(error or "normalization error")
                continue
            try:
                conn.execute(
                    """INSERT INTO cdm_records
                    (cdm_id,target_norad,target_name,secondary_norad,secondary_name,creation_date,tca,miss_distance_km,pc,relative_speed_km_s,object_type,object_class,object_class_source,source,raw_json,imported_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(cdm_id) DO UPDATE SET
                    target_norad=excluded.target_norad,target_name=excluded.target_name,secondary_norad=excluded.secondary_norad,secondary_name=excluded.secondary_name,creation_date=excluded.creation_date,tca=excluded.tca,miss_distance_km=excluded.miss_distance_km,pc=excluded.pc,relative_speed_km_s=excluded.relative_speed_km_s,object_type=excluded.object_type,object_class=excluded.object_class,object_class_source=excluded.object_class_source,source=excluded.source,raw_json=excluded.raw_json,imported_at=excluded.imported_at""",
                    tuple(normalized[k] for k in ("cdm_id","target_norad","target_name","secondary_norad","secondary_name","creation_date","tca","miss_distance_km","pc","relative_speed_km_s","object_type","object_class","object_class_source","source","raw_json")) + (imported_at,),
                )
                upserted += 1
            except Exception as exc:
                errors += 1
                if len(error_samples) < 10:
                    error_samples.append(str(exc))
        conn.commit()
    finally:
        conn.close()
    return {"ok": errors == 0, "received": len(records or []), "upserted": upserted, "skipped": skipped, "errors": errors, "error_samples": error_samples, "source": source, "imported_at": imported_at}


def cdm_status() -> dict[str, Any]:
    init_cdm_db()
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM cdm_records").fetchone()[0]
        targets = conn.execute("SELECT COUNT(DISTINCT target_norad) FROM cdm_records").fetchone()[0]
        bounds = conn.execute("SELECT MIN(tca),MAX(tca),MIN(creation_date),MAX(creation_date) FROM cdm_records").fetchone()
        return {"ok": True, "total_cdm_records": total, "unique_targets": targets, "tca_min": bounds[0], "tca_max": bounds[1], "creation_date_min": bounds[2], "creation_date_max": bounds[3]}
    finally:
        conn.close()


def _empty_by_class():
    return {cls: {"cdm": 0, "high_interest": 0, "maneuver": 0} for cls in OBJECT_CLASSES}


def _classify_event(pc, miss_km, cdm_pc_threshold, cdm_miss_distance_threshold_km, maneuver_pc_threshold, maneuver_miss_distance_threshold_km):
    pc_value = pc if pc is not None else -1.0
    miss_value = miss_km if miss_km is not None else math.inf
    if pc_value >= maneuver_pc_threshold or miss_value <= maneuver_miss_distance_threshold_km:
        return "maneuver"
    if pc_value >= cdm_pc_threshold or miss_value <= cdm_miss_distance_threshold_km:
        return "high_interest"
    return "cdm"


def run_historical_cdm_analysis(target_norad: str, lookback_days: float = 365, bucket_days: float = 30, time_axis: str = "creation_date", cdm_pc_threshold: float = 1e-7, cdm_miss_distance_threshold_km: float = 5, maneuver_pc_threshold: float = 1e-4, maneuver_miss_distance_threshold_km: float = 1, max_events: int = 200) -> dict[str, Any]:
    """Analyze actual CDM messages by creation time.

    All target rows are loaded first, then dates are parsed and filtered in Python.
    This avoids SQLite string-comparison failures caused by mixed timestamp formats.
    """
    init_cdm_db()
    target_norad = _normalize_norad(target_norad)
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=float(lookback_days))
    conn = get_connection()
    try:
        all_target_rows = [dict(row) for row in conn.execute("SELECT * FROM cdm_records WHERE target_norad=?", (target_norad,)).fetchall()]
    finally:
        conn.close()

    rows = []
    rejected_dates = []
    for row in all_target_rows:
        analysis_date = _parse_iso(row.get("creation_date")) or _parse_iso(row.get("imported_at"))
        if analysis_date is None:
            rejected_dates.append({"cdm_id": row.get("cdm_id"), "creation_date": row.get("creation_date"), "imported_at": row.get("imported_at")})
            continue
        row["analysis_date"] = analysis_date.isoformat()
        if start <= analysis_date <= now:
            rows.append(row)
    rows.sort(key=lambda row: row["analysis_date"])

    buckets = []
    all_events = []
    total_cdm = total_hi = total_man = 0
    cursor = start
    while cursor < now:
        b0 = cursor
        b1 = min(cursor + timedelta(days=float(bucket_days)), now)
        bucket_rows = [row for row in rows if b0 <= _parse_iso(row["analysis_date"]) < b1 or (b1 == now and _parse_iso(row["analysis_date"]) == b1)]
        by_class = _empty_by_class()
        high_interest = maneuvers = 0
        for row in bucket_rows:
            cls = row.get("object_class") or "unknown"
            by_class.setdefault(cls, {"cdm": 0, "high_interest": 0, "maneuver": 0})
            level = _classify_event(row.get("pc"), row.get("miss_distance_km"), cdm_pc_threshold, cdm_miss_distance_threshold_km, maneuver_pc_threshold, maneuver_miss_distance_threshold_km)
            by_class[cls]["cdm"] += 1
            if level in {"high_interest", "maneuver"}:
                by_class[cls]["high_interest"] += 1
                high_interest += 1
            if level == "maneuver":
                by_class[cls]["maneuver"] += 1
                maneuvers += 1
            all_events.append({"cdm_id": row.get("cdm_id"), "target_norad": row.get("target_norad"), "secondary_norad": row.get("secondary_norad"), "secondary_name": row.get("secondary_name"), "creation_date": row.get("creation_date"), "analysis_date": row.get("analysis_date"), "tca": row.get("tca"), "miss_distance_km": row.get("miss_distance_km"), "pc": row.get("pc"), "decision_level": level, "object_class": cls})
        count = len(bucket_rows)
        total_cdm += count
        total_hi += high_interest
        total_man += maneuvers
        buckets.append({"bucket_start": b0.isoformat(), "bucket_end": b1.isoformat(), "cdm_records": count, "high_interest_cdms": high_interest, "maneuver_candidates": maneuvers, "by_object_class": by_class})
        cursor = b1

    effective_days = max((now - start).total_seconds() / 86400, 0.0)
    annual = lambda count: None if effective_days <= 0 else round(float(count) * 365.25 / effective_days, 2)
    return {
        "ok": True,
        "mode": "historical_cdm_analysis",
        "target_norad": target_norad,
        "time_axis": "creation_date",
        "requested_time_axis": time_axis,
        "lookback_days": lookback_days,
        "bucket_days": bucket_days,
        "window_start": start.isoformat(),
        "window_end": now.isoformat(),
        "database_target_records": len(all_target_rows),
        "matched_records": len(rows),
        "rejected_date_records": rejected_dates[:10],
        "stored_date_samples": [{"cdm_id": row.get("cdm_id"), "creation_date": row.get("creation_date"), "imported_at": row.get("imported_at")} for row in all_target_rows[:10]],
        "thresholds": {"cdm_pc_threshold": cdm_pc_threshold, "cdm_miss_distance_threshold_km": cdm_miss_distance_threshold_km, "maneuver_pc_threshold": maneuver_pc_threshold, "maneuver_miss_distance_threshold_km": maneuver_miss_distance_threshold_km},
        "summary": {"cdm_records": total_cdm, "high_interest_cdms": total_hi, "maneuver_candidates": total_man, "annualized_cdm_records": annual(total_cdm), "annualized_high_interest_cdms": annual(total_hi), "annualized_maneuver_candidates": annual(total_man)},
        "time_series": buckets,
        "top_events": all_events[: int(max_events)],
        "methodology": {"source": "imported CDM records", "time_axis": "creation_date"},
        "limitations": ["Repeated CDM updates may describe the same physical conjunction."]
    }
