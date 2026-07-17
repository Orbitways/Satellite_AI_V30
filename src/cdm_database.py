"""
Storage and historical analysis for real CDM records.

The historical analysis preserves raw message-level metrics and additionally
clusters repeated CDM updates into deterministic conjunction episodes.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from tle_database import get_connection, init_db

OBJECT_CLASSES = ("debris", "inactive_satellite", "active_satellite", "unknown")
ACTIVE_OPS_STATUS = {"+", "P", "B", "S", "X"}
INACTIVE_OPS_STATUS = {"-", "D"}
CONJUNCTION_TCA_TOLERANCE_MINUTES = 10
_DECISION_SEVERITY = {"cdm": 0, "high_interest": 1, "maneuver": 2}


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
            CREATE INDEX IF NOT EXISTS idx_cdm_secondary ON cdm_records(secondary_norad);
            CREATE INDEX IF NOT EXISTS idx_cdm_object_class ON cdm_records(object_class);
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
    text = str(value).strip()
    return text or None


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
    text = str(value).strip()
    try:
        if "." in text:
            numeric = float(text)
            if numeric.is_integer():
                text = str(int(numeric))
    except Exception:
        pass
    return text.lstrip("0") or "0"


def _parse_dt(value) -> Optional[str]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            continue
    return text


def _parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _distance_km(row: dict[str, Any]) -> Optional[float]:
    value = _as_float(_coalesce(row, "miss_distance_km", "MISS_DISTANCE_KM", "MISS_DISTANCE", "min_miss_distance_km", "MISS_DISTANCE_VALUE"))
    if value is None:
        return None
    unit = str(_coalesce(row, "miss_distance_unit", "MISS_DISTANCE_UNIT", "MISS_DISTANCE_UNITS", "DISTANCE_UNIT") or "km").lower()
    return value / 1000.0 if unit in {"m", "meter", "meters", "metre", "metres"} else value


def _speed_km_s(row: dict[str, Any]) -> Optional[float]:
    value = _as_float(_coalesce(row, "relative_speed_km_s", "RELATIVE_SPEED_KM_S", "RELATIVE_SPEED", "RELATIVE_VELOCITY", "RELATIVE_VELOCITY_KM_S"))
    if value is None:
        return None
    unit = str(_coalesce(row, "relative_speed_unit", "RELATIVE_SPEED_UNIT", "RELATIVE_VELOCITY_UNIT") or "km/s").lower()
    return value / 1000.0 if unit in {"m/s", "mps", "meter/s", "meters/s"} else value


def _lookup_metadata(conn, norad: Optional[str]) -> Optional[dict[str, Any]]:
    if not norad:
        return None
    row = conn.execute(
        """SELECT norad_id, object_name, object_type, rcs_size, country,
                  launch_date, site, decay_date, ops_status_code, source, updated_at
           FROM object_metadata WHERE norad_id=?""",
        (str(norad),),
    ).fetchone()
    return dict(row) if row else None


def classify_secondary(row: dict[str, Any], metadata: Optional[dict[str, Any]] = None) -> tuple[str, str, Optional[str]]:
    explicit = _clean(_coalesce(row, "object_class", "secondary_object_class", "OBJECT_CLASS"))
    if explicit:
        normalized = explicit.lower().replace(" ", "_").replace("-", "_")
        if normalized in OBJECT_CLASSES:
            return normalized, "input_object_class", _clean(_coalesce(row, "object_type", "OBJECT_TYPE"))
        if "debris" in normalized:
            return "debris", "input_object_class", _clean(_coalesce(row, "object_type", "OBJECT_TYPE"))
        if "rocket" in normalized or "inactive" in normalized:
            return "inactive_satellite", "input_object_class", _clean(_coalesce(row, "object_type", "OBJECT_TYPE"))
        if "active" in normalized or "payload" in normalized:
            return "active_satellite", "input_object_class", _clean(_coalesce(row, "object_type", "OBJECT_TYPE"))

    object_type = _clean(_coalesce(row, "object_type", "OBJECT_TYPE", "secondary_object_type", "SECONDARY_OBJECT_TYPE"))
    decay_date = _clean(_coalesce(row, "decay_date", "DECAY", "DECAY_DATE"))
    ops_status = _clean(_coalesce(row, "ops_status_code", "OPS_STATUS_CODE"))
    if metadata:
        object_type = object_type or _clean(metadata.get("object_type"))
        decay_date = decay_date or _clean(metadata.get("decay_date"))
        ops_status = ops_status or _clean(metadata.get("ops_status_code"))

    upper_type = (object_type or "").upper()
    upper_status = (ops_status or "").upper()
    if "DEBRIS" in upper_type:
        return "debris", "spacetrack_satcat_object_type", object_type
    if "ROCKET" in upper_type or "R/B" in upper_type:
        return "inactive_satellite", "spacetrack_satcat_object_type", object_type
    if "PAYLOAD" in upper_type:
        if decay_date or upper_status in INACTIVE_OPS_STATUS:
            return "inactive_satellite", "spacetrack_satcat_payload_status", object_type
        return "active_satellite", "spacetrack_satcat_payload_status", object_type

    name = str(_coalesce(row, "secondary_name", "SECONDARY_NAME", "OBJECT_NAME", "SAT_2_NAME", "SATELLITE_2_NAME") or "").upper()
    if "DEB" in name or "DEBRIS" in name:
        return "debris", "name_heuristic", object_type
    if "R/B" in name or "ROCKET" in name or "OBJECT" in name:
        return "inactive_satellite", "name_heuristic", object_type
    if name:
        return "active_satellite", "name_heuristic", object_type
    return "unknown", "unknown", object_type


def _stable_cdm_id(row: dict[str, Any], normalized: dict[str, Any]) -> str:
    explicit = _clean(_coalesce(row, "cdm_id", "CDM_ID", "MESSAGE_ID", "CCSDS_CDM_ID", "id"))
    if explicit:
        return explicit
    base = "|".join(str(normalized.get(k) or "") for k in ["target_norad", "secondary_norad", "creation_date", "tca", "pc", "miss_distance_km"])
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]


def normalize_cdm_record(row: dict[str, Any], source: str, conn) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    target_norad = _normalize_norad(_coalesce(row, "target_norad", "TARGET_NORAD", "SAT_1_ID", "SAT1_ID", "OBJECT1_NORAD", "OBJECT_1_NORAD", "PRIMARY_NORAD", "TARGET_NORAD_CAT_ID"))
    secondary_norad = _normalize_norad(_coalesce(row, "secondary_norad", "SECONDARY_NORAD", "SAT_2_ID", "SAT2_ID", "OBJECT2_NORAD", "OBJECT_2_NORAD", "SECONDARY_NORAD_CAT_ID"))
    tca = _parse_dt(_coalesce(row, "tca", "TCA", "TIME_OF_CLOSEST_APPROACH"))
    if not target_norad:
        return None, "missing target NORAD"
    if not tca:
        return None, "missing TCA"

    metadata = _lookup_metadata(conn, secondary_norad)
    object_class, object_class_source, object_type = classify_secondary(row, metadata)
    normalized = {
        "target_norad": target_norad,
        "target_name": _clean(_coalesce(row, "target_name", "TARGET_NAME", "SAT_1_NAME", "OBJECT1_NAME", "PRIMARY_NAME")),
        "secondary_norad": secondary_norad,
        "secondary_name": _clean(_coalesce(row, "secondary_name", "SECONDARY_NAME", "SAT_2_NAME", "OBJECT2_NAME", "OBJECT_NAME")) or (metadata or {}).get("object_name"),
        "creation_date": _parse_dt(_coalesce(row, "creation_date", "CREATION_DATE", "MESSAGE_CREATION_DATE", "CDM_CREATION_DATE")),
        "tca": tca,
        "miss_distance_km": _distance_km(row),
        "pc": _as_float(_coalesce(row, "pc", "Pc", "PC", "COLLISION_PROBABILITY", "PROBABILITY_OF_COLLISION")),
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
    inserted = skipped = errors = 0
    error_samples: list[str] = []
    try:
        for row in records or []:
            normalized, error = normalize_cdm_record(row, source, conn)
            if error or not normalized:
                skipped += 1
                if len(error_samples) < 10:
                    error_samples.append(error or "unknown normalization error")
                continue
            try:
                conn.execute(
                    """INSERT INTO cdm_records
                       (cdm_id, target_norad, target_name, secondary_norad, secondary_name,
                        creation_date, tca, miss_distance_km, pc, relative_speed_km_s,
                        object_type, object_class, object_class_source, source, raw_json, imported_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(cdm_id) DO UPDATE SET
                        target_norad=excluded.target_norad,
                        target_name=excluded.target_name,
                        secondary_norad=excluded.secondary_norad,
                        secondary_name=excluded.secondary_name,
                        creation_date=excluded.creation_date,
                        tca=excluded.tca,
                        miss_distance_km=excluded.miss_distance_km,
                        pc=excluded.pc,
                        relative_speed_km_s=excluded.relative_speed_km_s,
                        object_type=excluded.object_type,
                        object_class=excluded.object_class,
                        object_class_source=excluded.object_class_source,
                        source=excluded.source,
                        raw_json=excluded.raw_json,
                        imported_at=excluded.imported_at""",
                    (
                        normalized["cdm_id"], normalized["target_norad"], normalized["target_name"],
                        normalized["secondary_norad"], normalized["secondary_name"], normalized["creation_date"],
                        normalized["tca"], normalized["miss_distance_km"], normalized["pc"],
                        normalized["relative_speed_km_s"], normalized["object_type"], normalized["object_class"],
                        normalized["object_class_source"], normalized["source"], normalized["raw_json"], imported_at,
                    ),
                )
                inserted += 1
            except Exception as exc:
                errors += 1
                if len(error_samples) < 10:
                    error_samples.append(str(exc))
        conn.commit()
    finally:
        conn.close()
    return {
        "ok": errors == 0,
        "received": len(records or []),
        "inserted": inserted,
        "updated": 0,
        "upserted": inserted,
        "skipped": skipped,
        "errors": errors,
        "error_samples": error_samples,
        "source": source,
        "imported_at": imported_at,
    }


def cdm_status() -> dict[str, Any]:
    init_cdm_db()
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM cdm_records").fetchone()[0]
        targets = conn.execute("SELECT COUNT(DISTINCT target_norad) FROM cdm_records").fetchone()[0]
        sources = [dict(r) for r in conn.execute("SELECT source, COUNT(*) AS count FROM cdm_records GROUP BY source ORDER BY count DESC").fetchall()]
        classes = [dict(r) for r in conn.execute("SELECT object_class, COUNT(*) AS count FROM cdm_records GROUP BY object_class ORDER BY count DESC").fetchall()]
        bounds = conn.execute("SELECT MIN(tca), MAX(tca), MIN(creation_date), MAX(creation_date) FROM cdm_records").fetchone()
        return {
            "ok": True,
            "total_cdm_records": total,
            "unique_targets": targets,
            "sources": sources,
            "object_classes": classes,
            "tca_min": bounds[0] if bounds else None,
            "tca_max": bounds[1] if bounds else None,
            "creation_date_min": bounds[2] if bounds else None,
            "creation_date_max": bounds[3] if bounds else None,
        }
    finally:
        conn.close()


def _classify_event(pc, miss_km, cdm_pc_threshold, cdm_miss_distance_threshold_km, maneuver_pc_threshold, maneuver_miss_distance_threshold_km):
    pc_value = pc if pc is not None else -1.0
    miss_value = miss_km if miss_km is not None else math.inf
    if pc_value >= maneuver_pc_threshold or miss_value <= maneuver_miss_distance_threshold_km:
        return "maneuver"
    if pc_value >= cdm_pc_threshold or miss_value <= cdm_miss_distance_threshold_km:
        return "high_interest"
    return "cdm"


def _event_from_row(row: dict[str, Any], level: str) -> dict[str, Any]:
    return {
        "cdm_id": row.get("cdm_id"),
        "target_norad": row.get("target_norad"),
        "secondary_norad": row.get("secondary_norad"),
        "secondary_name": row.get("secondary_name"),
        "object_class": row.get("object_class") or "unknown",
        "object_class_source": row.get("object_class_source"),
        "object_type": row.get("object_type"),
        "creation_date": row.get("creation_date"),
        "analysis_date": row.get("creation_date") or row.get("tca"),
        "tca": row.get("tca"),
        "miss_distance_km": row.get("miss_distance_km"),
        "pc": row.get("pc"),
        "pc_str": f"{float(row['pc']):.2e}" if row.get("pc") is not None else None,
        "relative_speed_km_s": row.get("relative_speed_km_s"),
        "decision_level": level,
        "source": row.get("source"),
    }


def _secondary_group_key(event: dict[str, Any]) -> Optional[str]:
    if event.get("secondary_norad"):
        return f"norad:{event['secondary_norad']}"
    name = " ".join(str(event.get("secondary_name") or "").upper().split())
    return f"name:{name}" if name else None


def _dominant_object_class(messages: list[dict[str, Any]]) -> str:
    valid = [m.get("object_class") for m in messages if m.get("object_class")]
    if not valid:
        return "unknown"
    counts = Counter(valid)
    max_count = max(counts.values())
    tied = {value for value, count in counts.items() if count == max_count}
    for message in reversed(messages):
        if message.get("object_class") in tied:
            return message["object_class"]
    return "unknown"


def _build_episode(messages: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(messages, key=lambda m: (_parse_iso(m["tca"]), m.get("creation_date") or "", m.get("cdm_id") or ""))
    creation_values = [m["creation_date"] for m in ordered if m.get("creation_date")]
    tca_values = [m["tca"] for m in ordered if m.get("tca")]
    pc_values = [float(m["pc"]) for m in ordered if m.get("pc") is not None]
    miss_values = [float(m["miss_distance_km"]) for m in ordered if m.get("miss_distance_km") is not None]
    latest_message = max(ordered, key=lambda m: (m.get("creation_date") or "", m.get("cdm_id") or ""))
    decision_level = max((m["decision_level"] for m in ordered), key=lambda level: _DECISION_SEVERITY[level])
    seed = "|".join([
        str(ordered[0].get("target_norad") or ""),
        str(_secondary_group_key(ordered[0]) or ordered[0].get("cdm_id") or ""),
        str(tca_values[0] if tca_values else ordered[0].get("cdm_id") or ""),
    ])
    return {
        "episode_id": hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20],
        "target_norad": ordered[0].get("target_norad"),
        "secondary_norad": ordered[0].get("secondary_norad"),
        "secondary_name": latest_message.get("secondary_name") or ordered[0].get("secondary_name"),
        "object_class": _dominant_object_class(ordered),
        "message_count": len(ordered),
        "first_creation_date": min(creation_values) if creation_values else None,
        "last_creation_date": max(creation_values) if creation_values else None,
        "first_tca": min(tca_values) if tca_values else None,
        "last_tca": max(tca_values) if tca_values else None,
        "representative_tca": max(tca_values) if tca_values else None,
        "max_pc": max(pc_values) if pc_values else None,
        "min_miss_distance_km": min(miss_values) if miss_values else None,
        "decision_level": decision_level,
        "latest_cdm_id": latest_message.get("cdm_id"),
        "messages": ordered,
    }


def _group_conjunction_episodes(events: list[dict[str, Any]], tolerance_minutes: int = CONJUNCTION_TCA_TOLERANCE_MINUTES) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    independent: list[dict[str, Any]] = []
    for event in events:
        secondary_key = _secondary_group_key(event)
        if not secondary_key or not event.get("tca"):
            independent.append(_build_episode([event]))
            continue
        grouped[(str(event.get("target_norad") or ""), secondary_key)].append(event)

    tolerance = timedelta(minutes=tolerance_minutes)
    episodes = independent
    for key in sorted(grouped):
        messages = sorted(grouped[key], key=lambda m: (_parse_iso(m["tca"]), m.get("creation_date") or "", m.get("cdm_id") or ""))
        cluster: list[dict[str, Any]] = []
        latest_tca: Optional[datetime] = None
        for message in messages:
            tca = _parse_iso(message["tca"])
            if cluster and latest_tca is not None and abs(tca - latest_tca) > tolerance:
                episodes.append(_build_episode(cluster))
                cluster = []
            cluster.append(message)
            latest_tca = tca
        if cluster:
            episodes.append(_build_episode(cluster))
    return episodes


def _empty_by_class():
    return {cls: {"cdm": 0, "high_interest": 0, "maneuver": 0} for cls in OBJECT_CLASSES}


def run_historical_cdm_analysis(
    target_norad: str,
    lookback_days: float = 365,
    bucket_days: float = 30,
    time_axis: str = "tca",
    cdm_pc_threshold: float = 1e-7,
    cdm_miss_distance_threshold_km: float = 5,
    maneuver_pc_threshold: float = 1e-4,
    maneuver_miss_distance_threshold_km: float = 1,
    max_events: int = 200,
) -> dict[str, Any]:
    init_cdm_db()
    target_norad = _normalize_norad(target_norad)
    axis = "creation_date" if time_axis == "creation_date" else "tca"
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=float(lookback_days))

    conn = get_connection()
    try:
        rows = [dict(row) for row in conn.execute(
            f"""SELECT * FROM cdm_records
                WHERE target_norad=? AND {axis} IS NOT NULL AND {axis} >= ? AND {axis} <= ?
                ORDER BY {axis} ASC""",
            (target_norad, start.isoformat(), now.isoformat()),
        ).fetchall()]
    finally:
        conn.close()

    events: list[dict[str, Any]] = []
    for row in rows:
        level = _classify_event(
            row.get("pc"), row.get("miss_distance_km"),
            cdm_pc_threshold, cdm_miss_distance_threshold_km,
            maneuver_pc_threshold, maneuver_miss_distance_threshold_km,
        )
        events.append(_event_from_row(row, level))

    episodes = _group_conjunction_episodes(events)
    total_cdm = len(events)
    total_hi = sum(event["decision_level"] in {"high_interest", "maneuver"} for event in events)
    total_man = sum(event["decision_level"] == "maneuver" for event in events)
    distinct = len(episodes)
    episode_hi = sum(episode["decision_level"] in {"high_interest", "maneuver"} for episode in episodes)
    episode_man = sum(episode["decision_level"] == "maneuver" for episode in episodes)

    buckets = []
    cursor = start
    while cursor < now:
        b0 = cursor
        b1 = min(cursor + timedelta(days=float(bucket_days)), now)
        bucket_events = []
        for event in events:
            value = event.get(axis)
            if not value:
                continue
            try:
                dt = _parse_iso(value)
            except Exception:
                continue
            if b0 <= dt < b1:
                bucket_events.append(event)

        bucket_episodes = []
        for episode in episodes:
            value = episode.get("first_creation_date") or episode.get("representative_tca")
            if not value:
                continue
            try:
                dt = _parse_iso(value)
            except Exception:
                continue
            if b0 <= dt < b1:
                bucket_episodes.append(episode)

        by_class = _empty_by_class()
        for event in bucket_events:
            cls = event.get("object_class") or "unknown"
            by_class.setdefault(cls, {"cdm": 0, "high_interest": 0, "maneuver": 0})
            by_class[cls]["cdm"] += 1
            if event["decision_level"] in {"high_interest", "maneuver"}:
                by_class[cls]["high_interest"] += 1
            if event["decision_level"] == "maneuver":
                by_class[cls]["maneuver"] += 1

        buckets.append({
            "bucket_start": b0.isoformat(),
            "bucket_end": b1.isoformat(),
            "cdm_records": len(bucket_events),
            "high_interest_cdms": sum(event["decision_level"] in {"high_interest", "maneuver"} for event in bucket_events),
            "maneuver_candidates": sum(event["decision_level"] == "maneuver" for event in bucket_events),
            "cdm_messages": len(bucket_events),
            "distinct_conjunctions": len(bucket_episodes),
            "high_interest_conjunctions": sum(episode["decision_level"] in {"high_interest", "maneuver"} for episode in bucket_episodes),
            "maneuver_candidate_conjunctions": sum(episode["decision_level"] == "maneuver" for episode in bucket_episodes),
            "by_object_class": by_class,
            "max_pc": max((float(event["pc"]) for event in bucket_events if event.get("pc") is not None), default=0.0),
            "min_miss_distance_km": min((float(event["miss_distance_km"]) for event in bucket_events if event.get("miss_distance_km") is not None), default=None),
        })
        cursor = b1

    events.sort(key=lambda event: (-(float(event.get("pc") or 0.0)), float(event.get("miss_distance_km") if event.get("miss_distance_km") is not None else 1e18)))
    episodes.sort(key=lambda episode: (
        -_DECISION_SEVERITY[episode["decision_level"]],
        -(float(episode["max_pc"]) if episode.get("max_pc") is not None else -1.0),
        float(episode["min_miss_distance_km"]) if episode.get("min_miss_distance_km") is not None else 1e18,
        episode.get("representative_tca") or "",
    ))

    effective_days = max(0.0, (now - start).total_seconds() / 86400)
    def annualize(value: int):
        return None if effective_days <= 0 else round(float(value) * 365.25 / effective_days, 2)

    average_messages = round(total_cdm / distinct, 2) if distinct else 0.0
    largest_episode = max((episode["message_count"] for episode in episodes), default=0)
    return {
        "ok": True,
        "mode": "historical_cdm_analysis",
        "target_norad": target_norad,
        "time_axis": axis,
        "lookback_days": lookback_days,
        "bucket_days": bucket_days,
        "thresholds": {
            "cdm_pc_threshold": cdm_pc_threshold,
            "cdm_pc_threshold_str": f"{cdm_pc_threshold:.2e}",
            "cdm_miss_distance_threshold_km": cdm_miss_distance_threshold_km,
            "maneuver_pc_threshold": maneuver_pc_threshold,
            "maneuver_pc_threshold_str": f"{maneuver_pc_threshold:.2e}",
            "maneuver_miss_distance_threshold_km": maneuver_miss_distance_threshold_km,
        },
        "summary": {
            "cdm_records": total_cdm,
            "high_interest_cdms": total_hi,
            "maneuver_candidates": total_man,
            "annualized_cdm_records": annualize(total_cdm),
            "annualized_high_interest_cdms": annualize(total_hi),
            "annualized_maneuver_candidates": annualize(total_man),
            "cdm_messages": total_cdm,
            "high_interest_messages": total_hi,
            "maneuver_threshold_messages": total_man,
            "distinct_conjunctions": distinct,
            "high_interest_conjunctions": episode_hi,
            "maneuver_candidate_conjunctions": episode_man,
            "annualized_cdm_messages": annualize(total_cdm),
            "annualized_distinct_conjunctions": annualize(distinct),
            "annualized_high_interest_conjunctions": annualize(episode_hi),
            "annualized_maneuver_candidate_conjunctions": annualize(episode_man),
        },
        "time_series": buckets,
        "top_events": events[: int(max_events)],
        "top_conjunctions": episodes[: int(max_events)],
        "diagnostics": {
            "grouping_method": "target_secondary_tca_tolerance",
            "tca_tolerance_minutes": CONJUNCTION_TCA_TOLERANCE_MINUTES,
            "database_messages": total_cdm,
            "database_conjunctions": distinct,
            "average_messages_per_conjunction": average_messages,
            "largest_episode_size": largest_episode,
        },
        "methodology": {
            "source": "imported CDM records",
            "time_axis": axis,
            "cdm_count_definition": "one imported CDM record",
            "conjunction_definition": "messages grouped by target, secondary object and TCA proximity",
            "episode_decision_definition": "most severe decision level among underlying CDM messages",
        },
        "limitations": [
            "This analysis depends on the completeness and quality of imported CDM records.",
            "Successive CDM updates are grouped by target, secondary object and TCA proximity.",
            "Episode grouping is a deterministic operational approximation and may not reproduce an operator's internal event identifiers.",
            "Maneuver-candidate conjunctions indicate threshold crossings, not confirmed executed maneuvers.",
            "Legacy maneuver fields remain message-level metrics for backward compatibility.",
        ],
    }
