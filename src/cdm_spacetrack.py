"""Space-Track public CDM synchronization helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from cdm_database import ingest_cdm_records


def _decode(raw) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")
    return str(raw)


def _clean(value):
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _coalesce(row: dict[str, Any], *keys: str):
    lower = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
        if key.lower() in lower and lower[key.lower()] not in (None, ""):
            return lower[key.lower()]
    return None


def _norad(value):
    if value in (None, ""):
        return None
    s = str(value).strip()
    try:
        if "." in s:
            f = float(s)
            if f.is_integer():
                s = str(int(f))
    except Exception:
        pass
    return s.lstrip("0") or "0"


def _float(value):
    if value in (None, ""):
        return None
    try:
        return float(str(value).strip().replace(",", ""))
    except Exception:
        return None


def _rows(raw_text: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads((raw_text or "").strip() or "[]")
    except Exception:
        return []
    if isinstance(parsed, list):
        return [r for r in parsed if isinstance(r, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    return []


def _env_aliases():
    if os.environ.get("SPACETRACK_USER") and not os.environ.get("SPACETRACK_EMAIL"):
        os.environ["SPACETRACK_EMAIL"] = os.environ["SPACETRACK_USER"]
    if os.environ.get("SPACETRACK_PASS") and not os.environ.get("SPACETRACK_PASSWORD"):
        os.environ["SPACETRACK_PASSWORD"] = os.environ["SPACETRACK_PASS"]


def normalize_spacetrack_cdm(row: dict[str, Any], target_norad: str) -> dict[str, Any]:
    target = _norad(target_norad)
    n1 = _norad(_coalesce(row, "SAT_1_ID", "SAT1_ID", "NORAD_CAT_ID_1", "OBJECT1_NORAD", "OBJECT_1_NORAD"))
    n2 = _norad(_coalesce(row, "SAT_2_ID", "SAT2_ID", "NORAD_CAT_ID_2", "OBJECT2_NORAD", "OBJECT_2_NORAD"))
    name1 = _clean(_coalesce(row, "SAT_1_NAME", "SAT1_NAME", "OBJECT1_NAME", "OBJECT_1_NAME", "OBJECT_NAME_1"))
    name2 = _clean(_coalesce(row, "SAT_2_NAME", "SAT2_NAME", "OBJECT2_NAME", "OBJECT_2_NAME", "OBJECT_NAME_2"))
    type1 = _clean(_coalesce(row, "SAT_1_OBJECT_TYPE", "OBJECT1_TYPE", "OBJECT_TYPE_1"))
    type2 = _clean(_coalesce(row, "SAT_2_OBJECT_TYPE", "OBJECT2_TYPE", "OBJECT_TYPE_2"))

    if n2 == target and n1:
        target_id, target_name, secondary_id, secondary_name, secondary_type = n2, name2, n1, name1, type1
    else:
        target_id, target_name, secondary_id, secondary_name, secondary_type = n1 or target, name1, n2, name2, type2

    return {
        "cdm_id": _clean(_coalesce(row, "CDM_ID", "CCSDS_CDM_ID", "MESSAGE_ID", "CDM_IDENTIFIER", "ID")),
        "target_norad": target_id,
        "target_name": target_name,
        "secondary_norad": secondary_id,
        "secondary_name": secondary_name,
        "creation_date": _clean(_coalesce(row, "CREATION_DATE", "CREATED", "MESSAGE_CREATION_DATE", "CDM_CREATION_DATE")),
        "tca": _clean(_coalesce(row, "TCA", "TIME_OF_CLOSEST_APPROACH")),
        "miss_distance_km": _float(_coalesce(row, "MISS_DISTANCE", "MISS_DISTANCE_KM", "MINIMUM_DISTANCE", "MISS_DISTANCE_VALUE")),
        "pc": _float(_coalesce(row, "PC", "Pc", "COLLISION_PROBABILITY", "PROBABILITY_OF_COLLISION")),
        "relative_speed_km_s": _float(_coalesce(row, "RELATIVE_SPEED", "RELATIVE_VELOCITY", "RELATIVE_SPEED_KM_S", "RELATIVE_VELOCITY_KM_S")),
        "object_type": secondary_type or _clean(_coalesce(row, "OBJECT_TYPE", "SECONDARY_OBJECT_TYPE")),
    }


def _query_paths(target: str, start: str, end: str, max_records: int) -> list[str]:
    limit = f"/limit/{int(max_records)}" if max_records else ""
    common = f"/TCA/{start}--{end}/orderby/TCA%20asc/format/json{limit}"
    return [
        f"/class/cdm_public/SAT_1_ID/{target}{common}",
        f"/class/cdm_public/SAT_2_ID/{target}{common}",
        f"/class/cdm_public/NORAD_CAT_ID_1/{target}{common}",
        f"/class/cdm_public/NORAD_CAT_ID_2/{target}{common}",
        f"/class/cdm_public{common}",
    ]


def fetch_spacetrack_public_cdms(target_norad: str, lookback_days: int = 365, max_records: int = 10000) -> dict[str, Any]:
    from spacetrack import SpaceTrackSession

    _env_aliases()
    target = _norad(target_norad)
    if not target:
        raise ValueError("target_norad is required")

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(lookback_days))
    start = start_dt.strftime("%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")

    raw_rows: list[dict[str, Any]] = []
    attempted: list[str] = []
    errors: list[str] = []
    base = "https://www.space-track.org/basicspacedata/query"

    with SpaceTrackSession() as client:
        for path in _query_paths(target, start, end, max_records):
            attempted.append(path)
            try:
                got = _rows(_decode(client._request(base + path)))
                raw_rows.extend(got)
            except Exception as exc:
                errors.append(str(exc))

    normalized: list[dict[str, Any]] = []
    seen = set()
    for row in raw_rows:
        rec = normalize_spacetrack_cdm(row, target)
        if rec.get("target_norad") != target:
            continue
        key = "|".join(str(rec.get(k) or "") for k in ["cdm_id", "secondary_norad", "creation_date", "tca", "pc", "miss_distance_km"])
        if key in seen:
            continue
        seen.add(key)
        normalized.append(rec)
        if max_records and len(normalized) >= int(max_records):
            break

    report = ingest_cdm_records(normalized, source="spacetrack_cdm_public") if normalized else {
        "ok": True,
        "received": 0,
        "upserted": 0,
        "skipped": 0,
        "errors": 0,
        "error_samples": [],
        "source": "spacetrack_cdm_public",
        "imported_at": datetime.now(timezone.utc).isoformat(),
    }
    return {
        "ok": report.get("errors", 0) == 0,
        "target_norad": target,
        "lookback_days": lookback_days,
        "window_start": start_dt.isoformat(),
        "window_end": end_dt.isoformat(),
        "raw_rows_fetched": len(raw_rows),
        "records_for_target": len(normalized),
        "import_report": report,
        "attempted_queries": attempted,
        "query_errors": errors[:10],
        "note": "Fetched from Space-Track cdm_public when available. Public CDMs may not include all operator-private CDM records.",
    }
