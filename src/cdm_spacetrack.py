"""Space-Track public CDM synchronization helpers.

A historical ingestion lookback describes when CDM messages were created, not
when the predicted conjunction reaches TCA. A CDM created today can refer to a
TCA several days in the future, so filtering the upstream feed by TCA would
silently exclude valid messages from the requested historical period.

The public CDM schema is not guaranteed to expose queryable NORAD fields, and
its returned column names can vary. Fetch a bounded CREATED window, inspect the
actual payload, then filter and normalize locally. Never report a successful
import when no record for the requested target was found.
"""

from __future__ import annotations

import json
import os
import re
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
    value = str(value).strip()
    return value or None


def _coalesce(row: dict[str, Any], *keys: str):
    lower = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
        value = lower.get(key.lower())
        if value not in (None, ""):
            return value
    return None


def _norad(value):
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
    except Exception as exc:
        raise ValueError(f"Space-Track returned non-JSON content: {exc}") from exc
    if isinstance(parsed, list):
        return [row for row in parsed if isinstance(row, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    return []


def _env_aliases():
    if os.environ.get("SPACETRACK_USER") and not os.environ.get("SPACETRACK_EMAIL"):
        os.environ["SPACETRACK_EMAIL"] = os.environ["SPACETRACK_USER"]
    if os.environ.get("SPACETRACK_PASS") and not os.environ.get("SPACETRACK_PASSWORD"):
        os.environ["SPACETRACK_PASSWORD"] = os.environ["SPACETRACK_PASS"]


_ID_KEYS_1 = (
    "SAT_1_ID", "SAT1_ID", "NORAD_CAT_ID_1", "NORAD_CAT_ID1",
    "OBJECT1_NORAD", "OBJECT_1_NORAD", "OBJECT1_NORAD_CAT_ID",
    "OBJECT_1_NORAD_CAT_ID", "PRIMARY_NORAD", "TARGET_NORAD_CAT_ID",
)
_ID_KEYS_2 = (
    "SAT_2_ID", "SAT2_ID", "NORAD_CAT_ID_2", "NORAD_CAT_ID2",
    "OBJECT2_NORAD", "OBJECT_2_NORAD", "OBJECT2_NORAD_CAT_ID",
    "OBJECT_2_NORAD_CAT_ID", "SECONDARY_NORAD", "SECONDARY_NORAD_CAT_ID",
)


def _candidate_ids(row: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key, value in row.items():
        key_upper = str(key).upper()
        if not re.search(r"(NORAD|CAT_ID|SAT_?\d?_?ID|OBJECT_?\d?_?ID)", key_upper):
            continue
        normalized = _norad(value)
        if normalized and normalized.isdigit():
            ids.add(normalized)
    return ids


def _pair_ids(row: dict[str, Any]) -> tuple[str | None, str | None]:
    return _norad(_coalesce(row, *_ID_KEYS_1)), _norad(_coalesce(row, *_ID_KEYS_2))


def normalize_spacetrack_cdm(row: dict[str, Any], target_norad: str) -> dict[str, Any] | None:
    target = _norad(target_norad)
    n1, n2 = _pair_ids(row)
    if target not in {n1, n2}:
        candidates = _candidate_ids(row)
        if target not in candidates:
            return None
        others = sorted(value for value in candidates if value != target)
        n1, n2 = target, (others[0] if others else None)

    name1 = _clean(_coalesce(row, "SAT_1_NAME", "SAT1_NAME", "OBJECT1_NAME", "OBJECT_1_NAME", "OBJECT_NAME_1", "PRIMARY_NAME"))
    name2 = _clean(_coalesce(row, "SAT_2_NAME", "SAT2_NAME", "OBJECT2_NAME", "OBJECT_2_NAME", "OBJECT_NAME_2", "SECONDARY_NAME"))
    type1 = _clean(_coalesce(row, "SAT_1_OBJECT_TYPE", "OBJECT1_TYPE", "OBJECT_TYPE_1", "PRIMARY_OBJECT_TYPE"))
    type2 = _clean(_coalesce(row, "SAT_2_OBJECT_TYPE", "OBJECT2_TYPE", "OBJECT_TYPE_2", "SECONDARY_OBJECT_TYPE"))

    if n2 == target:
        target_id, target_name = n2, name2
        secondary_id, secondary_name, secondary_type = n1, name1, type1
    else:
        target_id, target_name = n1, name1
        secondary_id, secondary_name, secondary_type = n2, name2, type2

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


def _query_path(start: str, end_exclusive: str, max_records: int) -> str:
    limit = f"/limit/{int(max_records)}" if max_records else ""
    return f"/class/cdm_public/CREATED/{start}--{end_exclusive}/orderby/CREATED%20asc/format/json{limit}"


def fetch_spacetrack_public_cdms(target_norad: str, lookback_days: int = 365, max_records: int = 10000) -> dict[str, Any]:
    from spacetrack import SpaceTrackSession

    _env_aliases()
    target = _norad(target_norad)
    if not target or not target.isdigit():
        raise ValueError("target_norad must be numeric")

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(lookback_days))
    start = start_dt.strftime("%Y-%m-%d")
    # Space-Track date-only range boundaries can omit records created later on
    # the end date. Query through tomorrow, while reporting the true UTC window.
    query_end_exclusive = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    base = "https://www.space-track.org/basicspacedata/query"
    path = _query_path(start, query_end_exclusive, max_records)

    with SpaceTrackSession() as client:
        raw_text = _decode(client._request(base + path))
    raw_rows = _rows(raw_text)

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in raw_rows:
        record = normalize_spacetrack_cdm(row, target)
        if not record:
            continue
        key = "|".join(str(record.get(k) or "") for k in ("cdm_id", "secondary_norad", "creation_date", "tca", "pc", "miss_distance_km"))
        if key in seen:
            continue
        seen.add(key)
        normalized.append(record)

    if normalized:
        report = ingest_cdm_records(normalized, source="spacetrack_cdm_public")
        status = "imported"
    else:
        report = {
            "ok": False,
            "received": 0,
            "upserted": 0,
            "skipped": 0,
            "errors": 0,
            "error_samples": [],
            "source": "spacetrack_cdm_public",
            "imported_at": datetime.now(timezone.utc).isoformat(),
        }
        status = "no_matching_public_cdms"

    sample_keys = sorted({str(key) for row in raw_rows[:20] for key in row.keys()})
    return {
        "ok": bool(normalized) and report.get("errors", 0) == 0,
        "status": status,
        "target_norad": target,
        "lookback_days": lookback_days,
        "window_axis": "creation_date",
        "window_start": start_dt.isoformat(),
        "window_end": end_dt.isoformat(),
        "raw_rows_fetched": len(raw_rows),
        "records_for_target": len(normalized),
        "import_report": report,
        "attempted_queries": [path],
        "query_errors": [],
        "sample_response_keys": sample_keys,
        "note": (
            "No public CDM created in the requested period matched this NORAD. This is a data-availability result, not a successful zero-event import."
            if not normalized
            else "Imported actual Space-Track cdm_public records created during the requested period for the requested target. Public coverage may be incomplete."
        ),
    }
