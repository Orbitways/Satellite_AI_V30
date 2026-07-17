"""Space-Track provider used by the durable all-constellation CDM archive."""

from __future__ import annotations

from typing import Any

from cdm_auto_sync import (
    DEFAULT_MAX_RECORDS,
    _credentials_configured,
    _decode,
    _env_aliases,
    _format_query_time,
    _json_rows,
)


def _coalesce(row: dict[str, Any], *keys: str):
    lowered = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
        value = lowered.get(key.lower())
        if value not in (None, ""):
            return value
    return None


def prepare_archive_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Add canonical aliases while preserving the complete provider payload."""
    target_norad = _coalesce(
        row,
        "SAT_1_ID",
        "SAT1_ID",
        "NORAD_CAT_ID_1",
        "NORAD_CAT_ID1",
        "OBJECT1_NORAD",
        "OBJECT_1_NORAD",
        "PRIMARY_NORAD",
    )
    secondary_norad = _coalesce(
        row,
        "SAT_2_ID",
        "SAT2_ID",
        "NORAD_CAT_ID_2",
        "NORAD_CAT_ID2",
        "OBJECT2_NORAD",
        "OBJECT_2_NORAD",
        "SECONDARY_NORAD",
    )
    tca = _coalesce(row, "TCA", "TIME_OF_CLOSEST_APPROACH")
    if target_norad in (None, "") or tca in (None, ""):
        return None

    prepared = dict(row)
    aliases = {
        "cdm_id": _coalesce(row, "CDM_ID", "MESSAGE_ID", "CCSDS_CDM_ID", "ID"),
        "target_norad": target_norad,
        "target_name": _coalesce(
            row,
            "SAT_1_NAME",
            "SAT1_NAME",
            "SAT1_OBJECT_NAME",
            "OBJECT1_NAME",
            "PRIMARY_NAME",
        ),
        "secondary_norad": secondary_norad,
        "secondary_name": _coalesce(
            row,
            "SAT_2_NAME",
            "SAT2_NAME",
            "SAT2_OBJECT_NAME",
            "OBJECT2_NAME",
            "SECONDARY_NAME",
        ),
        "creation_date": _coalesce(
            row,
            "CREATED",
            "CREATION_DATE",
            "MESSAGE_CREATION_DATE",
            "CDM_CREATION_DATE",
        ),
        "tca": tca,
        "miss_distance_km": _coalesce(
            row,
            "MIN_RNG",
            "MISS_DISTANCE",
            "MISS_DISTANCE_KM",
            "MINIMUM_RANGE",
        ),
        "pc": _coalesce(row, "PC", "COLLISION_PROBABILITY", "PROBABILITY_OF_COLLISION"),
        "relative_speed_km_s": _coalesce(
            row,
            "RELATIVE_SPEED",
            "RELATIVE_VELOCITY",
            "RELATIVE_SPEED_KM_S",
        ),
        "object_type": _coalesce(
            row,
            "SAT2_OBJECT_TYPE",
            "SAT_2_OBJECT_TYPE",
            "OBJECT2_TYPE",
            "SECONDARY_OBJECT_TYPE",
        ),
    }
    for key, value in aliases.items():
        if value not in (None, ""):
            prepared[key] = value
    return prepared


def _query_path(window_start, window_end, max_records: int) -> str:
    start = _format_query_time(window_start)
    end = _format_query_time(window_end)
    return (
        "/class/cdm_public"
        f"/CREATED/{start}--{end}"
        "/orderby/CREATED%20asc"
        "/format/json"
        f"/limit/{int(max_records)}"
    )


def fetch_available_cdms(
    window_start,
    window_end,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> tuple[list[dict[str, Any]], str]:
    """Retrieve and normalize all CDMs visible to the authenticated account."""
    from spacetrack import SpaceTrackSession

    _env_aliases()
    if not _credentials_configured():
        raise RuntimeError(
            "Space-Track credentials are missing. Configure SPACETRACK_EMAIL "
            "and SPACETRACK_PASSWORD (or SPACETRACK_USER/SPACETRACK_PASS)."
        )

    path = _query_path(window_start, window_end, max_records)
    url = "https://www.space-track.org/basicspacedata/query" + path
    with SpaceTrackSession() as client:
        raw_rows = _json_rows(_decode(client._request(url)))

    prepared_rows = [
        prepared
        for row in raw_rows
        if (prepared := prepare_archive_row(row)) is not None
    ]
    if raw_rows and not prepared_rows:
        sample_keys = sorted({str(key) for row in raw_rows[:20] for key in row})
        raise RuntimeError(
            "Space-Track returned CDM rows, but none could be normalized. "
            f"Observed response keys: {sample_keys[:80]}"
        )
    return prepared_rows, path
