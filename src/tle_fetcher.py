"""
tle_fetcher.py — TLE and Space-Track catalog metadata ingestion.

The refresh flow now stores two complementary datasets:
- TLE history from Space-Track gp / gp_history;
- SATCAT object metadata used to classify debris, rocket bodies and payloads.
"""

import json
import os
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)
PROGRESS_PATH = Path("data/refresh_progress.json")

# Backup TLE if no network / no database available.
FALLBACK_TLES = [
    (
        "ISS (ZARYA)",
        "1 25544U 98067A   24150.54097222  .00016717  00000+0  10270-3 0  9993",
        "2 25544  51.6416  21.5234 0006752  54.1234  45.6789 15.50012345678901",
    ),
]

TLEEntry = Tuple[str, str, str]


def _write_refresh_progress(**kwargs):
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": time.time(),
        "updated_at_iso": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }
    tmp = PROGRESS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(PROGRESS_PATH)


def load_refresh_progress():
    if not PROGRESS_PATH.exists():
        return {"state": "idle", "message": "No TLE refresh currently running."}
    try:
        return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        return {"state": "unknown", "error": str(e)}


def parse_tle_file(path: str) -> List[TLEEntry]:
    if not os.path.exists(path):
        logger.warning(f"TLE file not found: {path}. Using fallback.")
        return FALLBACK_TLES
    entries: List[TLEEntry] = []
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    i = 0
    while i + 2 < len(lines):
        name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]
        if _validate_tle(line1, line2):
            entries.append((name, line1, line2))
        i += 3
    return entries if entries else FALLBACK_TLES


def fetch_celestrak(category: str = "stations") -> List[TLEEntry]:
    try:
        import urllib.request
        url = "https://celestrak.org/pub/TLE/catalog.tle"
        with urllib.request.urlopen(url, timeout=10) as resp:
            content = resp.read().decode("utf-8")
        cache_path = f"data/celestrak_{category}.txt"
        os.makedirs("data", exist_ok=True)
        Path(cache_path).write_text(content, encoding="utf-8")
        return _parse_tle_string(content)
    except Exception as e:
        logger.warning(f"Celestrak unavailable ({e}). Using cache/fallback.")
        cache_path = f"data/celestrak_{category}.txt"
        if os.path.exists(cache_path):
            return parse_tle_file(cache_path)
        return FALLBACK_TLES


def _parse_tle_string(content: str) -> List[TLEEntry]:
    entries: List[TLEEntry] = []
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    i = 0
    while i + 2 < len(lines):
        name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]
        if _validate_tle(line1, line2):
            entries.append((name, line1, line2))
        i += 3
    return entries if entries else FALLBACK_TLES


def _validate_tle(line1: str, line2: str) -> bool:
    if not (line1.startswith("1 ") and line2.startswith("2 ")):
        return False
    if len(line1) < 69 or len(line2) < 69:
        return False
    return _checksum(line1) and _checksum(line2)


def _checksum(line: str) -> bool:
    if len(line) < 69:
        return False
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return (total % 10) == int(line[68])


def get_tle_epoch(line1: str) -> Optional[datetime]:
    try:
        epoch_str = line1[18:32].strip()
        year_2d = int(epoch_str[:2])
        year = 2000 + year_2d if year_2d < 57 else 1900 + year_2d
        day_of_year = float(epoch_str[2:])
        dt = datetime(year, 1, 1, tzinfo=timezone.utc)
        dt += timedelta(days=day_of_year - 1)
        return dt
    except Exception:
        return None


def _parse_tle_text(raw: str) -> list[TLEEntry]:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    tles = []
    i = 0
    while i + 2 < len(lines):
        name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]
        if line1.startswith("1 ") and line2.startswith("2 "):
            tles.append((name, line1, line2))
        i += 3
    return tles


def _decode_response(raw) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")
    return str(raw)


def _fetch_satcat_metadata(client, norad_ids: list[str], emit=None) -> list[dict]:
    """Fetch Space-Track SATCAT metadata for the supplied NORAD IDs."""
    metadata: list[dict] = []
    batch_size = 500
    batches = [norad_ids[i:i + batch_size] for i in range(0, len(norad_ids), batch_size)]
    for b_idx, batch in enumerate(batches):
        norad_str = ",".join(batch)
        pct = 76 + int(((b_idx + 1) / max(len(batches), 1)) * 3)
        url = (
            "https://www.space-track.org/basicspacedata/query"
            f"/class/satcat/NORAD_CAT_ID/{norad_str}"
            "/orderby/NORAD_CAT_ID%20asc/format/json"
        )
        try:
            raw = _decode_response(client._request(url))
            rows = json.loads(raw) if raw.strip().startswith("[") else []
            if isinstance(rows, list):
                metadata.extend(rows)
            if emit:
                emit(
                    f"SATCAT metadata batch {b_idx + 1}/{len(batches)}: +{len(rows) if isinstance(rows, list) else 0} records",
                    pct,
                    state="fetching_metadata",
                    metadata_fetched=len(metadata),
                    metadata_batches_done=b_idx + 1,
                    metadata_batches_total=len(batches),
                )
        except Exception as e:
            if emit:
                emit(
                    f"SATCAT metadata batch {b_idx + 1}/{len(batches)} failed: {e}",
                    pct,
                    state="fetching_metadata",
                    metadata_fetched=len(metadata),
                    last_metadata_error=str(e),
                )
    return metadata


def fetch_and_store(group: str = "starlink", days: int = 30):
    """
    Refresh the local TLE database using Space-Track gp / gp_history and enrich
    object metadata with Space-Track SATCAT.
    """
    from spacetrack import SpaceTrackSession
    from tle_database import ingest_tles, ingest_object_metadata, get_stats

    Path("data").mkdir(exist_ok=True)
    started_at = time.time()
    logs = []
    progress_context = {}

    def emit(msg, pct=None, state=None, **extra):
        logs.append(str(msg))
        if state is not None:
            progress_context["state"] = state
        for key, value in extra.items():
            if value is not None:
                progress_context[key] = value
        payload_extra = {k: v for k, v in progress_context.items() if k != "state"}
        _write_refresh_progress(
            state=progress_context.get("state", "running"),
            group=group,
            days=days,
            pct=pct,
            message=str(msg),
            logs_tail=logs[-10:],
            started_at=started_at,
            **payload_extra,
        )

    emit("Starting TLE + Space-Track metadata refresh.", pct=0, state="starting")

    try:
        if os.environ.get("SPACETRACK_USER") and not os.environ.get("SPACETRACK_EMAIL"):
            os.environ["SPACETRACK_EMAIL"] = os.environ["SPACETRACK_USER"]
        if os.environ.get("SPACETRACK_PASS") and not os.environ.get("SPACETRACK_PASSWORD"):
            os.environ["SPACETRACK_PASSWORD"] = os.environ["SPACETRACK_PASS"]

        all_tles: list[TLEEntry] = []
        metadata_rows: list[dict] = []
        source_used = "Space-Track gp_history + satcat"

        emit("Opening Space-Track session...", pct=2, state="opening_spacetrack_session")
        with SpaceTrackSession() as client:
            emit("Space-Track session opened.", pct=4, state="spacetrack_session_opened")
            now_dt = datetime.now(timezone.utc)
            end_str = now_dt.strftime("%Y-%m-%d")
            start_hist = (now_dt - timedelta(days=days)).strftime("%Y-%m-%d")

            emit("Querying Space-Track current LEO catalog...", pct=5, state="fetching_current_catalog")
            url_list = (
                "https://www.space-track.org/basicspacedata/query"
                "/class/gp/EPOCH/%3Enow-2"
                "/MEAN_MOTION/%3E11.25/ECCENTRICITY/%3C0.25"
                "/orderby/NORAD_CAT_ID/format/tle"
            )
            raw_list = _decode_response(client._request(url_list))
            tles_current = _parse_tle_text(raw_list) if raw_list and len(raw_list) > 200 else []
            norad_ids = sorted(set(tle[1][2:7].strip() for tle in tles_current))

            emit(
                f"{len(norad_ids)} current LEO objects identified.",
                pct=15,
                state="current_catalog_received",
                n_objects=len(norad_ids),
                fetched_tles=len(tles_current),
            )
            if not norad_ids:
                raise ValueError("No LEO object found via Space-Track gp")
            all_tles.extend(tles_current)

            emit("Fetching Space-Track SATCAT metadata...", pct=75, state="fetching_metadata", n_objects=len(norad_ids))
            metadata_rows = _fetch_satcat_metadata(client, norad_ids, emit=emit)

            batch_size = 500
            batches = [norad_ids[i:i + batch_size] for i in range(0, len(norad_ids), batch_size)]
            emit(
                f"{len(batches)} Space-Track history batches to fetch.",
                pct=20,
                state="fetching_history",
                n_objects=len(norad_ids),
                batches_done=0,
                batches_total=len(batches),
                fetched_tles=len(all_tles),
            )
            for b_idx, batch in enumerate(batches):
                pct = 20 + int(((b_idx + 1) / max(len(batches), 1)) * 55)
                norad_str = ",".join(batch)
                url_hist = (
                    "https://www.space-track.org/basicspacedata/query"
                    f"/class/gp_history/NORAD_CAT_ID/{norad_str}"
                    f"/EPOCH/{start_hist}--{end_str}"
                    "/orderby/NORAD_CAT_ID%20asc,EPOCH%20asc/format/tle"
                )
                try:
                    raw_h = _decode_response(client._request(url_hist))
                    batch_tles = _parse_tle_text(raw_h) if raw_h and len(raw_h) > 100 else []
                    all_tles.extend(batch_tles)
                    emit(
                        f"History batch {b_idx + 1}/{len(batches)}: +{len(batch_tles)} TLE",
                        pct=pct,
                        state="fetching_history",
                        n_objects=len(norad_ids),
                        batches_done=b_idx + 1,
                        batches_total=len(batches),
                        fetched_tles=len(all_tles),
                    )
                except Exception as batch_error:
                    emit(
                        f"History batch {b_idx + 1}/{len(batches)} failed: {batch_error}",
                        pct=pct,
                        state="fetching_history",
                        last_batch_error=str(batch_error),
                    )

        emit(f"Deduplicating {len(all_tles)} TLE records...", pct=77, state="deduplicating", fetched_tles=len(all_tles))
        seen = set()
        unique_tles = []
        for tle in all_tles:
            key = tle[1][2:7].strip() + "|" + tle[1][18:32]
            if key not in seen:
                seen.add(key)
                unique_tles.append(tle)
        all_tles = unique_tles

        metadata_report = {"upserted": 0, "skipped": 0, "errors": 0, "total": 0}
        if metadata_rows:
            emit(f"Ingesting {len(metadata_rows)} SATCAT metadata records...", pct=79, state="ingesting_metadata", metadata_expected=len(metadata_rows))
            metadata_report = ingest_object_metadata(metadata_rows, source="Space-Track satcat", emit=emit)

        emit(f"Ingesting {len(all_tles)} unique TLE records into local database...", pct=80, state="ingesting", fetched_tles=len(all_tles), expected=len(all_tles), ingested=0)
        report = ingest_tles(all_tles, source=source_used, emit=emit)
        stats = get_stats()

        status = {
            "ok": True,
            "group": group,
            "source": source_used,
            "days": days,
            "last_fetched_at": datetime.now(timezone.utc).isoformat(),
            "n_tles_fetched": len(all_tles),
            "n_metadata_fetched": len(metadata_rows),
            "metadata_report": metadata_report,
            "report": report,
            "stats": stats,
            "logs_tail": logs[-30:],
        }
        Path("data/tle_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

        _write_refresh_progress(
            state="done",
            group=group,
            days=days,
            pct=100,
            message="TLE + metadata refresh completed.",
            n_objects=progress_context.get("n_objects"),
            fetched_tles=len(all_tles),
            metadata_fetched=len(metadata_rows),
            metadata_report=metadata_report,
            ingested=report.get("total", len(all_tles)),
            expected=len(all_tles),
            added=report.get("added"),
            skipped=report.get("skipped"),
            errors=report.get("errors"),
            report=report,
            stats=stats,
            logs_tail=logs[-10:],
            started_at=started_at,
            finished_at=time.time(),
        )
        return status

    except Exception as e:
        logger.exception("TLE refresh failed")
        payload_extra = {k: v for k, v in progress_context.items() if k != "state"}
        _write_refresh_progress(
            state="error",
            group=group,
            days=days,
            pct=progress_context.get("pct", 0),
            message="TLE + metadata refresh failed.",
            error=str(e),
            logs_tail=logs[-10:],
            started_at=started_at,
            finished_at=time.time(),
            **payload_extra,
        )
        raise


def load_catalog_status():
    status_file = Path("data") / "tle_status.json"
    if not status_file.exists():
        return {"ok": False, "status": "not_refreshed_yet", "message": "No TLE refresh has been performed yet."}
    try:
        return json.loads(status_file.read_text(encoding="utf-8"))
    except Exception as e:
        return {"ok": False, "status": "status_read_error", "message": str(e)}
