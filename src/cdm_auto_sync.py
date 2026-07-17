"""Durable Space-Track CDM ingestion worker.

The worker downloads all ``cdm_public`` rows visible to the configured
Space-Track account, stores them through ``cdm_database.ingest_cdm_records`` and
keeps a durable sync history in the same SQLite database as the rest of the
backend.

Run it as a long-lived process or use ``--once`` from an external scheduler.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import quote

from cdm_database import cdm_status, ingest_cdm_records, init_cdm_db
from tle_database import get_connection

LOGGER = logging.getLogger("orbitways.cdm_auto_sync")
PROVIDER = "spacetrack_cdm_public"
DEFAULT_INTERVAL_HOURS = 8.0
DEFAULT_INITIAL_LOOKBACK_DAYS = 30
DEFAULT_OVERLAP_HOURS = 24.0
DEFAULT_MAX_LOOKBACK_DAYS = 30
DEFAULT_MAX_RECORDS = 50000
LEASE_MINUTES = 60


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    candidate = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _decode(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _json_rows(raw_text: str) -> list[dict[str, Any]]:
    parsed = json.loads((raw_text or "").strip() or "[]")
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [row for row in parsed if isinstance(row, dict)]
    raise ValueError("Space-Track returned JSON that is neither an object nor an array")


def _env_aliases() -> None:
    if os.environ.get("SPACETRACK_USER") and not os.environ.get("SPACETRACK_EMAIL"):
        os.environ["SPACETRACK_EMAIL"] = os.environ["SPACETRACK_USER"]
    if os.environ.get("SPACETRACK_PASS") and not os.environ.get("SPACETRACK_PASSWORD"):
        os.environ["SPACETRACK_PASSWORD"] = os.environ["SPACETRACK_PASS"]


def _credentials_configured() -> bool:
    user = os.environ.get("SPACETRACK_EMAIL") or os.environ.get("SPACETRACK_USER")
    password = os.environ.get("SPACETRACK_PASSWORD") or os.environ.get("SPACETRACK_PASS")
    return bool(user and password)


def init_sync_db() -> None:
    """Create scheduler state and audit tables without replacing existing data."""
    init_cdm_db()
    conn = get_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cdm_sync_state (
                provider TEXT PRIMARY KEY,
                last_started_at TEXT,
                last_success_at TEXT,
                last_window_start TEXT,
                last_window_end TEXT,
                last_raw_rows INTEGER DEFAULT 0,
                last_upserted INTEGER DEFAULT 0,
                last_error TEXT,
                next_due_at TEXT,
                lease_owner TEXT,
                lease_until TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cdm_sync_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                window_start TEXT NOT NULL,
                window_end TEXT NOT NULL,
                raw_rows INTEGER DEFAULT 0,
                upserted_rows INTEGER DEFAULT 0,
                skipped_rows INTEGER DEFAULT 0,
                error_rows INTEGER DEFAULT 0,
                error_text TEXT,
                worker_id TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_cdm_sync_runs_provider_started
                ON cdm_sync_runs(provider, started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_cdm_sync_runs_status
                ON cdm_sync_runs(status);
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO cdm_sync_state(provider, updated_at) VALUES (?, ?)",
            (PROVIDER, _iso(_utcnow())),
        )
        conn.commit()
    finally:
        conn.close()


def _load_state(conn) -> dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM cdm_sync_state WHERE provider=?", (PROVIDER,)
    ).fetchone()
    return dict(row) if row else {}


def _calculate_window(
    state: dict[str, Any],
    now: datetime,
    initial_lookback_days: int,
    overlap_hours: float,
    max_lookback_days: int,
) -> tuple[datetime, datetime, float]:
    last_success = _parse_dt(state.get("last_success_at"))
    retention_floor = now - timedelta(days=max_lookback_days)
    if last_success is None:
        start = now - timedelta(days=min(initial_lookback_days, max_lookback_days))
        coverage_gap_hours = 0.0
    else:
        desired_start = last_success - timedelta(hours=overlap_hours)
        start = max(desired_start, retention_floor)
        coverage_gap_hours = max(
            0.0, (retention_floor - desired_start).total_seconds() / 3600.0
        )
    return start, now, round(coverage_gap_hours, 3)


def _acquire_lease(
    worker_id: str,
    interval_hours: float,
    force: bool,
    initial_lookback_days: int,
    overlap_hours: float,
    max_lookback_days: int,
) -> dict[str, Any]:
    now = _utcnow()
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        state = _load_state(conn)
        lease_until = _parse_dt(state.get("lease_until"))
        lease_owner = state.get("lease_owner")
        if lease_until and lease_until > now and lease_owner != worker_id:
            conn.rollback()
            return {
                "acquired": False,
                "reason": "locked",
                "lease_owner": lease_owner,
                "lease_until": _iso(lease_until),
            }

        last_success = _parse_dt(state.get("last_success_at"))
        next_due = (
            last_success + timedelta(hours=interval_hours) if last_success else None
        )
        if not force and next_due and next_due > now:
            conn.rollback()
            return {
                "acquired": False,
                "reason": "not_due",
                "next_due_at": _iso(next_due),
            }

        window_start, window_end, gap_hours = _calculate_window(
            state=state,
            now=now,
            initial_lookback_days=initial_lookback_days,
            overlap_hours=overlap_hours,
            max_lookback_days=max_lookback_days,
        )
        lease_expiry = now + timedelta(minutes=LEASE_MINUTES)
        conn.execute(
            """
            UPDATE cdm_sync_state
            SET last_started_at=?, lease_owner=?, lease_until=?, updated_at=?
            WHERE provider=?
            """,
            (_iso(now), worker_id, _iso(lease_expiry), _iso(now), PROVIDER),
        )
        cursor = conn.execute(
            """
            INSERT INTO cdm_sync_runs(
                provider, started_at, status, window_start, window_end, worker_id
            ) VALUES (?, ?, 'running', ?, ?, ?)
            """,
            (
                PROVIDER,
                _iso(now),
                _iso(window_start),
                _iso(window_end),
                worker_id,
            ),
        )
        run_id = int(cursor.lastrowid)
        conn.commit()
        return {
            "acquired": True,
            "run_id": run_id,
            "window_start": window_start,
            "window_end": window_end,
            "coverage_gap_hours": gap_hours,
        }
    finally:
        conn.close()


def _format_query_time(value: datetime) -> str:
    raw = value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return quote(raw, safe="-T")


def _query_path(window_start: datetime, window_end: datetime, max_records: int) -> str:
    start = _format_query_time(window_start)
    end = _format_query_time(window_end)
    return (
        "/class/cdm_public"
        f"/CREATION_DATE/{start}--{end}"
        "/orderby/CREATION_DATE%20asc"
        "/format/json"
        f"/limit/{int(max_records)}"
    )


def fetch_available_cdms(
    window_start: datetime,
    window_end: datetime,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> tuple[list[dict[str, Any]], str]:
    """Retrieve all CDMs visible to the authenticated Space-Track account."""
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
        rows = _json_rows(_decode(client._request(url)))
    return rows, path


def _finish_run(
    *,
    run_id: int,
    worker_id: str,
    interval_hours: float,
    status: str,
    window_start: datetime,
    window_end: datetime,
    raw_rows: int = 0,
    report: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    now = _utcnow()
    report = report or {}
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            UPDATE cdm_sync_runs
            SET finished_at=?, status=?, raw_rows=?, upserted_rows=?,
                skipped_rows=?, error_rows=?, error_text=?
            WHERE id=?
            """,
            (
                _iso(now),
                status,
                int(raw_rows),
                int(report.get("upserted", 0) or 0),
                int(report.get("skipped", 0) or 0),
                int(report.get("errors", 0) or 0),
                error,
                run_id,
            ),
        )
        if status == "success":
            next_due = now + timedelta(hours=interval_hours)
            conn.execute(
                """
                UPDATE cdm_sync_state
                SET last_success_at=?, last_window_start=?, last_window_end=?,
                    last_raw_rows=?, last_upserted=?, last_error=NULL,
                    next_due_at=?, lease_owner=NULL, lease_until=NULL, updated_at=?
                WHERE provider=? AND lease_owner=?
                """,
                (
                    _iso(now),
                    _iso(window_start),
                    _iso(window_end),
                    int(raw_rows),
                    int(report.get("upserted", 0) or 0),
                    _iso(next_due),
                    _iso(now),
                    PROVIDER,
                    worker_id,
                ),
            )
        else:
            retry_at = now + timedelta(minutes=30)
            conn.execute(
                """
                UPDATE cdm_sync_state
                SET last_error=?, next_due_at=?, lease_owner=NULL,
                    lease_until=NULL, updated_at=?
                WHERE provider=? AND lease_owner=?
                """,
                (error, _iso(retry_at), _iso(now), PROVIDER, worker_id),
            )
        conn.commit()
    finally:
        conn.close()


def run_sync_once(
    *,
    force: bool = False,
    interval_hours: float = DEFAULT_INTERVAL_HOURS,
    initial_lookback_days: int = DEFAULT_INITIAL_LOOKBACK_DAYS,
    overlap_hours: float = DEFAULT_OVERLAP_HOURS,
    max_lookback_days: int = DEFAULT_MAX_LOOKBACK_DAYS,
    max_records: int = DEFAULT_MAX_RECORDS,
    worker_id: Optional[str] = None,
) -> dict[str, Any]:
    """Run one durable, deduplicated synchronization cycle."""
    init_sync_db()
    worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    lease = _acquire_lease(
        worker_id=worker_id,
        interval_hours=interval_hours,
        force=force,
        initial_lookback_days=initial_lookback_days,
        overlap_hours=overlap_hours,
        max_lookback_days=max_lookback_days,
    )
    if not lease.get("acquired"):
        return {"ok": True, "ran": False, **lease}

    run_id = int(lease["run_id"])
    window_start = lease["window_start"]
    window_end = lease["window_end"]
    try:
        rows, query_path = fetch_available_cdms(
            window_start=window_start,
            window_end=window_end,
            max_records=max_records,
        )
        report = ingest_cdm_records(rows, source=PROVIDER)
        if int(report.get("errors", 0) or 0) > 0:
            raise RuntimeError(
                f"CDM ingestion completed with {report.get('errors')} database errors: "
                f"{report.get('error_samples', [])[:3]}"
            )
        _finish_run(
            run_id=run_id,
            worker_id=worker_id,
            interval_hours=interval_hours,
            status="success",
            raw_rows=len(rows),
            report=report,
            window_start=window_start,
            window_end=window_end,
        )
        return {
            "ok": True,
            "ran": True,
            "run_id": run_id,
            "provider": PROVIDER,
            "window_start": _iso(window_start),
            "window_end": _iso(window_end),
            "coverage_gap_hours": lease.get("coverage_gap_hours", 0.0),
            "query_path": query_path,
            "raw_rows": len(rows),
            "import_report": report,
        }
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        _finish_run(
            run_id=run_id,
            worker_id=worker_id,
            interval_hours=interval_hours,
            status="failed",
            error=message[:2000],
            window_start=window_start,
            window_end=window_end,
        )
        raise


def sync_status(limit: int = 10) -> dict[str, Any]:
    init_sync_db()
    conn = get_connection()
    try:
        state = _load_state(conn)
        rows = conn.execute(
            """
            SELECT id, provider, started_at, finished_at, status, window_start,
                   window_end, raw_rows, upserted_rows, skipped_rows, error_rows,
                   error_text, worker_id
            FROM cdm_sync_runs
            WHERE provider=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (PROVIDER, max(1, min(int(limit), 100))),
        ).fetchall()
    finally:
        conn.close()
    return {
        "ok": True,
        "provider": PROVIDER,
        "credentials_configured": _credentials_configured(),
        "state": state,
        "recent_runs": [dict(row) for row in rows],
        "cdm_database": cdm_status(),
    }


def _next_delay(default_seconds: float = 300.0) -> float:
    status = sync_status(limit=1)
    next_due = _parse_dt(status.get("state", {}).get("next_due_at"))
    if next_due is None:
        return 60.0
    seconds = (next_due - _utcnow()).total_seconds()
    return max(30.0, min(seconds, default_seconds))


def run_forever(
    *,
    interval_hours: float,
    initial_lookback_days: int,
    overlap_hours: float,
    max_lookback_days: int,
    max_records: int,
) -> None:
    stop_event = threading.Event()

    def request_stop(signum, _frame):
        LOGGER.info("Received signal %s; stopping CDM synchronization worker", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    LOGGER.info(
        "Starting CDM synchronization worker: interval=%sh initial=%sd overlap=%sh",
        interval_hours,
        initial_lookback_days,
        overlap_hours,
    )
    while not stop_event.is_set():
        try:
            result = run_sync_once(
                interval_hours=interval_hours,
                initial_lookback_days=initial_lookback_days,
                overlap_hours=overlap_hours,
                max_lookback_days=max_lookback_days,
                max_records=max_records,
                worker_id=worker_id,
            )
            LOGGER.info("CDM synchronization result: %s", json.dumps(result, default=str))
        except Exception:
            LOGGER.exception("CDM synchronization failed; the worker will retry")
        stop_event.wait(_next_delay())


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synchronize Space-Track CDMs into SQLite")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="run one sync cycle and exit")
    mode.add_argument("--status", action="store_true", help="print scheduler/database status and exit")
    parser.add_argument("--force", action="store_true", help="ignore the next-due timestamp")
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=_env_float("CDM_SYNC_INTERVAL_HOURS", DEFAULT_INTERVAL_HOURS),
    )
    parser.add_argument(
        "--initial-lookback-days",
        type=int,
        default=_env_int("CDM_SYNC_INITIAL_LOOKBACK_DAYS", DEFAULT_INITIAL_LOOKBACK_DAYS),
    )
    parser.add_argument(
        "--overlap-hours",
        type=float,
        default=_env_float("CDM_SYNC_OVERLAP_HOURS", DEFAULT_OVERLAP_HOURS),
    )
    parser.add_argument(
        "--max-lookback-days",
        type=int,
        default=_env_int("CDM_SYNC_MAX_LOOKBACK_DAYS", DEFAULT_MAX_LOOKBACK_DAYS),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=_env_int("CDM_SYNC_MAX_RECORDS", DEFAULT_MAX_RECORDS),
    )
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.interval_hours <= 0:
        raise SystemExit("--interval-hours must be positive")
    if args.initial_lookback_days <= 0 or args.max_lookback_days <= 0:
        raise SystemExit("lookback days must be positive")
    if args.max_records <= 0:
        raise SystemExit("--max-records must be positive")

    if args.status:
        print(json.dumps(sync_status(), indent=2, default=str))
        return 0
    if args.once:
        try:
            result = run_sync_once(
                force=args.force,
                interval_hours=args.interval_hours,
                initial_lookback_days=args.initial_lookback_days,
                overlap_hours=args.overlap_hours,
                max_lookback_days=args.max_lookback_days,
                max_records=args.max_records,
            )
        except Exception as exc:
            LOGGER.exception("CDM synchronization failed")
            print(json.dumps({"ok": False, "error": str(exc)}))
            return 1
        print(json.dumps(result, indent=2, default=str))
        return 0

    run_forever(
        interval_hours=args.interval_hours,
        initial_lookback_days=args.initial_lookback_days,
        overlap_hours=args.overlap_hours,
        max_lookback_days=args.max_lookback_days,
        max_records=args.max_records,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
