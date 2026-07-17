"""Create consistent SQLite backups while the API is running."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOGGER = logging.getLogger("orbitways.sqlite_backup")
DEFAULT_SOURCE = Path("data/tle_database.sqlite")
DEFAULT_DESTINATION = Path("backups")


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


def create_backup(source: Path, destination: Path) -> Path | None:
    if not source.exists():
        LOGGER.warning("Database does not exist yet: %s", source)
        return None

    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = destination / f"tle_database-{timestamp}.sqlite"
    temporary = target.with_suffix(".sqlite.tmp")

    with sqlite3.connect(source) as source_db:
        with sqlite3.connect(temporary) as backup_db:
            source_db.backup(backup_db)
            result = backup_db.execute("PRAGMA integrity_check").fetchone()
            if not result or result[0] != "ok":
                raise RuntimeError(f"SQLite integrity check failed: {result}")

    temporary.replace(target)
    LOGGER.info("Created SQLite backup: %s", target)
    return target


def prune_backups(destination: Path, retention_days: int) -> int:
    if retention_days <= 0 or not destination.exists():
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    removed = 0
    for path in destination.glob("tle_database-*.sqlite"):
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if modified < cutoff:
            path.unlink()
            removed += 1
    if removed:
        LOGGER.info("Removed %s expired SQLite backup(s)", removed)
    return removed


def run_once(source: Path, destination: Path, retention_days: int) -> None:
    create_backup(source, destination)
    prune_backups(destination, retention_days)


def main() -> int:
    parser = argparse.ArgumentParser(description="Back up the Orbitways SQLite database")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=_env_float("SQLITE_BACKUP_INTERVAL_HOURS", 24.0),
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=_env_int("SQLITE_BACKUP_RETENTION_DAYS", 30),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.interval_hours <= 0:
        raise SystemExit("--interval-hours must be positive")

    stop_event = threading.Event()

    def stop(signum, _frame):
        LOGGER.info("Received signal %s; stopping backup worker", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while not stop_event.is_set():
        try:
            run_once(args.source, args.destination, args.retention_days)
        except Exception:
            LOGGER.exception("SQLite backup failed")
        if not args.loop:
            break
        stop_event.wait(args.interval_hours * 3600.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
