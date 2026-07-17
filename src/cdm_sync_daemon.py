"""Strict eight-hour daemon for automatic Space-Track CDM collection.

Unlike the interactive worker loop, this deployment entry point waits a full
configured interval after every attempt, including failed attempts. This keeps
all-constellation Space-Track calls within the intended three-per-day cadence.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import uuid

from cdm_auto_sync import (
    DEFAULT_INITIAL_LOOKBACK_DAYS,
    DEFAULT_INTERVAL_HOURS,
    DEFAULT_MAX_LOOKBACK_DAYS,
    DEFAULT_MAX_RECORDS,
    DEFAULT_OVERLAP_HOURS,
    run_sync_once,
)

LOGGER = logging.getLogger("orbitways.cdm_sync_daemon")


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


def main() -> int:
    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    interval_hours = _env_float("CDM_SYNC_INTERVAL_HOURS", DEFAULT_INTERVAL_HOURS)
    initial_lookback_days = _env_int(
        "CDM_SYNC_INITIAL_LOOKBACK_DAYS", DEFAULT_INITIAL_LOOKBACK_DAYS
    )
    overlap_hours = _env_float("CDM_SYNC_OVERLAP_HOURS", DEFAULT_OVERLAP_HOURS)
    max_lookback_days = _env_int(
        "CDM_SYNC_MAX_LOOKBACK_DAYS", DEFAULT_MAX_LOOKBACK_DAYS
    )
    max_records = _env_int("CDM_SYNC_MAX_RECORDS", DEFAULT_MAX_RECORDS)
    if interval_hours <= 0:
        raise SystemExit("CDM_SYNC_INTERVAL_HOURS must be positive")

    stop_event = threading.Event()

    def request_stop(signum, _frame):
        LOGGER.info("Received signal %s; stopping CDM synchronization daemon", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    worker_id = f"daemon:{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

    while not stop_event.is_set():
        try:
            result = run_sync_once(
                force=True,
                interval_hours=interval_hours,
                initial_lookback_days=initial_lookback_days,
                overlap_hours=overlap_hours,
                max_lookback_days=max_lookback_days,
                max_records=max_records,
                worker_id=worker_id,
            )
            LOGGER.info("CDM synchronization result: %s", result)
        except Exception:
            LOGGER.exception(
                "CDM synchronization failed; next attempt remains on the configured cadence"
            )

        stop_event.wait(interval_hours * 3600.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
