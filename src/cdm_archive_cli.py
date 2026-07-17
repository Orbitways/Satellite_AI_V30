"""CLI entry point using the production Space-Track CDM provider."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load_repository_env() -> None:
    """Load Space-Track credentials from the repository .env when not exported."""
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in {
            "SPACETRACK_EMAIL",
            "SPACETRACK_PASSWORD",
            "SPACETRACK_USER",
            "SPACETRACK_PASS",
        }:
            os.environ.setdefault(key, value)


_load_repository_env()

import cdm_auto_sync  # noqa: E402
from cdm_archive_provider import fetch_available_cdms  # noqa: E402


def main() -> int:
    cdm_auto_sync.fetch_available_cdms = fetch_available_cdms
    return cdm_auto_sync.main()


if __name__ == "__main__":
    raise SystemExit(main())
