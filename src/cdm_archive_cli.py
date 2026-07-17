"""CLI entry point using the production Space-Track CDM provider."""

from __future__ import annotations

import cdm_auto_sync
from cdm_archive_provider import fetch_available_cdms


def main() -> int:
    cdm_auto_sync.fetch_available_cdms = fetch_available_cdms
    return cdm_auto_sync.main()


if __name__ == "__main__":
    raise SystemExit(main())
