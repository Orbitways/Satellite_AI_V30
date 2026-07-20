#!/usr/bin/env bash
set -euo pipefail
A="${1:?archive path required}"
mkdir -p data "$A/data"
[[ ! -f "$A/data/tle_database.sqlite.gz" ]] || gzip -dc "$A/data/tle_database.sqlite.gz" > data/tle_database.sqlite
bash scripts/run_cdm_auto_sync.sh --once --force
bash scripts/run_cdm_auto_sync.sh --status > /tmp/cdm_status.json
python - <<'PY'
import sqlite3
s=sqlite3.connect('data/tle_database.sqlite'); d=sqlite3.connect('/tmp/cdm.sqlite'); s.backup(d); d.close(); s.close()
PY
gzip -9 -c /tmp/cdm.sqlite > "$A/data/tle_database.sqlite.gz"
cp /tmp/cdm_status.json "$A/data/cdm_archive_status.json"
cd "$A"
git config user.name github-actions[bot]
git config user.email 41898282+github-actions[bot]@users.noreply.github.com
git add -f data/tle_database.sqlite.gz data/cdm_archive_status.json
git diff --cached --quiet && exit 0
git commit -m "chore(cdm): update archive $(date -u +%Y-%m-%dT%H:%MZ)"
git push origin HEAD:cdm-archive
