#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REMOTE="${1:-origin}"
BRANCH="${2:-cdm-archive}"
ARCHIVE_PATH="data/tle_database.sqlite.gz"
DB_PATH="data/tle_database.sqlite"

mkdir -p data

echo "Fetching ${REMOTE}/${BRANCH}..."
git fetch "$REMOTE" "$BRANCH":refs/remotes/"$REMOTE"/"$BRANCH"

if ! git cat-file -e "$REMOTE/$BRANCH:$ARCHIVE_PATH" 2>/dev/null; then
  echo "Archive not found at $REMOTE/$BRANCH:$ARCHIVE_PATH" >&2
  exit 1
fi

tmp_file="$(mktemp "${TMPDIR:-/tmp}/tle_database.sqlite.XXXXXX")"
trap 'rm -f "$tmp_file"' EXIT

git show "$REMOTE/$BRANCH:$ARCHIVE_PATH" | gzip -dc > "$tmp_file"

python - "$tmp_file" <<'PY'
import sqlite3
import sys

path = sys.argv[1]
conn = sqlite3.connect(path)
try:
    result = conn.execute("PRAGMA integrity_check").fetchone()[0]
finally:
    conn.close()
if result != "ok":
    raise SystemExit(f"Downloaded SQLite archive failed integrity check: {result}")
PY

mv "$tmp_file" "$DB_PATH"
trap - EXIT
rm -f "${DB_PATH}-wal" "${DB_PATH}-shm"

echo "Restored $DB_PATH from $REMOTE/$BRANCH."
