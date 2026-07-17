# Automatic Space-Track CDM archive

The backend can now run a separate worker that retrieves every `cdm_public`
record visible to the configured Space-Track account and stores it in the
existing SQLite `cdm_records` table.

## Behaviour

- One all-constellation Space-Track request every 8 hours.
- First successful run backfills the available 30-day window.
- Later runs start from the previous success time minus a 24-hour overlap.
- Existing `cdm_id` uniqueness makes overlapping downloads idempotent.
- Every attempt is recorded in `cdm_sync_runs`.
- Durable scheduler state is stored in `cdm_sync_state`.
- A SQLite lease prevents duplicate downloads when two workers start.
- After a restart, the worker resumes from its persisted checkpoint.

Space-Track only returns CDMs that the account is entitled to access. Empty
results must therefore not be interpreted as zero conjunction risk.

## Required environment variables

```bash
export SPACETRACK_EMAIL="your-account@example.com"
export SPACETRACK_PASSWORD="your-password"
```

The existing aliases `SPACETRACK_USER` and `SPACETRACK_PASS` also work.

Optional configuration:

```bash
export CDM_SYNC_INTERVAL_HOURS=8
export CDM_SYNC_INITIAL_LOOKBACK_DAYS=30
export CDM_SYNC_OVERLAP_HOURS=24
export CDM_SYNC_MAX_LOOKBACK_DAYS=30
export CDM_SYNC_MAX_RECORDS=50000
```

## Start API and worker together

From the repository root:

```bash
./scripts/start_backend_with_cdm_sync.sh
```

The default API command is equivalent to:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001
```

Override `HOST`, `PORT`, or `UVICORN_APP` when required.

## Manual operations

Run one synchronization immediately:

```bash
PYTHONPATH=src python -m cdm_auto_sync --once --force
```

Show the durable scheduler history and CDM database status:

```bash
PYTHONPATH=src python -m cdm_auto_sync --status
```

## Important deployment limitation

The worker only runs while the backend host is awake. A GitHub Codespace that
is stopped or suspended cannot execute an 8-hour job. The worker catches up on
restart within Space-Track's available lookback window, but uninterrupted
collection requires an always-on host or an external scheduler running
`python -m cdm_auto_sync --once` against a persistent database volume.
