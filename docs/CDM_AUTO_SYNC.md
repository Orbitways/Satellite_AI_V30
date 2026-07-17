# Automatic Space-Track CDM archive

The backend includes a durable collector that retrieves every `cdm_public`
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
- The production provider maps the public `CREATED`, `MIN_RNG`, `PC`,
  `SAT_1_ID` and `SAT_2_ID` fields while retaining the complete raw response.

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

## Local or Codespaces operation

From the repository root:

```bash
bash scripts/start_backend_with_cdm_sync.sh
```

Run one synchronization immediately:

```bash
bash scripts/run_cdm_auto_sync.sh --once --force
```

Show the durable scheduler history and CDM database status:

```bash
bash scripts/run_cdm_auto_sync.sh --status
```

## Always-on production operation

Use the Docker VPS deployment under `deploy/`. It provides persistent host
storage, automatic HTTPS, restart policies, health checks and daily SQLite
backups. See `deploy/README.md` for the exact installation commands.

A GitHub Codespace is not a production host: when it stops or times out, all
running processes stop. Uninterrupted collection requires an always-on VPS or
another persistent server.
