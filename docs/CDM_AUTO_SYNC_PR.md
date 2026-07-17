## Change summary

This branch adds durable 8-hour Space-Track `cdm_public` synchronization, SQLite checkpoints and audit history, duplicate-worker protection, overlapping idempotent imports, restart recovery, command wrappers, documentation and focused unit tests.

## Validation

- Python syntax compilation passed.
- Shell syntax validation passed.
- Four focused tests passed.
- A live Space-Track call was not run because credentials are not available in the test environment.
