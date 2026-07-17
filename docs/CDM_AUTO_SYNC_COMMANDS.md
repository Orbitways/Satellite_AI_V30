# CDM synchronization commands

Start the API and the automatic 8-hour worker together:

```bash
bash scripts/start_backend_with_cdm_sync.sh
```

Run a synchronization immediately:

```bash
bash scripts/run_cdm_auto_sync.sh --once --force
```

Inspect scheduler and database status:

```bash
bash scripts/run_cdm_auto_sync.sh --status
```
