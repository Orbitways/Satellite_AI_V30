# Deployment note

The 8-hour worker requires an awake backend host and persistent SQLite storage. GitHub Codespaces suspends background processes when stopped, so uninterrupted collection requires an always-on deployment or an external scheduler invoking the one-shot worker.
