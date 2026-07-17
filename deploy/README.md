# Always-on deployment

This deployment runs the FastAPI backend, the Space-Track CDM collector and a
daily SQLite backup service on an always-on VPS. The database is stored on the
VPS under `deploy/persistent/data`, independently of container restarts.

## Recommended host

Use a small Docker VPS located in France. For the current MVP, 2 vCPU and 4 GB
RAM are sufficient for CDM collection and light API use. Choose 4 vCPU and 8 GB
RAM if the same server will regularly run full-catalog conjunction calculations.

## 1. Create the VPS and DNS record

Install Ubuntu 24.04 or Debian 12. Create an `A` record such as
`api.orbitways.com` pointing to the VPS IPv4 address.

## 2. Bootstrap the server

Connect by SSH and run:

```bash
git clone https://github.com/Orbitways/Satellite_AI_V30.git
cd Satellite_AI_V30
git checkout agent/cdm-auto-sync
sudo bash deploy/bootstrap_vps.sh
```

The script installs Docker when required, enables the firewall, opens SSH/HTTP/HTTPS,
and creates the persistent database and backup directories.

## 3. Configure secrets

Edit `/opt/orbitways/Satellite_AI_V30/deploy/.env` and replace every placeholder:

```bash
sudo nano /opt/orbitways/Satellite_AI_V30/deploy/.env
```

Generate the API token on the VPS:

```bash
openssl rand -hex 32
```

The populated `.env` file must never be committed.

## 4. Start the service

```bash
cd /opt/orbitways/Satellite_AI_V30/deploy
sudo docker compose up -d --build
```

Caddy obtains and renews HTTPS certificates automatically once DNS points to the
VPS and ports 80/443 are reachable.

## 5. Verify collection

```bash
sudo docker compose ps
sudo docker compose logs --tail=100 orbitways-api
sudo docker compose exec orbitways-api \
  bash scripts/run_cdm_auto_sync.sh --status
curl -sS https://api.orbitways.com/health
```

Run an immediate first import instead of waiting for the next cycle:

```bash
sudo docker compose exec orbitways-api \
  bash scripts/run_cdm_auto_sync.sh --once --force
```

## Persistence and backups

- Live database: `deploy/persistent/data/tle_database.sqlite`
- Daily backups: `deploy/persistent/backups/tle_database-*.sqlite`
- Default backup retention: 30 days
- LWS snapshots/backups are useful, but keep the application-level SQLite backups too.

## Updating after a merge

```bash
cd /opt/orbitways/Satellite_AI_V30
sudo BRANCH=main bash deploy/update_vps.sh
```

## Important

Do not use Codespaces as the production host. A stopped Codespace terminates all
running processes, so no background collector can run there.
