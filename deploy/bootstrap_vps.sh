#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script as root: sudo bash deploy/bootstrap_vps.sh" >&2
  exit 1
fi

REPO_DIR="${REPO_DIR:-/opt/orbitways/Satellite_AI_V30}"
BRANCH="${BRANCH:-agent/cdm-auto-sync}"

apt-get update
apt-get install -y ca-certificates curl git docker.io docker-compose-v2 ufw
systemctl enable --now docker

mkdir -p "$(dirname "$REPO_DIR")"
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone https://github.com/Orbitways/Satellite_AI_V30.git "$REPO_DIR"
fi

cd "$REPO_DIR"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

mkdir -p deploy/persistent/data deploy/persistent/backups
if [[ ! -f deploy/.env ]]; then
  cp deploy/.env.example deploy/.env
  chmod 600 deploy/.env
  echo
  echo "Edit $REPO_DIR/deploy/.env before starting the service."
  echo "Then run:"
  echo "  cd $REPO_DIR/deploy"
  echo "  docker compose up -d --build"
fi

ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

echo
echo "VPS bootstrap complete."
echo "Configuration file: $REPO_DIR/deploy/.env"
