#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
COMPOSE_FILE="${ROOT}/docker/docker-compose.yml"

echo "[destroy] Tearing down local stack and removing volumes..."
docker compose -f "${COMPOSE_FILE}" down -v


