#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
COMPOSE_FILE="${ROOT}/docker/docker-compose.yml"

echo "[run_local] Starting local stack (rust-core, orchestrator, prometheus)..."
docker compose -f "${COMPOSE_FILE}" up -d --build
echo "[run_local] Prometheus UI: http://localhost:9090"
echo "[run_local] Rust core metrics (if enabled): http://localhost:9100/metrics"


