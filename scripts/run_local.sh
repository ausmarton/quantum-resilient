#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
COMPOSE_FILE="${ROOT}/docker/docker-compose.yml"

echo "[run_local] Starting local stack (rust-core, orchestrator, prometheus)..."
docker compose -f "${COMPOSE_FILE}" up -d --build
echo "[run_local] Prometheus UI: http://localhost:9090"
echo "[run_local] Rust core metrics (if enabled): http://localhost:9100/metrics"

echo "[run_local] Ensuring PyO3 extension and orchestrator are available..."
if command -v python3 >/dev/null 2>&1; then
	if ! python3 -c 'import pqc_core' >/dev/null 2>&1; then
		echo "[run_local] Building and installing pqc_core (PyO3) via maturin..."
		python3 -m pip install -q --upgrade pip
		python3 -m pip install -q maturin
		python3 -m maturin develop -m "${ROOT}/src/rust_core/Cargo.toml"
	fi
	python3 -m pip install -q -e "${ROOT}/src/python_orchestrator" || true
	echo "[run_local] Emitting metrics and running comparisons..."
	cargo run --quiet --release --manifest-path "${ROOT}/src/rust_core/Cargo.toml" --bin emit_metrics || true
	cargo run --quiet --release --manifest-path "${ROOT}/src/rust_core/Cargo.toml" --bin run_comparisons || true
	echo "[run_local] Running orchestrator..."
	python3 -m python_orchestrator.cli --config "${ROOT}/configs/default.yaml" || true
	echo "[run_local] Results written to ${ROOT}/results"
else
	echo "[run_local] Python3 not found; skipping orchestrator execution."
fi


