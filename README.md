PQC Performance Benchmarking (Quantum-Resilient)
================================================

This repository provides a skeleton for a Post-Quantum Cryptography (PQC) performance benchmarking project. It includes placeholders for a Rust core, a Python orchestrator, containerization, CI, infrastructure-as-code, and deployment assets.

Getting Started
---------------
- Explore `docs/` for architecture, methodology, and reproducibility.
- Adjust configs in `configs/` (e.g., `default.yaml`).
- Use scripts in `./scripts` to run locally or with GCP.

Project Layout
--------------
- `src/rust_core/`: Rust core library (skeleton).
- `src/python_orchestrator/`: Python orchestrator (skeleton).
- `docker/`: Docker files (placeholders).
- `k8s/` and `helm/`: Kubernetes/Helm deployment manifests (placeholders).
- `terraform/gcp/`: Terraform for GKE deployment (placeholders).
- `.github/workflows/`: CI workflow skeleton.
- `configs/`, `examples/`, `results/`: Configuration, examples, and results directories.

Status
------
This is a scaffold only. No functional code is included yet.

Quickstart (Local)
------------------
Prereqs: Docker (optional), Rust toolchain, Python 3.11+ (for CLI).

- Emit metrics (optional):
  - `cargo run --quiet --release --manifest-path src/rust_core/Cargo.toml --bin emit_metrics`
- Run orchestrator:
  - `pip install -e src/python_orchestrator`
  - `pqc-orchestrator --config ./configs/default.yaml`
- Outputs: `./results/` contains `metrics.jsonl/csv`, `raw_events.csv`, `summary.csv/json/md`, `charts/`, `analysis.ipynb`, `report.zip`, `env.json`.
- Docker Compose (Prometheus + placeholders):
  - `bash scripts/run_local.sh`

Quickstart (GCP/GKE)
--------------------
- Provision & run:
  - `bash scripts/run_gcp.sh --config ./configs/default.yaml --project <PROJECT> --region us-central1 --cluster pqc-benchmark --bucket <GCS_BUCKET>`
- Results downloaded to `./results/gcp_<cluster>_<timestamp>/`.

Reproducibility
---------------
- Run twice and verify identity:
  - `bash scripts/reproduce.sh --config ./configs/default.yaml --mode local`


