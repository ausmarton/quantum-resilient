Reproducibility
===============

Local
-----
- Deterministic environment:
  - `TZ=UTC`, `LC_ALL=C`, `PYTHONHASHSEED=0`, and single-thread BLAS.
- Steps:
  1) Build/run local stack (optional for Prometheus):
     - `bash scripts/run_local.sh`
  2) Emit structured metrics (optional check):
     - `cargo run --quiet --release --manifest-path src/rust_core/Cargo.toml --bin emit_metrics`
  3) Run orchestrator:
     - `pqc-orchestrator --config ./configs/default.yaml` or `python -m python_orchestrator.cli -c ./configs/default.yaml`
  4) Artifacts in `./results/`: `metrics.jsonl/csv`, `raw_events.csv`, `summary.*`, `charts/`, `report.zip`, `env.json`, `metrics_validation.json`.

Reproduction script
-------------------
- Exact reproducibility check (runs twice and compares):
  - `bash scripts/reproduce.sh --config ./configs/default.yaml --mode local`
  - Compares canonicalized `environment.json`, checksums of `metrics.csv/jsonl`, `summary.*`, `analysis.ipynb`, and `charts/`.

GCP (GKE)
---------
- Provision + run:
  - `bash scripts/run_gcp.sh --config ./configs/default.yaml --project <PROJECT> --region us-central1 --cluster pqc-benchmark --bucket <GCS_BUCKET>`
  - Downloads `/results` from the orchestrator job pod into `./results/gcp_<cluster>_<timestamp>/`.
- Terraform creates:
  - GKE cluster, node pool, GCS bucket, service account with bucket object admin.
- Helm deploys:
  - Rust core Deployment and Service (metrics), orchestrator Job mounting ConfigMap with experiment YAML.

CI
--
- On each PR/main push:
  - Build: Rust + Python; lint; tests
  - Emit metrics via Rust helper; run orchestrator; reproducibility (local mode)
  - Upload `results/report.zip` as artifact

Notes
-----
- Keep `configs/default.yaml` as the canonical input for integration runs.
- Pin versions via CI + Terraform; capture `environment.json` for any run.


