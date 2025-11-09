Architecture
============

Overview
--------
This project benchmarks post-quantum cryptography (PQC) vs classical algorithms across two modes:
1) Hybrid TLS-like handshakes (KEM + symmetric; ECDHE/RSA + symmetric).
2) Application-layer streaming (AES-GCM payloads with deterministic seeds).

Components
----------
- Rust core (`src/rust_core/`):
  - `adapters/*`: Algorithm adapters implementing `CryptoAdapter` for Kyber/Dilithium/RSA/ECDHE/ECDSA.
  - `workload`: Deterministic streaming workloads, backpressure (Block/Drop), retries with backoff.
  - `modes`: Scenario wrappers (hybrid TLS-like, streaming, key-wrap and integrity like-for-like).
  - `metrics`: Collectors (JSONL, Prometheus) and resource sampling (CPU, RSS, disk, net).
  - `emit_metrics` binary: Emits sample JSON events for CI/integration runs.
- Python orchestrator (`src/python_orchestrator/`):
  - `config_loader`: Loads experiment YAML.
  - `runner`: Warmup, repetitions, aggregation, reporting, metrics validation.
  - `analysis`/`reporting`: Stats, charts, notebook, report.zip.
  - `schema_validate`: Validates JSON metrics against `configs/metrics_schema.yaml`.
- Packaging & Deploy:
  - Docker Compose: Rust core, orchestrator, Prometheus.
  - Helm (K8s): Deploys rust-core and orchestrator job; optional Prometheus scrape annotations.
  - Terraform (GCP): GKE cluster, GCS bucket, service account/IAM.
  - GitHub Actions: Build, lint, tests, short integration, reproducibility, report artifact.

Data Flow
---------
Experiment YAML → Python Orchestrator (PyO3 binding points) → Rust core (adapters/workload/modes) → JSONL metrics + Prometheus → Python aggregation/report → report.zip

- Parameters (workload, TPS, duration, chunk size, retries, backpressure) flow from YAML to orchestrator to Rust.
- Rust emits per-operation events (`metrics.jsonl`); Python converts to CSV (`metrics.csv`) and creates `raw_events.csv`.
- Aggregated outputs (`summary.csv`, `summary.json`) + charts + notebook + `report.zip`.

Extensibility
-------------
- Add new algorithms: implement `CryptoAdapter` and register in `adapters::mod`.
- Add new metrics sinks: implement `MetricsCollector`.
- Add new workloads/modes: extend `workload` or `modes` and expose via PyO3.


