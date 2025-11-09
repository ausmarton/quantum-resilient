Documentation
=============

This folder will contain project documentation, architecture notes, and design decisions.

Contents (planned)
------------------
- Architecture overview
- Benchmarking methodology
- Data handling and privacy guidance
- Deployment and operations notes

### Module responsibilities and interactions

- **Rust `pqc_core`**
  - **Adapters**: Thin wrappers around PQC algorithm implementations (e.g., external crates or FFI), exposing a uniform trait-based interface for keygen/sign/verify or KEM operations.
  - **Metrics**: Collect fine-grained timing, throughput, memory usage, error rates; expose counters/histograms for Prometheus via an optional HTTP endpoint.
  - **Workload engine**: Executes deterministic and stress workloads (single-threaded and concurrent), driven by parameters provided by the Python orchestrator.

- **Python orchestrator**
  - **Experiment execution**: Parses experiment YAML, validates schema, resolves datasets and parameters, and invokes `pqc_core` operations via PyO3 bindings.
  - **Analysis**: Post-processes raw metrics, aggregates statistics, and performs comparisons across algorithms/parameters/hardware.
  - **Reporting**: Writes results to `results/` in CSV/JSON; can render lightweight Markdown/HTML summaries for quick review.

- **Docker/K8s deployment & Terraform**
  - **Docker**: Container image wraps Python orchestrator and Rust core, providing reproducible runs locally and in CI.
  - **K8s/Helm**: Deploys the container to a cluster for scalable workloads; optional Service to expose the Prometheus `/metrics` endpoint.
  - **Terraform (GCP)**: Provisions a minimal GKE cluster and node pool for running orchestrated experiments at scale.

### Textual data-flow diagram

```text
Experiment YAML (configs/*.yaml)
        |
        v
Python Orchestrator
  - parse/validate YAML
  - build experiment plan
  - call Rust via PyO3
        |
        v
Rust pqc_core
  - adapters -> algorithm ops
  - workload engine executes runs
  - metrics collected (timings/mem/errs)
        |                          \
        |                           \ (optional) HTTP /metrics
        v                            \--> Prometheus scrape
Raw Metrics (in-memory / files)
        |
        v
Python Analysis & Reporting
  - aggregate/compare
  - write CSV/JSON to results/
  - generate brief reports (MD/HTML)
```

### Communication and metrics

- **PyO3 bindings**: The orchestrator imports a Python extension module compiled from `pqc_core`. Calls are in-process, avoiding serialization overhead. Data passed includes workload parameters and buffer payloads for crypto operations; results/metrics returned as structured objects.
- **Prometheus endpoint**: `pqc_core` can expose an optional HTTP server (configurable port) for counters/histograms during long runs. In K8s, this can be scraped by Prometheus via Service annotations; locally, it’s reachable from the host.

### Local vs cloud parity

- **Same image, same config**: Local `docker compose` and K8s/Helm use the same container image and experiment YAMLs, ensuring identical behavior across environments.
- **Environment-driven overrides**: Minor differences (e.g., concurrency, dataset paths, metrics port) are controlled via environment variables or Helm values, not code changes.
- **Results location**: Both local and cloud runs write to `results/` (bind mount or PVC) to preserve artifacts and facilitate comparison.


