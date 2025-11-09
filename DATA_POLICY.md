Data Policy
===========

Scope
-----
This project is a scaffold and does not process real data yet.

Guiding Principles
------------------
- Minimize data collection and storage.
- Prefer anonymized/synthetic inputs for benchmarking.
- Clearly document any datasets, sources, and licenses under `docs/` or `examples/`.

Storage & Retention
-------------------
- Results in `results/` should avoid sensitive data; use `.gitignore` for large or sensitive artifacts.
- Remove any accidental sensitive data immediately and rotate any exposed credentials.
 - Structured logs (`metrics.jsonl`) and summaries should remain synthetic/aggregated; do not include PII.

Security
--------
- Do not commit secrets. Use environment variables or secret managers.
- Follow least-privilege for any cloud resources provisioned during testing.

Telemetry & Metrics
-------------------
- Prometheus metrics and JSONL logs include performance counters and resource usage only.
- Disable or restrict metrics exposure in untrusted environments; scrape only within trusted networks.



