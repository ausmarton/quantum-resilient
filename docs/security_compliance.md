Security & Compliance
=====================

Synthetic Data Only
-------------------
- All benchmarks use synthetic or deterministic payloads; no PII or production data are processed.
- `DATA_POLICY.md` mandates synthetic inputs and prohibits secrets/PII in results.

License
-------
- Project is licensed under MIT (`LICENSE`). Switch to Apache-2.0 by replacing `LICENSE` and updating headers if desired.

Minimal IAM Roles
-----------------
- Terraform provisions a service account with `roles/storage.objectCreator` on the results GCS bucket.
  - Rationale: write-only (create new objects) minimizes blast radius; deletion/overwrite is not required.
  - If read-back from GCS is needed in-cluster, grant `roles/storage.objectViewer` to the workload identity or use short-lived signed URLs from CI.

Dependency Version Pinning
--------------------------
- Rust:
  - `src/rust_core/Cargo.lock` is checked in; Cargo uses it to pin transitive versions in CI.
  - Prefer updating via `cargo update -p <crate>@<version>` and commit the new lockfile.
- Python:
  - Use `src/python_orchestrator/constraints.txt` in CI to pin versions during installation.
  - For runtime reproducibility, create and ship a `requirements.txt` or a lockfile (e.g., `pip-tools`, `poetry`) and install with `-c constraints.txt` or equivalent.

Operational Guidance
--------------------
- Rotate credentials regularly; never commit secrets.
- Limit Prometheus `/metrics` exposure to trusted networks; disable or restrict in multi-tenant clusters.
- Regularly run `pip audit`/`cargo audit` and update pins as needed.


