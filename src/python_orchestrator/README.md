Python Orchestrator
===================

Skeleton for the Python orchestrator component. No functional code yet.

Notes
-----
- Configuration files are under `configs/`.
- Add orchestrator modules in future development phases.

CLI (skeleton)
--------------
- Install (editable):
  - `pip install -e .`
- Run:
  - `pqc-orchestrator --config ./configs/default.yaml`

Responsibilities (planned)
-------------------------
- Load experiment YAML.
- Dynamically load Rust adapters via PyO3 when available.
- Run warmup, repetitions, and delegate work to Rust core.
- Aggregate raw metrics (JSONL -> CSV) and capture environment snapshot.


