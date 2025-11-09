Contributing
============

Thank you for your interest in contributing! This repository is currently a scaffold with placeholders.

How to Contribute
-----------------
- Open an issue to propose changes.
- For documentation updates, modify files under `docs/` or top-level markdown files.
- For code scaffolding updates, keep changes minimal and non-functional as per current project phase.

Development Workflow
--------------------
- Rust:
  - Build: `cargo build --release` in `src/rust_core/`
  - Lint: `cargo fmt --all -- --check`, `cargo clippy`
  - Test: `cargo test --all`
- Python:
  - Install: `pip install -e src/python_orchestrator`
  - Lint: `ruff check`, `flake8`
  - Test: `pytest -q`
- Integration:
  - Emit metrics: `cargo run --manifest-path src/rust_core/Cargo.toml --bin emit_metrics`
  - Orchestrator: `pqc-orchestrator --config ./configs/default.yaml`

Code of Conduct
---------------
Be respectful and constructive. Assume good faith and collaborate openly.

License
-------
By contributing, you agree that your contributions will be licensed under the project’s LICENSE.


