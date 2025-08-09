## API Overview

### Python Orchestrator
- Module: `src/python_orchestrator/main.py`
  - Class `QuantumResilientFramework(config: dict)`
    - `run_objective_3_benchmarking() -> dict`
    - `run_objective_4_comparison() -> dict`
    - `run_focused_benchmarks() -> dict`
  - CLI subcommands:
    - `benchmark` — Objective 3
    - `compare` — Objective 4
    - `run` — both

- Module: `src/python_orchestrator/benchmarking.py`
  - Class `ComprehensiveBenchmarker(config: dict)`
    - `run_comprehensive_benchmarks() -> dict`
    - `generate_visualizations(output_dir: str = 'results') -> None`
    - `save_results(output_dir: str = 'results') -> None`
  - Data classes:
    - `BenchmarkResult`
    - `ResourceMetrics`

### Real-World AML Simulation
- Module: `src/real_world/aml_pipeline.py`
  - Class `AMLTransaction`
  - Class `RealWorldAMLPipeline`
    - `process_transaction(tx: AMLTransaction) -> dict`
    - `get_performance_metrics() -> dict`
    - `generate_compliance_report(transactions: list[AMLTransaction]) -> dict`
  - Function `run_aml_simulation()`

### Rust Core (PyO3 module)
- Module: `src/rust_core/src/lib.rs` (exposed as `pqc_core`)
  - Class `CryptoBenchmark(name: str, key_size: u32)`
    - `measure_operation(operation: str, data_size: usize) -> float`
    - `run_benchmark(iterations: usize, data_sizes: Vec<usize>) -> float`
    - `get_statistics() -> dict`
  - Class `PipelineSimulator(name: str, key_size: u32)`
    - `process_transaction(id: str, amount: f64, sender: str, recipient: str) -> float`
    - `get_statistics() -> dict`
  - Class `OQSCrypto(algorithm_name: str)`
    - `generate_keys() -> (bytes, bytes)`
    - `encapsulate(data: bytes, public_key: bytes) -> (bytes, bytes)`
    - `decapsulate(ciphertext: bytes, secret_key: bytes) -> bytes`
    - `sign(data: bytes, secret_key: bytes) -> bytes`
    - `verify(data: bytes, signature: bytes, public_key: bytes) -> bool`
    - Backward-compatible: `encrypt`, `decrypt`

### Results
- All APIs write results to `results/` by default; controlled via `config.yaml` or CLI flags.
