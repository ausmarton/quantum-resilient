## Benchmarking Guide

This document explains how to run benchmarks, what metrics are collected, and how to customize parameters.

### Running Benchmarks
- Run all benchmarks and comparisons:
```bash
python src/python_orchestrator/main.py run
```
- Benchmark only (Objective 3):
```bash
python src/python_orchestrator/main.py benchmark
```
- Comparison only (Objective 4):
```bash
python src/python_orchestrator/main.py compare
```
- One-step runner:
```bash
python run_benchmarks.py
```

### What’s Measured
- Latency metrics: mean, median, min, max, std, p95, p99, confidence intervals (95%, 99%)
- Throughput: operations per second under varying batch sizes and concurrency
- Resource utilization: CPU%, memory usage over time
- Stress behavior: operations, error rate, max connections reached

### Where Results Go
- Directory: `results/`
  - `benchmark_results.json|csv|xlsx` — raw and summarized results
  - `latency_comparison.png` — bar charts per data size
  - `throughput_analysis.png` — throughput line plots
  - `performance_trends.png` — trends for latency/throughput/variance
  - `algorithm_comparison_heatmap.png` — heatmap of mean latencies
  - `interactive_dashboard.html` — interactive Plotly dashboard

### Configuration
`config.yaml` controls:
- Algorithms (`algorithms.classical`, `algorithms.pqc`)
- Benchmarks:
  - `benchmarks.latency_test.iterations` — samples per data size
  - `benchmarks.latency_test.data_sizes` — payload sizes
  - `benchmarks.latency_test.warmup_iterations`
  - `benchmarks.throughput_test.batch_sizes`
  - `benchmarks.throughput_test.duration_seconds`
  - `benchmarks.throughput_test.concurrent_operations`
  - `benchmarks.resource_test.sampling_interval_ms`
  - `benchmarks.stress_test.*`
- Output dir: `framework.output_directory`

### Statistical Methods
- Descriptive statistics (mean, median, std, percentiles)
- Student’s t-based confidence intervals using sample SEM
- ANOVA across algorithms per data size
- Linear regression for scalability (latency vs size)

### Extending Benchmarks
- Add a new algorithm: update `config.yaml` and, if needed, Rust core mapping in `src/rust_core/src/lib.rs`.
- Add a metric: extend `ComprehensiveBenchmarker` in `src/python_orchestrator/benchmarking.py`.
- Add a chart: implement a new method under `generate_visualizations()`.

### Troubleshooting
- Empty results: check logs `pqc_research.log` and ensure the Rust module is importable (mocks are used otherwise).
- Slow runs: reduce `iterations`, `duration_seconds`, and `data_sizes` in `config.yaml`, or run `make quick-benchmark`.
