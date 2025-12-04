# Reproducibility and Stability Test Suite

Tools for validating experimental reproducibility and detecting performance regressions.

## Overview

This suite provides:
- **Multiple run execution**: Run experiments N times for statistical analysis
- **Variance analysis**: Measure run-to-run variability
- **Confidence intervals**: Bootstrap and parametric CIs
- **Stability testing**: Distribution consistency across runs
- **Regression detection**: Compare against baselines
- **Cluster scaling analysis**: Study performance vs cluster size

## Quick Start

### Run Reproducibility Test

```bash
# Execute experiment 20 times
python reproducibility/runner.py \
  --scenario scenarios/kyber_benchmark.yaml \
  --runs 20 \
  --replicas 30 \
  --exp-prefix kyber_stability_test

# Analyze variance
python reproducibility/variance.py \
  --input reproducibility/output/kyber_stability_test_20250201_120000 \
  --out reproducibility/output/kyber_stability_test_20250201_120000/analysis

# Compute confidence intervals
python reproducibility/confidence.py \
  --input reproducibility/output/kyber_stability_test_20250201_120000 \
  --method bca

# Analyze stability
python reproducibility/stability.py \
  --input reproducibility/output/kyber_stability_test_20250201_120000
```

### Integrated Pipeline

```bash
# Run complete reproducibility analysis
python research/scripts/pipeline_runner.py \
  --exp-id exp_001 \
  --reproducibility \
  --runs 10 \
  --generate-report
```

## Scripts

### `runner.py`

Execute experiments multiple times:

```bash
python runner.py \
  --scenario scenario.yaml \
  --runs 20 \
  --replicas 30 \
  --exp-prefix test_001 \
  --parallel 4 \
  --timeout 3600
```

Options:
- `--runs N`: Number of experiment iterations
- `--replicas N`: Worker replicas per run
- `--parallel N`: Run N experiments concurrently
- `--timeout`: Timeout per run in seconds
- `--no-retry`: Don't retry failed runs

### `variance.py`

Analyze variance across runs:

```bash
python variance.py --input batch_dir --out analysis/
```

Output:
- `variance_summary.json`: Statistics and CV values
- `variance_plots.png`: Visualization

### `confidence.py`

Compute confidence intervals:

```bash
python confidence.py \
  --input batch_dir \
  --confidence 0.95 \
  --method bca \
  --n-bootstrap 10000
```

Methods:
- `bca`: Bias-corrected and accelerated bootstrap (recommended)
- `percentile`: Bootstrap percentile method
- `basic`: Basic bootstrap
- `normal`: Normal approximation

### `stability.py`

Analyze distribution stability:

```bash
python stability.py \
  --input batch_dir \
  --metric latency_us \
  --sample-size 10000
```

Output:
- `stability_summary.json`: Pairwise comparisons, tail stability
- `stability_matrix.png`: KS/Wasserstein distance heatmap

### `regression.py`

Detect performance regressions:

```bash
# Compare to previous batch
python regression.py \
  --current batch_002 \
  --baseline batch_001

# Compare to reference baseline
python regression.py \
  --current batch_002 \
  --reference baselines/v1.0.json \
  --latency-threshold 15
```

Output:
- `regression_report.json`: All comparisons
- `regression_failures.txt`: Failed thresholds

### `cluster_scaling.py`

Analyze cluster scaling behavior:

```bash
python cluster_scaling.py \
  --input scaling_experiments/ \
  --cluster-sizes 2 5 10 20 40
```

Output:
- `scaling_summary.json`: Model fit, saturation point
- `scaling_curve.png`: Throughput and efficiency plots

## Output Structure

```
reproducibility/output/
└── batch_id/
    ├── batch_metadata.json
    ├── run_000/
    │   ├── run_metadata.json
    │   ├── merged/
    │   └── stats/
    ├── run_001/
    │   └── ...
    └── analysis/
        ├── variance_summary.json
        ├── variance_plots.png
        ├── confidence_intervals.json
        ├── stability_summary.json
        ├── stability_matrix.png
        ├── regression_report.json
        └── reproducibility_report.md
```

## Interpretation Guide

### Coefficient of Variation (CV)

| CV Range | Interpretation |
|----------|----------------|
| < 10% | Low variability (excellent) |
| 10-25% | Moderate variability (acceptable) |
| > 25% | High variability (investigate) |

### Stability Assessment

- **KS p-value > 0.05**: Distributions are similar
- **Wasserstein distance**: Lower is better (in original units)
- **Tail stability (CV < 20%)**: p99/p99.9 are consistent

### Regression Thresholds

Default thresholds:
- Latency increase > 10%: Regression
- Throughput decrease > 10%: Regression
- Variance increase > 50%: Regression
- Tail (p99) increase > 15%: Regression

## Integration

### With Research Pipeline

```bash
python research/scripts/pipeline_runner.py \
  --exp-id exp_001 \
  --generate-all \
  --reproducibility \
  --runs 10
```

### With Packaging

Reproducibility artifacts are automatically included in bundles:

```bash
python -m packaging bundle exp_001
```

Bundle includes:
- `reproducibility/summary.json`
- `reproducibility/confidence_intervals.json`
- `reproducibility/stability_matrix.png`
- `reproducibility/reproducibility_report.md`

## Requirements

```
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.0.0
matplotlib>=3.8.0
statsmodels>=0.14.0
rich>=13.0.0
pyyaml>=6.0
requests>=2.31.0
```

