# Quantum-Resilient Analysis Suite

Research analysis environment for processing and visualizing PQC benchmark results.

## Overview

This analysis suite provides tools for:
- Fetching experiment results from local filesystem, S3/MinIO, or GCS
- Merging and validating distributed worker JSONL files
- Computing statistical summaries and distributions
- Calculating effect sizes for algorithm comparisons
- Generating publication-quality figures
- Reproducible analysis pipelines

## Directory Structure

```
analysis/
├── notebooks/              # Jupyter notebooks for interactive analysis
│   ├── 00_setup.ipynb
│   ├── 01_load_results.ipynb
│   ├── 02_latency_analysis.ipynb
│   ├── 03_throughput_analysis.ipynb
│   ├── 04_queue_delay_analysis.ipynb
│   ├── 05_adapter_comparison.ipynb
│   ├── 06_effect_size.ipynb
│   ├── 07_cluster_scaling_behavior.ipynb
│   └── 99_generate_figures.ipynb
├── scripts/                # Command-line analysis tools
│   ├── fetch_results.py
│   ├── merge_jsonl.py
│   ├── compute_statistics.py
│   ├── effect_sizes.py
│   ├── plot_latency.py
│   ├── plot_throughput.py
│   ├── plot_queue_delay.py
│   └── export_dataset.py
├── data/                   # Downloaded and processed data (gitignored)
├── figures/                # Generated figures
├── requirements.txt
├── pyproject.toml
├── run_full_pipeline.sh
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Or with optional ML dependencies
pip install -r requirements.txt
pip install scikit-learn statsmodels
```

## Quick Start

### 1. Fetch Results

Download experiment results from storage:

```bash
# From local filesystem
python scripts/fetch_results.py \
  --experiment-id exp_2025_01_01_001 \
  --uri file:///path/to/results \
  --out data/exp_2025_01_01_001/

# From GCS
python scripts/fetch_results.py \
  --experiment-id exp_2025_01_01_001 \
  --uri gs://qr-results/exp_2025_01_01_001 \
  --out data/exp_2025_01_01_001/

# From S3/MinIO
python scripts/fetch_results.py \
  --experiment-id exp_2025_01_01_001 \
  --uri s3://qr-results/exp_2025_01_01_001 \
  --out data/exp_2025_01_01_001/
```

### 2. Merge JSONL Files

Combine worker results into a single timeline:

```bash
python scripts/merge_jsonl.py \
  --input data/exp_2025_01_01_001/raw/ \
  --output data/exp_2025_01_01_001/merged/
```

### 3. Compute Statistics

Generate statistical summaries and plots:

```bash
python scripts/compute_statistics.py \
  --input data/exp_2025_01_01_001/merged/merged.jsonl \
  --output data/exp_2025_01_01_001/stats/
```

### 4. Calculate Effect Sizes

Compare two experiments:

```bash
python scripts/effect_sizes.py \
  --exp-a data/exp_rsa/merged/merged.jsonl \
  --exp-b data/exp_kyber/merged/merged.jsonl \
  --metric latency_us \
  --out data/comparisons/rsa_vs_kyber.json
```

### 5. Generate Plots

```bash
python scripts/plot_latency.py \
  --input data/exp_2025_01_01_001/merged/merged.jsonl \
  --output figures/exp_2025_01_01_001/

python scripts/plot_throughput.py \
  --input data/exp_2025_01_01_001/merged/merged.jsonl \
  --output figures/exp_2025_01_01_001/
```

## Full Pipeline

Run the complete analysis pipeline:

```bash
./run_full_pipeline.sh exp_2025_01_01_001 gs://qr-results/exp_2025_01_01_001
```

This will:
1. Fetch remote results
2. Merge JSONL files
3. Compute statistics
4. Generate plots
5. Export to Parquet format

## Jupyter Notebooks

Start JupyterLab for interactive analysis:

```bash
jupyter lab
```

### Notebook Guide

| Notebook | Purpose |
|----------|---------|
| `00_setup.ipynb` | Environment check and authentication |
| `01_load_results.ipynb` | Load and preview data |
| `02_latency_analysis.ipynb` | Latency distributions and analysis |
| `03_throughput_analysis.ipynb` | Throughput over time |
| `04_queue_delay_analysis.ipynb` | Queue delay analysis |
| `05_adapter_comparison.ipynb` | RSA vs ECDSA vs Kyber comparison |
| `06_effect_size.ipynb` | Statistical significance testing |
| `07_cluster_scaling_behavior.ipynb` | Kubernetes scaling analysis |
| `99_generate_figures.ipynb` | Publication-quality figures |

## Output Formats

### Statistical Summary (summary.json)

```json
{
  "experiment_id": "exp_2025_01_01_001",
  "total_events": 520000,
  "duration_sec": 60.5,
  "latency": {
    "mean": 523.4,
    "std": 142.8,
    "p50": 498.0,
    "p90": 712.0,
    "p95": 845.0,
    "p99": 1124.0
  },
  "throughput": {
    "mean": 8595.0,
    "max": 9234.0,
    "min": 7823.0
  }
}
```

### Effect Sizes (comparison.json)

```json
{
  "experiment_a": "rsa_baseline",
  "experiment_b": "kyber_hybrid",
  "metric": "latency_us",
  "cohens_d": 0.85,
  "hedges_g": 0.84,
  "glass_delta": 0.92,
  "cliffs_delta": 0.71,
  "ks_statistic": 0.23,
  "wasserstein_distance": 145.3,
  "interpretation": "large effect"
}
```

## GCP Authentication

For GCS access, authenticate using Application Default Credentials:

```bash
gcloud auth application-default login
```

Or set a service account key:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

## Citation

If you use this analysis suite in your research, please cite:

```bibtex
@software{quantum_resilient_analysis,
  title = {Quantum-Resilient Cryptography Benchmark Analysis Suite},
  year = {2025},
  url = {https://github.com/quantum-resilient/quantum-resilient}
}
```
