# Analysis Pipeline

Python-based analysis pipeline for processing benchmark results.

## Containerized Execution (Recommended)

The analysis pipeline is containerized to ensure consistent Python dependencies across all environments.

**📖 For detailed containerization instructions, see: [Containerization Guide](../docs/guides/containerization.md)**

### Quick Start

```bash
# Run analysis script (automatically uses Podman/Docker)
./scripts/lib/run-python-container.sh analysis/scripts/compute_statistics.py \
  --input results/exp1/merged/merged.jsonl \
  --output results/exp1/stats

# Start Jupyter Lab
./scripts/start-jupyter.sh
# Access at http://localhost:8888
# Stop with: ./scripts/start-jupyter.sh --stop
```

The wrapper scripts automatically detect Podman (Fedora) or Docker and build images on first use.

## Host Python Execution (Alternative)

If you prefer to install dependencies directly:

```bash
cd analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 scripts/compute_statistics.py --help
```

## Scripts

- `scripts/compute_statistics.py` - Generate summary statistics from JSONL data
- `scripts/merge_jsonl.py` - Merge multiple JSONL files
- `scripts/plot_*.py` - Generate visualizations
- `aggregate_results.py` - Aggregate statistics across experiments
- `compare_all_environments.py` - Cross-environment comparison

## Notebooks

Jupyter notebooks for interactive analysis:
- `notebooks/01_load_results.ipynb` - Load and explore results
- `notebooks/02_latency_analysis.ipynb` - Latency distribution analysis
- `notebooks/03_throughput_analysis.ipynb` - Throughput analysis
- `notebooks/04_queue_delay_analysis.ipynb` - Queue delay analysis
- `notebooks/07_cluster_scaling_behavior.ipynb` - Scaling analysis

## Environment Variables

- `QR_USE_CONTAINER` - Set to `false` to disable containerization
- `QR_ANALYSIS_IMAGE` - Override container image name (default: `quantum-resilient-analysis`)
