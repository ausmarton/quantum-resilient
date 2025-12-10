# Analysis Pipeline

Python-based analysis pipeline for processing benchmark results.

## Containerized Execution (Recommended)

The analysis pipeline is containerized to ensure consistent Python dependencies across all environments.

### Quick Start

```bash
# Build the analysis container (wrapper script detects podman/docker automatically)
# The wrapper script will build automatically on first use, or build manually:
podman build -t quantum-resilient-analysis -f analysis/Dockerfile analysis/

# Run analysis script in container (automatically uses podman if available)
./scripts/lib/run-python-container.sh analysis/scripts/compute_statistics.py \
  --input results/exp1/merged/merged.jsonl \
  --output results/exp1/stats

# Start Jupyter Lab
# Option 1: Using helper script (recommended - detects podman/docker automatically)
./scripts/start-jupyter.sh
# Stop with: ./scripts/start-jupyter.sh --stop

# Option 2: Using podman-compose (if installed)
podman-compose up jupyter

# Option 3: Using podman directly
podman run -d --name quantum-resilient-jupyter \
  -p 8888:8888 \
  -v "$PWD/results:/workspace/results:ro" \
  -v "$PWD/analysis:/workspace/analysis:rw" \
  -v "$PWD/final-results:/workspace/final-results:rw" \
  -w /workspace \
  -e PYTHONPATH=/workspace/analysis:/workspace/analysis/scripts \
  quantum-resilient-jupyter:latest

# Access at http://localhost:8888
```

### Using the Wrapper Script

The `scripts/lib/run-python-container.sh` wrapper automatically:
- Builds the container image if it doesn't exist
- Mounts project directories
- Runs Python scripts with all dependencies

```bash
# Run any Python script
./scripts/lib/run-python-container.sh analysis/scripts/merge_jsonl.py --help

# Disable containerization (use host Python)
QR_USE_CONTAINER=false python3 analysis/scripts/compute_statistics.py --help
```

### Docker Compose / Podman Compose

**Note**: On Fedora with Podman, you can use `podman-compose` (if installed) or run containers directly with `podman`.

```bash
# Option 1: Using podman-compose (if installed)
podman-compose up jupyter

# Option 2: Using podman directly (recommended for Fedora)
# Build Jupyter image first:
podman build -t quantum-resilient-jupyter -f analysis/Dockerfile.jupyter analysis/

# Run Jupyter Lab
podman run -d --name quantum-resilient-jupyter \
  -p 8888:8888 \
  -v "$PWD/results:/workspace/results:ro" \
  -v "$PWD/analysis:/workspace/analysis:rw" \
  -v "$PWD/final-results:/workspace/final-results:rw" \
  -w /workspace \
  quantum-resilient-jupyter:latest

# Run one-off analysis command
podman run --rm \
  -v "$PWD/results:/workspace/results:rw" \
  -v "$PWD/analysis:/workspace/analysis:ro" \
  -v "$PWD/final-results:/workspace/final-results:rw" \
  -w /workspace \
  quantum-resilient-analysis:latest \
  python3 analysis/scripts/compute_statistics.py \
  --input results/exp1/merged/merged.jsonl \
  --output results/exp1/stats
```

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
