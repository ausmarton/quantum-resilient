# Containerization Guide

This guide covers using the containerized analysis pipeline with Podman (Fedora) or Docker.

## Overview

The analysis pipeline is containerized to ensure consistent Python dependencies across all environments. All scripts automatically detect and use Podman if available (Fedora default), falling back to Docker if Podman is not found.

## Quick Start

### Run Analysis Scripts

```bash
# The wrapper automatically uses Podman/Docker
./scripts/lib/run-python-container.sh analysis/scripts/compute_statistics.py \
  --input results/exp1/merged/merged.jsonl \
  --output results/exp1/stats
```

The first run will automatically build the container image.

### Start Jupyter Lab

```bash
# Helper script automatically uses Podman/Docker
./scripts/start-jupyter.sh

# Access at http://localhost:8888
# Stop with: ./scripts/start-jupyter.sh --stop
```

## Automatic Detection

The wrapper scripts (`scripts/lib/run-python-container.sh` and `scripts/start-jupyter.sh`) automatically detect the container runtime:

```bash
# Checks in this order:
1. podman (preferred on Fedora)
2. docker (fallback)
```

## Podman vs Docker Compatibility

- ✅ All scripts work identically with Podman and Docker
- ✅ Podman is detected first (preferred on Fedora)
- ✅ No root privileges needed with Podman (daemonless)
- ✅ Same command syntax (Podman is Docker-compatible)

## Manual Container Commands

If you prefer to use Podman/Docker directly:

### Build Containers

```bash
# Build analysis container
podman build -t quantum-resilient-analysis -f analysis/Dockerfile analysis/

# Build Jupyter container
podman build -t quantum-resilient-jupyter -f analysis/Dockerfile.jupyter analysis/
```

### Run Analysis Scripts

```bash
podman run --rm \
  -v "$PWD/results:/workspace/results:rw" \
  -v "$PWD/analysis:/workspace/analysis:ro" \
  quantum-resilient-analysis:latest \
  python3 analysis/scripts/compute_statistics.py --help
```

### Run Jupyter Lab

```bash
podman run -d --name quantum-resilient-jupyter \
  -p 8888:8888 \
  -v "$PWD/results:/workspace/results:ro" \
  -v "$PWD/analysis:/workspace/analysis:rw" \
  -v "$PWD/final-results:/workspace/final-results:rw" \
  -w /workspace \
  quantum-resilient-jupyter:latest
```

## Troubleshooting

### Check Container Runtime

```bash
podman --version
# or
docker --version
```

### List Container Images

```bash
podman images | grep quantum-resilient
# or
docker images | grep quantum-resilient
```

### View Running Containers

```bash
podman ps
# or
docker ps
```

### View Container Logs

```bash
podman logs quantum-resilient-jupyter
# or
docker logs quantum-resilient-jupyter
```

### Remove Containers/Images

```bash
# Stop and remove Jupyter container
podman stop quantum-resilient-jupyter
podman rm quantum-resilient-jupyter

# Remove images
podman rmi quantum-resilient-analysis:latest
podman rmi quantum-resilient-jupyter:latest
```

## Disable Containerization

If you prefer to use host Python (not recommended):

```bash
# Set environment variable
export QR_USE_CONTAINER=false

# Or inline
QR_USE_CONTAINER=false ./scripts/lib/run-python-container.sh analysis/scripts/compute_statistics.py --help
```

## Environment Variables

- `QR_USE_CONTAINER` - Set to `false` to disable containerization (use host Python)
- `QR_ANALYSIS_IMAGE` - Override container image name (default: `quantum-resilient-analysis`)

## Related Documentation

- **[Analysis Pipeline](../analysis/README.md)** - Analysis scripts overview (in `analysis/` directory)
- **[Data Collection](data-collection.md)** - Running experiments and data collection
- **[Researcher Guide](researcher-guide.md)** - Comprehensive researcher guide

## Notes

- Podman doesn't require root privileges (unlike Docker)
- Podman uses the same command syntax as Docker
- All scripts automatically prefer Podman when available
- No configuration changes needed - it just works!
