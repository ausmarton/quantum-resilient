# Podman Usage Guide (Fedora)

This project fully supports Podman on Fedora. All scripts automatically detect and use Podman if available.

## Automatic Detection

The wrapper scripts (`scripts/lib/run-python-container.sh` and `scripts/start-jupyter.sh`) automatically detect Podman and use it instead of Docker:

```bash
# Checks in this order:
1. podman (preferred on Fedora)
2. docker (fallback)
```

## Quick Start with Podman

### 1. Run Analysis Scripts

```bash
# The wrapper automatically uses Podman
./scripts/lib/run-python-container.sh analysis/scripts/compute_statistics.py \
  --input results/exp1/merged/merged.jsonl \
  --output results/exp1/stats
```

The first run will automatically build the container image using Podman.

### 2. Start Jupyter Lab

```bash
# Helper script automatically uses Podman
./scripts/start-jupyter.sh

# Access at http://localhost:8888
# Stop with: ./scripts/start-jupyter.sh --stop
```

### 3. Manual Podman Commands

If you prefer to use Podman directly:

```bash
# Build analysis container
podman build -t quantum-resilient-analysis -f analysis/Dockerfile analysis/

# Build Jupyter container
podman build -t quantum-resilient-jupyter -f analysis/Dockerfile.jupyter analysis/

# Run analysis script
podman run --rm \
  -v "$PWD/results:/workspace/results:rw" \
  -v "$PWD/analysis:/workspace/analysis:ro" \
  quantum-resilient-analysis:latest \
  python3 analysis/scripts/compute_statistics.py --help

# Run Jupyter Lab
podman run -d --name quantum-resilient-jupyter \
  -p 8888:8888 \
  -v "$PWD/results:/workspace/results:ro" \
  -v "$PWD/analysis:/workspace/analysis:rw" \
  -v "$PWD/final-results:/workspace/final-results:rw" \
  -w /workspace \
  quantum-resilient-jupyter:latest
```

## Podman vs Docker Compatibility

- ✅ All scripts work identically with Podman and Docker
- ✅ Podman is detected first (preferred on Fedora)
- ✅ No root privileges needed (Podman advantage)
- ✅ Same command syntax (Podman is Docker-compatible)

## Troubleshooting

### Check Podman Installation

```bash
podman --version
# Should show: podman version 5.x.x
```

### List Container Images

```bash
podman images | grep quantum-resilient
```

### View Running Containers

```bash
podman ps
```

### View Container Logs

```bash
podman logs quantum-resilient-jupyter
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

## Notes

- Podman doesn't require root privileges (unlike Docker)
- Podman uses the same command syntax as Docker
- All scripts automatically prefer Podman when available
- No configuration changes needed - it just works!
