#!/usr/bin/env bash
# =============================================================================
# scripts/start-jupyter.sh - Start Jupyter Lab in container
#
# Starts Jupyter Lab using Podman (or Docker if Podman not available).
# Automatically builds the Jupyter image if it doesn't exist.
#
# Usage:
#   ./scripts/start-jupyter.sh
#   ./scripts/start-jupyter.sh --stop  # Stop running Jupyter container
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Detect container runtime
if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
elif command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
else
    echo "Error: Neither podman nor docker found. Install one or use host Python." >&2
    exit 1
fi

JUPYTER_IMAGE="quantum-resilient-jupyter:latest"
JUPYTER_CONTAINER="quantum-resilient-jupyter"

# Handle stop command
if [[ "${1:-}" == "--stop" ]]; then
    if $CONTAINER_CMD ps -a --format "{{.Names}}" | grep -q "^${JUPYTER_CONTAINER}$"; then
        echo "Stopping Jupyter container..."
        $CONTAINER_CMD stop "$JUPYTER_CONTAINER" 2>/dev/null || true
        $CONTAINER_CMD rm "$JUPYTER_CONTAINER" 2>/dev/null || true
        echo "Jupyter container stopped."
    else
        echo "Jupyter container not running."
    fi
    exit 0
fi

# Check if container is already running
if $CONTAINER_CMD ps --format "{{.Names}}" | grep -q "^${JUPYTER_CONTAINER}$"; then
    echo "Jupyter Lab is already running!"
    echo "Access at: http://localhost:8888"
    echo "Stop with: $0 --stop"
    exit 0
fi

# Build image if it doesn't exist
if ! $CONTAINER_CMD image inspect "$JUPYTER_IMAGE" &> /dev/null; then
    echo "Building Jupyter container image: $JUPYTER_IMAGE"
    $CONTAINER_CMD build -t "$JUPYTER_IMAGE" -f analysis/Dockerfile.jupyter analysis/
fi

# Start Jupyter Lab
echo "Starting Jupyter Lab..."
$CONTAINER_CMD run -d \
    --name "$JUPYTER_CONTAINER" \
    -p 8888:8888 \
    -v "$SCRIPT_DIR/results:/workspace/results:ro" \
    -v "$SCRIPT_DIR/analysis:/workspace/analysis:rw" \
    -v "$SCRIPT_DIR/final-results:/workspace/final-results:rw" \
    -w /workspace \
    -e PYTHONPATH=/workspace/analysis:/workspace/analysis/scripts \
    -e JUPYTER_ENABLE_LAB=yes \
    "$JUPYTER_IMAGE"

echo ""
echo "Jupyter Lab started!"
echo "Access at: http://localhost:8888"
echo ""
echo "To view logs: $CONTAINER_CMD logs -f $JUPYTER_CONTAINER"
echo "To stop: $0 --stop"
echo ""

# Wait a moment and show the access token
sleep 2
echo "Access token:"
$CONTAINER_CMD exec "$JUPYTER_CONTAINER" jupyter lab list 2>/dev/null | grep -oP 'token=\K[^\s]+' | head -1 || echo "(Check logs for token)"
