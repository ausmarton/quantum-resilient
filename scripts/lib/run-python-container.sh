#!/usr/bin/env bash
# =============================================================================
# scripts/lib/run-python-container.sh - Run Python scripts in analysis container
#
# Wrapper script to execute Python scripts in the containerized analysis
# environment. This ensures consistent Python dependencies across all machines.
#
# Usage:
#   ./scripts/lib/run-python-container.sh <script.py> [args...]
#   ./scripts/lib/run-python-container.sh scripts/compute_statistics.py --input data.jsonl --output stats/
#
# Environment Variables:
#   QR_ANALYSIS_IMAGE: Override container image name (default: quantum-resilient-analysis)
#   QR_USE_CONTAINER: Set to "false" to disable containerization (use host Python)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SCRIPT_DIR"

# Configuration
ANALYSIS_IMAGE="${QR_ANALYSIS_IMAGE:-quantum-resilient-analysis}"
USE_CONTAINER="${QR_USE_CONTAINER:-true}"

# Check if containerization is disabled
if [[ "${USE_CONTAINER}" == "false" ]]; then
    exec python3 "$@"
fi

# Check if Docker/Podman is available
if command -v podman &> /dev/null; then
    DOCKER_CMD="podman"
elif command -v docker &> /dev/null; then
    DOCKER_CMD="docker"
else
    echo "Error: Neither podman nor docker found. Install one or set QR_USE_CONTAINER=false" >&2
    exit 1
fi

# Check if image exists, build if not
if ! $DOCKER_CMD image inspect "$ANALYSIS_IMAGE" &> /dev/null; then
    echo "Building analysis container image: $ANALYSIS_IMAGE"
    $DOCKER_CMD build -t "$ANALYSIS_IMAGE" -f analysis/Dockerfile analysis/
fi

# Determine script path (relative to project root or absolute)
SCRIPT_ARG="$1"
shift || true

# If script is relative, make it relative to project root
if [[ "$SCRIPT_ARG" != /* ]]; then
    # Already relative, use as-is
    SCRIPT_PATH="$SCRIPT_ARG"
else
    # Absolute path, convert to relative if under project root
    SCRIPT_PATH="${SCRIPT_ARG#$SCRIPT_DIR/}"
fi

# Convert path arguments from host absolute paths to container paths
# Container mounts project root as /workspace, so paths under project root
# need to be converted from /absolute/host/path to /workspace/relative/path
convert_path_to_container() {
    local path="$1"
    if [[ "$path" == /* ]]; then
        # Absolute path - check if it's under project root
        if [[ "$path" == "$SCRIPT_DIR"* ]]; then
            # Convert to container path: /workspace/relative/path
            echo "/workspace/${path#$SCRIPT_DIR/}"
        else
            # Path outside project root - return as-is (might be /tmp, etc.)
            echo "$path"
        fi
    else
        # Already relative - assume it's relative to /workspace
        echo "$path"
    fi
}

# Convert path arguments from host absolute paths to container paths
# Only convert explicit path arguments (--input, --output, etc.) to avoid false positives
CONVERTED_ARGS=()
while [[ $# -gt 0 ]]; do
    arg="$1"
    shift
    
    # Check if this is a path argument (--input, --output, --input-dir, --output-dir, etc.)
    if [[ "$arg" =~ ^--(input|output|input-dir|output-dir|data-dir|stats-dir|figures-dir)(=.*)?$ ]]; then
        # This is a path argument
        if [[ "$arg" == *=* ]]; then
            # --arg=value format
            arg_name="${arg%%=*}"
            arg_value="${arg#*=}"
            converted_value=$(convert_path_to_container "$arg_value")
            CONVERTED_ARGS+=("${arg_name}=${converted_value}")
        else
            # --arg value format
            CONVERTED_ARGS+=("$arg")
            if [[ $# -gt 0 ]]; then
                next_arg="$1"
                shift
                converted_value=$(convert_path_to_container "$next_arg")
                CONVERTED_ARGS+=("$converted_value")
            fi
        fi
    else
        # Not a known path argument - pass as-is
        # (Don't try to auto-detect paths, as this could break non-path arguments)
        CONVERTED_ARGS+=("$arg")
    fi
done

# Run script in container
# Mount project root as /workspace
# Mount results directory if it exists
# Set working directory to /workspace
# Use :Z flag for Podman on SELinux systems (Fedora)
if [[ "$DOCKER_CMD" == "podman" ]]; then
    VOLUME_FLAGS="-v $SCRIPT_DIR:/workspace:rw,Z"
else
    VOLUME_FLAGS="-v $SCRIPT_DIR:/workspace:rw"
fi

$DOCKER_CMD run --rm \
    $VOLUME_FLAGS \
    -w /workspace \
    "$ANALYSIS_IMAGE" \
    "$SCRIPT_PATH" "${CONVERTED_ARGS[@]}"
