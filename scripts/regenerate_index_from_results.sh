#!/usr/bin/env bash
# =============================================================================
# regenerate_index_from_results.sh - Regenerate index.json from existing results
#
# When data collection is done separately for each environment, this script
# regenerates a combined index.json from all existing results directories.
#
# Usage:
#   ./scripts/regenerate_index_from_results.sh \
#     --matrix orchestration/experiment_matrix.yaml \
#     --output final-results/
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MATRIX="$SCRIPT_DIR/orchestration/experiment_matrix.yaml"
OUTPUT_DIR="$SCRIPT_DIR/final-results"
ENVS="native,minikube,gcp"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Regenerate index.json from existing results directories.

OPTIONS:
    --matrix PATH    Experiment matrix YAML (default: orchestration/experiment_matrix.yaml)
    --output DIR     Output directory for index.json (default: final-results/)
    --envs LIST      Comma-separated environments (default: native,minikube,gcp)
    -h, --help       Show this help message
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --matrix)
            MATRIX="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --envs)
            ENVS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# Get Python command (containerized if available, fallback to host Python)
PYTHON_CMD="python3"
if [[ -f "$SCRIPT_DIR/scripts/lib/run-python-container.sh" ]] && \
   [[ "${QR_USE_CONTAINER:-true}" != "false" ]]; then
    PYTHON_CMD="$SCRIPT_DIR/scripts/lib/run-python-container.sh"
fi

# Run the index regeneration script
$PYTHON_CMD "$SCRIPT_DIR/scripts/lib/regenerate_index.py" \
    "$SCRIPT_DIR" \
    "$OUTPUT_DIR" \
    "$MATRIX" \
    "$ENVS"

echo ""
echo "Index regenerated successfully!"
echo "You can now run analysis:"
echo "  ./run_all_experiments.sh --skip-generation --skip-native --skip-minikube --skip-gcp"

