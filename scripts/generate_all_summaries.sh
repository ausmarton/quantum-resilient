#!/usr/bin/env bash
# =============================================================================
# generate_all_summaries.sh - Generate summary.json files for all experiments
#
# Processes all raw JSONL files in results/ and generates summary.json files
# in the stats/ subdirectory of each experiment.
#
# Usage:
#   ./scripts/generate_all_summaries.sh [OPTIONS]
#
# Options:
#   --env ENV          Process only specific environment (native, minikube, gcp)
#   --parallel N       Number of parallel jobs (default: 4)
#   --dry-run          Show what would be processed without actually processing
#   -h, --help         Show this help message
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="$SCRIPT_DIR/results"
ENV_FILTER=""
PARALLEL_JOBS=4
DRY_RUN=false

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Generate summary.json files for all experiments from raw JSONL data.

OPTIONS:
    --env ENV          Process only specific environment (native, minikube, gcp)
    --parallel N       Number of parallel jobs (default: 4)
    --dry-run          Show what would be processed without actually processing
    -h, --help         Show this help message
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV_FILTER="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
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

# Get Python command (containerized if available)
PYTHON_CMD="$SCRIPT_DIR/scripts/lib/run-python-container.sh"
if [[ ! -f "$PYTHON_CMD" ]] || [[ "${QR_USE_CONTAINER:-true}" == "false" ]]; then
    PYTHON_CMD="python3"
fi

log_info "Finding all raw JSONL files..."

# Find all raw JSONL files
RAW_FILES=()
if [[ -n "$ENV_FILTER" ]]; then
    SEARCH_PATH="$RESULTS_DIR/$ENV_FILTER"
else
    SEARCH_PATH="$RESULTS_DIR"
fi

while IFS= read -r -d '' file; do
    RAW_FILES+=("$file")
done < <(find "$SEARCH_PATH" -path "*/raw/run.jsonl" -type f -print0 2>/dev/null || true)

TOTAL_FILES=${#RAW_FILES[@]}
log_info "Found $TOTAL_FILES raw JSONL files to process"

if [[ $TOTAL_FILES -eq 0 ]]; then
    log_warn "No raw JSONL files found. Nothing to process."
    exit 0
fi

# Function to process a single file
process_file() {
    local jsonl_file="$1"
    local exp_dir=$(dirname "$(dirname "$jsonl_file")")
    local stats_dir="$exp_dir/stats"
    local merged_dir="$exp_dir/merged"
    
    # Create stats directory
    mkdir -p "$stats_dir"
    
    # Check if summary already exists
    if [[ -f "$stats_dir/summary.json" ]]; then
        echo "SKIP: $exp_dir (summary already exists)"
        return 0
    fi
    
    # Create merged directory and copy/link the file
    mkdir -p "$merged_dir"
    local merged_file="$merged_dir/merged.jsonl"
    
    # If merged file doesn't exist, create it from raw file
    if [[ ! -f "$merged_file" ]]; then
        cp "$jsonl_file" "$merged_file"
    fi
    
    # Generate summary
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "WOULD PROCESS: $exp_dir"
    else
        if "$PYTHON_CMD" "$SCRIPT_DIR/analysis/scripts/compute_statistics.py" \
            --input "$merged_file" \
            --output "$stats_dir" \
            --experiment-id "$(basename "$exp_dir")" 2>/dev/null; then
            echo "OK: $exp_dir"
        else
            echo "FAILED: $exp_dir" >&2
            return 1
        fi
    fi
}

export -f process_file
export SCRIPT_DIR PYTHON_CMD DRY_RUN

log_info "Processing files (parallelism: $PARALLEL_JOBS)..."

if [[ "$DRY_RUN" == "true" ]]; then
    for file in "${RAW_FILES[@]}"; do
        process_file "$file"
    done
else
    # Process in parallel
    printf '%s\0' "${RAW_FILES[@]}" | xargs -0 -P "$PARALLEL_JOBS" -I {} bash -c 'process_file "$@"' _ {}
fi

log_success "Processing complete!"

# Count generated summaries
SUMMARY_COUNT=$(find "$SEARCH_PATH" -name "summary.json" -type f 2>/dev/null | wc -l || echo 0)
log_info "Generated $SUMMARY_COUNT summary.json files"
