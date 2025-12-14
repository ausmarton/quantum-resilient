#!/usr/bin/env bash
# =============================================================================
# generate_experiment_summaries.sh - Generate summary.json at experiment level
#
# Merges all runs for each experiment, then generates a single summary.json
# per experiment (not per run). This is more efficient and matches what
# aggregate_results.py expects.
#
# Usage:
#   ./scripts/generate_experiment_summaries.sh [OPTIONS]
#
# Options:
#   --env ENV          Process only specific environment (native, minikube, gcp)
#   --parallel N       Number of parallel jobs (default: auto-detect based on CPU)
#   --resume           Resume from last checkpoint (skip already-processed)
#   -h, --help         Show this help message
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="$SCRIPT_DIR/results"
ENV_FILTER=""
# Optimize: Use more parallel jobs (CPU cores - 1, min 4, max 16)
CPU_CORES=$(nproc 2>/dev/null || echo 4)
PARALLEL_JOBS=$((CPU_CORES > 1 ? CPU_CORES - 1 : 1))
PARALLEL_JOBS=$((PARALLEL_JOBS < 4 ? 4 : PARALLEL_JOBS))
PARALLEL_JOBS=$((PARALLEL_JOBS > 16 ? 16 : PARALLEL_JOBS))

# Get Python command
PYTHON_CMD="$SCRIPT_DIR/scripts/lib/run-python-container.sh"
if [[ ! -f "$PYTHON_CMD" ]] || [[ "${QR_USE_CONTAINER:-true}" == "false" ]]; then
    PYTHON_CMD="python3"
fi

log_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

log_success() {
    echo -e "\033[0;32m[OK]\033[0m $1"
}

# Find all experiment directories (not run directories)
log_info "Finding experiment directories..."

EXPERIMENT_DIRS=()
if [[ -n "$ENV_FILTER" ]]; then
    SEARCH_PATH="$RESULTS_DIR/$ENV_FILTER"
else
    SEARCH_PATH="$RESULTS_DIR"
fi

# Find experiment directories (those that contain run-* subdirectories)
# Structure: results/env/experiment/run-*/raw/run.jsonl
while IFS= read -r -d '' exp_dir; do
    # Check if this directory has run-* subdirectories with raw data
    has_runs=false
    for run_dir in "$exp_dir"/run-*/raw/run.jsonl; do
        if [[ -f "$run_dir" ]]; then
            has_runs=true
            break
        fi
    done
    
    if [[ "$has_runs" == "true" ]]; then
        EXPERIMENT_DIRS+=("$exp_dir")
    fi
done < <(find "$SEARCH_PATH" -mindepth 2 -maxdepth 2 -type d -print0 2>/dev/null || true)

TOTAL_EXPS=${#EXPERIMENT_DIRS[@]}
log_info "Found $TOTAL_EXPS experiment directories"

if [[ $TOTAL_EXPS -eq 0 ]]; then
    log_info "No experiments found. Nothing to process."
    exit 0
fi

# Function to process one experiment
process_experiment() {
    local exp_dir="$1"
    local exp_name=$(basename "$exp_dir")
    local merged_dir="$exp_dir/merged"
    local stats_dir="$exp_dir/stats"
    
    # Skip if summary already exists
    if [[ -f "$stats_dir/summary.json" ]]; then
        echo "SKIP: $exp_name"
        return 0
    fi
    
    # Create directories
    mkdir -p "$merged_dir" "$stats_dir"
    
    # Merge all run JSONL files
    local merged_file="$merged_dir/merged.jsonl"
    if [[ ! -f "$merged_file" ]] || [[ ! -s "$merged_file" ]]; then
        # Find all raw JSONL files for this experiment
        local raw_files=()
        while IFS= read -r -d '' file; do
            raw_files+=("$file")
        done < <(find "$exp_dir" -path "*/run-*/raw/run.jsonl" -type f -print0 2>/dev/null || true)
        
        if [[ ${#raw_files[@]} -eq 0 ]]; then
            echo "NO_DATA: $exp_name"
            return 0
        fi
        
        # Merge files (simple concatenation, they're already sorted)
        # Use parallel cat if available for faster I/O on large files
        if command -v parallel &> /dev/null && [[ ${#raw_files[@]} -gt 1 ]]; then
            parallel -j +0 cat ::: "${raw_files[@]}" > "$merged_file"
        else
            cat "${raw_files[@]}" > "$merged_file"
        fi
        
        # Ensure merged file is readable by container (SELinux compatibility)
        # Set permissions before container tries to read
        chmod 644 "$merged_file" 2>/dev/null || true
        chmod u+r "$merged_file" 2>/dev/null || true
    fi
    
    # Generate summary
    # Run the command and check if summary was created (regardless of exit code)
    # Some scripts may exit with non-zero due to warnings but still create the file
    # Capture stderr to check for errors
    local error_log=$(mktemp)
    local stdout_log=$(mktemp)
    
    "$PYTHON_CMD" "$SCRIPT_DIR/analysis/scripts/compute_statistics.py" \
        --input "$merged_file" \
        --output "$stats_dir" \
        --experiment-id "$exp_name" > "$stdout_log" 2>"$error_log" || true
    
    # Check for permission errors
    if grep -q "Permission denied" "$error_log" 2>/dev/null; then
        # Fix permissions and retry once
        chmod -R u+r "$exp_dir" 2>/dev/null || true
        "$PYTHON_CMD" "$SCRIPT_DIR/analysis/scripts/compute_statistics.py" \
            --input "$merged_file" \
            --output "$stats_dir" \
            --experiment-id "$exp_name" > /dev/null 2>&1 || true
    fi
    
    # Log errors for debugging if summary not created
    if [[ ! -f "$stats_dir/summary.json" ]]; then
        if [[ -s "$error_log" ]]; then
            echo "  Error log: $(head -3 "$error_log" | tr '\n' ' ')" >&2
        fi
    fi
    
    rm -f "$error_log" "$stdout_log"
    
    # Check if summary was actually created (this is the real test)
    # Add small delay to ensure file is written (container may need time to sync)
    sleep 0.5
    
    if [[ -f "$stats_dir/summary.json" ]]; then
        # Validate it's valid JSON
        if python3 -c "import json; json.load(open('$stats_dir/summary.json'))" 2>/dev/null; then
            echo "OK: $exp_name"
            return 0
        else
            echo "FAILED: $exp_name (invalid JSON)" >&2
            return 1
        fi
    else
        # Double-check after another brief delay (file system sync)
        sleep 0.5
        if [[ -f "$stats_dir/summary.json" ]]; then
            if python3 -c "import json; json.load(open('$stats_dir/summary.json'))" 2>/dev/null; then
                echo "OK: $exp_name"
                return 0
            fi
        fi
        echo "FAILED: $exp_name (summary.json not created)" >&2
        return 1
    fi
}

# Parse command-line arguments
RESUME=false
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
        --resume)
            RESUME=true
            shift
            ;;
        -h|--help)
            cat <<EOF
Usage: $0 [OPTIONS]

Generate summary.json files for all experiments from raw JSONL data.

OPTIONS:
    --env ENV          Process only specific environment (native, minikube, gcp)
    --parallel N       Number of parallel jobs (default: auto-detect, min 4, max 16)
    --resume           Skip already-processed experiments (default: enabled)
    -h, --help         Show this help message
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

export -f process_experiment
export SCRIPT_DIR PYTHON_CMD

# Filter out already-processed if resuming
if [[ "$RESUME" == "true" ]] || [[ "$RESUME" == "false" ]]; then
    # Always skip already-processed (resume is default behavior)
    FILTERED_DIRS=()
    for exp_dir in "${EXPERIMENT_DIRS[@]}"; do
        stats_dir="$exp_dir/stats"
        if [[ ! -f "$stats_dir/summary.json" ]]; then
            FILTERED_DIRS+=("$exp_dir")
        fi
    done
    EXPERIMENT_DIRS=("${FILTERED_DIRS[@]}")
    log_info "Skipping already-processed experiments. Remaining: ${#EXPERIMENT_DIRS[@]}"
fi

log_info "Processing ${#EXPERIMENT_DIRS[@]} experiments (parallelism: $PARALLEL_JOBS)..."

# Process in parallel with progress tracking
TOTAL=${#EXPERIMENT_DIRS[@]}
PROCESSED=0
FAILED=0
SKIPPED=0

# Use a temporary file to track progress
PROGRESS_FILE=$(mktemp)
trap "rm -f $PROGRESS_FILE" EXIT

process_with_progress() {
    local exp_dir="$1"
    local result=$(process_experiment "$exp_dir" 2>&1)
    local status=$?
    
    {
        flock -x 200
        if echo "$result" | grep -q "SKIP:"; then
            SKIPPED=$((SKIPPED + 1))
        elif [[ $status -eq 0 ]]; then
            PROCESSED=$((PROCESSED + 1))
        else
            FAILED=$((FAILED + 1))
        fi
        local current=$((PROCESSED + SKIPPED + FAILED))
        if [[ $((current % 10)) -eq 0 ]] || [[ $current -eq $TOTAL ]]; then
            echo "[$current/$TOTAL] Processed: $PROCESSED, Skipped: $SKIPPED, Failed: $FAILED"
        fi
    } 200>$PROGRESS_FILE
    
    echo "$result"
    return $status
}

export -f process_with_progress
export PROGRESS_FILE PROCESSED SKIPPED FAILED TOTAL

# Process in parallel
printf '%s\0' "${EXPERIMENT_DIRS[@]}" | xargs -0 -P "$PARALLEL_JOBS" -I {} bash -c 'process_with_progress "$@"' _ {}

log_success "Processing complete!"

# Count generated summaries
SUMMARY_COUNT=$(find "$SEARCH_PATH" -path "*/stats/summary.json" -type f 2>/dev/null | wc -l || echo 0)
log_info "Generated $SUMMARY_COUNT summary.json files"
