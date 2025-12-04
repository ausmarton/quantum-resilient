#!/usr/bin/env bash
#
# run_local.sh - Complete local microbenchmark experiment runner
#
# Runs PQC benchmark natively, collects results, and produces analysis outputs.
# Supports multiple repeated runs with aggregated statistics.
# Non-interactive - suitable for CI/CD and automated experiments.
#
# Usage:
#   ./run_local.sh --scenario scenarios/hybrid_kyber_dilithium.yaml --out ./results/exp1 --runs 5
#
# Outputs (single run):
#   <out>/raw/          - Raw JSONL telemetry
#   <out>/merged/       - Merged/sorted JSONL + Parquet
#   <out>/stats/        - Summary statistics (JSON)
#   <out>/figures/      - Publication-quality plots (PNG)
#
# Outputs (multiple runs):
#   <out>/run-1/        - Results from run 1
#   <out>/run-2/        - Results from run 2
#   ...
#   <out>/aggregated_stats.json   - Aggregated statistics
#   <out>/stability_report.json   - Stability analysis

set -euo pipefail

# Default values
SCENARIO=""
OUT_DIR=""
DURATION=""
SEED=""
RUNS=1
TIMEOUT=3600
SKIP_ANALYSIS=false
SKIP_AGGREGATION=false

# Color output helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_run() {
    echo -e "${CYAN}[RUN $1/$2]${NC} $3"
}

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Run a complete PQC microbenchmark experiment locally.

OPTIONS:
    --scenario PATH     Path to scenario YAML file (required)
    --out DIR           Output directory for results (required)
    --runs N            Number of repeated runs (default: 1)
    --duration SEC      Override duration from scenario (optional)
    --seed NUM          Base RNG seed (optional, each run gets seed+run_index)
    --timeout SEC       Timeout per run (default: 3600)
    --skip-analysis     Skip Python analysis step
    --skip-aggregation  Skip aggregation across runs
    -h, --help          Show this help message

EXAMPLES:
    # Single run
    $0 --scenario scenarios/hybrid_kyber_dilithium.yaml --out ./results/exp1

    # Five repeated runs with aggregation
    $0 --scenario scenarios/hybrid_kyber_dilithium.yaml --out ./results/exp1 --runs 5

    # With deterministic seeding
    $0 --scenario scenarios/hybrid_kyber_dilithium.yaml --out ./results/exp1 --runs 5 --seed 1234

OUTPUTS (multiple runs):
    <out>/run-1/raw/run.jsonl         Raw telemetry for run 1
    <out>/run-1/stats/summary.json    Statistics for run 1
    ...
    <out>/aggregated_stats.json       Aggregated across all runs
    <out>/stability_report.json       Stability analysis
EOF
    exit 1
}

# Run a single benchmark iteration
run_single_benchmark() {
    local run_index=$1
    local run_out_dir=$2
    local run_seed=$3
    
    # Create output directories
    mkdir -p "$run_out_dir/raw"
    mkdir -p "$run_out_dir/merged"
    mkdir -p "$run_out_dir/stats"
    mkdir -p "$run_out_dir/figures"
    
    # Prepare scenario with overrides
    TEMP_SCENARIO=$(mktemp)
    cp "$SCENARIO" "$TEMP_SCENARIO"
    
    # Override JSONL output path
    RAW_JSONL_PATH="$run_out_dir/raw/run.jsonl"
    if grep -q "jsonl_out:" "$TEMP_SCENARIO"; then
        sed -i "s|jsonl_out:.*|jsonl_out: \"$RAW_JSONL_PATH\"|" "$TEMP_SCENARIO"
    else
        if grep -q "metrics:" "$TEMP_SCENARIO"; then
            sed -i "/metrics:/a\\  jsonl_out: \"$RAW_JSONL_PATH\"" "$TEMP_SCENARIO"
        else
            echo -e "\nmetrics:\n  jsonl_out: \"$RAW_JSONL_PATH\"" >> "$TEMP_SCENARIO"
        fi
    fi
    
    # Override duration if specified
    if [[ -n "$DURATION" ]]; then
        sed -i "s/duration_sec:.*/duration_sec: $DURATION/" "$TEMP_SCENARIO"
    fi
    
    # Set seed for this run
    if [[ -n "$run_seed" ]]; then
        if grep -q "rng_seed:" "$TEMP_SCENARIO"; then
            sed -i "s/rng_seed:.*/rng_seed: $run_seed/" "$TEMP_SCENARIO"
        else
            sed -i "/^id:/a rng_seed: $run_seed" "$TEMP_SCENARIO"
        fi
    fi
    
    # Run the benchmark
    local start_time=$(date +%s)
    local start_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    set +e
    timeout "$TIMEOUT" "$BINARY" --scenario "$TEMP_SCENARIO"
    local bench_exit_code=$?
    set -e
    
    local end_time=$(date +%s)
    local end_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local elapsed=$((end_time - start_time))
    
    # Clean up temp scenario
    rm -f "$TEMP_SCENARIO"
    
    if [[ $bench_exit_code -eq 124 ]]; then
        log_error "Run $run_index timed out after ${TIMEOUT}s"
        return 1
    elif [[ $bench_exit_code -ne 0 ]]; then
        log_error "Run $run_index failed with exit code: $bench_exit_code"
        return 1
    fi
    
    # Verify raw output
    if [[ ! -f "$RAW_JSONL_PATH" ]]; then
        log_error "Run $run_index: Raw JSONL not found"
        return 1
    fi
    
    local raw_lines=$(wc -l < "$RAW_JSONL_PATH")
    
    # Generate manifest for this run
    cat > "$run_out_dir/manifest.json" <<EOF
{
    "run_id": "${EXP_ID}_run${run_index}",
    "run_index": $run_index,
    "scenario": "$SCENARIO",
    "git_commit": "$GIT_COMMIT",
    "rustc_version": "$RUSTC_VERSION",
    "start_time_utc": "$start_iso",
    "end_time_utc": "$end_iso",
    "duration_sec": $elapsed,
    "events_count": $raw_lines,
    "seed": ${run_seed:-null},
    "host": "$(hostname)",
    "platform": "$(uname -s)-$(uname -m)"
}
EOF
    
    # Run analysis for this run
    if [[ "$SKIP_ANALYSIS" != "true" ]]; then
        if [[ -f "${ANALYSIS_DIR}/scripts/merge_jsonl.py" ]]; then
            python3 "${ANALYSIS_DIR}/scripts/merge_jsonl.py" \
                --input "$run_out_dir/raw" \
                --output "$run_out_dir/merged" 2>/dev/null || true
        fi
        
        local input_file="$run_out_dir/merged/merged.parquet"
        [[ ! -f "$input_file" ]] && input_file="$run_out_dir/merged/merged.jsonl"
        [[ ! -f "$input_file" ]] && input_file="$run_out_dir/raw/run.jsonl"
        
        if [[ -f "${ANALYSIS_DIR}/scripts/compute_statistics.py" ]]; then
            python3 "${ANALYSIS_DIR}/scripts/compute_statistics.py" \
                --input "$input_file" \
                --output "$run_out_dir/stats" \
                --experiment-id "${EXP_ID}_run${run_index}" 2>/dev/null || true
        fi
    fi
    
    echo "$elapsed:$raw_lines"
    return 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --out)
            OUT_DIR="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --skip-analysis)
            SKIP_ANALYSIS=true
            shift
            ;;
        --skip-aggregation)
            SKIP_AGGREGATION=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$SCENARIO" ]]; then
    log_error "Missing required argument: --scenario"
    usage
fi

if [[ -z "$OUT_DIR" ]]; then
    log_error "Missing required argument: --out"
    usage
fi

if [[ ! -f "$SCENARIO" ]]; then
    log_error "Scenario file not found: $SCENARIO"
    exit 1
fi

# Extract experiment ID from output directory
EXP_ID=$(basename "$OUT_DIR")

log_info "=========================================="
log_info "PQC Microbenchmark Runner (Multi-Run)"
log_info "=========================================="
log_info "Scenario: $SCENARIO"
log_info "Output: $OUT_DIR"
log_info "Experiment ID: $EXP_ID"
log_info "Runs: $RUNS"
[[ -n "$DURATION" ]] && log_info "Duration override: ${DURATION}s"
[[ -n "$SEED" ]] && log_info "Base RNG seed: $SEED"
log_info ""

# Step 1: Build the Rust binary
log_info "Step 1: Building pqc-bench (release mode)..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARGO_MANIFEST="${SCRIPT_DIR}/rust-core/Cargo.toml"
ANALYSIS_DIR="${SCRIPT_DIR}/analysis"

if [[ ! -f "$CARGO_MANIFEST" ]]; then
    log_error "Cargo.toml not found at: $CARGO_MANIFEST"
    exit 1
fi

cargo build --release --manifest-path "$CARGO_MANIFEST" 2>&1 | while read -r line; do
    echo "  $line"
done

BINARY="${SCRIPT_DIR}/target/release/pqc-bench"
if [[ ! -x "$BINARY" ]]; then
    log_error "Binary not found after build: $BINARY"
    exit 1
fi
log_success "Binary built: $BINARY"

# Get metadata for manifest
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
RUSTC_VERSION=$(rustc --version 2>/dev/null || echo "unknown")

# Step 2: Create base output directory
log_info "Step 2: Creating output directories..."
mkdir -p "$OUT_DIR"
log_success "Base directory created"

# Step 3: Run experiments
log_info "Step 3: Running $RUNS experiment(s)..."
TOTAL_START=$(date +%s)

COMPLETED_RUNS=0
FAILED_RUNS=0
RUN_RESULTS=()

for ((i = 1; i <= RUNS; i++)); do
    log_run $i $RUNS "Starting..."
    
    # Determine output directory for this run
    if [[ $RUNS -eq 1 ]]; then
        RUN_OUT_DIR="$OUT_DIR"
    else
        RUN_OUT_DIR="$OUT_DIR/run-$i"
    fi
    
    # Compute seed for this run
    if [[ -n "$SEED" ]]; then
        RUN_SEED=$((SEED + i - 1))
    else
        RUN_SEED=""
    fi
    
    # Run benchmark
    if result=$(run_single_benchmark $i "$RUN_OUT_DIR" "$RUN_SEED"); then
        elapsed=$(echo "$result" | cut -d: -f1)
        events=$(echo "$result" | cut -d: -f2)
        log_run $i $RUNS "Completed in ${elapsed}s ($events events)"
        COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
        RUN_RESULTS+=("$RUN_OUT_DIR")
    else
        log_error "Run $i failed"
        FAILED_RUNS=$((FAILED_RUNS + 1))
    fi
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

log_info ""
log_info "Completed: $COMPLETED_RUNS/$RUNS runs"
[[ $FAILED_RUNS -gt 0 ]] && log_warn "Failed: $FAILED_RUNS runs"

# Step 4: Aggregate results (for multiple runs)
if [[ $RUNS -gt 1 ]] && [[ "$SKIP_AGGREGATION" != "true" ]] && [[ $COMPLETED_RUNS -gt 0 ]]; then
    log_info "Step 4: Aggregating results across runs..."
    
    if [[ -f "${ANALYSIS_DIR}/aggregate_runs.py" ]]; then
        python3 "${ANALYSIS_DIR}/aggregate_runs.py" \
            --input "$OUT_DIR" \
            --runs "$COMPLETED_RUNS" \
            --output "$OUT_DIR" 2>&1 | while read -r line; do
            echo "  $line"
        done || log_warn "Aggregation completed with warnings"
        
        log_success "Aggregation complete"
    else
        log_warn "aggregate_runs.py not found, skipping aggregation"
    fi
fi

# Step 5: Generate combined figures (for multiple runs)
if [[ $RUNS -gt 1 ]] && [[ "$SKIP_ANALYSIS" != "true" ]] && [[ $COMPLETED_RUNS -gt 0 ]]; then
    log_info "Step 5: Generating combined figures..."
    
    # Create figures directory at base level
    mkdir -p "$OUT_DIR/figures"
    
    # Copy best figures from runs (use run-1 as representative)
    if [[ -d "$OUT_DIR/run-1/figures" ]]; then
        cp -r "$OUT_DIR/run-1/figures/"* "$OUT_DIR/figures/" 2>/dev/null || true
    fi
    
    log_success "Figures generated"
fi

# Final summary
echo ""
log_info "=========================================="
log_info "EXPERIMENT COMPLETE"
log_info "=========================================="
log_info "Experiment ID: $EXP_ID"
log_info "Total Duration: ${TOTAL_ELAPSED}s"
log_info "Runs: $COMPLETED_RUNS completed, $FAILED_RUNS failed"
echo ""

if [[ $RUNS -eq 1 ]]; then
    log_info "Outputs:"
    echo "  Raw JSONL:     $OUT_DIR/raw/run.jsonl"
    [[ -f "$OUT_DIR/merged/merged.jsonl" ]] && echo "  Merged JSONL:  $OUT_DIR/merged/merged.jsonl"
    [[ -f "$OUT_DIR/stats/summary.json" ]] && echo "  Stats:         $OUT_DIR/stats/summary.json"
    [[ -f "$OUT_DIR/figures/latency_cdf.png" ]] && echo "  Latency CDF:   $OUT_DIR/figures/latency_cdf.png"
    echo "  Manifest:      $OUT_DIR/manifest.json"
else
    log_info "Outputs:"
    for ((i = 1; i <= COMPLETED_RUNS; i++)); do
        echo "  Run $i:         $OUT_DIR/run-$i/"
    done
    [[ -f "$OUT_DIR/aggregated_stats.json" ]] && echo "  Aggregated:    $OUT_DIR/aggregated_stats.json"
    [[ -f "$OUT_DIR/stability_report.json" ]] && echo "  Stability:     $OUT_DIR/stability_report.json"
    [[ -d "$OUT_DIR/figures" ]] && echo "  Figures:       $OUT_DIR/figures/"
fi
echo ""

if [[ -f "$OUT_DIR/aggregated_stats.json" ]]; then
    log_info "Aggregated Statistics:"
    python3 -c "
import json
with open('$OUT_DIR/aggregated_stats.json') as f:
    data = json.load(f)
lat = data.get('latency', {})
if 'p95' in lat:
    p95 = lat['p95']
    print(f\"  p95 latency: {p95['mean']:.0f} ± {p95['std']:.0f} μs (CV: {p95['cv']:.1%})\")
    print(f\"  95% CI: [{p95['ci_95_low']:.0f}, {p95['ci_95_high']:.0f}] μs\")
tput = data.get('throughput', {})
if 'mean' in tput:
    print(f\"  throughput:  {tput['mean']:.0f} ± {tput['std']:.0f} ops/s (CV: {tput['cv']:.1%})\")
" 2>/dev/null || true
fi

if [[ -f "$OUT_DIR/stability_report.json" ]]; then
    log_info "Stability:"
    python3 -c "
import json
with open('$OUT_DIR/stability_report.json') as f:
    data = json.load(f)
stable = '✓ Stable' if data.get('overall_stable', False) else '✗ Unstable'
print(f\"  Overall: {stable}\")
for w in data.get('warnings', [])[:3]:
    print(f\"  ⚠ {w}\")
" 2>/dev/null || true
fi

echo ""
log_success "Done!"

exit 0
