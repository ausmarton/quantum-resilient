#!/usr/bin/env bash
#
# run_local.sh - Complete local microbenchmark experiment runner
#
# Runs PQC benchmark natively, collects results, and produces analysis outputs.
# Non-interactive - suitable for CI/CD and automated experiments.
#
# Usage:
#   ./run_local.sh --scenario scenarios/hybrid_kyber_dilithium.yaml --out ./results/exp1 --duration 30 --seed 1234
#
# Outputs:
#   <out>/raw/          - Raw JSONL telemetry
#   <out>/merged/       - Merged/sorted JSONL + Parquet
#   <out>/stats/        - Summary statistics (JSON)
#   <out>/figures/      - Publication-quality plots (PNG)

set -euo pipefail

# Default values
SCENARIO=""
OUT_DIR=""
DURATION=""
SEED=""
TIMEOUT=3600
SKIP_ANALYSIS=false

# Color output helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Run a complete PQC microbenchmark experiment locally.

OPTIONS:
    --scenario PATH     Path to scenario YAML file (required)
    --out DIR           Output directory for results (required)
    --duration SEC      Override duration from scenario (optional)
    --seed NUM          RNG seed for reproducibility (optional, overrides scenario)
    --timeout SEC       Timeout for benchmark run (default: 3600)
    --skip-analysis     Skip Python analysis step
    -h, --help          Show this help message

EXAMPLE:
    $0 --scenario scenarios/hybrid_kyber_dilithium.yaml --out ./results/exp1 --duration 30 --seed 1234

OUTPUTS:
    <out>/raw/run.jsonl           Raw telemetry
    <out>/merged/merged.jsonl     Sorted merged JSONL
    <out>/merged/merged.parquet   Parquet format
    <out>/stats/summary.json      Statistics
    <out>/figures/latency_cdf.png CDF plot
    <out>/figures/throughput.png  Throughput plot
    <out>/manifest.json           Run metadata
EOF
    exit 1
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
log_info "PQC Microbenchmark Runner"
log_info "=========================================="
log_info "Scenario: $SCENARIO"
log_info "Output: $OUT_DIR"
log_info "Experiment ID: $EXP_ID"
[[ -n "$DURATION" ]] && log_info "Duration override: ${DURATION}s"
[[ -n "$SEED" ]] && log_info "RNG seed: $SEED"
log_info ""

# Step 1: Build the Rust binary
log_info "Step 1: Building pqc-bench (release mode)..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARGO_MANIFEST="${SCRIPT_DIR}/rust-core/Cargo.toml"

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

# Step 2: Create output directories
log_info "Step 2: Creating output directories..."
mkdir -p "$OUT_DIR/raw"
mkdir -p "$OUT_DIR/merged"
mkdir -p "$OUT_DIR/stats"
mkdir -p "$OUT_DIR/figures"
log_success "Directories created"

# Step 3: Prepare scenario with overrides
log_info "Step 3: Preparing scenario..."
TEMP_SCENARIO=$(mktemp)
cp "$SCENARIO" "$TEMP_SCENARIO"

# Override JSONL output path to write to raw directory
# We'll use sed to modify the jsonl_out path
RAW_JSONL_PATH="$OUT_DIR/raw/run.jsonl"
if grep -q "jsonl_out:" "$TEMP_SCENARIO"; then
    sed -i "s|jsonl_out:.*|jsonl_out: \"$RAW_JSONL_PATH\"|" "$TEMP_SCENARIO"
else
    # Add jsonl_out to metrics section
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

# Override seed if specified  
if [[ -n "$SEED" ]]; then
    if grep -q "rng_seed:" "$TEMP_SCENARIO"; then
        sed -i "s/rng_seed:.*/rng_seed: $SEED/" "$TEMP_SCENARIO"
    else
        # Add rng_seed at the top level
        sed -i "/^id:/a rng_seed: $SEED" "$TEMP_SCENARIO"
    fi
fi

log_success "Scenario prepared"

# Step 4: Run the benchmark
log_info "Step 4: Running pqc-bench..."
START_TIME=$(date +%s)
START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Run with timeout
set +e
timeout "$TIMEOUT" "$BINARY" --scenario "$TEMP_SCENARIO"
BENCH_EXIT_CODE=$?
set -e

END_TIME=$(date +%s)
END_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
ELAPSED=$((END_TIME - START_TIME))

# Clean up temp scenario
rm -f "$TEMP_SCENARIO"

if [[ $BENCH_EXIT_CODE -eq 124 ]]; then
    log_error "Benchmark timed out after ${TIMEOUT}s"
    exit 1
elif [[ $BENCH_EXIT_CODE -ne 0 ]]; then
    log_error "Benchmark failed with exit code: $BENCH_EXIT_CODE"
    exit $BENCH_EXIT_CODE
fi

log_success "Benchmark completed in ${ELAPSED}s"

# Step 5: Verify raw output
log_info "Step 5: Verifying raw output..."
if [[ ! -f "$RAW_JSONL_PATH" ]]; then
    log_error "Raw JSONL not found: $RAW_JSONL_PATH"
    exit 1
fi

RAW_LINES=$(wc -l < "$RAW_JSONL_PATH")
log_success "Raw JSONL: $RAW_LINES events"

# Step 6: Generate manifest
log_info "Step 6: Generating manifest..."
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
RUSTC_VERSION=$(rustc --version 2>/dev/null || echo "unknown")
SCENARIO_NAME=$(basename "$SCENARIO" .yaml)

cat > "$OUT_DIR/manifest.json" <<EOF
{
    "run_id": "$EXP_ID",
    "scenario": "$SCENARIO",
    "scenario_name": "$SCENARIO_NAME",
    "git_commit": "$GIT_COMMIT",
    "rustc_version": "$RUSTC_VERSION",
    "start_time_utc": "$START_ISO",
    "end_time_utc": "$END_ISO",
    "duration_sec": $ELAPSED,
    "events_count": $RAW_LINES,
    "seed": ${SEED:-null},
    "host": "$(hostname)",
    "platform": "$(uname -s)-$(uname -m)"
}
EOF
log_success "Manifest written"

# Step 7: Run analysis pipeline (if not skipped)
if [[ "$SKIP_ANALYSIS" == "true" ]]; then
    log_warn "Skipping analysis (--skip-analysis)"
else
    log_info "Step 7: Running analysis pipeline..."
    
    ANALYSIS_DIR="${SCRIPT_DIR}/analysis"
    
    if [[ -f "${ANALYSIS_DIR}/run_full_pipeline.sh" ]]; then
        # Run the full analysis pipeline
        bash "${ANALYSIS_DIR}/run_full_pipeline.sh" "$EXP_ID" "$OUT_DIR/raw"
    else
        # Fall back to running individual scripts
        log_info "  Running merge_jsonl.py..."
        if [[ -f "${ANALYSIS_DIR}/scripts/merge_jsonl.py" ]]; then
            python3 "${ANALYSIS_DIR}/scripts/merge_jsonl.py" \
                --input "$OUT_DIR/raw" \
                --output "$OUT_DIR/merged" 2>&1 | while read -r line; do
                echo "    $line"
            done || log_warn "merge_jsonl.py failed"
        fi
        
        log_info "  Running compute_stats.py..."
        if [[ -f "${ANALYSIS_DIR}/scripts/compute_statistics.py" ]]; then
            python3 "${ANALYSIS_DIR}/scripts/compute_statistics.py" \
                --input "$OUT_DIR/merged/merged.parquet" \
                --output "$OUT_DIR/stats" \
                --experiment-id "$EXP_ID" 2>&1 | while read -r line; do
                echo "    $line"
            done || log_warn "compute_statistics.py failed"
        fi
        
        log_info "  Running plot_latency.py..."
        if [[ -f "${ANALYSIS_DIR}/scripts/plot_latency.py" ]]; then
            python3 "${ANALYSIS_DIR}/scripts/plot_latency.py" \
                --input "$OUT_DIR/merged/merged.parquet" \
                --output "$OUT_DIR/figures" \
                --experiment-id "$EXP_ID" 2>&1 | while read -r line; do
                echo "    $line"
            done || log_warn "plot_latency.py failed"
        fi
        
        log_info "  Running plot_throughput.py..."
        if [[ -f "${ANALYSIS_DIR}/scripts/plot_throughput.py" ]]; then
            python3 "${ANALYSIS_DIR}/scripts/plot_throughput.py" \
                --input "$OUT_DIR/merged/merged.parquet" \
                --output "$OUT_DIR/figures" \
                --experiment-id "$EXP_ID" 2>&1 | while read -r line; do
                echo "    $line"
            done || log_warn "plot_throughput.py failed"
        fi
    fi
    
    log_success "Analysis complete"
fi

# Final summary
echo ""
log_info "=========================================="
log_info "EXPERIMENT COMPLETE"
log_info "=========================================="
log_info "Experiment ID: $EXP_ID"
log_info "Duration: ${ELAPSED}s"
log_info "Events: $RAW_LINES"
echo ""
log_info "Outputs:"
echo "  Raw JSONL:     $OUT_DIR/raw/run.jsonl"
[[ -f "$OUT_DIR/merged/merged.jsonl" ]] && echo "  Merged JSONL:  $OUT_DIR/merged/merged.jsonl"
[[ -f "$OUT_DIR/merged/merged.parquet" ]] && echo "  Parquet:       $OUT_DIR/merged/merged.parquet"
[[ -f "$OUT_DIR/stats/summary.json" ]] && echo "  Stats:         $OUT_DIR/stats/summary.json"
[[ -f "$OUT_DIR/figures/latency_cdf.png" ]] && echo "  Latency CDF:   $OUT_DIR/figures/latency_cdf.png"
[[ -f "$OUT_DIR/figures/throughput.png" ]] && echo "  Throughput:    $OUT_DIR/figures/throughput.png"
echo "  Manifest:      $OUT_DIR/manifest.json"
echo ""
log_success "Done!"

exit 0

