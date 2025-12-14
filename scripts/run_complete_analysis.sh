#!/usr/bin/env bash
# =============================================================================
# run_complete_analysis.sh - Run complete analysis pipeline once summaries ready
#
# This script runs the complete analysis pipeline:
# 1. Aggregate statistics
# 2. Generate all visualizations
# 3. Run hypothesis tests
# 4. Generate scaling plots
# 5. Create comparison tables
#
# Usage:
#   ./scripts/run_complete_analysis.sh [OPTIONS]
#
# Options:
#   --skip-aggregate    Skip aggregation step
#   --skip-plots        Skip visualization generation
#   --skip-hypothesis   Skip hypothesis tests
#   --skip-scaling      Skip scaling plots
#   -h, --help          Show this help message
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

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

SKIP_AGGREGATE=false
SKIP_PLOTS=false
SKIP_HYPOTHESIS=false
SKIP_SCALING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-aggregate)
            SKIP_AGGREGATE=true
            shift
            ;;
        --skip-plots)
            SKIP_PLOTS=true
            shift
            ;;
        --skip-hypothesis)
            SKIP_HYPOTHESIS=true
            shift
            ;;
        --skip-scaling)
            SKIP_SCALING=true
            shift
            ;;
        -h|--help)
            cat <<EOF
Usage: $0 [OPTIONS]

Run complete analysis pipeline.

OPTIONS:
    --skip-aggregate    Skip aggregation step
    --skip-plots        Skip visualization generation
    --skip-hypothesis   Skip hypothesis tests
    --skip-scaling      Skip scaling plots
    -h, --help          Show this help message
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Get Python command
PYTHON_CMD="$SCRIPT_DIR/scripts/lib/run-python-container.sh"
if [[ ! -f "$PYTHON_CMD" ]] || [[ "${QR_USE_CONTAINER:-true}" == "false" ]]; then
    PYTHON_CMD="python3"
fi

INDEX_FILE="final-results/index.json"
OUTPUT_DIR="final-results"
FIGURES_DIR="$OUTPUT_DIR/figures"

# Check prerequisites
if [[ ! -f "$INDEX_FILE" ]]; then
    log_warn "Index file not found: $INDEX_FILE"
    log_info "Regenerating index..."
    ./scripts/regenerate_index_from_results.sh \
        --matrix orchestration/experiment_matrix.yaml \
        --output "$OUTPUT_DIR"
fi

# Check summary count
SUMMARY_COUNT=$(find results -path "*/stats/summary.json" -type f 2>/dev/null | wc -l || echo 0)
TOTAL_EXPERIMENTS=$(python3 -c "import json; idx=json.load(open('$INDEX_FILE')); print(len(idx.get('experiments', [])))" 2>/dev/null || echo 396)

log_info "Found $SUMMARY_COUNT summaries out of $TOTAL_EXPERIMENTS experiments"

if [[ $SUMMARY_COUNT -lt $((TOTAL_EXPERIMENTS * 80 / 100)) ]]; then
    log_warn "Less than 80% summaries complete. Some analysis may be incomplete."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
log_info "=========================================="
log_info "Complete Analysis Pipeline"
log_info "=========================================="
echo ""

# Step 1: Aggregate Statistics
if [[ "$SKIP_AGGREGATE" != "true" ]]; then
    log_info "Step 1: Aggregating statistics..."
    "$PYTHON_CMD" analysis/aggregate_results.py \
        --index "$INDEX_FILE" \
        --output "$OUTPUT_DIR"
    
    if [[ -f "$OUTPUT_DIR/aggregated_stats.json" ]]; then
        AGG_COUNT=$(python3 -c "import json; d=json.load(open('$OUTPUT_DIR/aggregated_stats.json')); print(len(d.get('aggregated', [])))" 2>/dev/null || echo 0)
        log_success "Aggregated $AGG_COUNT experiment configurations"
    else
        log_warn "Aggregated stats file not generated"
    fi
    echo ""
fi

# Step 2: Generate CDF Plots
if [[ "$SKIP_PLOTS" != "true" ]]; then
    log_info "Step 2: Generating CDF plots..."
    mkdir -p "$FIGURES_DIR"
    "$PYTHON_CMD" analysis/plot_combined_cdfs.py \
        --index "$INDEX_FILE" \
        --output "$FIGURES_DIR"
    
    CDF_COUNT=$(find "$FIGURES_DIR" -name "*cdf*.png" -type f 2>/dev/null | wc -l || echo 0)
    log_success "Generated $CDF_COUNT CDF plots"
    echo ""
fi

# Step 3: Generate Scaling Curves
if [[ "$SKIP_PLOTS" != "true" ]]; then
    log_info "Step 3: Generating scaling curves..."
    "$PYTHON_CMD" analysis/plot_scaling_curves.py \
        --index "$INDEX_FILE" \
        --output "$FIGURES_DIR" 2>/dev/null || log_warn "Scaling curves generation failed or no scaling data"
    echo ""
fi

# Step 4: Generate Replica Scaling Plots
if [[ "$SKIP_SCALING" != "true" ]]; then
    log_info "Step 4: Generating replica scaling plots..."
    mkdir -p "$FIGURES_DIR/scaling"
    "$PYTHON_CMD" analysis/plot_replica_scaling.py \
        --index "$INDEX_FILE" \
        --output "$FIGURES_DIR/scaling" 2>/dev/null || log_warn "Replica scaling plots failed or no scaling data"
    echo ""
fi

# Step 5: Run Hypothesis Tests
if [[ "$SKIP_HYPOTHESIS" != "true" ]]; then
    log_info "Step 5: Running hypothesis tests..."
    "$PYTHON_CMD" analysis/hypothesis_tests.py \
        --index "$INDEX_FILE" \
        --matrix orchestration/experiment_matrix.yaml \
        --output "$OUTPUT_DIR"
    
    if [[ -f "$OUTPUT_DIR/hypothesis_tests.json" ]]; then
        TEST_COUNT=$(python3 -c "import json; d=json.load(open('$OUTPUT_DIR/hypothesis_tests.json')); print(len(d.get('tests', [])))" 2>/dev/null || echo 0)
        log_success "Generated $TEST_COUNT hypothesis test results"
    else
        log_warn "Hypothesis tests file not generated"
    fi
    echo ""
fi

# Step 6: Cross-Environment Comparison
if [[ "$SKIP_PLOTS" != "true" ]]; then
    log_info "Step 6: Generating environment comparisons..."
    if [[ -f "$OUTPUT_DIR/aggregated_stats.json" ]]; then
        "$PYTHON_CMD" analysis/compare_all_environments.py \
            --native "$OUTPUT_DIR/aggregated_stats.json" \
            --minikube "$OUTPUT_DIR/aggregated_stats.json" \
            --gcp "$OUTPUT_DIR/aggregated_stats.json" \
            --output "$OUTPUT_DIR/comparisons" 2>/dev/null || log_warn "Environment comparison failed"
    fi
    echo ""
fi

# Summary
log_success "=========================================="
log_success "Analysis Pipeline Complete!"
log_success "=========================================="
echo ""

log_info "Generated artifacts:"
[[ -f "$OUTPUT_DIR/aggregated_stats.json" ]] && log_info "  ✅ Aggregated statistics"
[[ -f "$OUTPUT_DIR/aggregated_stats.csv" ]] && log_info "  ✅ Aggregated statistics (CSV)"
[[ -f "$OUTPUT_DIR/hypothesis_tests.json" ]] && log_info "  ✅ Hypothesis tests"
[[ -f "$OUTPUT_DIR/hypothesis_table.csv" ]] && log_info "  ✅ Hypothesis tests (CSV)"
FIG_COUNT=$(find "$FIGURES_DIR" -name "*.png" -type f 2>/dev/null | wc -l || echo 0)
log_info "  ✅ $FIG_COUNT visualization files"

echo ""
log_info "Output directory: $OUTPUT_DIR"
log_info "Figures directory: $FIGURES_DIR"
echo ""
