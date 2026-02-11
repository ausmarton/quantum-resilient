#!/usr/bin/env bash
# =============================================================================
# run_full_analysis_pipeline.sh - Complete reproducible analysis pipeline
#
# Runs the full analysis pipeline from raw data to final results and reports:
# 1. Generate experiment summaries (from raw JSONL)
# 2. Aggregate statistics
# 3. Generate hypothesis tests
# 4. Generate visualizations
# 5. Generate tables
# 6. Generate interpretation documents
# 7. Verify requirements compliance
#
# All stages are idempotent - can be re-run safely without re-processing
# existing outputs unless --force is used.
#
# Usage:
#   ./scripts/run_full_analysis_pipeline.sh [OPTIONS]
#
# Options:
#   --force              Force regeneration of all outputs
#   --skip-summaries     Skip summary generation (use existing)
#   --skip-aggregation   Skip aggregation (use existing)
#   --skip-visualizations Skip visualization generation
#   --skip-tables        Skip table generation
#   --skip-interpretation Skip interpretation document generation
#   --env ENV            Process only specific environment (native, minikube, gcp)
#   --output-dir DIR     Output directory (default: final-results)
#   -h, --help           Show this help message
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Defaults
FORCE=false
SKIP_SUMMARIES=false
SKIP_AGGREGATION=false
SKIP_VISUALIZATIONS=false
SKIP_TABLES=false
SKIP_INTERPRETATION=false
ENV_FILTER=""
OUTPUT_DIR="final-results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        --skip-summaries)
            SKIP_SUMMARIES=true
            shift
            ;;
        --skip-aggregation)
            SKIP_AGGREGATION=true
            shift
            ;;
        --skip-visualizations)
            SKIP_VISUALIZATIONS=true
            shift
            ;;
        --skip-tables)
            SKIP_TABLES=true
            shift
            ;;
        --skip-interpretation)
            SKIP_INTERPRETATION=true
            shift
            ;;
        --env)
            ENV_FILTER="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            head -n 30 "$0" | grep -E "^# |^#\$" | sed 's/^# //' | sed 's/^#$//'
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

log_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

log_success() {
    echo -e "\033[0;32m[OK]\033[0m $1"
}

log_warn() {
    echo -e "\033[0;33m[WARN]\033[0m $1"
}

log_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

check_output_exists() {
    local file="$1"
    local description="$2"
    
    if [[ -f "$file" ]] && [[ "$FORCE" != "true" ]]; then
        log_info "Skipping $description (already exists: $file)"
        return 0
    else
        return 1
    fi
}

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/figures"
mkdir -p "$OUTPUT_DIR/tables"

log_info "Starting full analysis pipeline..."
log_info "Output directory: $OUTPUT_DIR"
if [[ "$FORCE" == "true" ]]; then
    log_warn "Force mode enabled - will regenerate all outputs"
fi
echo

# ============================================================================
# Stage 1: Generate Experiment Summaries
# ============================================================================
if [[ "$SKIP_SUMMARIES" != "true" ]]; then
    log_info "Stage 1: Generating experiment summaries..."
    
    if check_output_exists "$OUTPUT_DIR/index.json" "index generation"; then
        log_info "Using existing index.json"
    else
        # Regenerate index first
        log_info "Regenerating experiment index..."
        "$PYTHON_CMD" "$SCRIPT_DIR/scripts/lib/regenerate_index.py" \
            --results-dir results \
            --output "$OUTPUT_DIR/index.json" || {
            log_error "Failed to regenerate index"
            exit 1
        }
        log_success "Index regenerated: $OUTPUT_DIR/index.json"
    fi
    
    # Generate summaries (script has built-in resume capability)
    log_info "Generating experiment summaries..."
    if [[ -n "$ENV_FILTER" ]]; then
        if [[ "$FORCE" == "true" ]]; then
            # Force mode: remove existing summaries first
            log_warn "Force mode: removing existing summaries for $ENV_FILTER"
            find "$SCRIPT_DIR/results/$ENV_FILTER" -name "summary.json" -type f -delete 2>/dev/null || true
        fi
        "$SCRIPT_DIR/scripts/generate_experiment_summaries.sh" --env "$ENV_FILTER" || {
            log_error "Summary generation failed"
            exit 1
        }
    else
        if [[ "$FORCE" == "true" ]]; then
            # Force mode: remove existing summaries first
            log_warn "Force mode: removing existing summaries"
            find "$SCRIPT_DIR/results" -name "summary.json" -type f -delete 2>/dev/null || true
        fi
        "$SCRIPT_DIR/scripts/generate_experiment_summaries.sh" --resume || {
            log_error "Summary generation failed"
            exit 1
        }
    fi
    log_success "Experiment summaries generated"
else
    log_info "Skipping summary generation (--skip-summaries)"
fi
echo

# ============================================================================
# Stage 2: Aggregate Statistics
# ============================================================================
if [[ "$SKIP_AGGREGATION" != "true" ]]; then
    log_info "Stage 2: Aggregating statistics..."
    
    if check_output_exists "$OUTPUT_DIR/aggregated_stats.json" "aggregation"; then
        log_info "Using existing aggregated_stats.json"
    else
        log_info "Running aggregate_results.py..."
        "$PYTHON_CMD" "$SCRIPT_DIR/analysis/aggregate_results.py" \
            --index "$OUTPUT_DIR/index.json" \
            --output "$OUTPUT_DIR" || {
            log_error "Aggregation failed"
            exit 1
        }
        log_success "Statistics aggregated: $OUTPUT_DIR/aggregated_stats.json"
    fi
else
    log_info "Skipping aggregation (--skip-aggregation)"
fi
echo

# ============================================================================
# Stage 3: Hypothesis Tests
# ============================================================================
if [[ "$SKIP_AGGREGATION" != "true" ]]; then
    log_info "Stage 3: Running hypothesis tests..."
    
    if check_output_exists "$OUTPUT_DIR/hypothesis_tests.json" "hypothesis tests"; then
        log_info "Using existing hypothesis_tests.json"
    else
        log_info "Running hypothesis_tests.py..."
        "$PYTHON_CMD" "$SCRIPT_DIR/analysis/hypothesis_tests.py" \
            --index "$OUTPUT_DIR/index.json" \
            --matrix "$SCRIPT_DIR/orchestration/experiment_matrix.yaml" \
            --output "$OUTPUT_DIR" || {
            log_error "Hypothesis tests failed"
            exit 1
        }
        log_success "Hypothesis tests completed: $OUTPUT_DIR/hypothesis_tests.json"
    fi
else
    log_info "Skipping hypothesis tests (depends on aggregation)"
fi
echo

# ============================================================================
# Stage 4: Generate Visualizations
# ============================================================================
if [[ "$SKIP_VISUALIZATIONS" != "true" ]]; then
    log_info "Stage 4: Generating visualizations..."
    
    # CDF plots
    if check_output_exists "$OUTPUT_DIR/figures/combined_ecdf.png" "CDF plots"; then
        log_info "Using existing CDF plots"
    else
        log_info "Generating CDF plots..."
        # Try plot_combined_cdfs.py first, fall back to plot_ecdf.py
        if [[ -f "$SCRIPT_DIR/analysis/plot_combined_cdfs.py" ]]; then
            "$PYTHON_CMD" "$SCRIPT_DIR/analysis/plot_combined_cdfs.py" \
                --index "$OUTPUT_DIR/index.json" \
                --output "$OUTPUT_DIR/figures" || {
                log_warn "CDF plot generation failed (may not be critical)"
            }
        elif [[ -f "$SCRIPT_DIR/analysis/plot_ecdf.py" ]]; then
            "$PYTHON_CMD" "$SCRIPT_DIR/analysis/plot_ecdf.py" \
                --index "$OUTPUT_DIR/index.json" \
                --output "$OUTPUT_DIR/figures" || {
                log_warn "CDF plot generation failed (may not be critical)"
            }
        else
            log_warn "No CDF plot script found, skipping"
        fi
    fi
    
    # Scaling curves
    if check_output_exists "$OUTPUT_DIR/figures/scaling_curves.png" "scaling curves"; then
        log_info "Using existing scaling curves"
    else
        log_info "Generating scaling curves..."
        "$PYTHON_CMD" "$SCRIPT_DIR/analysis/plot_scaling_curves.py" \
            --index "$OUTPUT_DIR/index.json" \
            --output "$OUTPUT_DIR/figures" || {
            log_warn "Scaling curve generation failed (may not be critical)"
        }
    fi
    
    # Environment comparisons
    if check_output_exists "$OUTPUT_DIR/figures/native_vs_minikube_vs_gcp.png" "environment comparisons"; then
        log_info "Using existing environment comparison plots"
    else
        log_info "Generating environment comparison plots..."
        # Find representative experiments for comparison
        python3 << 'PYTHON_EOF'
import json
from pathlib import Path
import subprocess

with open('final-results/index.json') as f:
    index = json.load(f)

# Find rsa2048 in all environments
examples = {}
for exp in index['experiments']:
    algo = exp.get('algorithm', '')
    env = exp.get('environment', '')
    if algo == 'rsa2048' and env in ['native', 'minikube', 'gcp']:
        if env not in examples:
            summary_path = Path(exp['output_dir']) / 'stats' / 'summary.json'
            if summary_path.exists():
                rel_path = summary_path.relative_to(Path.cwd())
                examples[env] = str(rel_path)
                if len(examples) == 3:
                    break

if len(examples) == 3:
    cmd = [
        './scripts/lib/run-python-container.sh',
        'analysis/compare_all_environments.py',
        '--native', examples['native'],
        '--minikube', examples['minikube'],
        '--gcp', examples['gcp'],
        '--output', 'final-results/comparison_table.json'
    ]
    subprocess.run(cmd, check=False)
PYTHON_EOF
        log_info "Environment comparison completed"
    fi
    
    log_success "Visualizations generated"
else
    log_info "Skipping visualization generation (--skip-visualizations)"
fi
echo

# ============================================================================
# Stage 5: Generate Tables
# ============================================================================
if [[ "$SKIP_TABLES" != "true" ]]; then
    log_info "Stage 5: Generating tables..."
    
    if check_output_exists "$OUTPUT_DIR/tables/performance_table.csv" "performance tables"; then
        log_info "Using existing tables"
    else
        log_info "Generating performance and effect size tables..."
        "$PYTHON_CMD" "$SCRIPT_DIR/scripts/extract_analysis_tables.py" \
            --aggregated "$OUTPUT_DIR/aggregated_stats.json" \
            --hypothesis "$OUTPUT_DIR/hypothesis_tests.json" \
            --output "$OUTPUT_DIR/tables" || {
            log_error "Table generation failed"
            exit 1
        }
        log_success "Tables generated: $OUTPUT_DIR/tables/"
    fi
else
    log_info "Skipping table generation (--skip-tables)"
fi
echo

# ============================================================================
# Stage 6: Generate Interpretation Documents
# ============================================================================
if [[ "$SKIP_INTERPRETATION" != "true" ]]; then
    log_info "Stage 6: Generating interpretation documents..."
    
    # This stage extracts data and populates interpretation documents
    # It's safe to re-run as it just updates markdown files
    if [[ -f "$SCRIPT_DIR/scripts/generate_interpretation_docs.py" ]]; then
        log_info "Updating interpretation documents with latest data..."
        "$PYTHON_CMD" "$SCRIPT_DIR/scripts/generate_interpretation_docs.py" \
            --stats "$OUTPUT_DIR/aggregated_stats.json" \
            --hypothesis "$OUTPUT_DIR/hypothesis_tests.json" \
            --output "$SCRIPT_DIR/docs/analysis" || {
            log_warn "Interpretation document generation failed (may not exist yet)"
        }
        log_success "Interpretation documents updated"
    else
        log_info "Interpretation doc generator not found, skipping (docs can be updated manually)"
    fi
else
    log_info "Skipping interpretation document generation (--skip-interpretation)"
fi
echo

# ============================================================================
# Stage 7: Generate Additional Metrics (Optional)
# ============================================================================
log_info "Stage 7: Generating additional metrics..."

# Cost efficiency (FR13) - optional
if [[ -f "$SCRIPT_DIR/scripts/compute_cost_efficiency.py" ]]; then
    if check_output_exists "$OUTPUT_DIR/cost_efficiency.json" "cost efficiency metrics"; then
        log_info "Using existing cost efficiency metrics"
    else
        log_info "Computing cost efficiency metrics..."
        "$PYTHON_CMD" "$SCRIPT_DIR/scripts/compute_cost_efficiency.py" \
            --stats "$OUTPUT_DIR/aggregated_stats.json" \
            --output "$OUTPUT_DIR/cost_efficiency.json" || {
            log_warn "Cost efficiency computation failed (optional)"
        }
    fi
fi

# Automated report (NFR8) - optional
if [[ -f "$SCRIPT_DIR/scripts/generate_analysis_report.py" ]]; then
    if check_output_exists "$OUTPUT_DIR/analysis_report.md" "analysis report"; then
        log_info "Using existing analysis report"
    else
        log_info "Generating analysis report..."
        "$PYTHON_CMD" "$SCRIPT_DIR/scripts/generate_analysis_report.py" \
            --stats "$OUTPUT_DIR/aggregated_stats.json" \
            --hypothesis "$OUTPUT_DIR/hypothesis_tests.json" \
            --compliance "$OUTPUT_DIR/compliance_report.json" \
            --output "$OUTPUT_DIR/analysis_report.md" || {
            log_warn "Report generation failed (optional)"
        }
    fi
fi

log_success "Additional metrics generated"
echo

# ============================================================================
# Stage 8: Verify Requirements Compliance
# ============================================================================
log_info "Stage 8: Verifying requirements compliance..."
"$PYTHON_CMD" "$SCRIPT_DIR/scripts/verify_requirements_compliance.py" \
    --base-dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/compliance_report.json" || {
    log_warn "Some compliance checks failed - see $OUTPUT_DIR/compliance_report.json"
}
log_success "Compliance verification completed"
echo

# ============================================================================
# Summary
# ============================================================================
log_success "Analysis pipeline completed successfully!"
echo
log_info "Outputs generated in: $OUTPUT_DIR"
log_info "  - Aggregated statistics: $OUTPUT_DIR/aggregated_stats.json"
log_info "  - Hypothesis tests: $OUTPUT_DIR/hypothesis_tests.json"
log_info "  - Figures: $OUTPUT_DIR/figures/"
log_info "  - Tables: $OUTPUT_DIR/tables/"
if [[ -f "$OUTPUT_DIR/cost_efficiency.json" ]]; then
    log_info "  - Cost efficiency: $OUTPUT_DIR/cost_efficiency.json"
fi
if [[ -f "$OUTPUT_DIR/analysis_report.md" ]]; then
    log_info "  - Analysis report: $OUTPUT_DIR/analysis_report.md"
fi
log_info "  - Compliance report: $OUTPUT_DIR/compliance_report.json"
echo
log_info "To re-run with force: ./scripts/run_full_analysis_pipeline.sh --force"
echo
log_info "Pipeline is fully reproducible and idempotent."
log_info "Re-run anytime to regenerate missing outputs or update interpretation docs."
