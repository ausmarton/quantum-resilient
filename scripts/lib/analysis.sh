#!/usr/bin/env bash
# =============================================================================
# scripts/lib/analysis.sh - Analysis pipeline invocation utilities
#
# Provides unified functions for running the analysis pipeline across all
# environments. Ensures consistent analysis execution regardless of environment.
#
# Usage:
#   source "$SCRIPT_DIR/scripts/lib/analysis.sh"
#   run_analysis_pipeline "$OUT_DIR" "$EXP_ID"
# =============================================================================

# -----------------------------------------------------------------------------
# Analysis Pipeline Functions
# -----------------------------------------------------------------------------

run_analysis_pipeline() {
    local out_dir="$1"
    local exp_id="$2"
    local skip_analysis="${3:-false}"
    
    if [[ "$skip_analysis" == "true" ]]; then
        log_warn "Skipping analysis (--skip-analysis)"
        return 0
    fi
    
    if [[ -z "$out_dir" ]] || [[ -z "$exp_id" ]]; then
        log_error "Output directory and experiment ID required for analysis"
        return 1
    fi
    
    log_info "Running analysis pipeline..."
    
    local script_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
    local analysis_dir="$script_dir/analysis"
    
    # Determine input path (prefer merged, fallback to raw)
    local input_path=""
    if [[ -f "$out_dir/merged/merged.jsonl" ]]; then
        input_path="$out_dir/merged"
    elif [[ -d "$out_dir/merged" ]] && [[ -n "$(find "$out_dir/merged" -name "*.jsonl" -type f 2>/dev/null | head -1)" ]]; then
        input_path="$out_dir/merged"
    else
        input_path="$out_dir/raw"
    fi
    
    # Try full pipeline script first (if available)
    if [[ -f "$analysis_dir/run_full_pipeline.sh" ]]; then
        if bash "$analysis_dir/run_full_pipeline.sh" "$exp_id" "$input_path" 2>&1 | while read -r line; do
            echo "  $line"
        done; then
            log_success "Analysis pipeline complete"
            return 0
        else
            log_warn "Analysis pipeline completed with warnings"
            # Continue with individual scripts as fallback
        fi
    fi
    
    # Run individual analysis scripts
    log_info "Running individual analysis scripts..."
    
    # Merge JSONL files if needed
    if [[ ! -f "$out_dir/merged/merged.jsonl" ]] && [[ -d "$out_dir/raw" ]]; then
        log_info "Merging JSONL files..."
        python3 "$analysis_dir/scripts/merge_jsonl.py" \
            --input "$out_dir/raw" \
            --output "$out_dir/merged" 2>/dev/null || {
            log_warn "merge_jsonl.py failed or not available"
        }
    fi
    
    # Compute statistics
    local input_file=""
    if [[ -f "$out_dir/merged/merged.parquet" ]]; then
        input_file="$out_dir/merged/merged.parquet"
    elif [[ -f "$out_dir/merged/merged.jsonl" ]]; then
        input_file="$out_dir/merged/merged.jsonl"
    else
        log_warn "No merged data file found for statistics computation"
        return 0
    fi
    
    if [[ -n "$input_file" ]]; then
        log_info "Computing statistics..."
        python3 "$analysis_dir/scripts/compute_statistics.py" \
            --input "$input_file" \
            --output "$out_dir/stats" \
            --experiment-id "$exp_id" 2>/dev/null || {
            log_warn "compute_statistics.py failed or not available"
        }
    fi
    
    log_success "Analysis complete"
    return 0
}

