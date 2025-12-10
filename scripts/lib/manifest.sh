#!/usr/bin/env bash
# =============================================================================
# scripts/lib/manifest.sh - Manifest generation utilities
#
# Provides functions for generating experiment manifest files with consistent
# structure across all environments (native, Minikube, GCP).
#
# Usage:
#   source "$SCRIPT_DIR/scripts/lib/manifest.sh"
#   generate_manifest "$OUT_DIR" "$EXP_ID" "$SCENARIO" "$ENVIRONMENT"
# =============================================================================

# -----------------------------------------------------------------------------
# Manifest Generation Functions
# -----------------------------------------------------------------------------

generate_manifest() {
    local out_dir="$1"
    local exp_id="$2"
    local scenario_path="${3:-}"
    local environment="${4:-unknown}"
    local run_index="${5:-1}"
    local event_count="${6:-0}"
    local duration_sec="${7:-0}"
    local rng_seed="${8:-null}"
    local replicas="${9:-1}"
    local extra_fields="${10:-}"
    
    if [[ -z "$out_dir" ]] || [[ -z "$exp_id" ]]; then
        log_error "Output directory and experiment ID required for manifest"
        return 1
    fi
    
    local script_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
    local git_commit=$(git -C "$script_dir" rev-parse HEAD 2>/dev/null || echo "unknown")
    local rustc_version=$(rustc --version 2>/dev/null || echo "unknown")
    local start_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    local end_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    
    # Generate base manifest JSON
    cat > "$out_dir/manifest.json" <<EOF
{
    "run_id": "${exp_id}",
    "run_index": ${run_index},
    "scenario_id": "$(basename "$scenario_path" .yaml 2>/dev/null || echo "unknown")",
    "scenario_path": "${scenario_path}",
    "environment": "${environment}",
    "execution_type": "${environment}",
    "git_commit": "${git_commit}",
    "rustc_version": "${rustc_version}",
    "start_time_utc": "${start_time}",
    "end_time_utc": "${end_time}",
    "duration_sec": ${duration_sec},
    "events_count": ${event_count},
    "rng_seed": ${rng_seed},
    "replicas": ${replicas},
    "host": "$(hostname)",
    "platform": "$(uname -s)-$(uname -m)"${extra_fields}
}
EOF
    
    log_success "Manifest written: $out_dir/manifest.json"
    return 0
}

