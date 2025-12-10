#!/usr/bin/env bash
# =============================================================================
# scripts/lib/directories.sh - Directory creation utilities
#
# Provides functions for creating consistent output directory structures
# across all environments (native, Minikube, GCP).
#
# Usage:
#   source "$SCRIPT_DIR/scripts/lib/directories.sh"
#   create_output_directories "$OUT_DIR"
# =============================================================================

# -----------------------------------------------------------------------------
# Directory Creation Functions
# -----------------------------------------------------------------------------

create_output_directories() {
    local out_dir="$1"
    
    if [[ -z "$out_dir" ]]; then
        log_error "Output directory not specified"
        return 1
    fi
    
    # Create base directory
    mkdir -p "$out_dir"
    
    # Create standard subdirectories
    mkdir -p "$out_dir/raw"
    mkdir -p "$out_dir/merged"
    mkdir -p "$out_dir/stats"
    mkdir -p "$out_dir/figures"
    
    log_success "Created output directories: $out_dir/{raw,merged,stats,figures}"
    return 0
}

verify_output_directories() {
    local out_dir="$1"
    local missing_dirs=()
    
    for subdir in raw merged stats figures; do
        if [[ ! -d "$out_dir/$subdir" ]]; then
            missing_dirs+=("$subdir")
        fi
    done
    
    if [[ ${#missing_dirs[@]} -gt 0 ]]; then
        log_error "Missing directories: ${missing_dirs[*]}"
        return 1
    fi
    
    return 0
}

