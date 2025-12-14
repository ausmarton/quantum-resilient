#!/usr/bin/env bash
# =============================================================================
# cleanup_processed_data.sh - Safely remove processed/analyzed data
#
# Removes processed data (merged, stats, figures, exports) while preserving
# raw data in results/ directory. Also cleans final-results/ directory.
#
# SAFETY: This script will NEVER delete raw data from results/*/raw/
#
# Usage:
#   ./scripts/cleanup_processed_data.sh [OPTIONS]
#
# Options:
#   --dry-run          Show what would be deleted without actually deleting
#   --keep-index       Keep final-results/index.json (regenerate other files)
#   --keep-final-results  Don't delete final-results/ directory
#   -h, --help         Show this help message
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

DRY_RUN=false
KEEP_INDEX=false
KEEP_FINAL_RESULTS=false

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Safely remove processed/analyzed data while preserving raw data.

OPTIONS:
    --dry-run              Show what would be deleted without actually deleting
    --keep-index           Keep final-results/index.json (regenerate other files)
    --keep-final-results   Don't delete final-results/ directory
    -h, --help             Show this help message

EXAMPLES:
    # Preview what would be deleted
    ./scripts/cleanup_processed_data.sh --dry-run

    # Clean all processed data (keeps raw data)
    ./scripts/cleanup_processed_data.sh

    # Clean but keep index.json
    ./scripts/cleanup_processed_data.sh --keep-index
EOF
    exit 1
}

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

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --keep-index)
            KEEP_INDEX=true
            shift
            ;;
        --keep-final-results)
            KEEP_FINAL_RESULTS=true
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

# Counters
TOTAL_SIZE=0
DIRS_REMOVED=0
FILES_REMOVED=0

remove_if_exists() {
    local path="$1"
    local description="${2:-$path}"
    
    if [[ -e "$path" ]]; then
        if [[ -d "$path" ]]; then
            local size=$(du -sb "$path" 2>/dev/null | cut -f1 || echo 0)
            TOTAL_SIZE=$((TOTAL_SIZE + size))
            DIRS_REMOVED=$((DIRS_REMOVED + 1))
            
            if [[ "$DRY_RUN" == "true" ]]; then
                log_info "Would remove directory: $description ($(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo "${size}B"))"
            else
                rm -rf "$path"
                log_success "Removed directory: $description"
            fi
        elif [[ -f "$path" ]]; then
            local size=$(stat -f%z "$path" 2>/dev/null || stat -c%s "$path" 2>/dev/null || echo 0)
            TOTAL_SIZE=$((TOTAL_SIZE + size))
            FILES_REMOVED=$((FILES_REMOVED + 1))
            
            if [[ "$DRY_RUN" == "true" ]]; then
                log_info "Would remove file: $description ($(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo "${size}B"))"
            else
                rm -f "$path"
                log_success "Removed file: $description"
            fi
        fi
    fi
}

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Cleanup Processed Data${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    log_warn "DRY RUN MODE - No files will be deleted"
    echo ""
fi

# Step 1: Clean final-results/ directory
if [[ "$KEEP_FINAL_RESULTS" != "true" ]]; then
    log_info "Step 1: Cleaning final-results/ directory..."
    
    if [[ -d "final-results" ]]; then
        if [[ "$KEEP_INDEX" == "true" ]]; then
            # Remove everything except index.json
            for item in final-results/*; do
                if [[ -f "$item" ]] && [[ "$(basename "$item")" != "index.json" ]]; then
                    remove_if_exists "$item" "final-results/$(basename "$item")"
                elif [[ -d "$item" ]]; then
                    remove_if_exists "$item" "final-results/$(basename "$item")"
                fi
            done
        else
            # Remove entire directory
            remove_if_exists "final-results" "final-results/"
        fi
    else
        log_warn "final-results/ directory does not exist (skipping)"
    fi
    echo ""
fi

# Step 2: Clean processed data from results/ directory
log_info "Step 2: Cleaning processed data from results/ directory..."
log_info "  Preserving: results/*/raw/ (raw data)"
log_info "  Removing: results/*/merged/, results/*/stats/, results/*/figures/, results/*/exports/"

PROCESSED_DIRS=("merged" "stats" "figures" "exports")

# Find all experiment directories
EXPERIMENT_DIRS=()
while IFS= read -r -d '' dir; do
    EXPERIMENT_DIRS+=("$dir")
done < <(find results -type d -mindepth 2 -maxdepth 2 -print0 2>/dev/null || true)

# Also check for run-*/ subdirectories
while IFS= read -r -d '' dir; do
    EXPERIMENT_DIRS+=("$dir")
done < <(find results -type d -name "run-*" -print0 2>/dev/null || true)

TOTAL_EXPERIMENTS=${#EXPERIMENT_DIRS[@]}
log_info "  Found $TOTAL_EXPERIMENTS experiment directories to process"

for exp_dir in "${EXPERIMENT_DIRS[@]}"; do
    for proc_dir in "${PROCESSED_DIRS[@]}"; do
        proc_path="$exp_dir/$proc_dir"
        if [[ -d "$proc_path" ]]; then
            remove_if_exists "$proc_path" "$proc_path"
        fi
    done
done

# Also clean any top-level processed directories in results/
for env in native minikube gcp; do
    if [[ -d "results/$env" ]]; then
        for proc_dir in "${PROCESSED_DIRS[@]}"; do
            # Check for direct processed dirs (shouldn't exist, but clean if they do)
            proc_path="results/$env/$proc_dir"
            if [[ -d "$proc_path" ]]; then
                remove_if_exists "$proc_path" "$proc_path"
            fi
        done
    fi
done

echo ""

# Step 3: Clean test-results/ directory (optional)
log_info "Step 3: Cleaning test-results/ directory..."
if [[ -d "test-results" ]]; then
    remove_if_exists "test-results" "test-results/"
else
    log_warn "test-results/ directory does not exist (skipping)"
fi
echo ""

# Step 4: Verify raw data is preserved
log_info "Step 4: Verifying raw data is preserved..."
RAW_COUNT=$(find results -type d -name "raw" 2>/dev/null | wc -l || echo 0)
RAW_FILES=$(find results -path "*/raw/*.jsonl" -type f 2>/dev/null | wc -l || echo 0)

if [[ $RAW_COUNT -gt 0 ]]; then
    log_success "Found $RAW_COUNT raw/ directories with $RAW_FILES JSONL files (preserved)"
else
    log_warn "No raw/ directories found (this may be normal if no experiments have been run)"
fi
echo ""

# Summary
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Cleanup Summary${NC}"
echo -e "${CYAN}========================================${NC}"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    log_info "DRY RUN COMPLETE - No files were actually deleted"
    echo ""
    log_info "Would remove:"
    log_info "  - $DIRS_REMOVED directories"
    log_info "  - $FILES_REMOVED files"
    log_info "  - Total size: $(numfmt --to=iec-i --suffix=B $TOTAL_SIZE 2>/dev/null || echo "${TOTAL_SIZE}B")"
    echo ""
    log_info "To actually perform cleanup, run without --dry-run flag"
else
    echo ""
    log_success "Cleanup complete!"
    echo ""
    log_info "Removed:"
    log_info "  - $DIRS_REMOVED directories"
    log_info "  - $FILES_REMOVED files"
    log_info "  - Total size: $(numfmt --to=iec-i --suffix=B $TOTAL_SIZE 2>/dev/null || echo "${TOTAL_SIZE}B")"
    echo ""
    log_success "Raw data preserved: $RAW_COUNT raw/ directories, $RAW_FILES JSONL files"
    echo ""
    log_info "Next steps:"
    log_info "  1. Regenerate index: ./scripts/regenerate_index_from_results.sh"
    log_info "  2. Run analysis: Follow docs/analysis/dissertation-guide.md"
fi

echo ""
