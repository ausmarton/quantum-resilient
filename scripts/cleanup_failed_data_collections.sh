#!/usr/bin/env bash
# =============================================================================
# cleanup_failed_data_collections.sh - Clean up data-collection directories
#
# Keeps directories with at least one successful experiment and deletes
# directories that only contain failures.
#
# Usage:
#   ./scripts/cleanup_failed_data_collections.sh [OPTIONS]
#
# Options:
#   --dry-run          Show what would be deleted without actually deleting
#   --verbose          Show detailed analysis for each directory
#   --min-success N    Minimum number of successful experiments to keep (default: 1)
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

Clean up data-collection directories, keeping only those with successful experiments.

OPTIONS:
    --dry-run          Show what would be deleted without actually deleting
    --verbose          Show detailed analysis for each directory
    --min-success N    Minimum number of successful experiments to keep (default: 1)
    -h, --help         Show this help message

EXAMPLE:
    $0 --dry-run --verbose
    $0 --min-success 10
EOF
    exit 1
}

DRY_RUN=false
VERBOSE=false
MIN_SUCCESS=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --min-success)
            MIN_SUCCESS="$2"
            shift 2
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

echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Data Collection Cleanup${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
log_info "Scanning data-collection directories..."
log_info "Minimum successful experiments to keep: $MIN_SUCCESS"
[[ "$DRY_RUN" == "true" ]] && log_warn "DRY RUN MODE - No files will be deleted"
echo ""

# Find all data-collection directories
DATA_DIRS=$(find . -maxdepth 1 -type d -name "data-collection-*" | sort)

if [[ -z "$DATA_DIRS" ]]; then
    log_info "No data-collection directories found"
    exit 0
fi

TOTAL_DIRS=0
KEEP_DIRS=0
DELETE_DIRS=0
TOTAL_SUCCESS=0
TOTAL_FAILED=0

# Analyze each directory
while IFS= read -r dir; do
    [[ ! -d "$dir" ]] && continue
    
    TOTAL_DIRS=$((TOTAL_DIRS + 1))
    dir_name=$(basename "$dir")
    
    # Check for log files
    log_files=$(find "$dir" -maxdepth 1 -name "*_run.log" -type f)
    
    if [[ -z "$log_files" ]]; then
        # No log files - likely incomplete run, mark for deletion
        if [[ "$VERBOSE" == "true" ]]; then
            log_warn "$dir_name: No log files found"
        fi
        DELETE_DIRS=$((DELETE_DIRS + 1))
        if [[ "$DRY_RUN" == "false" ]]; then
            rm -rf "$dir"
            log_info "  Deleted: $dir_name (no log files)"
        else
            log_info "  Would delete: $dir_name (no log files)"
        fi
        continue
    fi
    
    # Analyze log files for success/failure
    total_success=0
    total_failed=0
    has_any_success=false
    
    while IFS= read -r log_file; do
        if [[ ! -f "$log_file" ]]; then
            continue
        fi
        
        # Extract completion statistics from log
        # Look for patterns like:
        # - "Completed: 489"
        # - "Failed: 6"
        # - "skipped: 486" (cached/skipped experiments are also success)
        # - "[OK] Done!"
        # - "EXPERIMENT SUITE COMPLETE"
        
        completed_count=$(grep -oP 'Completed:\s*\K\d+' "$log_file" 2>/dev/null | tail -1 || echo "0")
        failed_count=$(grep -oP 'Failed:\s*\K\d+' "$log_file" 2>/dev/null | tail -1 || echo "0")
        skipped_count=$(grep -oP 'skipped:\s*\K\d+' "$log_file" 2>/dev/null | tail -1 || echo "0")
        
        # Check for successful completion indicators
        # A run is successful if:
        # 1. It has completed experiments > 0
        # 2. It has skipped/cached experiments > 0 (means it found existing successful results)
        # 3. It ends with "EXPERIMENT SUITE COMPLETE" or "[OK] Done!"
        # 4. It doesn't end with errors
        
        if grep -qE 'EXPERIMENT SUITE COMPLETE|\[OK\].*Done!' "$log_file" 2>/dev/null; then
            has_any_success=true
        fi
        
        # If there are skipped experiments, that means some experiments were already successful
        if [[ $skipped_count -gt 0 ]]; then
            has_any_success=true
        fi
        
        # Check if log ends with error (indicates failure)
        if tail -5 "$log_file" 2>/dev/null | grep -qE '\[ERROR\]|Failed|failed'; then
            # Only mark as failure if no success indicators found
            if [[ $completed_count -eq 0 ]] && [[ $skipped_count -eq 0 ]] && [[ "$has_any_success" == "false" ]]; then
                # This might be a failed run, but we'll check other logs too
                :
            fi
        fi
        
        # Convert to integers
        completed_count=$((completed_count + 0))
        failed_count=$((failed_count + 0))
        
        total_success=$((total_success + completed_count))
        total_failed=$((total_failed + failed_count))
        
        if [[ "$VERBOSE" == "true" ]]; then
            log_file_name=$(basename "$log_file")
            echo "    $log_file_name: Completed=$completed_count, Failed=$failed_count"
        fi
        
    done <<< "$log_files"
    
    # Check summary.txt if it exists
    if [[ -f "$dir/summary.txt" ]]; then
        summary_content=$(cat "$dir/summary.txt")
        if echo "$summary_content" | grep -qiE 'completed|success'; then
            has_any_success=true
        fi
    fi
    
    TOTAL_SUCCESS=$((TOTAL_SUCCESS + total_success))
    TOTAL_FAILED=$((TOTAL_FAILED + total_failed))
    
    # Decision: keep or delete?
    # Keep if:
    # 1. Has minimum required successful experiments
    # 2. Has any success indicators (completed suite, skipped experiments, etc.)
    # 3. Has skipped/cached experiments (means it found existing successful results)
    
    should_keep=false
    
    if [[ $total_success -ge $MIN_SUCCESS ]]; then
        should_keep=true
    elif [[ "$has_any_success" == "true" ]]; then
        should_keep=true
    elif [[ $total_success -eq 0 ]] && [[ $total_failed -eq 0 ]]; then
        # Check if this is a run that only had skipped/cached experiments
        # (which means it found existing successful results)
        if grep -qE 'skipped:|cached|Found existing results' <<< "$log_files" 2>/dev/null; then
            # Check logs for skipped count
            total_skipped=0
            while IFS= read -r log_file; do
                skipped=$(grep -oP 'skipped:\s*\K\d+' "$log_file" 2>/dev/null | tail -1 || echo "0")
                total_skipped=$((total_skipped + skipped))
            done <<< "$log_files"
            
            if [[ $total_skipped -gt 0 ]]; then
                should_keep=true
            fi
        fi
    fi
    
    if [[ "$should_keep" == "true" ]]; then
        KEEP_DIRS=$((KEEP_DIRS + 1))
        if [[ "$VERBOSE" == "true" ]]; then
            log_success "$dir_name: KEEP (Success: $total_success, Failed: $total_failed)"
        fi
    else
        DELETE_DIRS=$((DELETE_DIRS + 1))
        if [[ "$DRY_RUN" == "false" ]]; then
            rm -rf "$dir"
            log_info "  Deleted: $dir_name (Success: $total_success, Failed: $total_failed)"
        else
            log_warn "  Would delete: $dir_name (Success: $total_success, Failed: $total_failed)"
        fi
    fi
    
done <<< "$DATA_DIRS"

# Summary
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
log_info "Summary:"
log_info "  Total directories scanned: $TOTAL_DIRS"
log_info "  Directories to keep: $KEEP_DIRS"
log_info "  Directories to delete: $DELETE_DIRS"
log_info "  Total successful experiments: $TOTAL_SUCCESS"
log_info "  Total failed experiments: $TOTAL_FAILED"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    log_warn "DRY RUN MODE - No files were actually deleted"
    log_info "Run without --dry-run to perform the cleanup"
fi

if [[ $DELETE_DIRS -eq 0 ]]; then
    log_success "No directories to delete - all have successful experiments!"
    exit 0
elif [[ "$DRY_RUN" == "false" ]]; then
    log_success "Cleanup complete!"
    exit 0
else
    exit 0
fi

