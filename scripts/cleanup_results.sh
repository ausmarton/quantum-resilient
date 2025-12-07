#!/usr/bin/env bash
# =============================================================================
# cleanup_results.sh - Safely delete experiment results for re-running
#
# This script helps you delete collected data so you can re-run all experiments
# from scratch. It provides options to:
# - Delete specific environments (native, minikube, gcp)
# - Delete all results
# - Archive before deleting (recommended)
# - Delete analysis outputs (final-results/)
#
# Usage:
#   ./scripts/cleanup_results.sh [OPTIONS]
#
# Examples:
#   # Archive and delete all native results
#   ./scripts/cleanup_results.sh --env native --archive
#
#   # Delete all results (no archive - be careful!)
#   ./scripts/cleanup_results.sh --all --no-archive
#
#   # Delete only analysis outputs (keep raw data)
#   ./scripts/cleanup_results.sh --analysis-only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_BASE="$SCRIPT_DIR/results"
FINAL_RESULTS="$SCRIPT_DIR/final-results"
FINAL_RESULTS_SMOKE="$SCRIPT_DIR/final-results-smoke"

# Options
ENV=""
DELETE_ALL=false
ARCHIVE=true
ANALYSIS_ONLY=false
DRY_RUN=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Safely delete experiment results for re-running.

OPTIONS:
    --env ENV              Delete specific environment: native, minikube, or gcp
    --all                  Delete all environments (native, minikube, gcp)
    --analysis-only        Delete only analysis outputs (final-results/, final-results-smoke/)
    --archive              Archive before deleting (default: true)
    --no-archive           Don't archive before deleting (use with caution!)
    --dry-run              Show what would be deleted without actually deleting
    -h, --help             Show this help message

EXAMPLES:
    # Archive and delete native results
    $0 --env native --archive

    # Delete all results with archive (safe)
    $0 --all --archive

    # Delete all results without archive (dangerous!)
    $0 --all --no-archive

    # Delete only analysis outputs (keep raw data)
    $0 --analysis-only

    # See what would be deleted
    $0 --all --dry-run
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV="$2"
            shift 2
            ;;
        --all)
            DELETE_ALL=true
            shift
            ;;
        --analysis-only)
            ANALYSIS_ONLY=true
            shift
            ;;
        --archive)
            ARCHIVE=true
            shift
            ;;
        --no-archive)
            ARCHIVE=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate options
if [[ -z "$ENV" ]] && [[ "$DELETE_ALL" == "false" ]] && [[ "$ANALYSIS_ONLY" == "false" ]]; then
    echo -e "${RED}Error: Must specify --env, --all, or --analysis-only${NC}"
    usage
fi

if [[ "$DELETE_ALL" == "true" ]] && [[ -n "$ENV" ]]; then
    echo -e "${YELLOW}Warning: --all overrides --env${NC}"
fi

# Archive function
archive_data() {
    local source_dir="$1"
    local archive_name="$2"
    
    if [[ ! -d "$source_dir" ]] || [[ -z "$(ls -A "$source_dir" 2>/dev/null)" ]]; then
        echo -e "${YELLOW}  No data to archive in $source_dir${NC}"
        return
    fi
    
    local archive_dir="$SCRIPT_DIR/archive"
    mkdir -p "$archive_dir"
    
    local archive_path="$archive_dir/${archive_name}-$(date +%Y%m%d-%H%M%S)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "${CYAN}  [DRY RUN] Would archive: $source_dir -> $archive_path${NC}"
    else
        echo -e "${BLUE}  Archiving: $source_dir -> $archive_path${NC}"
        cp -r "$source_dir" "$archive_path"
        echo -e "${GREEN}  ✓ Archived to: $archive_path${NC}"
    fi
}

# Delete function
delete_data() {
    local target_dir="$1"
    local description="$2"
    
    if [[ ! -d "$target_dir" ]]; then
        echo -e "${YELLOW}  Directory doesn't exist: $target_dir${NC}"
        return
    fi
    
    local size=$(du -sh "$target_dir" 2>/dev/null | cut -f1)
    local count=$(find "$target_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "${CYAN}  [DRY RUN] Would delete: $target_dir${NC}"
        echo -e "${CYAN}    Size: $size, Experiments: $count${NC}"
    else
        echo -e "${RED}  Deleting: $target_dir${NC}"
        echo -e "${YELLOW}    Size: $size, Experiments: $count${NC}"
        rm -rf "$target_dir"
        mkdir -p "$target_dir"  # Recreate empty directory
        echo -e "${GREEN}  ✓ Deleted${NC}"
    fi
}

# Main execution
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${CYAN}  Cleanup Results - DRY RUN${NC}"
else
    echo -e "${CYAN}  Cleanup Results${NC}"
fi
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

if [[ "$ANALYSIS_ONLY" == "true" ]]; then
    echo -e "${MAGENTA}Deleting Analysis Outputs Only${NC}"
    echo ""
    
    if [[ "$ARCHIVE" == "true" ]]; then
        echo -e "${BLUE}Archiving before deletion...${NC}"
        [[ -d "$FINAL_RESULTS" ]] && archive_data "$FINAL_RESULTS" "final-results"
        [[ -d "$FINAL_RESULTS_SMOKE" ]] && archive_data "$FINAL_RESULTS_SMOKE" "final-results-smoke"
        echo ""
    fi
    
    echo -e "${BLUE}Deleting analysis outputs...${NC}"
    delete_data "$FINAL_RESULTS" "Full-scale analysis outputs"
    delete_data "$FINAL_RESULTS_SMOKE" "Smoke-test analysis outputs"
    
elif [[ "$DELETE_ALL" == "true" ]]; then
    echo -e "${MAGENTA}Deleting All Environments${NC}"
    echo ""
    
    if [[ "$ARCHIVE" == "true" ]]; then
        echo -e "${BLUE}Archiving before deletion...${NC}"
        [[ -d "$RESULTS_BASE/native" ]] && archive_data "$RESULTS_BASE/native" "results-native"
        [[ -d "$RESULTS_BASE/minikube" ]] && archive_data "$RESULTS_BASE/minikube" "results-minikube"
        [[ -d "$RESULTS_BASE/gcp" ]] && archive_data "$RESULTS_BASE/gcp" "results-gcp"
        [[ -d "$FINAL_RESULTS" ]] && archive_data "$FINAL_RESULTS" "final-results"
        [[ -d "$FINAL_RESULTS_SMOKE" ]] && archive_data "$FINAL_RESULTS_SMOKE" "final-results-smoke"
        echo ""
    fi
    
    echo -e "${BLUE}Deleting all results...${NC}"
    delete_data "$RESULTS_BASE/native" "Native results"
    delete_data "$RESULTS_BASE/minikube" "Minikube results"
    delete_data "$RESULTS_BASE/gcp" "GCP results"
    delete_data "$FINAL_RESULTS" "Full-scale analysis outputs"
    delete_data "$FINAL_RESULTS_SMOKE" "Smoke-test analysis outputs"
    
else
    # Specific environment
    if [[ "$ENV" != "native" ]] && [[ "$ENV" != "minikube" ]] && [[ "$ENV" != "gcp" ]]; then
        echo -e "${RED}Error: Invalid environment: $ENV${NC}"
        echo "Valid environments: native, minikube, gcp"
        exit 1
    fi
    
    echo -e "${MAGENTA}Deleting ${ENV^^} Environment${NC}"
    echo ""
    
    if [[ "$ARCHIVE" == "true" ]]; then
        echo -e "${BLUE}Archiving before deletion...${NC}"
        [[ -d "$RESULTS_BASE/$ENV" ]] && archive_data "$RESULTS_BASE/$ENV" "results-$ENV"
        echo ""
    fi
    
    echo -e "${BLUE}Deleting $ENV results...${NC}"
    delete_data "$RESULTS_BASE/$ENV" "$ENV results"
fi

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${CYAN}DRY RUN complete - no data was deleted${NC}"
    echo -e "${BLUE}Run without --dry-run to actually delete${NC}"
else
    echo -e "${GREEN}Cleanup complete!${NC}"
    if [[ "$ARCHIVE" == "true" ]]; then
        echo -e "${GREEN}Data has been archived to: $SCRIPT_DIR/archive/${NC}"
    fi
    echo ""
    echo -e "${BLUE}You can now re-run experiments:${NC}"
    echo "  ./run_full_scale_data_collection.sh --env native"
    echo "  ./run_full_scale_data_collection.sh --env minikube"
    echo "  ./run_full_scale_data_collection.sh --env gcp --project <project> --bucket <bucket>"
fi

echo ""

