#!/usr/bin/env bash
# =============================================================================
# cleanup_experiment_artifacts.sh - Clean up all experiment artifacts
#
# Removes:
# - Raw experiment data (results/<env>/)
# - Processed data (merged, stats, figures per experiment)
# - Final analysis artifacts (final-results/)
# - Progress tracking files (.progress_*.jsonl)
# - Generated scenarios (generated-scenarios/)
# - Research artifacts (research/output/, analysis/data/)
# - Packaging artifacts (packaging/output/)
# - Reproducibility artifacts (reproducibility/output/)
#
# Usage:
#   # Clean all environments
#   ./scripts/cleanup_experiment_artifacts.sh --all
#
#   # Clean specific environment(s)
#   ./scripts/cleanup_experiment_artifacts.sh --envs native,minikube
#   ./scripts/cleanup_experiment_artifacts.sh --envs gcp
#
#   # Clean everything (all environments + final results)
#   ./scripts/cleanup_experiment_artifacts.sh --all --include-final
#
# Options:
#   --envs <list>        Comma-separated list of environments (native,minikube,gcp)
#   --all                Clean all environments
#   --include-final      Also clean final-results/ (dissertation artifacts)
#   --include-scenarios  Also clean generated-scenarios/
#   --dry-run            Show what would be deleted without actually deleting
#   --quiet              Suppress confirmation prompts
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Default values
CLEAN_ENVS=""
CLEAN_ALL=false
INCLUDE_FINAL=false
INCLUDE_SCENARIOS=false
DRY_RUN=false
QUIET=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --envs)
            CLEAN_ENVS="$2"
            shift 2
            ;;
        --all)
            CLEAN_ALL=true
            shift
            ;;
        --include-final)
            INCLUDE_FINAL=true
            shift
            ;;
        --include-scenarios)
            INCLUDE_SCENARIOS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        -h|--help)
            cat <<EOF
Usage: $0 [OPTIONS]

Clean up experiment artifacts (raw data, processed data, dissertation artifacts).

Options:
  --envs <list>        Comma-separated list of environments (native,minikube,gcp)
  --all                Clean all environments
  --include-final      Also clean final-results/ (dissertation artifacts)
  --include-scenarios  Also clean generated-scenarios/
  --dry-run            Show what would be deleted without actually deleting
  --quiet              Suppress confirmation prompts
  -h, --help           Show this help message

Examples:
  # Clean all environments
  $0 --all

  # Clean specific environments
  $0 --envs native,minikube
  $0 --envs gcp

  # Clean everything including final results
  $0 --all --include-final

  # Dry run to see what would be deleted
  $0 --all --dry-run
EOF
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate arguments
if [[ "$CLEAN_ALL" == "false" ]] && [[ -z "$CLEAN_ENVS" ]]; then
    log_error "Must specify either --all or --envs <list>"
    exit 1
fi

# Determine which environments to clean
if [[ "$CLEAN_ALL" == "true" ]]; then
    ENVS_TO_CLEAN=("native" "minikube" "gcp")
else
    IFS=',' read -ra ENVS_TO_CLEAN <<< "$CLEAN_ENVS"
fi

# Function to delete directory or file
delete_path() {
    local path="$1"
    local description="${2:-$path}"
    
    if [[ ! -e "$path" ]]; then
        return 0  # Already doesn't exist
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would delete: $description"
        if [[ -d "$path" ]]; then
            local count=$(find "$path" -type f 2>/dev/null | wc -l)
            log_info "  (contains $count files)"
        fi
    else
        if [[ -d "$path" ]]; then
            rm -rf "$path"
            log_success "Deleted directory: $description"
        elif [[ -f "$path" ]]; then
            rm -f "$path"
            log_success "Deleted file: $description"
        fi
    fi
}

# Function to get size of directory
get_dir_size() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        du -sh "$dir" 2>/dev/null | cut -f1 || echo "0"
    else
        echo "0"
    fi
}

# Show what will be cleaned
log_info "=== Experiment Artifact Cleanup ==="
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    log_warn "DRY RUN MODE - No files will be deleted"
    echo ""
fi

log_info "Environments to clean: ${ENVS_TO_CLEAN[*]}"
if [[ "$INCLUDE_FINAL" == "true" ]]; then
    log_info "Will also clean: final-results/ (dissertation artifacts)"
fi
if [[ "$INCLUDE_SCENARIOS" == "true" ]]; then
    log_info "Will also clean: generated-scenarios/"
fi
echo ""

# Calculate total size before cleanup
TOTAL_SIZE=0
for env in "${ENVS_TO_CLEAN[@]}"; do
    if [[ -d "results/$env" ]]; then
        env_size=$(du -sb "results/$env" 2>/dev/null | cut -f1 || echo "0")
        TOTAL_SIZE=$((TOTAL_SIZE + env_size))
    fi
done
if [[ -d "final-results" ]] && [[ "$INCLUDE_FINAL" == "true" ]]; then
    final_size=$(du -sb "final-results" 2>/dev/null | cut -f1 || echo "0")
    TOTAL_SIZE=$((TOTAL_SIZE + final_size))
fi

if [[ $TOTAL_SIZE -gt 0 ]]; then
    TOTAL_SIZE_MB=$((TOTAL_SIZE / 1024 / 1024))
    log_info "Total size to clean: ~${TOTAL_SIZE_MB} MB"
    echo ""
fi

# Confirm unless --quiet
if [[ "$QUIET" == "false" ]] && [[ "$DRY_RUN" == "false" ]]; then
    echo -n "Continue with cleanup? [y/N] "
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log_info "Cleanup cancelled"
        exit 0
    fi
    echo ""
fi

# Clean environment-specific results
log_info "Cleaning environment-specific results..."
for env in "${ENVS_TO_CLEAN[@]}"; do
    env_dir="results/$env"
    if [[ -d "$env_dir" ]]; then
        env_size=$(get_dir_size "$env_dir")
        log_info "Cleaning $env environment results (${env_size})..."
        delete_path "$env_dir" "results/$env/"
        
        # Count experiments cleaned
        if [[ "$DRY_RUN" == "false" ]] && [[ -d "$env_dir" ]]; then
            exp_count=$(find "$env_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
            log_info "  Cleaned $exp_count experiment(s) from $env"
        fi
    else
        log_info "No results found for $env (directory doesn't exist)"
    fi
done
echo ""

# Clean progress tracking files
log_info "Cleaning progress tracking files..."
for env in "${ENVS_TO_CLEAN[@]}"; do
    progress_file="final-results/.progress_${env}.jsonl"
    delete_path "$progress_file" "Progress file for $env"
done
# Also clean any other progress files
for progress_file in final-results/.progress_*.jsonl .progress_*.jsonl; do
    if [[ -f "$progress_file" ]]; then
        delete_path "$progress_file" "Progress file: $(basename "$progress_file")"
    fi
done
echo ""

# Clean final results (dissertation artifacts)
if [[ "$INCLUDE_FINAL" == "true" ]]; then
    log_info "Cleaning final-results/ (dissertation artifacts)..."
    if [[ -d "final-results" ]]; then
        final_size=$(get_dir_size "final-results")
        log_info "Cleaning final-results/ (${final_size})..."
        delete_path "final-results" "final-results/"
    else
        log_info "No final-results/ directory found"
    fi
    echo ""
fi

# Clean generated scenarios
if [[ "$INCLUDE_SCENARIOS" == "true" ]]; then
    log_info "Cleaning generated-scenarios/..."
    if [[ -d "generated-scenarios" ]]; then
        scenarios_size=$(get_dir_size "generated-scenarios")
        log_info "Cleaning generated-scenarios/ (${scenarios_size})..."
        delete_path "generated-scenarios" "generated-scenarios/"
    else
        log_info "No generated-scenarios/ directory found"
    fi
    echo ""
fi

# Clean additional artifact directories
log_info "Cleaning additional artifact directories..."

# Research artifacts
if [[ -d "research/output" ]]; then
    delete_path "research/output" "research/output/"
fi

# Analysis data
if [[ -d "analysis/data" ]]; then
    delete_path "analysis/data" "analysis/data/"
fi

# Packaging artifacts
if [[ -d "packaging/output" ]]; then
    delete_path "packaging/output" "packaging/output/"
fi

# Reproducibility artifacts
if [[ -d "reproducibility/output" ]]; then
    delete_path "reproducibility/output" "reproducibility/output/"
fi

# Local results (if exists)
if [[ -d "local_results" ]]; then
    delete_path "local_results" "local_results/"
fi

echo ""

# Summary
log_success "=== Cleanup Complete ==="
if [[ "$DRY_RUN" == "true" ]]; then
    log_info "This was a dry run - no files were actually deleted"
    log_info "Run without --dry-run to perform the actual cleanup"
else
    log_success "All specified artifacts have been cleaned"
fi
