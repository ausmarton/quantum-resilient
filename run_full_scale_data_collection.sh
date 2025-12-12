#!/usr/bin/env bash
# =============================================================================
# run_full_scale_data_collection.sh - Collect raw data from full-scale runs
#
# Runs full-scale benchmarks separately for each environment to avoid resource
# throttling. Captures all raw data needed for dissertation analysis.
#
# This script:
# 1. Runs experiments WITHOUT analysis (--skip-analysis)
# 2. Captures all raw JSONL data
# 3. Generates individual experiment statistics
# 4. Creates a data manifest for later analysis
# 5. Allows running each environment separately
#
# Usage:
#   # Run all environments sequentially
#   ./run_full_scale_data_collection.sh --all --project <gcp-project> --bucket <gcs-bucket>
#
#   # Run individual environment
#   ./run_full_scale_data_collection.sh --env native
#   ./run_full_scale_data_collection.sh --env minikube
#   ./run_full_scale_data_collection.sh --env gcp --project <project> --bucket <bucket>
#
# Output:
#   - Individual experiment data: results/<env>/<scenario-id>/
#   - Data manifest: data-collection-<timestamp>/manifest.json
#   - Summary: data-collection-<timestamp>/summary.txt
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
MATRIX="$SCRIPT_DIR/orchestration/experiment_matrix.yaml"
ENVS=""
RUN_ALL=false
PROJECT=""
BUCKET=""
REGION="us-central1"
PARALLEL_JOBS=1
SKIP_ANALYSIS=true  # Always skip analysis - we'll do it later
SKIP_NATIVE=false
SKIP_MINIKUBE=false
SKIP_GCP=false
CONTINUE_ON_ERROR=true
MAX_RETRIES=2
CHECK_SYSTEM_LOAD=true  # Check system load before native/minikube runs

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
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

log_step() {
    echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}\n"
}

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Collect raw data from full-scale benchmark runs (no analysis).

OPTIONS:
    --all                    Run all environments sequentially (native, minikube, gcp)
    --env ENV                Run single environment: native, minikube, or gcp
    --matrix PATH            Experiment matrix YAML (default: orchestration/experiment_matrix.yaml)
    --project ID             GCP project ID (required for gcp env)
    --bucket NAME            GCS bucket name (required for gcp env)
    --region REGION          GCP region (default: us-central1)
    --parallel N             Parallel jobs per environment (default: 1)
    --skip-native            Skip native experiments
    --skip-minikube          Skip Minikube experiments
    --skip-gcp               Skip GCP experiments
    --continue-on-error       Continue if individual experiments fail (default: true)
    --max-retries N           Max retries per failed experiment (default: 2)
    -h, --help               Show this help message

EXAMPLES:
    # Run all environments sequentially
    $0 --all --project my-gcp-project --bucket pqc-bench-results

    # Run only native (local machine)
    $0 --env native

    # Run only minikube (local machine)
    $0 --env minikube

    # Run only GCP (cloud)
    $0 --env gcp --project my-project --bucket my-bucket

NOTES:
    - This script skips analysis phase (--skip-analysis) to focus on data collection
    - All raw data is stored in results/<env>/<scenario-id>/
    - Analysis can be run later using: ./run_all_experiments.sh --skip-generation --skip-native --skip-minikube --skip-gcp
    - Each environment can be run separately to avoid resource throttling
EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            shift
            ;;
        --env)
            ENVS="$2"
            shift 2
            ;;
        --matrix)
            MATRIX="$2"
            shift 2
            ;;
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --bucket)
            BUCKET="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --skip-native)
            SKIP_NATIVE=true
            shift
            ;;
        --skip-minikube)
            SKIP_MINIKUBE=true
            shift
            ;;
        --skip-gcp)
            SKIP_GCP=true
            shift
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=true
            shift
            ;;
        --max-retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        --no-check-load)
            CHECK_SYSTEM_LOAD=false
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

# Validate
if [[ "$RUN_ALL" == "true" ]]; then
    ENVS="native,minikube,gcp"
elif [[ -z "$ENVS" ]]; then
    log_error "Must specify --all or --env"
    usage
fi

if [[ "$ENVS" == *"gcp"* ]]; then
    if [[ -z "$PROJECT" ]]; then
        log_error "GCP environment requested but --project not provided"
        exit 1
    fi
    if [[ -z "$BUCKET" ]]; then
        log_error "GCP environment requested but --bucket not provided"
        exit 1
    fi
fi

if [[ ! -f "$MATRIX" ]]; then
    log_error "Matrix file not found: $MATRIX"
    exit 1
fi

# Create data collection directory with timestamp
COLLECTION_TIMESTAMP=$(date +%Y%m%d-%H%M%S)
COLLECTION_DIR="$SCRIPT_DIR/data-collection-${COLLECTION_TIMESTAMP}"
mkdir -p "$COLLECTION_DIR"

log_step "Full-Scale Data Collection Run"
log_info "Collection directory: $COLLECTION_DIR"
log_info "Matrix: $MATRIX"
log_info "Environments: $ENVS"
log_info "Mode: DATA COLLECTION ONLY (analysis skipped)"
log_info "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# Parse environments
IFS=',' read -ra ENV_ARRAY <<< "$ENVS"

# Run experiments for each environment
for env in "${ENV_ARRAY[@]}"; do
    # Skip if flagged
    case $env in
        native)
            [[ "$SKIP_NATIVE" == "true" ]] && { log_warn "Skipping native"; continue; }
            ;;
        minikube)
            [[ "$SKIP_MINIKUBE" == "true" ]] && { log_warn "Skipping minikube"; continue; }
            ;;
        gcp)
            [[ "$SKIP_GCP" == "true" ]] && { log_warn "Skipping gcp"; continue; }
            ;;
    esac
    
    log_step "Environment: ${env^^} - Data Collection"
    
    # Check system load for native/minikube (optional)
    if [[ "$CHECK_SYSTEM_LOAD" == "true" ]] && ([[ "$env" == "native" ]] || [[ "$env" == "minikube" ]]); then
        log_info "Checking system load..."
        if ! "$SCRIPT_DIR/scripts/check_system_load.sh" --warn-threshold 1.0 --fail-threshold 2.0; then
            log_warn "System load check failed, but continuing anyway"
            log_info "Use --no-check-load to skip this check"
            echo ""
        fi
    fi
    
    # Build command
    CMD=(
        "$SCRIPT_DIR/run_all_experiments.sh"
        --matrix "$MATRIX"
        --envs "$env"
        --parallel "$PARALLEL_JOBS"
        --skip-analysis  # Skip analysis - we'll do it later
        --continue-on-error
        --max-retries "$MAX_RETRIES"
    )
    
    # For minikube and GCP, include scaling replicas (1,2,4,8) for scaling experiments
    # Native doesn't support replicas > 1, so only pass for containerized environments
    if [[ "$env" == "minikube" ]] || [[ "$env" == "gcp" ]]; then
        CMD+=(--replicas "1,2,4,8")
    fi
    
    if [[ "$env" == "gcp" ]]; then
        CMD+=(--project "$PROJECT" --bucket "$BUCKET" --region "$REGION")
    fi
    
    # Run and capture output
    LOG_FILE="$COLLECTION_DIR/${env}_run.log"
    log_info "Running experiments for $env..."
    log_info "Log file: $LOG_FILE"
    
    if "${CMD[@]}" 2>&1 | tee "$LOG_FILE"; then
        log_success "$env data collection completed"
        
        # Validate data collection for this environment
        log_step "Validating $env data collection"
        VALIDATION_LOG="$COLLECTION_DIR/${env}_validation.log"
        
        # Use check_progress.sh for validation (it supports --env and --matrix)
        log_info "Running progress check for $env..."
        if "$SCRIPT_DIR/scripts/check_progress.sh" --env "$env" --matrix "$MATRIX" 2>&1 | tee "$VALIDATION_LOG"; then
            log_success "$env data validation completed"
        else
            log_warn "$env data validation had warnings (check $VALIDATION_LOG)"
        fi
    else
        log_error "$env data collection failed (check $LOG_FILE)"
        if [[ "$CONTINUE_ON_ERROR" != "true" ]]; then
            exit 1
        fi
    fi
    
    echo ""
done

# Generate data collection manifest
log_step "Generating Data Collection Manifest"

MANIFEST_FILE="$COLLECTION_DIR/manifest.json"
SUMMARY_FILE="$COLLECTION_DIR/summary.txt"

python3 <<EOF
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

script_dir = Path("$SCRIPT_DIR")
collection_dir = Path("$COLLECTION_DIR")
manifest_file = Path("$MANIFEST_FILE")
summary_file = Path("$SUMMARY_FILE")
envs = "$ENVS".split(",")

manifest = {
    "collection_timestamp": datetime.now(timezone.utc).isoformat(),
    "collection_dir": str(collection_dir),
    "matrix_file": "$MATRIX",
    "environments": envs,
    "mode": "data_collection_only",
    "analysis_skipped": True,
    "experiments": {}
}

# Count experiments per environment
for env in envs:
    env_results_dir = script_dir / "results" / env
    if not env_results_dir.exists():
        continue
    
    experiments = []
    for exp_dir in sorted(env_results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        # Check for key data files
        merged_file = exp_dir / "merged" / "merged.jsonl"
        stats_file = exp_dir / "stats" / "summary.json"
        raw_file = exp_dir / "raw" / "run.jsonl"
        
        has_data = merged_file.exists() or stats_file.exists() or raw_file.exists()
        
        if has_data:
            exp_info = {
                "scenario_id": exp_dir.name,
                "path": str(exp_dir),
                "has_merged_jsonl": merged_file.exists(),
                "has_stats": stats_file.exists(),
                "has_raw_jsonl": raw_file.exists(),
            }
            
            # Get file sizes
            if merged_file.exists():
                exp_info["merged_jsonl_size_bytes"] = merged_file.stat().st_size
            if stats_file.exists():
                exp_info["stats_file_size_bytes"] = stats_file.stat().st_size
            if raw_file.exists():
                exp_info["raw_jsonl_size_bytes"] = raw_file.exists() and raw_file.stat().st_size
            
            experiments.append(exp_info)
    
    manifest["experiments"][env] = {
        "count": len(experiments),
        "experiments": experiments
    }

# Write manifest
with open(manifest_file, 'w') as f:
    json.dump(manifest, f, indent=2)

# Generate summary
total_experiments = sum(m["count"] for m in manifest["experiments"].values())

summary_lines = [
    "=" * 70,
    "Full-Scale Data Collection Summary",
    "=" * 70,
    "",
    f"Collection timestamp: {manifest['collection_timestamp']}",
    f"Collection directory: {collection_dir}",
    f"Matrix file: {manifest['matrix_file']}",
    f"Environments: {', '.join(envs)}",
    f"Mode: Data collection only (analysis skipped)",
    "",
    "Experiment Counts:",
    "-" * 70,
]

for env in envs:
    if env in manifest["experiments"]:
        count = manifest["experiments"][env]["count"]
        summary_lines.append(f"  {env:12s}: {count:4d} experiments")
    else:
        summary_lines.append(f"  {env:12s}:    0 experiments")

summary_lines.extend([
    "",
    "-" * 70,
    f"Total experiments: {total_experiments}",
    "",
    "Data Locations:",
    "-" * 70,
])

for env in envs:
    if env in manifest["experiments"]:
        summary_lines.append(f"  {env}: {script_dir}/results/{env}/")
        for exp in manifest["experiments"][env]["experiments"][:5]:  # Show first 5
            summary_lines.append(f"    - {exp['scenario_id']}")
        if manifest["experiments"][env]["count"] > 5:
            summary_lines.append(f"    ... and {manifest['experiments'][env]['count'] - 5} more")

summary_lines.extend([
    "",
    "Next Steps:",
    "-" * 70,
    "1. Verify data collection:",
    f"   ./scripts/verify_experiments.sh {script_dir}/results/",
    "",
    "2. Run analysis later (when all environments are complete):",
    "   ./run_all_experiments.sh",
    "     --skip-generation",
    "     --skip-native --skip-minikube --skip-gcp",
    f"     --matrix {manifest['matrix_file']}",
    "",
    "   Or analyze specific environment:",
    "   ./run_all_experiments.sh",
    "     --skip-generation",
    "     --envs native",
    f"     --matrix {manifest['matrix_file']}",
    "",
    "=" * 70,
])

with open(summary_file, 'w') as f:
    f.write("\n".join(summary_lines))

print(f"Manifest written: {manifest_file}")
print(f"Summary written: {summary_file}")
print(f"\nTotal experiments collected: {total_experiments}")
EOF

log_success "Data collection manifest generated"

# Display summary
echo ""
cat "$SUMMARY_FILE"

log_step "Data Collection Complete"
log_info "All raw data is stored in: results/<env>/<scenario-id>/"
log_info "Collection manifest: $MANIFEST_FILE"
log_info "Summary: $SUMMARY_FILE"
echo ""

# Run validation
log_step "Validating Data Collection"
log_info "Checking that all required data is present..."

if "$SCRIPT_DIR/scripts/validate_data_collection.sh" \
    --matrix "$MATRIX" \
    --results-dir "$SCRIPT_DIR/results" \
    --envs "$ENVS"; then
    log_success "Validation passed - all required data is present!"
    echo ""
    log_info "✅ Ready for analysis!"
    log_info ""
    log_info "To run analysis:"
    echo "  ./run_all_experiments.sh --skip-generation --skip-native --skip-minikube --skip-gcp"
else
    log_warn "Validation found missing or incomplete data"
    log_info ""
    log_info "Options:"
    echo ""
    echo "  1. Complete analysis for incomplete experiments (have raw data, missing merged/stats):"
    echo "     ./scripts/complete_incomplete_experiments.sh --env $env"
    echo ""
    echo "  2. Re-run data collection (will skip completed ones, resume from where it left off):"
    echo "     ./run_full_scale_data_collection.sh --env $env"
    echo ""
    echo "  3. Run analysis anyway (will only analyze available data):"
    echo "     ./run_all_experiments.sh --skip-generation --skip-native --skip-minikube --skip-gcp"
    echo ""
    log_info "Check validation details above for counts of missing vs incomplete experiments."
fi
echo ""

exit 0

