#!/usr/bin/env bash
# =============================================================================
# run_all_experiments.sh - Master orchestration script
#
# Executes all benchmarking scenarios across all environments (native, Minikube,
# GCP), runs multiple repeats, collects data, and produces dissertation-ready
# final-results/ directory.
#
# Usage:
#   ./run_all_experiments.sh \
#     --project <gcp-project> \
#     --bucket <gcs-bucket> \
#     --matrix orchestration/experiment_matrix.yaml \
#     --envs native,minikube,gcp
#
# Requirements:
#   - Python 3.10+ with analysis dependencies
#   - For native: Rust toolchain
#   - For Minikube: Minikube + Podman
#   - For GCP: gcloud, Terraform, Podman
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATED_SCENARIOS_DIR="$SCRIPT_DIR/generated-scenarios"
RESULTS_BASE="$SCRIPT_DIR/results"

# Default values
MATRIX="$SCRIPT_DIR/orchestration/experiment_matrix.yaml"
ENVS="native"
PROJECT=""
BUCKET=""
REGION="us-central1"
PARALLEL_JOBS=1
REPLICAS="1"  # Comma-separated list: 1,2,4,8
SKIP_GENERATION=false
SKIP_NATIVE=false
SKIP_MINIKUBE=false
SKIP_GCP=false
SKIP_ANALYSIS=false
SKIP_SCALING=false
DRY_RUN=false
CONTINUE_ON_ERROR=true
MAX_RETRIES=2
SMOKE_TEST=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Tracking
TOTAL_SCENARIOS=0
COMPLETED_SCENARIOS=0
FAILED_SCENARIOS=0
MASTER_INDEX=()

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
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

log_phase() {
    echo -e "\n${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}  PHASE: $1${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Execute all benchmarking scenarios across environments.

OPTIONS:
    --matrix PATH           Experiment matrix YAML (default: orchestration/experiment_matrix.yaml)
    --envs LIST             Comma-separated environments: native,minikube,gcp (default: native)
    --project ID            GCP project ID (required for gcp env)
    --bucket NAME           GCS bucket name (required for gcp env)
    --region REGION         GCP region (default: us-central1)
    --parallel N            Parallel jobs per environment (default: 1)
    --replicas LIST         Comma-separated replica counts for scaling tests: 1,2,4,8
    --skip-generation       Skip scenario generation
    --skip-native           Skip native experiments
    --skip-minikube         Skip Minikube experiments
    --skip-gcp              Skip GCP experiments
    --skip-analysis         Skip final analysis
    --skip-scaling          Skip scaling analysis
    --dry-run               Show what would be executed
    --continue-on-error     Continue if individual experiments fail (default: true)
    --max-retries N         Max retries per failed experiment (default: 2)
    --smoke-test            Enable smoke-test mode (reduced scale, minimal cost)
    -h, --help              Show this help message

EXAMPLE:
    # Standard experiments
    $0 --envs native,minikube,gcp \\
       --project my-gcp-project \\
       --bucket pqc-bench-results \\
       --matrix orchestration/experiment_matrix.yaml

    # Scaling experiments with multiple replica counts
    $0 --envs minikube,gcp \\
       --replicas 1,2,4,8 \\
       --project my-gcp-project \\
       --bucket pqc-bench-results
EOF
    exit 1
}

# Progress tracking with time estimates
START_TIME=$(date +%s)
LAST_PROGRESS_UPDATE=$(date +%s)

update_progress() {
    local current=$1
    local total=$2
    local env=$3
    local scenario=$4
    
    local pct=$((current * 100 / total))
    local now=$(date +%s)
    local elapsed=$((now - START_TIME))
    
    # Calculate ETA
    local ETA_STR="calculating..."
    if [[ $current -gt 0 ]] && [[ $elapsed -gt 0 ]]; then
        local rate=$((current * 100 / elapsed))  # scenarios per 100 seconds
        if [[ $rate -gt 0 ]]; then
            local remaining=$((total - current))
            local eta_seconds=$((remaining * 100 / rate))
            local eta_minutes=$((eta_seconds / 60))
            local eta_hours=$((eta_minutes / 60))
            
            if [[ $eta_hours -gt 0 ]]; then
                local eta_remaining_minutes=$((eta_minutes % 60))
                ETA_STR="${eta_hours}h ${eta_remaining_minutes}m"
            elif [[ $eta_minutes -gt 0 ]]; then
                ETA_STR="${eta_minutes}m"
            else
                ETA_STR="${eta_seconds}s"
            fi
        fi
    fi
    
    # Format elapsed time
    local elapsed_hours=$((elapsed / 3600))
    local elapsed_minutes=$(((elapsed % 3600) / 60))
    local elapsed_seconds=$((elapsed % 60))
    
    local ELAPSED_STR
    if [[ $elapsed_hours -gt 0 ]]; then
        ELAPSED_STR="${elapsed_hours}h ${elapsed_minutes}m"
    elif [[ $elapsed_minutes -gt 0 ]]; then
        ELAPSED_STR="${elapsed_minutes}m ${elapsed_seconds}s"
    else
        ELAPSED_STR="${elapsed_seconds}s"
    fi
    
    # Update every 5 seconds or on every scenario
    if [[ $((now - LAST_PROGRESS_UPDATE)) -ge 5 ]] || [[ $current -eq $total ]]; then
        printf "\r${CYAN}[%3d%%]${NC} [%s] %s | Elapsed: %s | ETA: %s | %d/%d" \
            "$pct" "$env" "$scenario" "$ELAPSED_STR" "$ETA_STR" "$current" "$total"
        LAST_PROGRESS_UPDATE=$now
        
        if [[ $current -eq $total ]]; then
            echo ""  # New line when complete
        fi
    fi
}

# Signal handler for graceful stop
cleanup_on_exit() {
    echo ""
    log_warn "Received interrupt signal. Saving progress..."
    log_info "Completed experiments are saved and will be skipped on resume."
    log_info "To resume, simply re-run the same command:"
    echo "  $0 ${ORIGINAL_ARGS[*]}"
    exit 130  # Exit code 130 = terminated by SIGINT
}

trap cleanup_on_exit INT TERM
ORIGINAL_ARGS=("$@")

# Run single experiment with retries
run_experiment() {
    local env=$1
    local scenario_path=$2
    local scenario_id=$3
    local output_dir=$4
    local replicas=${5:-1}
    local retries=0
    
    while [[ $retries -le $MAX_RETRIES ]]; do
        local exit_code=0
        
        case $env in
            native)
                # Native doesn't support replicas
                "$SCRIPT_DIR/run_local.sh" \
                    --scenario "$scenario_path" \
                    --out "$output_dir" \
                    --duration 30 \
                    $([ "$SMOKE_TEST" == "true" ] && echo "--smoke-test" || echo "") 2>&1 || exit_code=$?
                ;;
            minikube)
                "$SCRIPT_DIR/run_minikube.sh" \
                    --scenario "$scenario_path" \
                    --out "$output_dir" \
                    --replicas "$replicas" \
                    --exp-id "$scenario_id" \
                    $([ "$SMOKE_TEST" == "true" ] && echo "--smoke-test" || echo "") 2>&1 || exit_code=$?
                ;;
            gcp)
                # Unified execution: Always use Kubernetes Job submission
                # PARALLEL_JOBS=1 = sequential (submit one, wait, submit next)
                # PARALLEL_JOBS>1 = parallel (submit multiple, wait for all)
                # This eliminates redundant code paths
                
                if [[ "${GCP_USE_PERSISTENT_CLUSTER:-false}" == "true" ]]; then
                    # Persistent cluster mode: use direct Kubernetes Job submission
                    if [[ -z "${GCP_IMAGE_NAME:-}" ]]; then
                        # Get image name if not set
                        IMAGE_REPO="${REGION}-docker.pkg.dev/${PROJECT}/pqc"
                        GCP_IMAGE_NAME="${IMAGE_REPO}/pqc-bench:latest"
                    fi
                    
                    # Determine namespace
                    if [[ "$SMOKE_TEST" == "true" ]]; then
                        GCP_NAMESPACE="pqc-smoke-test"
                    else
                        GCP_NAMESPACE="default"
                    fi
                    
                    # Submit job (non-blocking for parallel, blocking for sequential)
                    JOB_NAME=$("$SCRIPT_DIR/scripts/submit_gcp_job_parallel.sh" \
                        --scenario "$scenario_path" \
                        --exp-id "$scenario_id" \
                        --project "$PROJECT" \
                        --bucket "$BUCKET" \
                        --region "$REGION" \
                        --image "$GCP_IMAGE_NAME" \
                        --namespace "$GCP_NAMESPACE" \
                        --replicas "$replicas" \
                        $([ "$SMOKE_TEST" == "true" ] && echo "--smoke-test" || echo "") 2>&1) || exit_code=$?
                    
                    if [[ $exit_code -eq 0 ]]; then
                        # Store job for tracking (used for parallel mode)
                        if [[ $PARALLEL_JOBS -gt 1 ]]; then
                            echo "$JOB_NAME|$scenario_id|$output_dir" >> "${JOB_TRACKING_FILE:-/tmp/gcp_jobs_${env}.txt}"
                            # For parallel mode, wait for all jobs at the end
                            exit_code=0  # Job submission succeeded
                        else
                            # Sequential mode: wait for this job to complete, then download
                            log_info "  Waiting for job $JOB_NAME to complete..."
                            while true; do
                                STATUS=$(kubectl get job "$JOB_NAME" -n "$GCP_NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || echo "")
                                FAILED_STATUS=$(kubectl get job "$JOB_NAME" -n "$GCP_NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || echo "")
                                
                                if [[ "$STATUS" == "True" ]]; then
                                    # Job completed, download results
                                    if "$SCRIPT_DIR/fetch_and_analyse_from_gcs.sh" \
                                        --exp-id "$scenario_id" \
                                        --bucket "$BUCKET" \
                                        --out "$output_dir" 2>&1; then
                                        exit_code=0
                                    else
                                        exit_code=1
                                    fi
                                    break
                                elif [[ "$FAILED_STATUS" == "True" ]]; then
                                    log_error "  Job $JOB_NAME failed"
                                    exit_code=1
                                    break
                                fi
                                sleep 5
                            done
                        fi
                    fi
                else
                    # Ephemeral mode: use deploy_gcp.sh (for single experiments or when cluster doesn't exist)
                    GCP_ARGS=(
                        --scenario "$scenario_path"
                        --exp-id "$scenario_id"
                        --project "$PROJECT"
                        --bucket "$BUCKET"
                        --region "$REGION"
                        --replicas "$replicas"
                        --ephemeral
                    )
                    [ "$SMOKE_TEST" == "true" ] && GCP_ARGS+=(--smoke-test)
                    
                    "$SCRIPT_DIR/deploy_gcp.sh" "${GCP_ARGS[@]}" 2>&1 || exit_code=$?
                    
                    if [[ $exit_code -eq 0 ]]; then
                        "$SCRIPT_DIR/fetch_and_analyse_from_gcs.sh" \
                            --exp-id "$scenario_id" \
                            --bucket "$BUCKET" \
                            --out "$output_dir" 2>&1 || exit_code=$?
                    fi
                fi
                ;;
        esac
        
        if [[ $exit_code -eq 0 ]]; then
            return 0
        fi
        
        retries=$((retries + 1))
        if [[ $retries -le $MAX_RETRIES ]]; then
            log_warn "Experiment failed, retrying ($retries/$MAX_RETRIES)..."
            sleep 5
        fi
    done
    
    return 1
}

# Add entry to master index
add_to_index() {
    local scenario_id=$1
    local env=$2
    local algorithm=$3
    local payload=$4
    local rate=$5
    local output_dir=$6
    local status=$7
    local replicas=${8:-1}
    
    MASTER_INDEX+=("{\"scenario_id\":\"$scenario_id\",\"environment\":\"$env\",\"algorithm\":\"$algorithm\",\"payload_size\":$payload,\"rate\":$rate,\"replicas\":$replicas,\"output_dir\":\"$output_dir\",\"status\":\"$status\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}")
}

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --matrix)
            MATRIX="$2"
            shift 2
            ;;
        --envs)
            ENVS="$2"
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
        --replicas)
            REPLICAS="$2"
            shift 2
            ;;
        --skip-generation)
            SKIP_GENERATION=true
            shift
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
        --skip-analysis)
            SKIP_ANALYSIS=true
            shift
            ;;
        --skip-scaling)
            SKIP_SCALING=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
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
        --smoke-test)
            SMOKE_TEST=true
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

# Validate required arguments
if [[ ! -f "$MATRIX" ]]; then
    log_error "Matrix file not found: $MATRIX"
    exit 1
fi

# Check if GCP env is requested but missing credentials
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

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
START_TIME=$(date +%s)
START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo -e "${MAGENTA}"
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║      PQC Benchmark - Complete Experiment Orchestration Suite          ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

log_info "Matrix: $MATRIX"
log_info "Environments: $ENVS"
log_info "Replicas: $REPLICAS"
[[ "$SMOKE_TEST" == "true" ]] && log_info "Mode: SMOKE-TEST (reduced scale, minimal cost)"
log_info "Started: $START_ISO"

# Parse replicas into array
IFS=',' read -ra REPLICA_ARRAY <<< "$REPLICAS"

if [[ "$DRY_RUN" == "true" ]]; then
    log_warn "DRY RUN MODE - No experiments will be executed"
fi

# =============================================================================
# Phase 1: Generate Scenarios
# =============================================================================
log_phase "1. Scenario Generation"

if [[ "$SKIP_GENERATION" == "true" ]]; then
    log_warn "Skipping scenario generation (--skip-generation)"
else
    log_info "Generating scenarios from matrix..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        python3 "$SCRIPT_DIR/orchestration/generate_scenarios.py" \
            --matrix "$MATRIX" \
            --output "$GENERATED_SCENARIOS_DIR" \
            --dry-run \
            $([ "$SMOKE_TEST" == "true" ] && echo "--smoke-test" || echo "")
    else
        python3 "$SCRIPT_DIR/orchestration/generate_scenarios.py" \
            --matrix "$MATRIX" \
            --output "$GENERATED_SCENARIOS_DIR" \
            $([ "$SMOKE_TEST" == "true" ] && echo "--smoke-test" || echo "")
    fi
    
    log_success "Scenarios generated"
fi

# Count scenarios
if [[ -f "$GENERATED_SCENARIOS_DIR/manifest.json" ]]; then
    TOTAL_SCENARIOS=$(python3 -c "import json; print(json.load(open('$GENERATED_SCENARIOS_DIR/manifest.json'))['total_scenarios'])")
    log_info "Total scenarios: $TOTAL_SCENARIOS"
fi

# Set final results directory based on smoke-test mode
if [[ "$SMOKE_TEST" == "true" ]]; then
    FINAL_RESULTS_DIR="$SCRIPT_DIR/final-results-smoke"
    REPLICAS="1"  # Force replicas to 1 in smoke-test mode
else
    FINAL_RESULTS_DIR="$SCRIPT_DIR/final-results"
fi

# =============================================================================
# Phase 2: Create Output Directories
# =============================================================================
log_phase "2. Initialize Output Directories"

mkdir -p "$FINAL_RESULTS_DIR/figures"
mkdir -p "$FINAL_RESULTS_DIR/stats"
mkdir -p "$FINAL_RESULTS_DIR/tables"
mkdir -p "$RESULTS_BASE/native"
mkdir -p "$RESULTS_BASE/minikube"
mkdir -p "$RESULTS_BASE/gcp"

log_success "Output directories created"

# =============================================================================
# Phase 3: Execute Experiments
# =============================================================================
log_phase "3. Execute Experiments"

# Parse environments
IFS=',' read -ra ENV_ARRAY <<< "$ENVS"

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
    
    log_step "Environment: ${env^^}"
    
    # Read scenarios from manifest
    if [[ ! -f "$GENERATED_SCENARIOS_DIR/manifest.json" ]]; then
        log_error "Scenario manifest not found. Run without --skip-generation first."
        exit 1
    fi
    
    # Initialize progress tracking for this environment
    START_TIME=$(date +%s)
    LAST_PROGRESS_UPDATE=$(date +%s)
    
    # Count total scenarios for this environment
    ENV_TOTAL_SCENARIOS=$(python3 -c "
import json
with open('$GENERATED_SCENARIOS_DIR/manifest.json') as f:
    manifest = json.load(f)
count = sum(1 for s in manifest['scenarios'])
print(count)
")
    
    # Calculate actual number of experiments that will run (accounting for replicas)
    # This is different from scenario count because:
    # - Scaling experiments run with multiple replicas (1, 2, 4, 8)
    # - Non-scaling experiments run with replica 1 only
    # - Native environment only runs with replica 1
    ENV_TOTAL_EXPERIMENTS=$(python3 -c "
import json
import sys

with open('$GENERATED_SCENARIOS_DIR/manifest.json') as f:
    manifest = json.load(f)

env = '$env'
replicas = [int(r) for r in '$REPLICAS'.split(',')]
scaling_replicas = [r for r in replicas if r > 1]  # [2, 4, 8] if REPLICAS='1,2,4,8'

total_experiments = 0
for s in manifest['scenarios']:
    is_scaling = s.get('scaling_experiment', False)
    
    if env == 'native':
        # Native only runs with replica 1
        total_experiments += 1
    elif is_scaling:
        # Scaling experiments run with all replicas (1, 2, 4, 8)
        total_experiments += len(replicas)
    else:
        # Non-scaling experiments run with replica 1 only
        total_experiments += 1

print(total_experiments)
")
    
    log_info "Total scenarios for ${env}: $ENV_TOTAL_SCENARIOS"
    log_info "Total experiments to run: $ENV_TOTAL_EXPERIMENTS (accounting for replicas)"
    log_info "Progress will be shown every 5 seconds"
    echo ""
    
    # For GCP batch runs (multiple experiments), use persistent cluster mode for efficiency
    # This creates the cluster once, reuses it for all experiments, then destroys it
    # Export variables so they're available in run_experiment function
    export GCP_USE_PERSISTENT_CLUSTER=false
    export GCP_CLUSTER_EXISTS=false
    
    if [[ "$env" == "gcp" ]] && [[ $ENV_TOTAL_EXPERIMENTS -gt 1 ]]; then
        log_info "Detected batch run with $ENV_TOTAL_EXPERIMENTS experiments"
        log_info "Using persistent cluster mode for efficiency (cluster created once, reused, destroyed at end)"
        
        # Determine cluster name based on smoke-test mode
        if [[ "$SMOKE_TEST" == "true" ]]; then
            CLUSTER_NAME="pqc-smoke-test"
        else
            CLUSTER_NAME="pqc-bench-gke"
        fi
        
        # Check if cluster already exists
        if gcloud container clusters describe "$CLUSTER_NAME" \
            --region "$REGION" \
            --project "$PROJECT" &>/dev/null 2>&1; then
            log_warn "Cluster $CLUSTER_NAME already exists"
            log_info "Will reuse existing cluster (use --skip-gcp and destroy manually if needed)"
            export GCP_USE_PERSISTENT_CLUSTER=true
            export GCP_CLUSTER_EXISTS=true
        else
            # Calculate node count based on parallel jobs for proper isolation
            # Each job needs ~1 CPU (800m request), n2-standard-2 has 2 vCPUs
            # To avoid noisy neighbor: max 1 job per node (or 2 if we allow some sharing)
            # For safety and isolation: use 1 job per node
            # Add 1 extra node for system overhead
            if [[ $PARALLEL_JOBS -gt 1 ]]; then
                CALCULATED_NODE_COUNT=$((PARALLEL_JOBS + 1))
                log_info "Calculated node count: $CALCULATED_NODE_COUNT (for $PARALLEL_JOBS parallel jobs + 1 overhead)"
            else
                CALCULATED_NODE_COUNT="${NODE_COUNT:-1}"
                log_info "Using node count: $CALCULATED_NODE_COUNT (sequential mode)"
            fi
            
            log_info "Creating persistent cluster for batch run..."
            if "$SCRIPT_DIR/deploy_gcp.sh" \
                --create-cluster \
                --project "$PROJECT" \
                --bucket "$BUCKET" \
                --region "$REGION" \
                --machine-type "${MACHINE_TYPE:-n2-standard-2}" \
                --node-count "$CALCULATED_NODE_COUNT" \
                $([ "$SMOKE_TEST" == "true" ] && echo "--smoke-test" || echo ""); then
                log_success "Cluster created successfully"
                export GCP_USE_PERSISTENT_CLUSTER=true
                export GCP_CLUSTER_EXISTS=false  # We just created it
            else
                log_error "Failed to create cluster, falling back to ephemeral mode"
                export GCP_USE_PERSISTENT_CLUSTER=false
                export GCP_CLUSTER_EXISTS=false
            fi
        fi
        
        # Build image once for the batch (if cluster exists or was just created)
        if [[ "${GCP_USE_PERSISTENT_CLUSTER:-false}" == "true" ]]; then
            log_info "Building container image once for batch run..."
            # Get first scenario for image build
            FIRST_SCENARIO=$(python3 -c "
import json
with open('$GENERATED_SCENARIOS_DIR/manifest.json') as f:
    manifest = json.load(f)
if manifest['scenarios']:
    print(manifest['scenarios'][0]['path'])
" 2>/dev/null || echo "")
            
            if [[ -n "$FIRST_SCENARIO" ]] && [[ -f "$FIRST_SCENARIO" ]]; then
                # Get image name from deploy script
                IMAGE_REPO="${REGION}-docker.pkg.dev/${PROJECT}/pqc"
                export GCP_IMAGE_NAME="${IMAGE_REPO}/pqc-bench:latest"
                
                if "$SCRIPT_DIR/deploy_gcp.sh" \
                    --scenario "$FIRST_SCENARIO" \
                    --exp-id "batch-setup-$(date +%s)" \
                    --project "$PROJECT" \
                    --bucket "$BUCKET" \
                    --region "$REGION" \
                    --skip-terraform \
                    --skip-aggregation \
                    $([ "$SMOKE_TEST" == "true" ] && echo "--smoke-test" || echo "") 2>&1 | grep -E "(Building|Pushing|Image|ERROR|WARN)" || true; then
                    log_success "Image built and ready for batch: $GCP_IMAGE_NAME"
                else
                    log_warn "Image build had issues, but continuing (will build per-experiment if needed)"
                    # Try to get image name from terraform output
                    export GCP_IMAGE_NAME="${IMAGE_REPO}/pqc-bench:latest"
                fi
            fi
        fi
    else
        # Single experiment or non-GCP: use default behavior (ephemeral for GCP)
        export GCP_USE_PERSISTENT_CLUSTER=false
        export GCP_CLUSTER_EXISTS=false
    fi
    
    # For GCP execution with persistent cluster, create temp file to track jobs
    # Used for both sequential (PARALLEL_JOBS=1) and parallel (PARALLEL_JOBS>1) modes
    if [[ "$env" == "gcp" ]] && [[ "${GCP_USE_PERSISTENT_CLUSTER:-false}" == "true" ]]; then
        TEMP_DIR=$(mktemp -d)
        export TEMP_DIR
        export JOB_TRACKING_FILE="${TEMP_DIR}/gcp_jobs_${env}.txt"
        > "$JOB_TRACKING_FILE"  # Create empty file
        if [[ $PARALLEL_JOBS -gt 1 ]]; then
            log_info "Parallel execution mode: Jobs will be submitted in batches of $PARALLEL_JOBS"
        else
            log_info "Sequential execution mode: Jobs will be submitted one at a time"
        fi
    fi
    
    # Process scenarios
    scenario_count=0
    experiment_count=0  # Track actual experiments (not scenarios)
    env_completed=0
    env_failed=0
    env_skipped=0
    
    # Extract scenarios using Python for reliable JSON parsing
    # Include scaling_experiment flag (defaults to False if not present)
    scenarios=$(python3 -c "
import json
with open('$GENERATED_SCENARIOS_DIR/manifest.json') as f:
    manifest = json.load(f)
for s in manifest['scenarios']:
    scaling = s.get('scaling_experiment', False)
    print(f\"{s['id']}|{s['path']}|{s['algorithm']}|{s['payload_size']}|{s['rate']}|{scaling}\")
")
    
    while IFS='|' read -r scenario_id scenario_path algorithm payload rate is_scaling; do
        scenario_count=$((scenario_count + 1))
        
        # Iterate over replica counts
        for replica_count in "${REPLICA_ARRAY[@]}"; do
            # For native, only run with 1 replica
            if [[ "$env" == "native" ]] && [[ "$replica_count" -gt 1 ]]; then
                continue
            fi
            
            # In smoke-test mode, only run with 1 replica
            if [[ "$SMOKE_TEST" == "true" ]] && [[ "$replica_count" -gt 1 ]]; then
                continue
            fi
            
            # For replicas > 1, only run scaling experiments
            # This ensures we don't run all 468 experiments with replicas 2,4,8
            if [[ "$replica_count" -gt 1 ]] && [[ "$is_scaling" != "True" ]]; then
                continue
            fi
            
            # Generate unique output dir and ID for scaling experiments
            if [[ "$replica_count" -gt 1 ]]; then
                output_dir="$RESULTS_BASE/$env/${scenario_id}_r${replica_count}"
                run_scenario_id="${scenario_id}_r${replica_count}"
            else
                output_dir="$RESULTS_BASE/$env/$scenario_id"
                run_scenario_id="$scenario_id"
            fi
            
            # Increment experiment count (not scenario count) for progress tracking
            # This must happen BEFORE any continue statements to ensure accurate counting
            experiment_count=$((experiment_count + 1))
            
            if [[ "$DRY_RUN" == "true" ]]; then
                update_progress $experiment_count $ENV_TOTAL_EXPERIMENTS "$env" "$run_scenario_id"
                log_info "  Would run: $run_scenario_id (replicas: $replica_count)"
                add_to_index "$run_scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "dry_run" "$replica_count"
                continue
            fi
            
            # Check if already completed (resume capability)
            # When --skip-analysis is used, only raw data exists
            # When analysis is enabled, check for merged/stats files
            is_complete=false
            raw_file="$output_dir/raw/run.jsonl"
            stats_file="$output_dir/stats/summary.json"
            merged_file="$output_dir/merged/merged.jsonl"
            
            if [[ "$SKIP_ANALYSIS" == "true" ]]; then
                # In data collection mode: check for raw data
                if [[ -f "$raw_file" ]] && [[ -s "$raw_file" ]]; then
                    is_complete=true
                fi
            else
                # In full mode: check for analysis outputs
                if [[ -f "$stats_file" ]] && [[ -s "$stats_file" ]]; then
                    is_complete=true
                elif [[ -f "$merged_file" ]] && [[ -s "$merged_file" ]]; then
                    is_complete=true
                fi
            fi
            
            if [[ "$is_complete" == "true" ]]; then
                # Update progress with completed count (not just experiment count)
                env_completed=$((env_completed + 1))
                update_progress $env_completed $ENV_TOTAL_EXPERIMENTS "$env" "$run_scenario_id (cached)"
                add_to_index "$run_scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "cached" "$replica_count"
                env_skipped=$((env_skipped + 1))
                continue
            elif [[ -d "$output_dir" ]]; then
                # Directory exists but incomplete - might be partial data
                if [[ -f "$raw_file" ]] && [[ ! -s "$raw_file" ]]; then
                    log_warn "  Found empty raw data for $run_scenario_id, will re-run"
                    rm -rf "$output_dir"
                elif [[ "$SKIP_ANALYSIS" != "true" ]] && [[ ! -f "$raw_file" ]]; then
                    log_warn "  Found incomplete results for $run_scenario_id, will re-run"
                    rm -rf "$output_dir"
                fi
            fi
            
            # Run experiment
            # For GCP with persistent cluster: job submission is handled in run_experiment
            # For parallel mode: jobs are submitted and tracked, then waited on at the end
            # For sequential mode: each job is submitted and waited on immediately
            update_progress $env_completed $ENV_TOTAL_EXPERIMENTS "$env" "$run_scenario_id"
            
            if run_experiment "$env" "$scenario_path" "$run_scenario_id" "$output_dir" "$replica_count"; then
                # Validate data integrity immediately after collection
                raw_file="$output_dir/raw/run.jsonl"
                if [[ -f "$raw_file" ]]; then
                    file_size=$(stat -f%z "$raw_file" 2>/dev/null || stat -c%s "$raw_file" 2>/dev/null || echo 0)
                    if [[ $file_size -eq 0 ]]; then
                        log_error "  Data integrity check FAILED: $run_scenario_id has 0-byte file!"
                        log_error "  This experiment will be marked as failed and can be retried"
                        add_to_index "$run_scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "failed" "$replica_count"
                        env_failed=$((env_failed + 1))
                        FAILED_SCENARIOS=$((FAILED_SCENARIOS + 1))
                        # Remove empty file so it can be retried
                        rm -f "$raw_file"
                        continue
                    else
                        line_count=$(wc -l < "$raw_file" 2>/dev/null || echo 0)
                        if [[ $line_count -eq 0 ]]; then
                            log_error "  Data integrity check FAILED: $run_scenario_id has no JSONL lines!"
                            log_error "  File size: $file_size bytes, but no lines found"
                            add_to_index "$run_scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "failed" "$replica_count"
                            env_failed=$((env_failed + 1))
                            FAILED_SCENARIOS=$((FAILED_SCENARIOS + 1))
                            rm -f "$raw_file"
                            continue
                        fi
                        log_success "  Completed: $run_scenario_id (replicas: $replica_count) - $file_size bytes, $line_count events"
                    fi
                else
                    log_error "  Data integrity check FAILED: $run_scenario_id - raw file not found!"
                    add_to_index "$run_scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "failed" "$replica_count"
                    env_failed=$((env_failed + 1))
                    FAILED_SCENARIOS=$((FAILED_SCENARIOS + 1))
                    continue
                fi
                
                    add_to_index "$run_scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "success" "$replica_count"
                    env_completed=$((env_completed + 1))
                    COMPLETED_SCENARIOS=$((COMPLETED_SCENARIOS + 1))
                    # Update progress with completed count after successful run
                    update_progress $env_completed $ENV_TOTAL_EXPERIMENTS "$env" "$run_scenario_id"
                else
                    log_error "  Failed: $run_scenario_id"
                    add_to_index "$run_scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "failed" "$replica_count"
                    env_failed=$((env_failed + 1))
                    FAILED_SCENARIOS=$((FAILED_SCENARIOS + 1))
                    
                    if [[ "$CONTINUE_ON_ERROR" != "true" ]]; then
                        log_error "Stopping due to failure (use --continue-on-error to ignore)"
                        exit 1
                    fi
                fi
        done
        
    done <<< "$scenarios"
    
    # For GCP execution with persistent cluster and parallel jobs, wait for all jobs to complete
    # For sequential mode (PARALLEL_JOBS=1), jobs are already waited on individually
    if [[ "$env" == "gcp" ]] && [[ "${GCP_USE_PERSISTENT_CLUSTER:-false}" == "true" ]] && [[ $PARALLEL_JOBS -gt 1 ]] && [[ -f "${JOB_TRACKING_FILE:-}" ]]; then
        log_info "Waiting for all parallel jobs to complete..."
        
        # Determine namespace
        if [[ "$SMOKE_TEST" == "true" ]]; then
            GCP_NAMESPACE="pqc-smoke-test"
        else
            GCP_NAMESPACE="default"
        fi
        
        # Track job completion
        TOTAL_JOBS=$(wc -l < "$JOB_TRACKING_FILE" 2>/dev/null || echo "0")
        COMPLETED_JOBS=0
        FAILED_JOBS=0
        
        while IFS='|' read -r job_name scenario_id output_dir; do
            # Wait for job to complete
            while true; do
                STATUS=$(kubectl get job "$job_name" -n "$GCP_NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || echo "")
                FAILED_STATUS=$(kubectl get job "$job_name" -n "$GCP_NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || echo "")
                
                if [[ "$STATUS" == "True" ]]; then
                    # Job completed successfully, download results
                    if "$SCRIPT_DIR/fetch_and_analyse_from_gcs.sh" \
                        --exp-id "$scenario_id" \
                        --bucket "$BUCKET" \
                        --out "$output_dir" 2>&1; then
                        COMPLETED_JOBS=$((COMPLETED_JOBS + 1))
                        env_completed=$((env_completed + 1))
                        log_success "  Completed: $scenario_id"
                    else
                        FAILED_JOBS=$((FAILED_JOBS + 1))
                        env_failed=$((env_failed + 1))
                        log_error "  Failed to download results for: $scenario_id"
                    fi
                    break
                elif [[ "$FAILED_STATUS" == "True" ]]; then
                    FAILED_JOBS=$((FAILED_JOBS + 1))
                    env_failed=$((env_failed + 1))
                    log_error "  Job failed: $scenario_id"
                    break
                fi
                sleep 5
            done
            
            # Update progress
            update_progress $env_completed $ENV_TOTAL_EXPERIMENTS "$env" "$scenario_id"
        done < "$JOB_TRACKING_FILE"
        
        log_info "Parallel execution complete: $COMPLETED_JOBS succeeded, $FAILED_JOBS failed"
        rm -rf "$TEMP_DIR"
    fi
    
    echo ""
    log_info "Environment $env summary:"
    log_info "  Completed: $env_completed (skipped: $env_skipped, new: $((env_completed - env_skipped)))"
    log_info "  Failed: $env_failed"
    log_info "  Progress: $((env_completed * 100 / ENV_TOTAL_EXPERIMENTS))%"
    echo ""
    
    # Cleanup: Destroy persistent cluster if we created it for this batch
    if [[ "$env" == "gcp" ]] && [[ "${GCP_USE_PERSISTENT_CLUSTER:-false}" == "true" ]] && [[ "${GCP_CLUSTER_EXISTS:-true}" == "false" ]]; then
        log_info "Destroying persistent cluster after batch completion..."
        if "$SCRIPT_DIR/deploy_gcp.sh" \
            --destroy-cluster \
            --project "$PROJECT" \
            --bucket "$BUCKET" \
            --region "$REGION" \
            $([ "$SMOKE_TEST" == "true" ] && echo "--smoke-test" || echo ""); then
            log_success "Cluster destroyed successfully"
        else
            log_warn "Cluster destruction had errors (cluster may still exist)"
            log_info "You can destroy it manually with:"
            echo "  ./deploy_gcp.sh --destroy-cluster --project $PROJECT --bucket $BUCKET --region $REGION"
        fi
    fi
    
done

# =============================================================================
# Phase 4: Write Master Index
# =============================================================================
log_phase "4. Write Master Index"

# Write index.json
INDEX_FILE="$FINAL_RESULTS_DIR/index.json"
{
    echo "{"
    echo "  \"generated_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"matrix_file\": \"$MATRIX\","
    echo "  \"environments\": [\"${ENV_ARRAY[*]// /\", \"}\"],"
    echo "  \"total_scenarios\": $TOTAL_SCENARIOS,"
    echo "  \"completed_scenarios\": $COMPLETED_SCENARIOS,"
    echo "  \"failed_scenarios\": $FAILED_SCENARIOS,"
    echo "  \"experiments\": ["
    
    first=true
    for entry in "${MASTER_INDEX[@]}"; do
        if [[ "$first" == "true" ]]; then
            first=false
        else
            echo ","
        fi
        echo -n "    $entry"
    done
    
    echo ""
    echo "  ]"
    echo "}"
} > "$INDEX_FILE"

log_success "Master index: $INDEX_FILE"

# =============================================================================
# Phase 5: Statistical Aggregation
# =============================================================================
log_phase "5. Statistical Aggregation"

if [[ "$SKIP_ANALYSIS" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
    log_warn "Skipping aggregation"
else
    log_info "Aggregating results across experiments..."
    
    python3 "$SCRIPT_DIR/analysis/aggregate_results.py" \
        --index "$INDEX_FILE" \
        --output "$FINAL_RESULTS_DIR" 2>&1 || log_warn "Aggregation completed with warnings"
    
    log_success "Aggregation complete"
fi

# =============================================================================
# Phase 6: Generate Combined Figures
# =============================================================================
log_phase "6. Generate Combined Figures"

if [[ "$SKIP_ANALYSIS" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
    log_warn "Skipping figure generation"
else
    log_info "Generating combined CDF plots..."
    python3 "$SCRIPT_DIR/analysis/plot_combined_cdfs.py" \
        --index "$INDEX_FILE" \
        --output "$FINAL_RESULTS_DIR/figures" 2>&1 || log_warn "CDF plots completed with warnings"
    
    log_info "Generating scaling curves..."
    python3 "$SCRIPT_DIR/analysis/plot_scaling_curves.py" \
        --index "$INDEX_FILE" \
        --output "$FINAL_RESULTS_DIR/figures" 2>&1 || log_warn "Scaling plots completed with warnings"
    
    log_success "Figures generated"
fi

# =============================================================================
# Phase 7: Hypothesis Testing
# =============================================================================
log_phase "7. Hypothesis Testing"

if [[ "$SKIP_ANALYSIS" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
    log_warn "Skipping hypothesis tests"
else
    log_info "Running statistical hypothesis tests..."
    log_info "  - Kolmogorov-Smirnov test (distribution shape)"
    log_info "  - Mann-Whitney U test (distribution location)"
    log_info "  - Welch's t-test (mean difference)"
    log_info "  - Cohen's d with 95% CI (effect size)"
    log_info "  - Holm-Bonferroni correction (multiple comparisons)"
    
    python3 "$SCRIPT_DIR/analysis/hypothesis_tests.py" \
        --index "$INDEX_FILE" \
        --matrix "$MATRIX" \
        --output "$FINAL_RESULTS_DIR" 2>&1 || log_warn "Hypothesis tests completed with warnings"
    
    # Report results
    if [[ -f "$FINAL_RESULTS_DIR/hypothesis_tests.json" ]]; then
        TOTAL_TESTS=$(python3 -c "import json; print(json.load(open('$FINAL_RESULTS_DIR/hypothesis_tests.json'))['total_comparisons'])" 2>/dev/null || echo "?")
        SIG_TESTS=$(python3 -c "import json; print(json.load(open('$FINAL_RESULTS_DIR/hypothesis_tests.json'))['significant_comparisons'])" 2>/dev/null || echo "?")
        log_info "  Total comparisons: $TOTAL_TESTS"
        log_info "  Significant (α=0.05, corrected): $SIG_TESTS"
    fi
    
    log_success "Hypothesis tests complete"
fi

# =============================================================================
# Phase 8: Build Final Report (PDF)
# =============================================================================
log_phase "8. Build Final Report (PDF)"

if [[ "$SKIP_ANALYSIS" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
    log_warn "Skipping report generation"
else
    log_info "Building dissertation-ready PDF report..."
    
    python3 "$SCRIPT_DIR/analysis/build_final_report.py" \
        --results-dir "$FINAL_RESULTS_DIR" \
        --output "$FINAL_RESULTS_DIR/report.pdf" 2>&1 || log_warn "Report generation completed with warnings"
    
    if [[ -f "$FINAL_RESULTS_DIR/report.pdf" ]]; then
        REPORT_SIZE=$(du -h "$FINAL_RESULTS_DIR/report.pdf" | cut -f1)
        log_success "PDF report generated: $FINAL_RESULTS_DIR/report.pdf ($REPORT_SIZE)"
    else
        log_warn "PDF generation may have failed (reportlab required)"
    fi
fi

# =============================================================================
# Phase 9: Replica Scaling Analysis
# =============================================================================
log_phase "9. Replica Scaling Analysis"

if [[ "$SKIP_ANALYSIS" == "true" ]] || [[ "$SKIP_SCALING" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
    log_warn "Skipping scaling analysis"
else
    # Check if we have scaling experiments (replicas > 1)
    HAS_SCALING=false
    for r in "${REPLICA_ARRAY[@]}"; do
        if [[ "$r" -gt 1 ]]; then
            HAS_SCALING=true
            break
        fi
    done
    
    if [[ "$HAS_SCALING" == "true" ]]; then
        log_info "Generating replica scaling plots..."
        
        mkdir -p "$FINAL_RESULTS_DIR/figures/scaling"
        
        python3 "$SCRIPT_DIR/analysis/plot_replica_scaling.py" \
            --index "$INDEX_FILE" \
            --output "$FINAL_RESULTS_DIR/figures/scaling" 2>&1 || log_warn "Scaling plots completed with warnings"
        
        # List generated files
        if [[ -d "$FINAL_RESULTS_DIR/figures/scaling" ]]; then
            SCALING_FILES=$(ls "$FINAL_RESULTS_DIR/figures/scaling/"*.png 2>/dev/null | wc -l)
            log_success "Generated $SCALING_FILES scaling figures"
        fi
    else
        log_info "No scaling experiments (replicas > 1) detected, skipping"
    fi
fi

# =============================================================================
# Final Summary
# =============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║                    EXPERIMENT SUITE COMPLETE                          ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

log_info "Duration: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
log_info "Total scenarios: $TOTAL_SCENARIOS"
log_info "Completed: $COMPLETED_SCENARIOS"
log_info "Failed: $FAILED_SCENARIOS"
echo ""

log_info "Final results location:"
echo "  $FINAL_RESULTS_DIR/"
echo ""
log_info "📌 IMPORTANT: All dissertation-ready outputs are in the directory above!"
log_info "   Individual experiment results are in: results/<env>/<scenario-id>/"
echo ""

log_info "Key outputs:"
[[ -f "$FINAL_RESULTS_DIR/index.json" ]] && echo "  ├── index.json (master experiment index)"
[[ -f "$FINAL_RESULTS_DIR/aggregated_stats.json" ]] && echo "  ├── aggregated_stats.json"
[[ -f "$FINAL_RESULTS_DIR/aggregated_stats.csv" ]] && echo "  ├── aggregated_stats.csv"
[[ -f "$FINAL_RESULTS_DIR/hypothesis_tests.json" ]] && echo "  ├── hypothesis_tests.json (statistical tests)"
[[ -f "$FINAL_RESULTS_DIR/hypothesis_table.csv" ]] && echo "  ├── hypothesis_table.csv"
[[ -f "$FINAL_RESULTS_DIR/hypothesis_interpretation.txt" ]] && echo "  ├── hypothesis_interpretation.txt"
[[ -d "$FINAL_RESULTS_DIR/figures" ]] && echo "  ├── figures/ (✅ Use these for dissertation!)"
[[ -d "$FINAL_RESULTS_DIR/figures/scaling" ]] && echo "  │   └── scaling/ (throughput, latency, efficiency)"
[[ -d "$FINAL_RESULTS_DIR/stats" ]] && echo "  ├── stats/"
[[ -d "$FINAL_RESULTS_DIR/tables" ]] && echo "  ├── tables/"
[[ -f "$FINAL_RESULTS_DIR/report.pdf" ]] && echo "  └── report.pdf (dissertation-ready)"
echo ""

# Optionally copy figures to analysis/figures/dissertation for notebook compatibility
if [[ -d "$FINAL_RESULTS_DIR/figures" ]] && [[ -d "$SCRIPT_DIR/analysis/figures/dissertation" ]]; then
    FIGURE_COUNT=$(find "$FINAL_RESULTS_DIR/figures" -name "*.png" -type f 2>/dev/null | wc -l)
    if [[ $FIGURE_COUNT -gt 0 ]]; then
        log_info "Copying figures to analysis/figures/dissertation/ for notebook compatibility..."
        cp -n "$FINAL_RESULTS_DIR/figures"/*.png "$SCRIPT_DIR/analysis/figures/dissertation/" 2>/dev/null || true
        log_success "Figures also available in: analysis/figures/dissertation/"
        echo ""
    fi
fi

if [[ $FAILED_SCENARIOS -gt 0 ]]; then
    log_warn "$FAILED_SCENARIOS experiment(s) failed. Check logs for details."
fi

log_success "Done!"

exit 0

