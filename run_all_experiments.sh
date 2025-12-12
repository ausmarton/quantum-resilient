#!/usr/bin/env bash
# =============================================================================
# run_all_experiments.sh - Unified Benchmark Orchestration Script
#
# Executes benchmarking scenarios across all environments (native, Minikube, GCP),
# runs multiple repeats, collects data, and produces dissertation-ready results.
#
# This script handles both smoke-test and full-scale benchmarks using the same flow:
# - Same scenario generation (orchestration/generate_scenarios.py)
# - Same results structure (results/<env>/<scenario-id>/)
# - Same analysis pipeline
# - Only differs in experiment matrix filtering and parameters (controlled by --smoke-test)
#
# Usage:
#   # Full-scale benchmarks
#   ./run_all_experiments.sh \
#     --project <gcp-project> \
#     --bucket <gcs-bucket> \
#     --matrix orchestration/experiment_matrix.yaml \
#     --envs native,minikube,gcp
#
#   # Smoke-test benchmarks (same command, add --smoke-test)
#   ./run_all_experiments.sh \
#     --smoke-test \
#     --envs native,minikube,gcp
#
# Results Structure (same for both smoke-test and full-scale):
#   - Individual experiments: results/<env>/<scenario-id>/
#   - Final results: final-results/ (unified for both modes)
#   - Scenario IDs are identical format regardless of mode
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

# Source unified Kubernetes job management functions
source "$SCRIPT_DIR/scripts/lib/k8s-job.sh" 2>/dev/null || true
source "$SCRIPT_DIR/scripts/lib/k8s-image.sh" 2>/dev/null || true
source "$SCRIPT_DIR/scripts/lib/analysis.sh" 2>/dev/null || true

# Default values
MATRIX="$SCRIPT_DIR/orchestration/experiment_matrix.yaml"
ENVS="native"
PROJECT=""
BUCKET=""
REGION="us-central1"
PARALLEL_JOBS=1
REPLICAS="1"  # Comma-separated list: 1,2,4,8
REPLICAS_EXPLICITLY_SET=false  # Track if user explicitly set --replicas
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
MINI_SMOKE_TEST=false

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

# Get Python command (containerized if available, fallback to host Python)
get_python_cmd() {
    if [[ -f "$SCRIPT_DIR/scripts/lib/run-python-container.sh" ]] && \
       [[ "${QR_USE_CONTAINER:-true}" != "false" ]]; then
        echo "$SCRIPT_DIR/scripts/lib/run-python-container.sh"
    else
        echo "python3"
    fi
}

# Convert absolute path to relative path (for containerized scripts)
# Container mounts project root as /workspace, so absolute paths need to be relative
to_relative_path() {
    local path="$1"
    if [[ "$path" == /* ]]; then
        # Absolute path - convert to relative if under project root
        if [[ "$path" == "$SCRIPT_DIR"* ]]; then
            echo "${path#$SCRIPT_DIR/}"
        else
            # Path outside project root - return as-is (might be /tmp, etc.)
            echo "$path"
        fi
    else
        # Already relative
        echo "$path"
    fi
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
                            Uses exact same flow as full-scale, only differs in:
                            - Experiment matrix filtering (subset of algorithms/experiments)
                            - Reduced parameters (2 payloads, 2 rates, 1 run, 5s duration)
                            Scenario IDs, results structure, and directories are identical.
    --mini-smoke-test       Enable minimal smoke-test mode (2 experiments: 1 classical, 1 PQC)
                            Even smaller than --smoke-test: only rsa2048 and kyber512,
                            1 payload (256B), 1 rate (100 msg/s), 1 run per algorithm.
                            Perfect for quick end-to-end validation.
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
        # Calculate seconds per scenario (use floating point via bc if available, else integer math)
        if command -v bc &>/dev/null; then
            local rate=$(echo "scale=2; $elapsed / $current" | bc)
            local remaining=$((total - current))
            local eta_seconds=$(echo "scale=0; $rate * $remaining" | bc | cut -d. -f1)
        else
            # Integer math fallback: calculate seconds per scenario (rounded)
            local rate=$((elapsed / current))
            local remaining=$((total - current))
            local eta_seconds=$((rate * remaining))
        fi
        
        if [[ $eta_seconds -gt 0 ]]; then
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

# Extract base experiment ID (without run_index suffix)
# Scenario IDs include "_run1", "_run2", etc., but we want base ID for output directories
# Format: <algorithm>_p<payload>_r<rate>_run<N>_<hash> -> <algorithm>_p<payload>_r<rate>_<hash>
# Handles patterns like burst, scaling, duration suffixes
extract_base_experiment_id() {
    local scenario_id=$1
    # Remove _run<N> pattern (where N is 1-9 or 10+)
    # Pattern: _run followed by digits, then _hash
    echo "$scenario_id" | sed -E 's/_run[0-9]+(_[a-f0-9]{8})$/\1/' || echo "$scenario_id"
}

# Run single experiment with retries
run_experiment() {
    local env=$1
    local scenario_path=$2
    local scenario_id=$3
    local output_dir=$4
    local replicas=${5:-1}
    local total_runs=${6:-5}  # Default to 5 runs if not provided
    local retries=0
    
    while [[ $retries -le $MAX_RETRIES ]]; do
        local exit_code=0
        
        case $env in
            native)
                # Native doesn't support replicas
                # For native benchmarks, we now process only run-1 scenarios and run them with --runs parameter
                # This groups multiple runs into single experiments, reducing 468 scenarios to ~94 experiments
                # Scenario extraction already filters to run-1 scenarios for native environment
                
                # Determine number of runs (use total_runs parameter, default to 5 for full-scale, 1 for smoke-test)
                local runs_param=""
                if [[ "$SMOKE_TEST" == "true" ]]; then
                    runs_param="--runs 1"
                else
                    # Get total_runs from function parameter (6th argument)
                    local total_runs_from_param="${total_runs:-5}"
                    runs_param="--runs $total_runs_from_param"
                fi
                
                # Extract seed from scenario file if available
                SCENARIO_SEED=""
                if [[ -f "$scenario_path" ]]; then
                    SCENARIO_SEED=$(grep -E "^rng_seed:" "$scenario_path" | head -1 | awk '{print $2}' | tr -d '"' || echo "")
                fi
                
                # Run with --runs parameter to execute multiple runs in a single experiment
                RUN_ARGS=(
                    --scenario "$scenario_path"
                    --out "$output_dir"
                    --duration 30
                    $runs_param
                )
                [[ -n "$SCENARIO_SEED" ]] && RUN_ARGS+=(--seed "$SCENARIO_SEED")
                [[ "$SMOKE_TEST" == "true" ]] && RUN_ARGS+=(--smoke-test)
                
                "$SCRIPT_DIR/run_local.sh" "${RUN_ARGS[@]}" 2>&1 || exit_code=$?
                ;;
            minikube)
                # Minikube now handles multiple runs internally via --runs parameter
                # The scenario_path passed here is always for run-1 due to filtering above
                # Each run still creates a separate Kubernetes Job, maintaining isolation
                # Runs execute sequentially (one job completes before next starts), matching native behavior
                
                # Determine number of runs (5 for full-scale, 1 for smoke-test)
                local runs_param=""
                if [[ "$SMOKE_TEST" == "true" ]]; then
                    runs_param="--runs 1"
                else
                    # Get total_runs from function parameter (6th argument)
                    local total_runs_from_param="${total_runs:-5}"
                    runs_param="--runs $total_runs_from_param"
                fi
                
                # Minikube supports conditional parallelism (limited to prevent overutilization)
                # Check if parallelism is enabled and if we should use it
                if [[ "${MINIKUBE_USE_PARALLELISM:-false}" == "true" ]]; then
                    # Parallel mode: Submit job non-blocking and track it
                    # Use unified job submission function
                    if [[ -z "${MINIKUBE_IMAGE_NAME:-}" ]]; then
                        # Build and load image (this should be done once before the loop)
                        log_error "Minikube image not set for parallel mode - this should not happen"
                        exit_code=1
                    else
                        # Use consistent namespace for all test types
                        MINIKUBE_NAMESPACE="default"
                        
                        # Submit job using unified function (non-blocking)
                        # Note: Parallel mode doesn't support --runs yet (would need multiple job submissions)
                        # For now, parallel mode runs single runs only
                        JOB_NAME=$(submit_k8s_job \
                            "minikube" \
                            "$scenario_path" \
                            "$scenario_id" \
                            "$MINIKUBE_IMAGE_NAME" \
                            "$MINIKUBE_NAMESPACE" \
                            "$replicas" \
                            "$SMOKE_TEST" \
                            "" \
                            "" 2>&1) || exit_code=$?
                        
                        if [[ $exit_code -eq 0 ]] && [[ -n "$JOB_NAME" ]]; then
                            # Track job for batch waiting
                            echo "$JOB_NAME|$scenario_id|$output_dir" >> "${MINIKUBE_JOB_TRACKING_FILE:-/tmp/minikube_jobs.txt}"
                            exit_code=0
                        else
                            log_error "Job submission failed for $scenario_id"
                            exit_code=1
                        fi
                    fi
                else
                    # Sequential mode: Run experiment with --runs parameter and wait for completion
                    # Use --quiet flag to suppress verbose output, allowing progress updates to show
                    # Errors are still logged by run_minikube.sh internally
                    # Redirect stderr to capture errors but allow stdout for job name capture
                    if "$SCRIPT_DIR/run_minikube.sh" \
                        --scenario "$scenario_path" \
                        --out "$output_dir" \
                        --replicas "$replicas" \
                        --exp-id "$scenario_id" \
                        $runs_param \
                        --quiet \
                        $([ "$SMOKE_TEST" == "true" ] && echo "--smoke-test" || echo "") >/dev/null 2>/tmp/minikube_${scenario_id}.log; then
                        exit_code=0
                    else
                        exit_code=$?
                        # Log errors if job failed
                        if [[ -f /tmp/minikube_${scenario_id}.log ]]; then
                            log_warn "Minikube experiment failed, last 10 lines:"
                            tail -10 /tmp/minikube_${scenario_id}.log | while read line; do
                                log_warn "  $line"
                            done
                        fi
                    fi
                fi
                ;;
            gcp)
                # GCP now handles multiple runs internally via --runs parameter
                # The scenario_path passed here is always for run-1 due to filtering above
                # Each run still creates a separate Kubernetes Job, maintaining isolation
                # Runs execute sequentially (one job completes before next starts), matching native behavior
                
                # Determine number of runs (5 for full-scale, 1 for smoke-test)
                local runs_param=""
                if [[ "$SMOKE_TEST" == "true" ]]; then
                    runs_param="--runs 1"
                else
                    # Get total_runs from function parameter (6th argument)
                    local total_runs_from_param="${total_runs:-5}"
                    runs_param="--runs $total_runs_from_param"
                fi
                
                # Unified execution: Always use Kubernetes Job submission
                # For persistent cluster mode: Submit ALL jobs at once, let Kubernetes scheduler handle parallelism
                # For ephemeral mode: Use deploy_gcp.sh (creates/destroys cluster per experiment)
                
                if [[ "${GCP_USE_PERSISTENT_CLUSTER:-false}" == "true" ]]; then
                    # Persistent cluster mode: Submit all jobs immediately (non-blocking)
                    # Kubernetes scheduler will determine parallelism based on available nodes
                    # Support multiple runs by submitting separate jobs for each run
                    if [[ -z "${GCP_IMAGE_NAME:-}" ]]; then
                        # Get image name if not set
                        IMAGE_REPO="${REGION}-docker.pkg.dev/${PROJECT}/pqc"
                        GCP_IMAGE_NAME="${IMAGE_REPO}/pqc-bench:latest"
                    fi
                    
                    # Use consistent namespace for all test types
                    GCP_NAMESPACE="default"
                    
                    # Determine number of runs (5 for full-scale, 1 for smoke-test)
                    num_runs=1
                    if [[ "$SMOKE_TEST" != "true" ]]; then
                        num_runs="${total_runs:-5}"
                    fi
                    
                    # Submit multiple jobs (one per run) to support multiple runs in parallel mode
                    # Each run gets a unique experiment ID with _run<N> suffix
                    # Check GCS for existing runs and only submit missing ones
                    export GCP_CLUSTER_NAME="$CLUSTER_NAME"
                    runs_submitted=0
                    runs_skipped=0
                    for ((run_idx = 1; run_idx <= num_runs; run_idx++)); do
                        if [[ $num_runs -gt 1 ]]; then
                            RUN_EXP_ID="${run_scenario_id}_run${run_idx}"
                        else
                            RUN_EXP_ID="$run_scenario_id"
                        fi
                        
                        # Check if this specific run already exists in GCS (resume capability)
                        RUN_GCS_PATH="gs://${BUCKET}/experiments/${RUN_EXP_ID}"
                        if gsutil -q ls "$RUN_GCS_PATH/raw/run.jsonl" &>/dev/null 2>&1; then
                            log_info "  Skipping run $run_idx/$num_runs for $run_scenario_id (already exists in GCS)"
                            runs_skipped=$((runs_skipped + 1))
                            continue
                        fi
                        
                        # Submit job immediately (non-blocking) - all jobs submitted, then wait for all at end
                        log_info "  Submitting run $run_idx/$num_runs for $run_scenario_id (EXP_ID: $RUN_EXP_ID)"
                        JOB_SUBMIT_OUTPUT=$("$SCRIPT_DIR/scripts/submit_gcp_job_parallel.sh" \
                            --scenario "$scenario_path" \
                            --exp-id "$RUN_EXP_ID" \
                            --project "$PROJECT" \
                            --bucket "$BUCKET" \
                            --region "$REGION" \
                            --image "$GCP_IMAGE_NAME" \
                            --namespace "$GCP_NAMESPACE" \
                            --replicas "$replicas" \
                            $([ "$SMOKE_TEST" == "true" ] && echo "--smoke-test" || echo "") 2>&1) || run_exit_code=$?
                        
                        if [[ ${run_exit_code:-0} -ne 0 ]]; then
                            log_error "Job submission failed for $RUN_EXP_ID (replicas: $replicas, run $run_idx/$num_runs)"
                            # Print all output lines (errors go to stderr, which is captured)
                            if [[ -n "$JOB_SUBMIT_OUTPUT" ]]; then
                                echo "$JOB_SUBMIT_OUTPUT" | while IFS= read -r line || [[ -n "$line" ]]; do
                                    log_error "  $line"
                                done
                            else
                                log_error "  No error output captured (check kubectl connectivity and cluster status)"
                            fi
                            exit_code=${run_exit_code:-1}
                        else
                            JOB_NAME=$(echo "$JOB_SUBMIT_OUTPUT" | tail -1)
                            if [[ -z "$JOB_NAME" ]]; then
                                log_error "Job submission returned success but no job name for $RUN_EXP_ID"
                                exit_code=1
                            else
                                # Track job for batch waiting (all jobs submitted, wait for all at end)
                                echo "$JOB_NAME|$RUN_EXP_ID|$output_dir" >> "${JOB_TRACKING_FILE:-/tmp/gcp_jobs_${env}.txt}"
                                runs_submitted=$((runs_submitted + 1))
                            fi
                        fi
                    done
                    
                    if [[ $runs_skipped -gt 0 ]]; then
                        log_info "  Skipped $runs_skipped existing run(s), submitted $runs_submitted new run(s) for $run_scenario_id"
                    fi
                    
                    # Note: exit_code is set by the loop above
                else
                    # Ephemeral mode: use deploy_gcp.sh with --runs parameter
                    GCP_ARGS=(
                        --scenario "$scenario_path"
                        --exp-id "$run_scenario_id"
                        --project "$PROJECT"
                        --bucket "$BUCKET"
                        --region "$REGION"
                        --replicas "$replicas"
                        $runs_param
                        --ephemeral
                    )
                    [ "$SMOKE_TEST" == "true" ] && GCP_ARGS+=(--smoke-test)
                    
                    "$SCRIPT_DIR/deploy_gcp.sh" "${GCP_ARGS[@]}" 2>&1 || exit_code=$?
                    
                    if [[ $exit_code -eq 0 ]]; then
                        "$SCRIPT_DIR/fetch_and_analyse_from_gcs.sh" \
                            --exp-id "$run_scenario_id" \
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
    
    local entry="{\"scenario_id\":\"$scenario_id\",\"environment\":\"$env\",\"algorithm\":\"$algorithm\",\"payload_size\":$payload,\"rate\":$rate,\"replicas\":$replicas,\"output_dir\":\"$output_dir\",\"status\":\"$status\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
    MASTER_INDEX+=("$entry")
    
    # Write progress incrementally to a progress file for resume capability
    # This allows us to track progress even if the script is interrupted
    # Ensure FINAL_RESULTS_DIR is set (may not be set on first call, initialize it)
    if [[ -z "${FINAL_RESULTS_DIR:-}" ]]; then
        FINAL_RESULTS_DIR="$SCRIPT_DIR/final-results"
    fi
    mkdir -p "$FINAL_RESULTS_DIR"
    local progress_file="$FINAL_RESULTS_DIR/.progress_${env}.jsonl"
    echo "$entry" >> "$progress_file"
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
            REPLICAS_EXPLICITLY_SET=true
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
        --mini-smoke-test)
            MINI_SMOKE_TEST=true
            SMOKE_TEST=true  # Mini smoke test implies smoke test
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
if [[ "$MINI_SMOKE_TEST" == "true" ]]; then
    log_info "Mode: MINI SMOKE-TEST (2 experiments: 1 classical, 1 PQC)"
elif [[ "$SMOKE_TEST" == "true" ]]; then
    log_info "Mode: SMOKE-TEST (reduced scale, minimal cost)"
fi
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
    
    PYTHON_CMD=$(get_python_cmd)
    # Convert paths to relative for containerized scripts
    SCENARIO_SCRIPT=$(to_relative_path "$SCRIPT_DIR/orchestration/generate_scenarios.py")
    MATRIX_REL=$(to_relative_path "$MATRIX")
    OUTPUT_REL=$(to_relative_path "$GENERATED_SCENARIOS_DIR")
    
    if [[ "$DRY_RUN" == "true" ]]; then
        $PYTHON_CMD "$SCENARIO_SCRIPT" \
            --matrix "$MATRIX_REL" \
            --output "$OUTPUT_REL" \
            --dry-run \
            $([ "$SMOKE_TEST" == "true" ] && echo "--smoke-test" || echo "") \
            $([ "$MINI_SMOKE_TEST" == "true" ] && echo "--mini-smoke-test" || echo "")
    else
        $PYTHON_CMD "$SCENARIO_SCRIPT" \
            --matrix "$MATRIX_REL" \
            --output "$OUTPUT_REL" \
            $([ "$SMOKE_TEST" == "true" ] && echo "--smoke-test" || echo "") \
            $([ "$MINI_SMOKE_TEST" == "true" ] && echo "--mini-smoke-test" || echo "")
    fi
    
    log_success "Scenarios generated"
fi

# Count scenarios
if [[ -f "$GENERATED_SCENARIOS_DIR/manifest.json" ]]; then
    PYTHON_CMD=$(get_python_cmd)
    MANIFEST_REL=$(to_relative_path "$GENERATED_SCENARIOS_DIR/manifest.json")
    TOTAL_SCENARIOS=$($PYTHON_CMD -c "import json; print(json.load(open('$MANIFEST_REL'))['total_scenarios'])")
    log_info "Total scenarios: $TOTAL_SCENARIOS"
fi

# Create run tracking directory with timestamp
RUN_TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR="$SCRIPT_DIR/run-${RUN_TIMESTAMP}"
mkdir -p "$RUN_DIR"

# Unified final results directory (same for both smoke-test and full-scale)
# Initialize early so progress tracking can use it
FINAL_RESULTS_DIR="$SCRIPT_DIR/final-results"

# Force replicas to 1 in smoke-test mode
if [[ "$SMOKE_TEST" == "true" ]]; then
    REPLICAS="1"
fi

# =============================================================================
# Phase 2: Create Output Directories
# =============================================================================
log_phase "2. Initialize Output Directories"

# Determine final results directory based on smoke-test mode
# Use same structure: final-results/ for both smoke-test and full-scale
# Unified final results directory (same for both smoke-test and full-scale)
# The distinction comes from scenario filtering, not directory structure
# (FINAL_RESULTS_DIR already initialized above)

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

# Auto-set replicas for full-scale runs on minikube/GCP (if not explicitly set)
# This ensures consistency with run_full_scale_data_collection.sh
if [[ "$REPLICAS_EXPLICITLY_SET" == "false" ]] && [[ "$SMOKE_TEST" != "true" ]]; then
    # Check if minikube or gcp is in the environment list
    if [[ "$ENVS" == *"minikube"* ]] || [[ "$ENVS" == *"gcp"* ]]; then
        REPLICAS="1,2,4,8"
        log_info "Auto-setting replicas to 1,2,4,8 for full-scale minikube/GCP runs"
        log_info "  (Use --replicas to override, or --smoke-test uses 1 replica)"
    fi
fi

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
    
    # For Minikube: Ensure Minikube is running, switch context, and ensure namespace exists
    if [[ "$env" == "minikube" ]]; then
        # Check if Minikube is running
        MINIKUBE_STATUS=$(minikube status --format='{{.Host}}' 2>/dev/null || echo "Stopped")
        if [[ "$MINIKUBE_STATUS" != "Running" ]]; then
            log_info "Minikube not running. Starting Minikube..."
            minikube start --driver=podman --cpus=4 --memory=8g || {
                log_error "Failed to start Minikube. Please run: minikube start --driver=podman"
                exit 1
            }
            log_success "Minikube started"
        else
            log_info "Minikube is already running"
        fi
        
        # Switch kubectl context to Minikube
        log_info "Switching kubectl context to Minikube..."
        if kubectl config use-context minikube &>/dev/null; then
            log_success "kubectl context switched to Minikube"
        else
            log_warn "Failed to switch to Minikube context (may not exist yet)"
            # Wait a moment for Minikube to fully initialize
            sleep 2
            kubectl config use-context minikube || {
                log_error "Failed to switch to Minikube context after waiting"
                exit 1
            }
        fi
        
        # Ensure namespace exists for smoke tests
        if [[ "$SMOKE_TEST" == "true" ]]; then
            # Use consistent namespace for all test types
            MINIKUBE_NAMESPACE="default"
            # Ensure default namespace exists (it should, but check anyway)
            if ! kubectl get namespace "$MINIKUBE_NAMESPACE" &>/dev/null; then
                log_info "Creating namespace '$MINIKUBE_NAMESPACE'..."
                kubectl create namespace "$MINIKUBE_NAMESPACE" || {
                    log_error "Failed to create namespace '$MINIKUBE_NAMESPACE'"
                    exit 1
                }
                log_success "Namespace '$MINIKUBE_NAMESPACE' created"
            else
                log_info "Namespace '$MINIKUBE_NAMESPACE' already exists"
            fi
        fi
    fi
    
    # For Minikube: Check if parallelism should be enabled
    if [[ "$env" == "minikube" ]]; then
        MINIKUBE_USE_PARALLELISM=false
        MINIKUBE_MAX_PARALLEL=4
        
        if [[ $PARALLEL_JOBS -gt 1 ]]; then
            # Limit parallelism for Minikube to prevent overutilization
            if [[ $PARALLEL_JOBS -gt $MINIKUBE_MAX_PARALLEL ]]; then
                log_warn "Limiting Minikube parallelism to $MINIKUBE_MAX_PARALLEL (requested: $PARALLEL_JOBS)"
                PARALLEL_JOBS=$MINIKUBE_MAX_PARALLEL
            fi
            
            # Check system load before enabling parallelism
            if [[ -f "$SCRIPT_DIR/scripts/check_system_load.sh" ]]; then
                if "$SCRIPT_DIR/scripts/check_system_load.sh" --warn-threshold 1.0 --fail-threshold 2.0 >/dev/null 2>&1; then
                    MINIKUBE_USE_PARALLELISM=true
                    log_info "Minikube parallelism enabled: $PARALLEL_JOBS jobs (system load check passed)"
                else
                    log_warn "System load check failed or load is high - using sequential mode for Minikube"
                    log_info "  (Minikube should be closer to native to avoid noisy neighbors)"
                    MINIKUBE_USE_PARALLELISM=false
                fi
            else
                # If check script doesn't exist, enable parallelism with warning
                log_warn "System load check script not found - enabling parallelism with caution"
                MINIKUBE_USE_PARALLELISM=true
            fi
            
            if [[ "$MINIKUBE_USE_PARALLELISM" == "true" ]]; then
                # Create job tracking file for Minikube parallel mode
                TEMP_DIR=$(mktemp -d)
                export TEMP_DIR
                export MINIKUBE_JOB_TRACKING_FILE="${TEMP_DIR}/minikube_jobs.txt"
                > "$MINIKUBE_JOB_TRACKING_FILE"
                
                # Build and load image once for all jobs
                log_info "Building and loading Minikube image for parallel execution..."
                IMAGE_NAME="pqc-bench"
                IMAGE_TAG="latest"
                # Save original stderr to fd 3 before command substitution
                exec 3>&2
                MINIKUBE_IMAGE_NAME=$(build_and_load_image_minikube \
                    "$IMAGE_NAME" \
                    "$IMAGE_TAG" \
                    "Containerfile" \
                    "false" \
                    "false" 2>&3)
                BUILD_EXIT=$?
                exec 3>&-  # Close fd 3
                if [[ $BUILD_EXIT -ne 0 ]]; then
                    log_error "Failed to build and load Minikube image"
                    MINIKUBE_USE_PARALLELISM=false
                fi
                export MINIKUBE_IMAGE_NAME
                if [[ -n "$MINIKUBE_IMAGE_NAME" ]]; then
                    log_success "Minikube image ready: $MINIKUBE_IMAGE_NAME"
                fi
            fi
        else
            log_info "Minikube: Sequential mode (PARALLEL_JOBS=1, closer to native baseline)"
        fi
        export MINIKUBE_USE_PARALLELISM
    fi
    
    # Read scenarios from manifest
    if [[ ! -f "$GENERATED_SCENARIOS_DIR/manifest.json" ]]; then
        log_error "Scenario manifest not found. Run without --skip-generation first."
        exit 1
    fi
    
    # Initialize progress tracking for this environment
    START_TIME=$(date +%s)
    LAST_PROGRESS_UPDATE=$(date +%s)
    
    # Count total scenarios for this environment
    PYTHON_CMD=$(get_python_cmd)
    MANIFEST_REL=$(to_relative_path "$GENERATED_SCENARIOS_DIR/manifest.json")
    ENV_TOTAL_SCENARIOS=$($PYTHON_CMD -c "
import json
with open('$MANIFEST_REL') as f:
    manifest = json.load(f)
count = sum(1 for s in manifest['scenarios'])
print(count)
")
    
    # Calculate actual number of experiments that will run (accounting for replicas)
    # This is different from scenario count because:
    # - Scaling experiments run with multiple replicas (1, 2, 4, 8)
    # - Non-scaling experiments run with replica 1 only
    # - Native environment only runs with replica 1
    # - For native: Only run-1 scenarios are processed (others handled by --runs parameter)
    PYTHON_CMD=$(get_python_cmd)
    MANIFEST_REL=$(to_relative_path "$GENERATED_SCENARIOS_DIR/manifest.json")
    ENV_TOTAL_EXPERIMENTS=$($PYTHON_CMD -c "
import json
import sys

with open('$MANIFEST_REL') as f:
    manifest = json.load(f)

env = '$env'
replicas = [int(r) for r in '$REPLICAS'.split(',')]
scaling_replicas = [r for r in replicas if r > 1]  # [2, 4, 8] if REPLICAS='1,2,4,8'

total_experiments = 0
seen_configs = set()  # Track unique configurations for native

for s in manifest['scenarios']:
    is_scaling = s.get('scaling_experiment', False)
    run_index = s.get('run_index', 1)
    
    if env == 'native' or env == 'minikube' or env == 'gcp':
        # For native, minikube, and GCP: Only count run-1 scenarios (others handled by --runs parameter)
        # Group by configuration to avoid counting duplicates
        # Each experiment invocation handles multiple runs internally, maintaining isolation
        # Include workload_pattern and duration_sec in config_key to distinguish burst/sustained experiments
        if run_index == 1:
            workload_pattern = s.get('workload_pattern', 'constant')
            duration_sec = s.get('duration_sec', 30)
            config_key = (s['algorithm'], s['payload_size'], s['rate'], is_scaling, workload_pattern, duration_sec)
            if config_key not in seen_configs:
                seen_configs.add(config_key)
                # Count experiments accounting for replicas
                # Baseline experiments: 1 replica each
                # Scaling experiments: multiple replicas (1, 2, 4, 8) = 4 experiments each
                if is_scaling:
                    # Scaling experiments run with all replicas
                    total_experiments += len(replicas)
                else:
                    # Baseline experiments run with replica 1 only
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
        
        # Use consistent cluster name for all test types and environments
        CLUSTER_NAME="pqc-bench"
        
        # Check if cluster already exists
        # Calculate node count for reasonable parallelism - Kubernetes will queue and schedule jobs
        # We submit ALL jobs at once, but only create enough nodes for reasonable parallelism
        # Kubernetes scheduler will queue jobs and schedule them as nodes become available
        # Each job needs ~1 CPU (800m request), n2-standard-2 has ~1.5 vCPUs available
        # So 1 job per node for optimal isolation
        # Add 1 extra node for system overhead
        if [[ $PARALLEL_JOBS -gt 1 ]]; then
            # For explicit parallelism, use PARALLEL_JOBS as a hint
            CALCULATED_NODE_COUNT=$((PARALLEL_JOBS + 1))
            log_info "Calculated node count: $CALCULATED_NODE_COUNT (based on PARALLEL_JOBS=$PARALLEL_JOBS)"
        else
            # For PARALLEL_JOBS=1, use a small but reasonable node count for queuing
            # This allows Kubernetes to queue jobs and schedule them as nodes become available
            # Without creating a massive cluster for all jobs to run simultaneously
            CALCULATED_NODE_COUNT="${NODE_COUNT:-3}"
            log_info "Using node count: $CALCULATED_NODE_COUNT (default for queuing, not all-at-once execution)"
            log_info "  All $ENV_TOTAL_EXPERIMENTS jobs will be submitted at once"
            log_info "  Kubernetes will queue and schedule them as nodes become available"
            log_info "  This provides reasonable parallelism without over-provisioning the cluster"
        fi
        
        # Check if cluster exists (suppress stderr to avoid noise, but capture exit code)
        CLUSTER_EXISTS=false
        if gcloud container clusters describe "$CLUSTER_NAME" \
            --region "$REGION" \
            --project "$PROJECT" &>/dev/null 2>&1; then
            CLUSTER_EXISTS=true
        fi
        
        if [[ "$CLUSTER_EXISTS" == "true" ]]; then
            log_warn "Cluster $CLUSTER_NAME already exists"
            
            # Detect the actual node pool name (now defaults to "default-pool" via Terraform)
            NODE_POOL_NAME=$(gcloud container node-pools list \
                --cluster "$CLUSTER_NAME" \
                --region "$REGION" \
                --project "$PROJECT" \
                --format="value(name)" 2>/dev/null | head -1 || echo "default-pool")
            
            if [[ -z "$NODE_POOL_NAME" ]]; then
                log_warn "Cluster exists but has no node pools. Creating node pool..."
                
                # Create node pool using Terraform (only the node pool, not the cluster)
                TERRAFORM_DIR="$SCRIPT_DIR/terraform/gke"
                cd "$TERRAFORM_DIR"
                
                # Initialize Terraform if needed
                if [[ ! -d ".terraform" ]]; then
                    log_info "Initializing Terraform..."
                    terraform init -input=false >/dev/null 2>&1 || {
                        log_error "Terraform initialization failed"
                        cd "$SCRIPT_DIR"
                        log_error "Please manually create a node pool or fix the cluster"
                        export GCP_USE_PERSISTENT_CLUSTER=false
                        export GCP_CLUSTER_EXISTS=false
                        continue
                    }
                fi
                
                # Import the existing cluster into Terraform state (if not already imported)
                log_info "Ensuring cluster is in Terraform state..."
                if ! terraform state show google_container_cluster.primary >/dev/null 2>&1; then
                    log_info "Importing existing cluster into Terraform state..."
                    terraform import google_container_cluster.primary "projects/${PROJECT}/locations/${REGION}/clusters/${CLUSTER_NAME}" >/dev/null 2>&1 || {
                        log_warn "Could not import cluster into Terraform state (may already exist)"
                    }
                fi
                
                # For Terraform with regional clusters, node_count is TOTAL nodes (not per zone)
                # However, based on quota errors, it seems Terraform might interpret it as per-zone
                # So we'll use gcloud directly which is more explicit about per-zone vs total
                # Calculate nodes per zone for gcloud (which requires per-zone specification)
                ZONES=$(gcloud compute regions describe "$REGION" --project "$PROJECT" --format="value(zones)" 2>/dev/null | tr ';' '\n' | wc -l || echo "3")
                NODES_PER_ZONE=$(( (CALCULATED_NODE_COUNT + ZONES - 1) / ZONES ))
                TOTAL_NODES=$((NODES_PER_ZONE * ZONES))
                log_info "Creating node pool with $NODES_PER_ZONE nodes per zone (total: $TOTAL_NODES nodes across $ZONES zones)..."
                
                # Note: With Terraform managing the default pool, we don't need to create a separate pool
                # The default pool is configured via node_config in the cluster resource
                # If autoscaling is needed, configure it via gcloud:
                #   gcloud container node-pools update default-pool --cluster <cluster> --region <region> \
                #     --enable-autoscaling --min-nodes <min> --max-nodes <max>
                log_info "Default node pool is managed by Terraform (configured in cluster resource)"
                log_info "If autoscaling is needed, configure it manually via gcloud commands"
                NODE_POOL_NAME="default-pool"
                
                cd "$SCRIPT_DIR"
                
                # After creating node pool, verify it exists and continue with scaling
                if [[ -n "$NODE_POOL_NAME" ]]; then
                    log_info "Node pool created: $NODE_POOL_NAME"
                    # Wait a moment for node pool to be ready
                    sleep 5
                fi
            fi
            
            # Now proceed with scaling if we have a node pool
            if [[ -n "$NODE_POOL_NAME" ]]; then
                log_info "Detected node pool: $NODE_POOL_NAME"
                
                # Always check and scale cluster to match calculated node count
                CURRENT_NODE_COUNT=$(gcloud container node-pools describe "$NODE_POOL_NAME" \
                    --cluster "$CLUSTER_NAME" \
                    --region "$REGION" \
                    --project "$PROJECT" \
                    --format="value(initialNodeCount)" 2>/dev/null || echo "0")
                
                # For regional clusters, --num-nodes is per zone, so we need to calculate nodes per zone
                # Get the number of zones the cluster spans
                # First try to get from cluster description
                CLUSTER_LOCATIONS=$(gcloud container clusters describe "$CLUSTER_NAME" \
                    --region "$REGION" \
                    --project "$PROJECT" \
                    --format="value(locations)" 2>/dev/null || echo "")
                
                if [[ -n "$CLUSTER_LOCATIONS" ]]; then
                    # Count zones from cluster locations
                    # Locations can be comma or semicolon separated
                    if echo "$CLUSTER_LOCATIONS" | grep -q ';'; then
                        ZONES=$(echo "$CLUSTER_LOCATIONS" | tr ';' '\n' | wc -l)
                    else
                        ZONES=$(echo "$CLUSTER_LOCATIONS" | tr ',' '\n' | wc -l)
                    fi
                    log_info "Cluster spans zones: $CLUSTER_LOCATIONS ($ZONES zones)"
                else
                    # Fallback: get zones from region description (more reliable)
                    ZONES=$(gcloud compute regions describe "$REGION" \
                        --project "$PROJECT" \
                        --format="value(zones)" 2>/dev/null | tr ';' '\n' | wc -l)
                    if [[ -z "$ZONES" ]] || [[ "$ZONES" -eq 0 ]]; then
                        # Final fallback: assume 3 zones for regional clusters
                        ZONES=3
                        log_warn "Could not detect zone count, assuming 3 zones (regional cluster default)"
                    else
                        log_info "Detected $ZONES zones in region $REGION"
                    fi
                fi
                
                # Calculate nodes per zone (round up to ensure we have enough total nodes)
                NODES_PER_ZONE=$(( (CALCULATED_NODE_COUNT + ZONES - 1) / ZONES ))
                TOTAL_NODES=$((NODES_PER_ZONE * ZONES))
                
                log_info "Calculated: $NODES_PER_ZONE nodes per zone × $ZONES zones = $TOTAL_NODES total nodes"
                
                # For Terraform regional clusters, node_count is PER ZONE, not total
                # So CURRENT_NODE_COUNT is already per-zone, multiply by zones to get total
                CURRENT_TOTAL_NODES=$((CURRENT_NODE_COUNT * ZONES))
                
                # Compare current total nodes with required total nodes
                # CALCULATED_NODE_COUNT is the total we want, convert to per-zone for Terraform
                if [[ "$CURRENT_TOTAL_NODES" -ne "$CALCULATED_NODE_COUNT" ]]; then
                    log_info "Current cluster state: $CURRENT_NODE_COUNT nodes per zone (total: $CURRENT_TOTAL_NODES nodes)"
                    log_info "Target cluster state: $CALCULATED_NODE_COUNT total nodes ($NODES_PER_ZONE per zone across $ZONES zones)"
                    log_info "Scaling cluster from $CURRENT_TOTAL_NODES to $CALCULATED_NODE_COUNT total nodes"
                    
                    # Estimate resource requirements for quota checking
                    # n2-standard-2 has 2 vCPUs and 8GB RAM per node
                    REQUIRED_CPUS=$((CALCULATED_NODE_COUNT * 2))
                    REQUIRED_DISK_GB=$((CALCULATED_NODE_COUNT * 50))  # 50GB per node (default disk size)
                    
                    log_info "Resource requirements: $REQUIRED_CPUS vCPUs, $REQUIRED_DISK_GB GB disk (for $CALCULATED_NODE_COUNT n2-standard-2 nodes)"
                    log_info "If scaling fails due to quota, reduce PARALLEL_JOBS or increase GCP quotas"
                    
                    # Use Terraform to scale the cluster (keeps state in sync)
                    log_info "Scaling cluster using Terraform (node_count=$NODES_PER_ZONE per zone = $CALCULATED_NODE_COUNT total)..."
                    TERRAFORM_DIR="$SCRIPT_DIR/terraform/gke"
                    cd "$TERRAFORM_DIR"
                    
                    # Initialize Terraform if needed
                    if [[ ! -d .terraform ]]; then
                        log_info "Initializing Terraform..."
                        terraform init -input=false >/dev/null 2>&1 || true
                    fi
                    
                    # Terraform now manages the default pool directly (configured in cluster resource)
                    # The default pool name is "default-pool" (GKE standard)
                    TERRAFORM_EXPECTED_POOL="default-pool"
                    
                    # Check if cluster is in Terraform state
                    CLUSTER_IN_STATE=$(terraform state list 2>/dev/null | grep -q "google_container_cluster.primary" && echo "yes" || echo "no")
                    
                    # Since default pool is managed via cluster resource, we don't check for separate node pool resource
                    # Terraform manages the default pool configuration via node_config in the cluster resource
                    USE_TERRAFORM_FOR_NODE_POOL=true
                    
                    # Always try to import cluster into Terraform state (for consistency)
                    if [[ "$CLUSTER_IN_STATE" == "no" ]]; then
                        log_info "Cluster exists but not in Terraform state. Importing..."
                        CLUSTER_RESOURCE_ID="projects/$PROJECT/locations/$REGION/clusters/$CLUSTER_NAME"
                        if terraform import \
                            -var="project_id=$PROJECT" \
                            -var="region=$REGION" \
                            -var="bucket_name=$BUCKET" \
                            -var="machine_type=${MACHINE_TYPE:-n2-standard-2}" \
                            -var="node_count=$CURRENT_TOTAL_NODES" \
                            -var="cluster_name=$CLUSTER_NAME" \
                            -var="smoke_test=$SMOKE_TEST" \
                            -var="disk_size_gb=${DISK_SIZE_GB:-50}" \
                            -var="ephemeral=false" \
                            google_container_cluster.primary "$CLUSTER_RESOURCE_ID" >/dev/null 2>&1; then
                            log_success "Cluster imported into Terraform state"
                        else
                            log_warn "Failed to import cluster into Terraform state (may already exist or be managed elsewhere)"
                        fi
                    fi
                    
                    # Only try to import/manage node pool with Terraform if name matches
                    if [[ "$USE_TERRAFORM_FOR_NODE_POOL" == "true" ]]; then
                        # Import node pool if it exists but isn't in state
                        if [[ "$NODE_POOL_IN_STATE" == "no" ]]; then
                            log_info "Node pool exists but not in Terraform state. Importing..."
                            NODE_POOL_RESOURCE_ID="projects/$PROJECT/locations/$REGION/clusters/$CLUSTER_NAME/nodePools/$NODE_POOL_NAME"
                            if terraform import \
                                -var="project_id=$PROJECT" \
                                -var="region=$REGION" \
                                -var="bucket_name=$BUCKET" \
                                -var="machine_type=${MACHINE_TYPE:-n2-standard-2}" \
                                -var="node_count=$CURRENT_TOTAL_NODES" \
                                -var="cluster_name=$CLUSTER_NAME" \
                                -var="smoke_test=$SMOKE_TEST" \
                                -var="disk_size_gb=${DISK_SIZE_GB:-50}" \
                                -var="ephemeral=false" \
                                google_container_node_pool.primary "$NODE_POOL_RESOURCE_ID" >/dev/null 2>&1; then
                                log_success "Node pool imported into Terraform state"
                            else
                                log_warn "Failed to import node pool. Will try to update anyway."
                            fi
                        fi
                    fi
                    
                    # Apply scaling - use Terraform if node pool name matches, otherwise use gcloud
                    # For regional clusters, Terraform's node_count is TOTAL nodes (distributed across zones)
                    set +e  # Don't exit on error, we'll handle it
                    if [[ "$USE_TERRAFORM_FOR_NODE_POOL" == "false" ]]; then
                        # Use gcloud for scaling when node pool name doesn't match Terraform expectation
                        log_info "Using gcloud to scale node pool '$NODE_POOL_NAME' (not managed by Terraform)"
                        SCALE_OUTPUT=$(timeout 300 gcloud container clusters resize "$CLUSTER_NAME" \
                            --node-pool "$NODE_POOL_NAME" \
                            --num-nodes "$NODES_PER_ZONE" \
                            --region "$REGION" \
                            --project "$PROJECT" \
                            --quiet 2>&1)
                        SCALE_EXIT_CODE=$?
                    else
                        # Use Terraform to update only the node pool (not the cluster)
                        # Terraform node_count is PER ZONE for regional clusters
                        SCALE_OUTPUT=$(timeout 600 terraform apply -auto-approve \
                            -target=google_container_node_pool.primary \
                            -var="project_id=$PROJECT" \
                            -var="region=$REGION" \
                            -var="bucket_name=$BUCKET" \
                            -var="machine_type=${MACHINE_TYPE:-n2-standard-2}" \
                            -var="node_count=$NODES_PER_ZONE" \
                            -var="cluster_name=$CLUSTER_NAME" \
                            -var="smoke_test=$SMOKE_TEST" \
                            -var="disk_size_gb=${DISK_SIZE_GB:-50}" \
                            -var="ephemeral=false" 2>&1)
                        SCALE_EXIT_CODE=$?
                    fi
                    set -e  # Re-enable exit on error
                    
                    cd "$SCRIPT_DIR"
                    
                    # Always log output for debugging
                    if [[ -n "$SCALE_OUTPUT" ]]; then
                        if [[ "$USE_TERRAFORM_FOR_NODE_POOL" == "false" ]]; then
                            log_info "gcloud scaling command output:"
                        else
                            log_info "Terraform apply output:"
                        fi
                        echo "$SCALE_OUTPUT" | tail -30 | while IFS= read -r line || [[ -n "$line" ]]; do
                            log_info "  $line"
                        done
                    fi
                    
                    if [[ $SCALE_EXIT_CODE -eq 0 ]]; then
                        if [[ "$USE_TERRAFORM_FOR_NODE_POOL" == "false" ]]; then
                            log_success "Cluster scaling initiated using gcloud"
                        else
                            log_success "Cluster scaled successfully using Terraform"
                        fi
                        log_info "Target: $CALCULATED_NODE_COUNT total nodes (distributed across $ZONES zones)"
                        log_info "Note: Actual scaling may take 5-10 minutes to complete"
                    elif [[ $SCALE_EXIT_CODE -eq 124 ]]; then
                        if [[ "$USE_TERRAFORM_FOR_NODE_POOL" == "false" ]]; then
                            log_warn "gcloud scaling command timed out after 300 seconds (5 minutes)"
                            log_info "This may indicate the operation is still in progress (scaling can take 5-10 minutes)"
                        else
                            log_warn "Terraform apply timed out after 600 seconds (10 minutes)"
                            log_info "This may indicate the operation is still in progress"
                        fi
                        
                        # Check if there are ongoing operations
                        log_info "Checking for ongoing cluster operations..."
                        ONGOING_OPS=$(gcloud container operations list \
                            --cluster "$CLUSTER_NAME" \
                            --region "$REGION" \
                            --project "$PROJECT" \
                            --filter="status=RUNNING" \
                            --format="value(name)" 2>/dev/null | wc -l | tr -d '[:space:]' || echo "0")
                        ONGOING_OPS=${ONGOING_OPS:-0}
                        
                        if [[ "$ONGOING_OPS" =~ ^[0-9]+$ ]] && [[ "$ONGOING_OPS" -gt 0 ]]; then
                            log_info "Found $ONGOING_OPS ongoing operation(s) - scaling is likely still in progress"
                            log_info "Waiting up to 10 minutes for scaling to complete..."
                            
                            # Wait for operations to complete (with timeout)
                            MAX_WAIT=600  # 10 minutes
                            WAIT_START=$(date +%s)
                            while [[ $(($(date +%s) - WAIT_START)) -lt $MAX_WAIT ]]; do
                                ONGOING_OPS=$(gcloud container operations list \
                                    --cluster "$CLUSTER_NAME" \
                                    --region "$REGION" \
                                    --project "$PROJECT" \
                                    --filter="status=RUNNING" \
                                    --format="value(name)" 2>/dev/null | wc -l | tr -d '[:space:]' || echo "0")
                                ONGOING_OPS=${ONGOING_OPS:-0}
                                
                                if [[ "$ONGOING_OPS" =~ ^[0-9]+$ ]] && [[ "$ONGOING_OPS" -eq 0 ]]; then
                                    log_success "Scaling operation completed"
                                    break
                                fi
                                
                                ELAPSED=$(($(date +%s) - WAIT_START))
                                if [[ $((ELAPSED % 60)) -eq 0 ]] && [[ $ELAPSED -gt 0 ]]; then
                                    log_info "Still waiting... ($((ELAPSED / 60)) minutes elapsed, $ONGOING_OPS operation(s) still running)"
                                fi
                                sleep 10
                            done
                            
                            if [[ $(($(date +%s) - WAIT_START)) -ge $MAX_WAIT ]]; then
                                log_warn "Timeout waiting for scaling to complete. Proceeding with current cluster state."
                            fi
                        else
                            log_info "No ongoing operations detected. Scaling may have completed or failed."
                        fi
                        
                        # CRITICAL: Wait for nodes to be fully Ready before submitting jobs
                        # This prevents "unschedulable" pods when jobs are submitted before nodes are ready
                        log_info "Waiting for nodes to be Ready before submitting jobs..."
                        MAX_NODE_WAIT=600  # 10 minutes
                        NODE_WAIT_START=$(date +%s)
                        READY_NODES=0
                        REQUIRED_READY_NODES=$CALCULATED_NODE_COUNT
                        
                        while [[ $(($(date +%s) - NODE_WAIT_START)) -lt $MAX_NODE_WAIT ]]; do
                            # Get ready nodes count (nodes in Ready state)
                            READY_NODES=$(kubectl get nodes --no-headers 2>/dev/null | grep -c " Ready " || echo "0")
                            READY_NODES=${READY_NODES:-0}
                            
                            if [[ "$READY_NODES" =~ ^[0-9]+$ ]] && [[ "$READY_NODES" -ge "$REQUIRED_READY_NODES" ]]; then
                                log_success "All $REQUIRED_READY_NODES nodes are Ready (found $READY_NODES ready nodes)"
                                break
                            fi
                            
                            ELAPSED=$(($(date +%s) - NODE_WAIT_START))
                            if [[ $((ELAPSED % 30)) -eq 0 ]] && [[ $ELAPSED -gt 0 ]]; then
                                log_info "Waiting for nodes to be Ready... ($READY_NODES/$REQUIRED_READY_NODES ready, $((ELAPSED / 60))m ${ELAPSED}s elapsed)"
                                # Show node status for debugging
                                kubectl get nodes --no-headers 2>/dev/null | head -5 | while IFS= read -r line || [[ -n "$line" ]]; do
                                    log_info "  Node: $line"
                                done || true
                            fi
                            sleep 5
                        done
                        
                        if [[ $(($(date +%s) - NODE_WAIT_START)) -ge $MAX_NODE_WAIT ]]; then
                            log_warn "Timeout waiting for nodes to be Ready. Found $READY_NODES/$REQUIRED_READY_NODES ready nodes."
                            log_warn "Proceeding anyway, but some pods may be unschedulable."
                        else
                            # Double-check nodes are actually ready
                            READY_NODES=$(kubectl get nodes --no-headers 2>/dev/null | grep -c " Ready " || echo "0")
                            if [[ "$READY_NODES" -lt "$REQUIRED_READY_NODES" ]]; then
                                log_warn "Only $READY_NODES nodes are Ready (expected $REQUIRED_READY_NODES). Some jobs may be unschedulable."
                            fi
                        fi
                        
                        log_info "Check cluster status with:"
                        log_info "  gcloud container node-pools describe $NODE_POOL_NAME --cluster $CLUSTER_NAME --region $REGION"
                        log_warn "Continuing with existing cluster size ($CURRENT_TOTAL_NODES nodes). Performance may be degraded."
                    else
                        if echo "$SCALE_OUTPUT" | grep -q "INSUFFICIENT_QUOTA\|quota"; then
                            log_error "Failed to scale cluster due to insufficient quota"
                            log_info "Required: $REQUIRED_CPUS vCPUs, $REQUIRED_DISK_GB GB disk (for $CALCULATED_NODE_COUNT n2-standard-2 nodes)"
                            
                            # Extract specific quota information from error message
                            if echo "$SCALE_OUTPUT" | grep -q "request requires.*short"; then
                                log_info ""
                                log_info "Quota details from error:"
                                echo "$SCALE_OUTPUT" | grep -E "request requires|short|quota of|available" | while IFS= read -r line || [[ -n "$line" ]]; do
                                    log_info "  $line"
                                done
                                
                                # Try to extract the shortfall amount and quota name
                                SHORTFALL=$(echo "$SCALE_OUTPUT" | grep -oP "short '\K[0-9.]+" | head -1 || echo "")
                                QUOTA_NAME=$(echo "$SCALE_OUTPUT" | grep -oP 'resource "\K[^"]+' | head -1 || echo "")
                                
                                if [[ -n "$SHORTFALL" ]]; then
                                    log_info ""
                                    log_info "You need to free up or increase quota by approximately $SHORTFALL GB"
                                    
                                    # Provide specific guidance based on quota type
                                    if echo "$SCALE_OUTPUT" | grep -q "SSD_TOTAL_GB"; then
                                        log_info ""
                                        log_info "To find 'SSD_TOTAL_GB' quota in GCP Console:"
                                        log_info "  1. Go to: https://console.cloud.google.com/iam-admin/quotas?usage=USED&project=$PROJECT"
                                        log_info "  2. Filter by:"
                                        log_info "     - Service: 'Compute Engine API'"
                                        log_info "     - Location: '$REGION' (regional quota)"
                                        log_info "  3. Search for: 'Persistent Disk SSD' or 'SSD_TOTAL_GB'"
                                        log_info "     (It may also appear as 'Persistent Disk SSD (GB)' or 'SSD persistent disk')"
                                        log_info "  4. Click on the quota and request an increase"
                                        log_info ""
                                        log_info "Alternative: Use gcloud to find the quota:"
                                        log_info "  gcloud compute project-info describe --project $PROJECT --format='table(quotas.metric,quotas.limit,quotas.usage)' | grep -i ssd"
                                    elif echo "$SCALE_OUTPUT" | grep -q "DISKS_TOTAL_GB"; then
                                        log_info ""
                                        log_info "To find 'DISKS_TOTAL_GB' quota in GCP Console:"
                                        log_info "  1. Go to: https://console.cloud.google.com/iam-admin/quotas?usage=USED&project=$PROJECT"
                                        log_info "  2. Filter by:"
                                        log_info "     - Service: 'Compute Engine API'"
                                        log_info "     - Location: '$REGION' (regional quota)"
                                        log_info "  3. Search for: 'Persistent Disk' or 'DISKS_TOTAL_GB'"
                                    fi
                                    
                                    log_info ""
                                    log_info "Check current disk usage:"
                                    log_info "  gcloud compute disks list --project $PROJECT --format='table(name,sizeGb,zone)'"
                                    log_info "Delete unused disks to free up quota, or request an increase at:"
                                    log_info "  https://console.cloud.google.com/iam-admin/quotas?usage=USED&project=$PROJECT"
                                fi
                            fi
                            
                            # Check if current cluster size is sufficient for desired parallelism
                            # Each node can handle ~1-2 jobs (conservative estimate: 1 job per node)
                            MAX_JOBS_WITH_CURRENT_CLUSTER=$CURRENT_TOTAL_NODES
                            
                            if [[ $PARALLEL_JOBS -gt $MAX_JOBS_WITH_CURRENT_CLUSTER ]]; then
                                log_error ""
                                log_error "CRITICAL: Cluster has only $CURRENT_TOTAL_NODES nodes but $PARALLEL_JOBS parallel jobs requested!"
                                log_error "Experiments will likely fail due to insufficient resources."
                                log_info ""
                                log_info "Immediate options:"
                                log_info "  1. Reduce PARALLEL_JOBS to $MAX_JOBS_WITH_CURRENT_CLUSTER or less:"
                                log_info "     PARALLEL_JOBS=$MAX_JOBS_WITH_CURRENT_CLUSTER ./run_full_scale_data_collection.sh --env gcp ..."
                                log_info "  2. Free up disk quota or increase GCP quotas:"
                                log_info "     https://console.cloud.google.com/iam-admin/quotas?usage=USED&project=$PROJECT"
                                log_info "  3. Use smaller machine type (currently: ${MACHINE_TYPE:-n2-standard-2})"
                                log_info ""
                                if [[ "$USE_TERRAFORM_FOR_NODE_POOL" == "false" ]]; then
                                    log_info "To scale manually after fixing quota:"
                                    log_info "  gcloud container clusters resize $CLUSTER_NAME --node-pool $NODE_POOL_NAME --num-nodes $NODES_PER_ZONE --region $REGION"
                                else
                                    log_info "To scale manually after fixing quota:"
                                    log_info "  cd terraform/gke && terraform apply -var='node_count=$NODES_PER_ZONE' -auto-approve"
                                fi
                                log_warn "Continuing with existing cluster size, but experiments may fail!"
                            else
                                log_warn "Cluster size ($CURRENT_TOTAL_NODES nodes) should be sufficient for $PARALLEL_JOBS parallel jobs"
                                log_info "Options to scale up later:"
                                log_info "  1. Free up disk quota or increase GCP quotas:"
                                log_info "     https://console.cloud.google.com/iam-admin/quotas?usage=USED&project=$PROJECT"
                                log_info "  2. Use smaller machine type (currently: ${MACHINE_TYPE:-n2-standard-2})"
                                log_warn "Continuing with existing cluster size. Performance may be degraded."
                            fi
                        else
                            if [[ "$USE_TERRAFORM_FOR_NODE_POOL" == "false" ]]; then
                                log_error "Failed to scale cluster with gcloud. Error:"
                            else
                                log_error "Failed to scale cluster with Terraform. Error:"
                            fi
                            echo "$SCALE_OUTPUT" | grep -i "error\|failed" | head -10
                            if [[ "$USE_TERRAFORM_FOR_NODE_POOL" == "false" ]]; then
                                log_info "To scale manually:"
                                log_info "  gcloud container clusters resize $CLUSTER_NAME --node-pool $NODE_POOL_NAME --num-nodes $NODES_PER_ZONE --region $REGION"
                            else
                                log_info "To scale manually using Terraform:"
                                log_info "  cd terraform/gke && terraform apply -var='node_count=$NODES_PER_ZONE' -auto-approve"
                            fi
                            log_warn "Continuing with existing cluster size ($CURRENT_TOTAL_NODES nodes). Performance may be degraded."
                        fi
                    fi
                else
                    log_info "Cluster already has $CURRENT_TOTAL_NODES total nodes (matches required $CALCULATED_NODE_COUNT)"
                fi
            else
                log_error "No node pool available. Cannot proceed with cluster scaling."
                export GCP_USE_PERSISTENT_CLUSTER=false
            fi
            
            # Verify bucket access before starting experiments
            log_info "Verifying GCS bucket access..."
            if gsutil ls -b "gs://${BUCKET}" &>/dev/null; then
                log_success "Bucket gs://${BUCKET} is accessible"
            else
                log_error "Cannot access bucket gs://${BUCKET}"
                log_info "Please verify:"
                log_info "  1. Bucket exists: gsutil ls -b gs://${BUCKET}"
                log_info "  2. You have permissions: gsutil iam get gs://${BUCKET}"
                log_info "  3. Service account has access (if using Workload Identity)"
                export GCP_USE_PERSISTENT_CLUSTER=false
            fi
            
            # CRITICAL: Ensure cluster is in RUNNING state before proceeding
            log_info "Verifying cluster is in RUNNING state..."
            CLUSTER_STATUS=$(gcloud container clusters describe "$CLUSTER_NAME" \
                --region "$REGION" \
                --project "$PROJECT" \
                --format="value(status)" 2>/dev/null || echo "")
            
            if [[ "$CLUSTER_STATUS" != "RUNNING" ]]; then
                log_warn "Cluster status is '$CLUSTER_STATUS' (expected: RUNNING)"
                
                if [[ "$CLUSTER_STATUS" == "RECONCILING" ]] || [[ "$CLUSTER_STATUS" == "PROVISIONING" ]]; then
                    log_info "Cluster is being provisioned/reconciled. Waiting up to 10 minutes for RUNNING state..."
                    MAX_WAIT=600  # 10 minutes
                    WAIT_START=$(date +%s)
                    while [[ $(($(date +%s) - WAIT_START)) -lt $MAX_WAIT ]]; do
                        CLUSTER_STATUS=$(gcloud container clusters describe "$CLUSTER_NAME" \
                            --region "$REGION" \
                            --project "$PROJECT" \
                            --format="value(status)" 2>/dev/null || echo "")
                        
                        if [[ "$CLUSTER_STATUS" == "RUNNING" ]]; then
                            log_success "Cluster is now RUNNING"
                            break
                        fi
                        
                        ELAPSED=$(($(date +%s) - WAIT_START))
                        if [[ $((ELAPSED % 60)) -eq 0 ]] && [[ $ELAPSED -gt 0 ]]; then
                            log_info "Still waiting... ($((ELAPSED / 60)) minutes elapsed, status: $CLUSTER_STATUS)"
                        fi
                        sleep 15
                    done
                fi
                
                # Final check
                CLUSTER_STATUS=$(gcloud container clusters describe "$CLUSTER_NAME" \
                    --region "$REGION" \
                    --project "$PROJECT" \
                    --format="value(status)" 2>/dev/null || echo "")
                
                if [[ "$CLUSTER_STATUS" != "RUNNING" ]]; then
                    log_error "Cluster is not in RUNNING state (current: $CLUSTER_STATUS)"
                    log_error "Cannot proceed with experiments. Cluster must be RUNNING."
                    log_info "Check cluster status:"
                    log_info "  gcloud container clusters describe $CLUSTER_NAME --region $REGION --project $PROJECT"
                    log_info "Wait for cluster to be RUNNING, then retry."
                    log_info "If cluster is stuck, you may need to delete and recreate it:"
                    log_info "  gcloud container clusters delete $CLUSTER_NAME --region $REGION --project $PROJECT"
                    exit 1
                fi
            else
                log_success "Cluster is RUNNING"
            fi
            
            # CRITICAL: Configure kubectl credentials BEFORE checking node readiness
            # This ensures we're checking the correct cluster's nodes, not Minikube
            log_info "Configuring kubectl credentials..."
            if ! gcloud container clusters get-credentials "$CLUSTER_NAME" \
                --region "$REGION" \
                --project "$PROJECT" 2>&1; then
                log_error "Failed to configure kubectl credentials"
                log_info "Please ensure you have access to the cluster:"
                log_info "  gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project $PROJECT"
                exit 1
            fi
            
            # Verify kubectl connectivity
            log_info "Verifying kubectl connectivity..."
            if ! kubectl cluster-info &>/dev/null; then
                log_error "Cannot connect to Kubernetes cluster via kubectl"
                log_info "This may be due to:"
                log_info "  1. Private cluster endpoint (ensure enable_private_endpoint=false or configure authorized networks)"
                log_info "  2. Network connectivity issues"
                log_info "  3. Authentication problems"
                exit 1
            fi
            log_success "kubectl is connected to cluster"
            
            # Now verify nodes are ready (kubectl context is correctly set to GCP cluster)
            if [[ -n "$CURRENT_TOTAL_NODES" ]] && [[ "$CURRENT_TOTAL_NODES" -gt 0 ]]; then
                log_info "Verifying nodes are Ready before submitting jobs..."
                READY_NODES=$(kubectl get nodes --no-headers 2>/dev/null | grep -c " Ready " 2>/dev/null || echo "0")
                READY_NODES=$(echo "$READY_NODES" | tr -d '\n' | xargs)
                READY_NODES=${READY_NODES:-0}
                
                if [[ "$READY_NODES" -lt "$CURRENT_TOTAL_NODES" ]]; then
                    log_warn "Only $READY_NODES/$CURRENT_TOTAL_NODES nodes are Ready. Waiting up to 5 minutes for nodes to be ready..."
                    MAX_NODE_WAIT=300  # 5 minutes
                    NODE_WAIT_START=$(date +%s)
                    
                    while [[ $(($(date +%s) - NODE_WAIT_START)) -lt $MAX_NODE_WAIT ]]; do
                        READY_NODES=$(kubectl get nodes --no-headers 2>/dev/null | grep -c " Ready " 2>/dev/null || echo "0")
                        READY_NODES=$(echo "$READY_NODES" | tr -d '\n' | xargs)
                        if [[ "$READY_NODES" -ge "$CURRENT_TOTAL_NODES" ]]; then
                            log_success "All $CURRENT_TOTAL_NODES nodes are Ready"
                            break
                        fi
                        sleep 10
                    done
                    
                    if [[ "$READY_NODES" -lt "$CURRENT_TOTAL_NODES" ]]; then
                        log_warn "Only $READY_NODES/$CURRENT_TOTAL_NODES nodes are Ready. Some jobs may be unschedulable."
                    fi
                else
                    log_success "All $CURRENT_TOTAL_NODES nodes are Ready"
                fi
            fi
            
            log_info "Will reuse existing cluster (use --skip-gcp and destroy manually if needed)"
            export GCP_USE_PERSISTENT_CLUSTER=true
            export GCP_CLUSTER_EXISTS=true
        else
            # Cluster doesn't exist - create it
            log_info "Cluster $CLUSTER_NAME does not exist"
            log_info "Creating persistent cluster for batch run..."
            # For Terraform regional clusters, node_count is PER ZONE, not total
            # We need to calculate nodes per zone from total nodes
            # Example: 45 total nodes ÷ 3 zones = 15 nodes per zone
            ZONES=$(gcloud compute regions describe "$REGION" --project "$PROJECT" --format="value(zones)" 2>/dev/null | tr ';' '\n' | wc -l || echo "3")
            NODES_PER_ZONE=$(( (CALCULATED_NODE_COUNT + ZONES - 1) / ZONES ))
            TOTAL_NODES=$((NODES_PER_ZONE * ZONES))
            log_info "Will create cluster with $TOTAL_NODES total nodes ($NODES_PER_ZONE per zone across $ZONES zones)"
            log_info "  Terraform node_count=$NODES_PER_ZONE (per zone) → $TOTAL_NODES total nodes"
            if "$SCRIPT_DIR/deploy_gcp.sh" \
                --create-cluster \
                --project "$PROJECT" \
                --bucket "$BUCKET" \
                --region "$REGION" \
                --machine-type "${MACHINE_TYPE:-n2-standard-2}" \
                --node-count "$NODES_PER_ZONE" \
                $([ "$SMOKE_TEST" == "true" ] && echo "--smoke-test" || echo ""); then
                log_success "Cluster created successfully"
                
                # Wait for cluster to be RUNNING
                log_info "Waiting for cluster to be RUNNING..."
                MAX_WAIT=900  # 15 minutes for new cluster
                WAIT_START=$(date +%s)
                while [[ $(($(date +%s) - WAIT_START)) -lt $MAX_WAIT ]]; do
                    CLUSTER_STATUS=$(gcloud container clusters describe "$CLUSTER_NAME" \
                        --region "$REGION" \
                        --project "$PROJECT" \
                        --format="value(status)" 2>/dev/null || echo "")
                    
                    if [[ "$CLUSTER_STATUS" == "RUNNING" ]]; then
                        log_success "Cluster is RUNNING"
                        break
                    fi
                    
                    ELAPSED=$(($(date +%s) - WAIT_START))
                    if [[ $((ELAPSED % 60)) -eq 0 ]] && [[ $ELAPSED -gt 0 ]]; then
                        log_info "Still waiting... ($((ELAPSED / 60)) minutes elapsed, status: $CLUSTER_STATUS)"
                    fi
                    sleep 15
                done
                
                # Final check
                CLUSTER_STATUS=$(gcloud container clusters describe "$CLUSTER_NAME" \
                    --region "$REGION" \
                    --project "$PROJECT" \
                    --format="value(status)" 2>/dev/null || echo "")
                
                if [[ "$CLUSTER_STATUS" != "RUNNING" ]]; then
                    log_error "Cluster creation completed but cluster is not RUNNING (status: $CLUSTER_STATUS)"
                    log_error "Cannot proceed with experiments."
                    exit 1
                fi
                
                # Configure kubectl credentials for the newly created cluster
                log_info "Configuring kubectl credentials for new cluster..."
                if ! gcloud container clusters get-credentials "$CLUSTER_NAME" \
                    --region "$REGION" \
                    --project "$PROJECT" 2>&1; then
                    log_error "Failed to configure kubectl credentials"
                    exit 1
                fi
                
                # Verify kubectl connectivity
                log_info "Verifying kubectl connectivity..."
                if ! kubectl cluster-info &>/dev/null; then
                    log_error "Cannot connect to Kubernetes cluster via kubectl"
                    log_info "This may be due to private cluster endpoint configuration"
                    exit 1
                fi
                log_success "kubectl is connected to cluster"
                
                # CRITICAL: Wait for nodes to be Ready before proceeding
                # Newly created clusters need time for nodes to become Ready
                log_info "Waiting for nodes to be Ready after cluster creation..."
                MAX_NODE_WAIT=600  # 10 minutes
                NODE_WAIT_START=$(date +%s)
                READY_NODES=0
                REQUIRED_READY_NODES=$CALCULATED_NODE_COUNT
                
                while [[ $(($(date +%s) - NODE_WAIT_START)) -lt $MAX_NODE_WAIT ]]; do
                    # Get ready nodes count (nodes in Ready state)
                    READY_NODES=$(kubectl get nodes --no-headers 2>/dev/null | grep -c " Ready " || echo "0")
                    READY_NODES=${READY_NODES:-0}
                    
                    if [[ "$READY_NODES" =~ ^[0-9]+$ ]] && [[ "$READY_NODES" -ge "$REQUIRED_READY_NODES" ]]; then
                        log_success "All $REQUIRED_READY_NODES nodes are Ready (found $READY_NODES ready nodes)"
                        break
                    fi
                    
                    ELAPSED=$(($(date +%s) - NODE_WAIT_START))
                    if [[ $((ELAPSED % 30)) -eq 0 ]] && [[ $ELAPSED -gt 0 ]]; then
                        log_info "Waiting for nodes to be Ready... ($READY_NODES/$REQUIRED_READY_NODES ready, $((ELAPSED / 60))m ${ELAPSED}s elapsed)"
                        # Show node status for debugging
                        kubectl get nodes --no-headers 2>/dev/null | head -5 | while IFS= read -r line || [[ -n "$line" ]]; do
                            log_info "  Node: $line"
                        done || true
                    fi
                    sleep 5
                done
                
                if [[ $(($(date +%s) - NODE_WAIT_START)) -ge $MAX_NODE_WAIT ]]; then
                    log_warn "Timeout waiting for nodes to be Ready. Found $READY_NODES/$REQUIRED_READY_NODES ready nodes."
                    log_warn "Proceeding anyway, but some pods may be unschedulable."
                else
                    # Double-check nodes are actually ready
                    READY_NODES=$(kubectl get nodes --no-headers 2>/dev/null | grep -c " Ready " || echo "0")
                    if [[ "$READY_NODES" -lt "$REQUIRED_READY_NODES" ]]; then
                        log_warn "Only $READY_NODES nodes are Ready (expected $REQUIRED_READY_NODES). Some jobs may be unschedulable."
                    fi
                fi
                
                export GCP_USE_PERSISTENT_CLUSTER=true
                export GCP_CLUSTER_EXISTS=false  # We just created it
            else
                log_error "Failed to create persistent cluster for batch run"
                log_warn "Falling back to ephemeral mode (creates/destroys cluster per experiment)"
                log_info "This will be slower but should work if cluster creation issues are transient"
                export GCP_USE_PERSISTENT_CLUSTER=false
                export GCP_CLUSTER_EXISTS=false
            fi
        fi
        
        # Verify cluster exists and is ready (only for persistent cluster mode)
        if [[ "${GCP_USE_PERSISTENT_CLUSTER:-false}" == "true" ]]; then
            # Double-check cluster actually exists and is accessible
            if ! gcloud container clusters describe "$CLUSTER_NAME" \
                --region "$REGION" \
                --project "$PROJECT" &>/dev/null 2>&1; then
                log_error "Cluster $CLUSTER_NAME does not exist despite GCP_USE_PERSISTENT_CLUSTER=true"
                log_warn "Falling back to ephemeral mode for this batch run"
                export GCP_USE_PERSISTENT_CLUSTER=false
                export GCP_CLUSTER_EXISTS=false
            else
                # Verify cluster is RUNNING
                CLUSTER_STATUS=$(gcloud container clusters describe "$CLUSTER_NAME" \
                    --region "$REGION" \
                    --project "$PROJECT" \
                    --format="value(status)" 2>/dev/null || echo "")
                
                if [[ "$CLUSTER_STATUS" != "RUNNING" ]]; then
                    log_warn "Cluster $CLUSTER_NAME exists but is not RUNNING (status: $CLUSTER_STATUS)"
                    log_warn "Falling back to ephemeral mode for this batch run"
                    export GCP_USE_PERSISTENT_CLUSTER=false
                    export GCP_CLUSTER_EXISTS=false
                else
                    log_success "Cluster $CLUSTER_NAME verified: exists and is RUNNING"
                fi
            fi
        fi
        
        # Build image once for the batch (if cluster exists or was just created)
        if [[ "${GCP_USE_PERSISTENT_CLUSTER:-false}" == "true" ]]; then
            log_info "Building container image once for batch run..."
            # Get first scenario for image build
            PYTHON_CMD=$(get_python_cmd)
            MANIFEST_REL=$(to_relative_path "$GENERATED_SCENARIOS_DIR/manifest.json")
            FIRST_SCENARIO=$($PYTHON_CMD -c "
import json
with open('$MANIFEST_REL') as f:
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
                    --skip-job \
                    $([ "$SMOKE_TEST" == "true" ] && echo "--smoke-test" || echo "") \
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
    # All jobs are submitted immediately, then we wait for all to complete
    # Kubernetes scheduler determines parallelism based on available nodes
    if [[ "$env" == "gcp" ]] && [[ "${GCP_USE_PERSISTENT_CLUSTER:-false}" == "true" ]]; then
        TEMP_DIR=$(mktemp -d)
        export TEMP_DIR
        export JOB_TRACKING_FILE="${TEMP_DIR}/gcp_jobs_${env}.txt"
        > "$JOB_TRACKING_FILE"  # Create empty file
        log_info "All jobs will be submitted immediately; Kubernetes scheduler will determine parallelism based on available nodes"
    fi
    
    # Process scenarios
    scenario_count=0
    experiment_count=0  # Track actual experiments (not scenarios)
    env_completed=0
    env_failed=0
    env_skipped=0
    
    # Extract scenarios using Python for reliable JSON parsing
    # Include scaling_experiment flag (defaults to False if not present)
    # For native environment: Only process run-1 scenarios and run with --runs parameter
    # This groups multiple runs into single experiments, reducing 468 scenarios to ~94 experiments
    PYTHON_CMD=$(get_python_cmd)
    MANIFEST_REL=$(to_relative_path "$GENERATED_SCENARIOS_DIR/manifest.json")
    if [[ "$env" == "native" ]] || [[ "$env" == "minikube" ]] || [[ "$env" == "gcp" ]]; then
        # For native, minikube, and GCP: Only process run-1 scenarios (will run with --runs parameter)
        # This groups multiple runs into single experiments, reducing overhead while maintaining isolation
        # Each run still creates a separate Kubernetes Job (for minikube/GCP) or process execution (for native)
        scenarios=$($PYTHON_CMD -c "
import json
with open('$MANIFEST_REL') as f:
    manifest = json.load(f)
seen_configs = set()
for s in manifest['scenarios']:
    # Only process run-1 scenarios (others will be handled by --runs parameter)
    run_index = s.get('run_index', 1)
    if run_index == 1:
        # Create unique config key to avoid duplicates
        # Include workload_pattern and duration_sec to distinguish burst/sustained experiments
        workload_pattern = s.get('workload_pattern', 'constant')
        duration_sec = s.get('duration_sec', 30)
        config_key = (s['algorithm'], s['payload_size'], s['rate'], s.get('scaling_experiment', False), workload_pattern, duration_sec)
        if config_key not in seen_configs:
            seen_configs.add(config_key)
            scaling = s.get('scaling_experiment', False)
            total_runs = s.get('total_runs', 1)
            print(f\"{s['id']}|{s['path']}|{s['algorithm']}|{s['payload_size']}|{s['rate']}|{scaling}|{total_runs}\")
")
    else
        # For other environments: Process all scenarios as before
        scenarios=$($PYTHON_CMD -c "
import json
with open('$MANIFEST_REL') as f:
    manifest = json.load(f)
for s in manifest['scenarios']:
    scaling = s.get('scaling_experiment', False)
    total_runs = s.get('total_runs', 1)
    print(f\"{s['id']}|{s['path']}|{s['algorithm']}|{s['payload_size']}|{s['rate']}|{scaling}|{total_runs}\")
")
    fi
    
    while IFS='|' read -r scenario_id scenario_path algorithm payload rate is_scaling total_runs; do
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
            # Use base experiment ID (without run_index) for output directory
            # This ensures consistency: one experiment directory = multiple runs internally
            base_experiment_id=$(extract_base_experiment_id "$scenario_id")
            
            if [[ "$replica_count" -gt 1 ]]; then
                output_dir="$RESULTS_BASE/$env/${base_experiment_id}_r${replica_count}"
                run_scenario_id="${base_experiment_id}_r${replica_count}"
            else
                output_dir="$RESULTS_BASE/$env/$base_experiment_id"
                run_scenario_id="$base_experiment_id"
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
            # For GCP: also check GCS bucket for existing results
            # Note: For experiments with multiple runs, check for aggregated stats or any run data
            is_complete=false
            raw_file="$output_dir/raw/run.jsonl"
            stats_file="$output_dir/stats/summary.json"
            merged_file="$output_dir/merged/merged.jsonl"
            aggregated_file="$output_dir/aggregated_stats.json"
            
            # For multi-run experiments, check if ALL runs have raw data
            if [[ "$total_runs" -gt 1 ]]; then
                all_runs_complete=true
                for ((run_idx = 1; run_idx <= total_runs; run_idx++)); do
                    run_raw_file="$output_dir/run-${run_idx}/raw/run.jsonl"
                    if [[ ! -f "$run_raw_file" ]] || [[ ! -s "$run_raw_file" ]]; then
                        all_runs_complete=false
                        break
                    fi
                done
                if [[ "$all_runs_complete" == "true" ]]; then
                    # All runs have raw data - experiment is complete (even if analysis hasn't run)
                    is_complete=true
                fi
            fi
            
            # For GCP: check GCS bucket first (results are stored there, not locally)
            # BUCKET is set as a parameter and should be available in this scope
            if [[ "$env" == "gcp" ]] && [[ -n "${BUCKET:-}" ]]; then
                # For multi-run experiments in persistent cluster mode, each run has its own GCS directory
                # Format: gs://bucket/experiments/<base_id>_run<N>/
                # For single-run or ephemeral mode: gs://bucket/experiments/<base_id>/
                if [[ "$total_runs" -gt 1 ]] && [[ "${GCP_USE_PERSISTENT_CLUSTER:-false}" == "true" ]]; then
                    # Check each individual run in GCS
                    all_runs_in_gcs=true
                    for ((run_idx = 1; run_idx <= total_runs; run_idx++)); do
                        RUN_GCS_PATH="gs://${BUCKET}/experiments/${run_scenario_id}_run${run_idx}"
                        if ! gsutil -q ls "$RUN_GCS_PATH/raw/run.jsonl" &>/dev/null 2>&1; then
                            all_runs_in_gcs=false
                            break
                        fi
                    done
                    if [[ "$all_runs_in_gcs" == "true" ]]; then
                        is_complete=true
                        log_info "  Found all ${total_runs} runs in GCS for $run_scenario_id"
                    fi
                else
                    # Single-run or ephemeral mode: check base experiment path
                    GCS_EXP_PATH="gs://${BUCKET}/experiments/${run_scenario_id}"
                    
                    # Check if experiment exists in GCS (quiet check to avoid noise)
                    if gsutil -q ls "$GCS_EXP_PATH/merged.jsonl" &>/dev/null 2>&1; then
                        # Merged file exists in GCS
                        is_complete=true
                        log_info "  Found existing results in GCS for $run_scenario_id (merged.jsonl)"
                    elif gsutil -q ls "$GCS_EXP_PATH/raw/run.jsonl" &>/dev/null 2>&1; then
                        # Raw file exists in GCS
                        is_complete=true
                        log_info "  Found existing results in GCS for $run_scenario_id (raw/run.jsonl)"
                    fi
                fi
            fi
            
            # Also check local files (for all environments, including GCP if downloaded)
            if [[ "$is_complete" != "true" ]]; then
                if [[ "$SKIP_ANALYSIS" == "true" ]]; then
                    # In data collection mode: check for raw data
                    # For multi-run experiments, check if ALL runs have raw data
                    if [[ "$total_runs" -gt 1 ]]; then
                        all_runs_have_data=true
                        for ((run_idx = 1; run_idx <= total_runs; run_idx++)); do
                            run_raw_file="$output_dir/run-${run_idx}/raw/run.jsonl"
                            if [[ ! -f "$run_raw_file" ]] || [[ ! -s "$run_raw_file" ]]; then
                                all_runs_have_data=false
                                break
                            fi
                        done
                        if [[ "$all_runs_have_data" == "true" ]]; then
                            is_complete=true
                        fi
                    elif [[ -f "$raw_file" ]] && [[ -s "$raw_file" ]]; then
                        is_complete=true
                    fi
                else
                    # In full mode: check for analysis outputs first
                    if [[ -f "$stats_file" ]] && [[ -s "$stats_file" ]]; then
                        is_complete=true
                    elif [[ -f "$merged_file" ]] && [[ -s "$merged_file" ]]; then
                        is_complete=true
                    elif [[ -f "$aggregated_file" ]]; then
                        # Aggregated stats exist (multi-run experiment completed)
                        is_complete=true
                    elif [[ "$total_runs" -gt 1 ]] && [[ -f "$output_dir/run-1/raw/run.jsonl" ]]; then
                        # Multi-run experiment: raw data exists but analysis hasn't run
                        log_info "  Found raw data for $run_scenario_id (multi-run experiment), skipping benchmark run, will complete analysis only"
                        # Mark that we need to run analysis only (don't mark as complete yet)
                        # We'll handle this after the check block
                    elif [[ -f "$raw_file" ]] && [[ -s "$raw_file" ]]; then
                        # Single-run experiment: raw data exists but analysis hasn't run
                        log_info "  Found raw data for $run_scenario_id, skipping benchmark run, will complete analysis only"
                        # Mark that we need to run analysis only (don't mark as complete yet)
                        # We'll handle this after the check block
                    fi
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
                    # In full mode and no raw file - directory exists but no data at all
                    log_warn "  Found incomplete results for $run_scenario_id (no raw data), will re-run"
                    rm -rf "$output_dir"
                elif [[ "$SKIP_ANALYSIS" != "true" ]] && [[ -f "$raw_file" ]] && [[ -s "$raw_file" ]]; then
                    # Raw data exists but analysis hasn't run - skip benchmark, run analysis only
                    log_info "  Found raw data for $run_scenario_id, skipping benchmark run, will complete analysis only (data collection already done)"
                    # Mark that we need to run analysis only (don't mark as complete yet)
                    # We'll handle this after the check block
                fi
            fi
            
            # Check if we should skip benchmark run and only run analysis
            # This happens when raw data exists but analysis hasn't been completed
            SKIP_BENCHMARK_RUN=false
            if [[ "$SKIP_ANALYSIS" != "true" ]] && [[ -f "$raw_file" ]] && [[ -s "$raw_file" ]]; then
                # Check if analysis is already complete
                if [[ ! -f "$stats_file" ]] && [[ ! -f "$merged_file" ]]; then
                    # Raw data exists but analysis not done - skip benchmark, run analysis only
                    SKIP_BENCHMARK_RUN=true
                fi
            fi
            
            if [[ "$SKIP_BENCHMARK_RUN" == "true" ]]; then
                # Skip benchmark run, but run analysis pipeline
                log_info "  Skipping benchmark run for $run_scenario_id (raw data exists), running analysis only..."
                update_progress $env_completed $ENV_TOTAL_EXPERIMENTS "$env" "$run_scenario_id (analysis only)"
                
                # Run analysis pipeline directly
                if run_analysis_pipeline "$output_dir" "$run_scenario_id" "$SKIP_ANALYSIS"; then
                    # Validate that analysis produced expected outputs
                    if [[ -f "$stats_file" ]] || [[ -f "$merged_file" ]]; then
                        log_success "  Completed analysis for $run_scenario_id (replicas: $replica_count)"
                        add_to_index "$run_scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "success" "$replica_count"
                        env_completed=$((env_completed + 1))
                        COMPLETED_SCENARIOS=$((COMPLETED_SCENARIOS + 1))
                        update_progress $env_completed $ENV_TOTAL_EXPERIMENTS "$env" "$run_scenario_id"
                    else
                        log_warn "  Analysis completed but expected outputs not found for $run_scenario_id"
                        add_to_index "$run_scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "partial" "$replica_count"
                        env_completed=$((env_completed + 1))
                    fi
                else
                    log_error "  Analysis pipeline failed for $run_scenario_id"
                    add_to_index "$run_scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "failed" "$replica_count"
                    env_failed=$((env_failed + 1))
                    FAILED_SCENARIOS=$((FAILED_SCENARIOS + 1))
                fi
                continue
            fi
            
            # Run experiment (benchmark + analysis)
            # For GCP with persistent cluster: all jobs are submitted immediately (non-blocking), then waited on at the end
            # For Minikube with parallelism: jobs are submitted immediately (non-blocking), then waited on at the end
            # For other environments: jobs run sequentially
            update_progress $env_completed $ENV_TOTAL_EXPERIMENTS "$env" "$run_scenario_id"
            
            # Pass total_runs to run_experiment function (defaults to 5 if not set or empty)
            if run_experiment "$env" "$scenario_path" "$run_scenario_id" "$output_dir" "$replica_count" "${total_runs:-5}"; then
                # Validate data integrity immediately after collection
                # NOTE: For GCP persistent cluster mode and Minikube parallel mode, skip this check - validation happens later
                if [[ "$env" == "gcp" ]] && [[ "${GCP_USE_PERSISTENT_CLUSTER:-false}" == "true" ]]; then
                    # For GCP persistent cluster, all jobs are submitted and tracked, then waited on at the end
                    # Just mark as submitted for now
                    log_info "  Submitted: $run_scenario_id (will be validated after job completion)"
                elif [[ "$env" == "minikube" ]] && [[ "${MINIKUBE_USE_PARALLELISM:-false}" == "true" ]]; then
                    # For Minikube parallel mode, all jobs are submitted and tracked, then waited on at the end
                    # Just mark as submitted for now
                    log_info "  Submitted: $run_scenario_id (will be validated after job completion)"
                else
                    # For sequential jobs or non-GCP, check immediately
                    # For multi-run experiments, check for aggregated stats or run-1 data
                    if [[ "$total_runs" -gt 1 ]]; then
                        # Multi-run experiment: check for aggregated stats or first run data
                        if [[ -f "$output_dir/aggregated_stats.json" ]]; then
                            log_success "  Completed: $run_scenario_id (replicas: $replica_count) - multi-run experiment with aggregated stats"
                        elif [[ -f "$output_dir/run-1/raw/run.jsonl" ]]; then
                            file_size=$(stat -f%z "$output_dir/run-1/raw/run.jsonl" 2>/dev/null || stat -c%s "$output_dir/run-1/raw/run.jsonl" 2>/dev/null || echo 0)
                            if [[ $file_size -gt 0 ]]; then
                                log_success "  Completed: $run_scenario_id (replicas: $replica_count) - multi-run experiment, $file_size bytes in run-1"
                            else
                                log_error "  Data integrity check FAILED: $run_scenario_id has 0-byte file in run-1!"
                                add_to_index "$run_scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "failed" "$replica_count"
                                env_failed=$((env_failed + 1))
                                FAILED_SCENARIOS=$((FAILED_SCENARIOS + 1))
                                continue
                            fi
                        else
                            log_error "  Data integrity check FAILED: $run_scenario_id - no data found for multi-run experiment!"
                            add_to_index "$run_scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "failed" "$replica_count"
                            env_failed=$((env_failed + 1))
                            FAILED_SCENARIOS=$((FAILED_SCENARIOS + 1))
                            continue
                        fi
                    else
                        # Single-run experiment: check for raw file
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
                    fi
                
                    add_to_index "$run_scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "success" "$replica_count"
                    env_completed=$((env_completed + 1))
                    COMPLETED_SCENARIOS=$((COMPLETED_SCENARIOS + 1))
                    # Update progress with completed count after successful run
                    update_progress $env_completed $ENV_TOTAL_EXPERIMENTS "$env" "$run_scenario_id"
                fi  # End of else block for non-parallel GCP jobs
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
    
    # For GCP execution with persistent cluster, wait for all jobs to complete
    # All jobs were submitted immediately; now wait for all to finish
    if [[ "$env" == "gcp" ]] && [[ "${GCP_USE_PERSISTENT_CLUSTER:-false}" == "true" ]] && [[ -f "${JOB_TRACKING_FILE:-}" ]]; then
        TOTAL_SUBMITTED=$(wc -l < "$JOB_TRACKING_FILE" 2>/dev/null || echo "0")
        if [[ $TOTAL_SUBMITTED -gt 0 ]]; then
            log_info "Waiting for all $TOTAL_SUBMITTED submitted jobs to complete (Kubernetes scheduler determines parallelism)..."
        
        # Use consistent namespace for all test types
        GCP_NAMESPACE="default"
        
        # Track job completion
        TOTAL_JOBS=$(wc -l < "$JOB_TRACKING_FILE" 2>/dev/null || echo "0")
        COMPLETED_JOBS=0
        FAILED_JOBS=0
        
        while IFS='|' read -r job_name scenario_id output_dir; do
            # Extract algorithm, payload, rate, and replica count from scenario_id or output_dir
            # Format: <algorithm>_p<payload>_r<rate>_run<N>_<hash> or <algorithm>_p<payload>_r<rate>_run<N>_<hash>_r<replicas>
            algorithm="unknown"
            payload=0
            rate=0
            replica_count=1
            
            # Parse scenario_id: <algorithm>_p<payload>_r<rate>_run<N>_<hash> or with _r<replicas> suffix
            # Remove replica suffix if present to get base scenario_id for parsing
            base_scenario_id="$scenario_id"
            if [[ "$scenario_id" =~ _r([0-9]+)$ ]]; then
                replica_count="${BASH_REMATCH[1]}"
                base_scenario_id="${scenario_id%_r*}"
            fi
            
            # Parse base scenario_id: <algorithm>_p<payload>_r<rate>_<hash> (base experiment ID)
            # Note: Output directories now use base IDs without run_index
            if [[ "$base_scenario_id" =~ ^([^_]+)_p([0-9]+)_r([0-9]+)_ ]]; then
                algorithm="${BASH_REMATCH[1]}"
                payload="${BASH_REMATCH[2]}"
                rate="${BASH_REMATCH[3]}"
            fi
            
            # If replica_count wasn't set from scenario_id, check output_dir
            if [[ "$replica_count" -eq 1 ]] && [[ "$output_dir" =~ _r([0-9]+)$ ]]; then
                replica_count="${BASH_REMATCH[1]}"
            fi
            
            # Wait for job to complete using unified function
            if wait_for_job "$job_name" "$GCP_NAMESPACE" "900s" "false"; then
                # Job completed successfully, download results
                if "$SCRIPT_DIR/fetch_and_analyse_from_gcs.sh" \
                    --exp-id "$scenario_id" \
                    --bucket "$BUCKET" \
                    --out "$output_dir" 2>&1; then
                    # Validate data integrity after download
                    raw_file="$output_dir/raw/run.jsonl"
                    if [[ -f "$raw_file" ]]; then
                        file_size=$(stat -f%z "$raw_file" 2>/dev/null || stat -c%s "$raw_file" 2>/dev/null || echo 0)
                        if [[ $file_size -gt 0 ]]; then
                            line_count=$(wc -l < "$raw_file" 2>/dev/null || echo 0)
                            if [[ $line_count -gt 0 ]]; then
                                COMPLETED_JOBS=$((COMPLETED_JOBS + 1))
                                env_completed=$((env_completed + 1))
                                log_success "  Completed: $scenario_id - $file_size bytes, $line_count events"
                                # Add to index with success status
                                add_to_index "$scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "success" "$replica_count"
                            else
                                FAILED_JOBS=$((FAILED_JOBS + 1))
                                env_failed=$((env_failed + 1))
                                log_error "  Data integrity check FAILED: $scenario_id has no JSONL lines!"
                                add_to_index "$scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "failed" "$replica_count"
                            fi
                        else
                            FAILED_JOBS=$((FAILED_JOBS + 1))
                            env_failed=$((env_failed + 1))
                            log_error "  Data integrity check FAILED: $scenario_id has 0-byte file!"
                            add_to_index "$scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "failed" "$replica_count"
                        fi
                    else
                        FAILED_JOBS=$((FAILED_JOBS + 1))
                        env_failed=$((env_failed + 1))
                        log_error "  Data integrity check FAILED: $scenario_id - raw file not found after download!"
                        add_to_index "$scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "failed" "$replica_count"
                    fi
                else
                    FAILED_JOBS=$((FAILED_JOBS + 1))
                    env_failed=$((env_failed + 1))
                    log_error "  Failed to download results for: $scenario_id"
                    add_to_index "$scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "failed" "$replica_count"
                fi
            else
                # Job failed or timed out
                FAILED_JOBS=$((FAILED_JOBS + 1))
                env_failed=$((env_failed + 1))
                log_error "  Job failed or timed out: $scenario_id"
                add_to_index "$scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "failed" "$replica_count"
            fi
            
            # Update progress
            update_progress $env_completed $ENV_TOTAL_EXPERIMENTS "$env" "$scenario_id"
        done < "$JOB_TRACKING_FILE"
        
            log_info "Batch execution complete: $COMPLETED_JOBS succeeded, $FAILED_JOBS failed"
            rm -rf "$TEMP_DIR"
        else
            log_warn "No jobs were tracked for GCP persistent cluster mode"
        fi
    fi
    
    # For Minikube execution with parallelism, wait for all jobs to complete
    # All jobs were submitted immediately; now wait for all to finish
    if [[ "$env" == "minikube" ]] && [[ "${MINIKUBE_USE_PARALLELISM:-false}" == "true" ]] && [[ -f "${MINIKUBE_JOB_TRACKING_FILE:-}" ]]; then
        TOTAL_SUBMITTED=$(wc -l < "$MINIKUBE_JOB_TRACKING_FILE" 2>/dev/null || echo "0")
        if [[ $TOTAL_SUBMITTED -gt 0 ]]; then
            log_info "Waiting for all $TOTAL_SUBMITTED submitted Minikube jobs to complete..."
            
            # Determine namespace
            if [[ "$SMOKE_TEST" == "true" ]]; then
                # Use consistent namespace for all test types
                MINIKUBE_NAMESPACE="default"
            else
                MINIKUBE_NAMESPACE="default"
            fi
            
            # Track job completion
            TOTAL_JOBS=$(wc -l < "$MINIKUBE_JOB_TRACKING_FILE" 2>/dev/null || echo "0")
            COMPLETED_JOBS=0
            FAILED_JOBS=0
            
            while IFS='|' read -r job_name scenario_id output_dir; do
                # Extract algorithm, payload, rate, and replica count from scenario_id
                algorithm="unknown"
                payload=0
                rate=0
                replica_count=1
                
                # Remove replica suffix if present to get base scenario_id for parsing
                base_scenario_id="$scenario_id"
                if [[ "$scenario_id" =~ _r([0-9]+)$ ]]; then
                    replica_count="${BASH_REMATCH[1]}"
                    base_scenario_id="${scenario_id%_r*}"
                fi
                
                # Parse base scenario_id: <algorithm>_p<payload>_r<rate>_<hash> (base experiment ID)
                # Note: Output directories now use base IDs without run_index
                if [[ "$base_scenario_id" =~ ^([^_]+)_p([0-9]+)_r([0-9]+)_ ]]; then
                    algorithm="${BASH_REMATCH[1]}"
                    payload="${BASH_REMATCH[2]}"
                    rate="${BASH_REMATCH[3]}"
                fi
                
                # If replica_count wasn't set from scenario_id, check output_dir
                if [[ "$replica_count" -eq 1 ]] && [[ "$output_dir" =~ _r([0-9]+)$ ]]; then
                    replica_count="${BASH_REMATCH[1]}"
                fi
                
                # Check if scenario_id has replica suffix: _r<replicas>
                if [[ "$scenario_id" =~ _r([0-9]+)$ ]]; then
                    replica_count="${BASH_REMATCH[1]}"
                fi
                
                # Wait for job to complete using unified function
                if wait_for_job "$job_name" "$MINIKUBE_NAMESPACE" "900s" "false"; then
                    # Job completed successfully, retrieve results
                    if retrieve_job_results "minikube" "$job_name" "$MINIKUBE_NAMESPACE" "$output_dir" 2>&1; then
                        # Validate data integrity after retrieval
                        raw_file="$output_dir/raw/run.jsonl"
                        if [[ -f "$raw_file" ]]; then
                            file_size=$(stat -f%z "$raw_file" 2>/dev/null || stat -c%s "$raw_file" 2>/dev/null || echo 0)
                            if [[ $file_size -gt 0 ]]; then
                                line_count=$(wc -l < "$raw_file" 2>/dev/null || echo 0)
                                if [[ $line_count -gt 0 ]]; then
                                    COMPLETED_JOBS=$((COMPLETED_JOBS + 1))
                                    env_completed=$((env_completed + 1))
                                    log_success "  Completed: $scenario_id - $file_size bytes, $line_count events"
                                    add_to_index "$scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "success" "$replica_count"
                                else
                                    FAILED_JOBS=$((FAILED_JOBS + 1))
                                    env_failed=$((env_failed + 1))
                                    log_error "  Data integrity check FAILED: $scenario_id has no JSONL lines!"
                                    add_to_index "$scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "failed" "$replica_count"
                                fi
                            else
                                FAILED_JOBS=$((FAILED_JOBS + 1))
                                env_failed=$((env_failed + 1))
                                log_error "  Data integrity check FAILED: $scenario_id has 0-byte file!"
                                add_to_index "$scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "failed" "$replica_count"
                            fi
                        else
                            FAILED_JOBS=$((FAILED_JOBS + 1))
                            env_failed=$((env_failed + 1))
                            log_error "  Data integrity check FAILED: $scenario_id - raw file not found after retrieval!"
                            add_to_index "$scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "failed" "$replica_count"
                        fi
                    else
                        FAILED_JOBS=$((FAILED_JOBS + 1))
                        env_failed=$((env_failed + 1))
                        log_error "  Failed to retrieve results for: $scenario_id"
                        add_to_index "$scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "failed" "$replica_count"
                    fi
                else
                    # Job failed or timed out
                    FAILED_JOBS=$((FAILED_JOBS + 1))
                    env_failed=$((env_failed + 1))
                    log_error "  Job failed or timed out: $scenario_id"
                    add_to_index "$scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "failed" "$replica_count"
                fi
                
                # Update progress
                update_progress $env_completed $ENV_TOTAL_EXPERIMENTS "$env" "$scenario_id"
            done < "$MINIKUBE_JOB_TRACKING_FILE"
            
            log_info "Minikube parallel execution complete: $COMPLETED_JOBS succeeded, $FAILED_JOBS failed"
            rm -rf "$TEMP_DIR"
        else
            log_warn "No jobs were tracked for Minikube parallel mode"
        fi
    fi
    
    echo ""
    log_info "Environment $env summary:"
    log_info "  Completed: $env_completed (skipped: $env_skipped, new: $((env_completed - env_skipped)))"
    log_info "  Failed: $env_failed"
    log_info "  Progress: $((env_completed * 100 / ENV_TOTAL_EXPERIMENTS))%"
    echo ""
    
    # Cleanup: Always destroy GKE cluster and all underlying resources after completion
    # This ensures no residual GCP resources remain (except GCS bucket), regardless of benchmark type
    # All resources are ephemeral: cluster, nodes, disks, service accounts, Artifact Registry, etc.
    # Only the GCS bucket persists to store experiment results
    if [[ "$env" == "gcp" ]]; then
        log_info "Destroying GKE cluster and all underlying resources after completion..."
        log_info "This ensures no residual GCP resources remain (cluster, nodes, disks, service accounts, Artifact Registry, etc.)"
        log_info "Only the GCS bucket will persist to store experiment results"
        
        # Use consistent cluster name for all test types and environments
        CLUSTER_NAME="pqc-bench"
        
        if "$SCRIPT_DIR/deploy_gcp.sh" \
            --destroy-cluster \
            --project "$PROJECT" \
            --bucket "$BUCKET" \
            --region "$REGION" \
            $([ "$SMOKE_TEST" == "true" ] && echo "--smoke-test" || echo ""); then
            log_success "Cluster and resources destroyed successfully"
        else
            log_warn "Cluster destruction had errors (cluster may still exist)"
            log_info "Running cleanup script to catch any orphaned resources..."
            if "$SCRIPT_DIR/scripts/cleanup_gcp_resources.sh" \
                --project "$PROJECT" \
                --region "$REGION" \
                --cluster-name "$CLUSTER_NAME"; then
                log_success "Cleanup script completed"
            else
                log_warn "Cleanup script had errors"
            fi
            log_info "You can destroy resources manually with:"
            echo "  ./deploy_gcp.sh --destroy-cluster --project $PROJECT --bucket $BUCKET --region $REGION"
            echo "  Or use: ./scripts/cleanup_gcp_resources.sh --project $PROJECT --region $REGION"
        fi
    fi
    
done

# =============================================================================
# Phase 4: Write Master Index
# =============================================================================
log_phase "4. Write Master Index"

# Write index.json
INDEX_FILE="$FINAL_RESULTS_DIR/index.json"

# If progress files exist, merge them with in-memory index for complete picture
# This ensures we capture all experiments even if script was interrupted
for env_item in "${ENV_ARRAY[@]}"; do
    progress_file="$FINAL_RESULTS_DIR/.progress_${env_item}.jsonl"
    if [[ -f "$progress_file" ]]; then
        # Read progress file and add to MASTER_INDEX if not already present
        while IFS= read -r line; do
            if [[ -n "$line" ]]; then
                # Check if this entry is already in MASTER_INDEX
                found=false
                for existing in "${MASTER_INDEX[@]}"; do
                    if [[ "$existing" == "$line" ]]; then
                        found=true
                        break
                    fi
                done
                if [[ "$found" == "false" ]]; then
                    MASTER_INDEX+=("$line")
                fi
            fi
        done < "$progress_file"
    fi
done

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

# Clean up progress files now that we've written the final index
for env_item in "${ENV_ARRAY[@]}"; do
    progress_file="$FINAL_RESULTS_DIR/.progress_${env_item}.jsonl"
    [[ -f "$progress_file" ]] && rm -f "$progress_file"
done

# =============================================================================
# Phase 5: Statistical Aggregation
# =============================================================================
log_phase "5. Statistical Aggregation"

if [[ "$SKIP_ANALYSIS" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
    log_warn "Skipping aggregation"
else
    log_info "Aggregating results across experiments..."
    
    PYTHON_CMD=$(get_python_cmd)
    AGGREGATE_SCRIPT=$(to_relative_path "$SCRIPT_DIR/analysis/aggregate_results.py")
    INDEX_REL=$(to_relative_path "$INDEX_FILE")
    OUTPUT_REL=$(to_relative_path "$FINAL_RESULTS_DIR")
    $PYTHON_CMD "$AGGREGATE_SCRIPT" \
        --index "$INDEX_REL" \
        --output "$OUTPUT_REL" 2>&1 || log_warn "Aggregation completed with warnings"
    
    log_success "Aggregation complete"
fi

# =============================================================================
# Phase 6: Generate Combined Figures
# =============================================================================
log_phase "6. Generate Combined Figures"

if [[ "$SKIP_ANALYSIS" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
    log_warn "Skipping figure generation"
else
    PYTHON_CMD=$(get_python_cmd)
    INDEX_REL=$(to_relative_path "$INDEX_FILE")
    FIGURES_REL=$(to_relative_path "$FINAL_RESULTS_DIR/figures")
    log_info "Generating combined CDF plots..."
    $PYTHON_CMD "$(to_relative_path "$SCRIPT_DIR/analysis/plot_combined_cdfs.py")" \
        --index "$INDEX_REL" \
        --output "$FIGURES_REL" 2>&1 || log_warn "CDF plots completed with warnings"
    
    log_info "Generating scaling curves..."
    $PYTHON_CMD "$(to_relative_path "$SCRIPT_DIR/analysis/plot_scaling_curves.py")" \
        --index "$INDEX_REL" \
        --output "$FIGURES_REL" 2>&1 || log_warn "Scaling plots completed with warnings"
    
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
    
    PYTHON_CMD=$(get_python_cmd)
    $PYTHON_CMD "$SCRIPT_DIR/analysis/hypothesis_tests.py" \
        --index "$INDEX_FILE" \
        --matrix "$MATRIX" \
        --output "$FINAL_RESULTS_DIR" 2>&1 || log_warn "Hypothesis tests completed with warnings"
    
    # Report results
    if [[ -f "$FINAL_RESULTS_DIR/hypothesis_tests.json" ]]; then
        TOTAL_TESTS=$($PYTHON_CMD -c "import json; print(json.load(open('$FINAL_RESULTS_DIR/hypothesis_tests.json'))['total_comparisons'])" 2>/dev/null || echo "?")
        SIG_TESTS=$($PYTHON_CMD -c "import json; print(json.load(open('$FINAL_RESULTS_DIR/hypothesis_tests.json'))['significant_comparisons'])" 2>/dev/null || echo "?")
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
    
    PYTHON_CMD=$(get_python_cmd)
    OUTPUT_REL=$(to_relative_path "$FINAL_RESULTS_DIR")
    $PYTHON_CMD "$(to_relative_path "$SCRIPT_DIR/analysis/build_final_report.py")" \
        --results-dir "$OUTPUT_REL" \
        --output "$OUTPUT_REL/report.pdf" 2>&1 || log_warn "Report generation completed with warnings"
    
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
        
        PYTHON_CMD=$(get_python_cmd)
        INDEX_REL=$(to_relative_path "$INDEX_FILE")
        SCALING_OUTPUT_REL=$(to_relative_path "$FINAL_RESULTS_DIR/figures/scaling")
        $PYTHON_CMD "$(to_relative_path "$SCRIPT_DIR/analysis/plot_replica_scaling.py")" \
            --index "$INDEX_REL" \
            --output "$SCALING_OUTPUT_REL" 2>&1 || log_warn "Scaling plots completed with warnings"
        
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
log_info "Run directory: $RUN_DIR"
echo ""
log_phase "Final Summary"
echo ""

# Calculate completion percentage
if [[ $TOTAL_SCENARIOS -gt 0 ]]; then
    COMPLETION_PCT=$((COMPLETED_SCENARIOS * 100 / TOTAL_SCENARIOS))
else
    COMPLETION_PCT=0
fi

log_info "Overall Statistics:"
echo "  Total scenarios: $TOTAL_SCENARIOS"
echo "  Completed: $COMPLETED_SCENARIOS ($COMPLETION_PCT%)"
echo "  Failed: $FAILED_SCENARIOS"
if [[ $COMPLETED_SCENARIOS -lt $TOTAL_SCENARIOS ]]; then
    REMAINING=$((TOTAL_SCENARIOS - COMPLETED_SCENARIOS - FAILED_SCENARIOS))
    if [[ $REMAINING -gt 0 ]]; then
        echo "  Remaining: $REMAINING"
    fi
fi
echo ""

# Per-environment breakdown
log_info "Per-Environment Breakdown:"
for env_item in "${ENV_ARRAY[@]}"; do
    # Count experiments for this environment from results
    env_completed=$(find "results/$env_item" -name "run.jsonl" -type f 2>/dev/null | wc -l)
    env_failed=$(find "results/$env_item" -type d -mindepth 1 -maxdepth 1 2>/dev/null | while read dir; do
        if [[ ! -f "$dir/raw/run.jsonl" ]] && [[ -d "$dir" ]]; then
            echo "failed"
        fi
    done | wc -l)
    
    if [[ $env_completed -gt 0 ]] || [[ $env_failed -gt 0 ]]; then
        echo "  $env_item: $env_completed completed, $env_failed failed"
    fi
done
echo ""

# Check if all phases completed
if [[ -f "$FINAL_RESULTS_DIR/index.json" ]]; then
    INDEX_SIZE=$(stat -c%s "$FINAL_RESULTS_DIR/index.json" 2>/dev/null || stat -f%z "$FINAL_RESULTS_DIR/index.json" 2>/dev/null || echo "0")
    if [[ $INDEX_SIZE -gt 100 ]]; then
        INDEX_COUNT=$(python3 -c "import json; data=json.load(open('$FINAL_RESULTS_DIR/index.json')); print(len(data.get('experiments', [])))" 2>/dev/null || echo "0")
        log_success "✅ All phases completed successfully"
        echo "  Master index contains $INDEX_COUNT experiments"
    else
        log_warn "⚠️  Test did not complete all phases"
        echo "  Stopped before Phase 4: Write Master Index"
        echo "  To check what was completed, run: ./scripts/check_progress.sh --env <env>"
    fi
else
    log_warn "⚠️  Test did not complete all phases"
    echo "  Stopped before Phase 4: Write Master Index"
    echo "  To check what was completed, run: ./scripts/check_progress.sh --env <env>"
fi
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

# =============================================================================
# Generate Run Summary
# =============================================================================
SUMMARY_FILE="$RUN_DIR/summary.txt"
MANIFEST_FILE="$RUN_DIR/manifest.json"

{
    echo "======================================================================"
    echo "Experiment Run Summary"
    echo "======================================================================"
    echo ""
    echo "Run timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Run directory: $RUN_DIR"
    echo "Matrix file: $MATRIX"
    echo "Environments: ${ENVS}"
    echo "Mode: $([ "$SMOKE_TEST" == "true" ] && echo "Smoke test" || echo "Full scale")"
    echo "Started: $(date -u -d "@$START_TIME" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo "N/A")"
    echo "Duration: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
    echo ""
    echo "Experiment Counts:"
    echo "----------------------------------------------------------------------"
    for env_item in "${ENV_ARRAY[@]}"; do
        env_completed=$(find "results/$env_item" -name "run.jsonl" -type f 2>/dev/null | wc -l)
        env_failed=$(find "results/$env_item" -type d -mindepth 1 -maxdepth 1 2>/dev/null | while read dir; do
            if [[ ! -f "$dir/raw/run.jsonl" ]] && [[ -d "$dir" ]]; then
                echo "failed"
            fi
        done | wc -l)
        if [[ $env_completed -gt 0 ]] || [[ $env_failed -gt 0 ]]; then
            echo "  ${env_item}: $env_completed completed, $env_failed failed"
        fi
    done
    echo ""
    echo "----------------------------------------------------------------------"
    echo "Total scenarios: $TOTAL_SCENARIOS"
    echo "Completed: $COMPLETED_SCENARIOS"
    echo "Failed: $FAILED_SCENARIOS"
    echo ""
    echo "Output Locations:"
    echo "----------------------------------------------------------------------"
    echo "  Final results: $FINAL_RESULTS_DIR/"
    echo "  Individual experiments: $RESULTS_BASE/<env>/<scenario-id>/"
    echo ""
    if [[ -f "$FINAL_RESULTS_DIR/index.json" ]]; then
        echo "Key Outputs:"
        echo "  - index.json: $FINAL_RESULTS_DIR/index.json"
        [[ -f "$FINAL_RESULTS_DIR/aggregated_stats.json" ]] && echo "  - aggregated_stats.json"
        [[ -f "$FINAL_RESULTS_DIR/hypothesis_tests.json" ]] && echo "  - hypothesis_tests.json"
        [[ -d "$FINAL_RESULTS_DIR/figures" ]] && echo "  - figures/ (dissertation-ready)"
        echo ""
    fi
    echo "======================================================================"
} > "$SUMMARY_FILE"

# Create manifest JSON
python3 <<EOF
import json
from pathlib import Path
from datetime import datetime, timezone

manifest = {
    "run_timestamp": datetime.now(timezone.utc).isoformat(),
    "run_directory": str(Path("$RUN_DIR")),
    "matrix_file": "$MATRIX",
    "environments": "$ENVS".split(","),
    "smoke_test": $([ "$SMOKE_TEST" == "true" ] && echo "True" || echo "False"),
    "total_scenarios": $TOTAL_SCENARIOS,
    "completed_scenarios": $COMPLETED_SCENARIOS,
    "failed_scenarios": $FAILED_SCENARIOS,
    "duration_seconds": $ELAPSED,
    "final_results_dir": str(Path("$FINAL_RESULTS_DIR")),
    "results_base": str(Path("$RESULTS_BASE")),
}

with open("$MANIFEST_FILE", 'w') as f:
    json.dump(manifest, f, indent=2)
EOF

log_info "Run summary: $SUMMARY_FILE"
log_info "Run manifest: $MANIFEST_FILE"

exit 0

