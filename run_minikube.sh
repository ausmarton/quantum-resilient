#!/usr/bin/env bash
# =============================================================================
# run_minikube.sh - End-to-end Kubernetes experiment runner
#
# Builds container, deploys to Minikube, runs benchmark, retrieves results,
# and produces analysis outputs.
#
# Usage:
#   ./run_minikube.sh --scenario scenarios/hybrid_kyber_dilithium.yaml \
#                     --out results/exp2 --exp-id exp2
#
# Requirements:
#   - Podman >= 4.0 (rootless recommended)
#   - Minikube >= 1.37 (tested) with containerd runtime
#   - kubectl
#   - Python 3.10+ with analysis dependencies
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common libraries
source "$SCRIPT_DIR/scripts/lib/common.sh"
source "$SCRIPT_DIR/scripts/lib/directories.sh"
source "$SCRIPT_DIR/scripts/lib/analysis.sh"
source "$SCRIPT_DIR/scripts/lib/manifest.sh"
source "$SCRIPT_DIR/scripts/lib/k8s-configmap.sh"
source "$SCRIPT_DIR/scripts/lib/k8s-job.sh"

IMAGE_NAME="pqc-bench"
IMAGE_TAG="latest"
JOB_NAME="pqc-bench-worker"
CONFIGMAP_NAME="pqc-bench-scenario"
PVC_NAME="pqc-bench-results"
NAMESPACE="default"
JOB_TIMEOUT="600s"

# Default values
SCENARIO=""
OUT_DIR=""
EXP_ID=""
RUNS=1
SEED=""
REPLICAS=1
SKIP_BUILD=false
FORCE_BUILD=false
TAG_WITH_GIT=false
SKIP_ANALYSIS=false
SKIP_AGGREGATION=false
KEEP_JOB=false
SMOKE_TEST=false

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Run PQC benchmark experiment in Minikube Kubernetes cluster.
Supports multiple repeated runs and horizontal scaling experiments.

OPTIONS:
    --scenario PATH     Path to scenario YAML file (required)
    --out DIR           Output directory for results (required)
    --exp-id ID         Experiment identifier (required)
    --runs N            Number of repeated runs (default: 1)
    --replicas N        Number of parallel pod replicas (default: 1)
    --seed NUM          Base RNG seed (each run gets seed+run_index)
    --skip-build        Skip container image build (uses existing if available)
    --force-build       Force rebuild even if image exists
    --tag-git           Tag image with git commit hash for reproducibility
    --skip-analysis     Skip Python analysis after run
    --skip-aggregation  Skip aggregation across runs
    --keep-job          Don't delete Job after completion
    --timeout SEC       Job timeout in seconds (default: 600)
    --smoke-test        Enable smoke-test mode (reduced duration/scale)
    -h, --help          Show this help message

EXAMPLES:
    # Single run
    $0 --scenario scenarios/hybrid_kyber_dilithium.yaml \\
       --out results/k8s_exp1 --exp-id k8s_exp1

    # Five repeated runs
    $0 --scenario scenarios/hybrid_kyber_dilithium.yaml \\
       --out results/k8s_exp1 --exp-id k8s_exp1 --runs 5

    # Scaling test with 4 replicas
    $0 --scenario scenarios/hybrid_kyber_dilithium.yaml \\
       --out results/scale_test --exp-id scale_4x --replicas 4

PREREQUISITES (recommended for Podman rootless):
    1. Start Minikube with Podman driver, containerd, kindnet CNI, and larger pod CIDR:
       MINIKUBE_ROOTLESS=true minikube start --driver=podman --rootless \\
         --kubernetes-version=v1.32.0 \\
         --container-runtime=containerd \\
         --cni=kindnet \\
         --extra-config=controller-manager.cluster-cidr=10.244.0.0/16 \\
         --extra-config=kube-proxy.cluster-cidr=10.244.0.0/16 \\
         --extra-config=kubelet.pod-cidr=10.244.0.0/16

    2. Ensure kubectl is configured:
       kubectl cluster-info
EOF
    exit 1
}

cleanup() {
    log_info "Cleaning up previous resources..."
    
    # If JOB_NAME is set, try to delete that specific job
    if [[ -n "${JOB_NAME:-}" ]]; then
        kubectl delete job "$JOB_NAME" --ignore-not-found=true -n "$NAMESPACE" 2>/dev/null || true
        # Wait a bit for pods to be terminated
        sleep 2
    fi
    
    # Also clean up any failed/completed jobs and pods with our labels
    # This handles cases where JOB_NAME wasn't set or cleanup failed
    log_info "Cleaning up any leftover jobs and pods..."
    kubectl delete jobs -l app=pqc-bench,component=worker -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true
    kubectl delete pods -l app=pqc-bench,component=worker -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true
    
    # Wait for pods to be fully terminated
    sleep 2
    
    # Don't delete PVC - we need to keep results across retries
}

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --out)
            OUT_DIR="$2"
            shift 2
            ;;
        --exp-id)
            EXP_ID="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --replicas)
            REPLICAS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --force-build)
            FORCE_BUILD=true
            shift
            ;;
        --tag-git)
            TAG_WITH_GIT=true
            shift
            ;;
        --skip-analysis)
            SKIP_ANALYSIS=true
            shift
            ;;
        --skip-aggregation)
            SKIP_AGGREGATION=true
            shift
            ;;
        --keep-job)
            KEEP_JOB=true
            shift
            ;;
        --timeout)
            JOB_TIMEOUT="${2}s"
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
if [[ -z "$SCENARIO" ]]; then
    log_error "Missing required argument: --scenario"
    usage
fi

if [[ -z "$OUT_DIR" ]]; then
    log_error "Missing required argument: --out"
    usage
fi

if [[ -z "$EXP_ID" ]]; then
    log_error "Missing required argument: --exp-id"
    usage
fi

if [[ ! -f "$SCENARIO" ]]; then
    log_error "Scenario file not found: $SCENARIO"
    exit 1
fi

# Make paths absolute
SCENARIO="$(cd "$(dirname "$SCENARIO")" && pwd)/$(basename "$SCENARIO")"
OUT_DIR="$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)"

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
START_TIME=$(date +%s)
START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           PQC Benchmark - Minikube Experiment Runner         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

log_info "Experiment ID: $EXP_ID"
log_info "Scenario: $SCENARIO"
log_info "Output: $OUT_DIR"
log_info "Runs: $RUNS"
log_info "Replicas: $REPLICAS"
[[ "$SMOKE_TEST" == "true" ]] && log_info "Mode: SMOKE-TEST (reduced scale)"
[[ -n "$SEED" ]] && log_info "Base RNG seed: $SEED"
log_info "Started: $START_ISO"

# Determine if we're doing a scaling test
SCALING_MODE=false
if [[ $REPLICAS -gt 1 ]] && [[ "$SMOKE_TEST" != "true" ]]; then
    SCALING_MODE=true
    log_info "Mode: Scaling test (parallel job with $REPLICAS pods)"
fi

# Override for smoke-test mode
if [[ "$SMOKE_TEST" == "true" ]]; then
    RUNS=1
    REPLICAS=1
    log_info "Smoke-test mode: forcing runs=1, replicas=1"
fi

# =============================================================================
# Step 1: Verify prerequisites
# =============================================================================
log_step "Step 1/9: Verifying prerequisites"

# Check Podman
if ! command -v podman &> /dev/null; then
    log_error "Podman not found. Please install Podman >= 4.0"
    exit 1
fi
PODMAN_VERSION=$(podman --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
log_success "Podman: $PODMAN_VERSION"

# Check Minikube
if ! command -v minikube &> /dev/null; then
    log_error "Minikube not found. Please install Minikube >= 1.34"
    exit 1
fi
MINIKUBE_VERSION=$(minikube version --short 2>/dev/null | grep -oE 'v[0-9]+\.[0-9]+' | head -1)
log_success "Minikube: $MINIKUBE_VERSION"

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    log_error "kubectl not found"
    exit 1
fi
log_success "kubectl: $(kubectl version --client --short 2>/dev/null || kubectl version --client -o yaml | grep gitVersion | head -1)"

# Check Minikube status
MINIKUBE_STATUS=$(minikube status --format='{{.Host}}' 2>/dev/null || echo "Stopped")
if [[ "$MINIKUBE_STATUS" != "Running" ]]; then
    log_warn "Minikube not running. Attempting to start..."
    minikube start --driver=podman --cpus=4 --memory=8g || {
        log_error "Failed to start Minikube. Please run: minikube start --driver=podman"
        exit 1
    }
fi

# CRITICAL: Ensure kubectl is pointing to Minikube (not GCP or other cluster)
log_info "Ensuring kubectl context is set to Minikube..."
minikube update-context >/dev/null 2>&1 || {
    log_warn "minikube update-context failed, trying manual switch..."
    kubectl config use-context minikube >/dev/null 2>&1 || {
        log_error "Failed to switch to Minikube context"
        log_info "Current context: $(kubectl config current-context 2>/dev/null || echo 'unknown')"
        log_info "Available contexts:"
        kubectl config get-contexts 2>&1 | head -5 || true
        exit 1
    }
}

# Verify we're connected to Minikube
CURRENT_CONTEXT=$(kubectl config current-context 2>/dev/null || echo "")
if [[ "$CURRENT_CONTEXT" != "minikube" ]]; then
    log_error "kubectl context is not Minikube (current: $CURRENT_CONTEXT)"
    log_error "This will cause connection timeouts to GCP API server"
    log_info "Please run: kubectl config use-context minikube"
    exit 1
fi

log_success "Minikube cluster is running (context: $CURRENT_CONTEXT)"

# =============================================================================
# Step 2: Create output directories
# =============================================================================
log_step "Step 2/9: Creating output directories"

mkdir -p "$OUT_DIR/raw"
mkdir -p "$OUT_DIR/merged"
mkdir -p "$OUT_DIR/stats"
mkdir -p "$OUT_DIR/figures"

log_success "Created: $OUT_DIR/{raw,merged,stats,figures}"

# =============================================================================
# Step 3: Build container image
# =============================================================================
log_step "Step 3/9: Building container image"

# Check if image already exists
if podman image exists "$IMAGE_NAME:$IMAGE_TAG" 2>/dev/null; then
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_info "Image $IMAGE_NAME:$IMAGE_TAG already exists, skipping build (--skip-build)"
    else
        log_info "Image $IMAGE_NAME:$IMAGE_TAG already exists"
        log_info "Use --skip-build to avoid this check, or rebuild with --force-build"
        
        # Check if we should rebuild anyway (e.g., if source changed)
        if [[ "${FORCE_BUILD:-false}" == "true" ]]; then
            log_info "Force rebuilding image (--force-build)..."
            cd "$SCRIPT_DIR"
            podman build -t "$IMAGE_NAME:$IMAGE_TAG" -f Containerfile . 2>&1 | while read -r line; do
                echo "  $line"
            done
            
            if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
                log_error "Container build failed"
                exit 1
            fi
            
            log_success "Image rebuilt: $IMAGE_NAME:$IMAGE_TAG"
        else
            log_success "Using existing image: $IMAGE_NAME:$IMAGE_TAG"
            log_info "  (To rebuild, use --force-build flag)"
        fi
    fi
else
    log_info "Building $IMAGE_NAME:$IMAGE_TAG with Podman..."
    
    cd "$SCRIPT_DIR"
    podman build -t "$IMAGE_NAME:$IMAGE_TAG" -f Containerfile . 2>&1 | while read -r line; do
        echo "  $line"
    done
    
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        log_error "Container build failed"
        exit 1
    fi
    
    log_success "Image built: $IMAGE_NAME:$IMAGE_TAG"
    
    # Optionally tag with git commit hash for reproducibility
    if [[ "${TAG_WITH_GIT:-false}" == "true" ]]; then
        GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        if [[ "$GIT_COMMIT" != "unknown" ]]; then
            GIT_TAG="${IMAGE_NAME}:git-${GIT_COMMIT}"
            log_info "Tagging image with git commit: $GIT_TAG"
            podman tag "$IMAGE_NAME:$IMAGE_TAG" "$GIT_TAG" 2>/dev/null || true
        fi
    fi
fi

# =============================================================================
# Step 4: Load image into Minikube
# =============================================================================
log_step "Step 4/9: Loading image into Minikube"

log_info "Loading image into Minikube..."
# Tag with localhost/ prefix (Minikube expects this for local images)
LOCAL_IMAGE="localhost/${IMAGE_NAME}:${IMAGE_TAG}"
if ! podman image exists "$LOCAL_IMAGE" 2>/dev/null; then
    log_info "Tagging image as $LOCAL_IMAGE..."
    podman tag "$IMAGE_NAME:$IMAGE_TAG" "$LOCAL_IMAGE" 2>/dev/null || {
        log_error "Failed to tag image"
        exit 1
    }
fi

# Load into Minikube using podman save/load
TEMP_TAR=$(mktemp --suffix=.tar)
if podman save "$LOCAL_IMAGE" -o "$TEMP_TAR" 2>/dev/null; then
    if minikube image load "$TEMP_TAR" >/dev/null 2>&1; then
log_success "Image loaded into Minikube"
    else
        log_error "Failed to load image into Minikube"
        rm -f "$TEMP_TAR"
        exit 1
    fi
    rm -f "$TEMP_TAR"
else
    log_error "Failed to save image to tar"
    exit 1
fi


# =============================================================================
# Multi-Run Execution Loop
# =============================================================================

COMPLETED_RUNS=0
FAILED_RUNS=0
TOTAL_RUN_START=$(date +%s)

for ((RUN_INDEX = 1; RUN_INDEX <= RUNS; RUN_INDEX++)); do
    if [[ $RUNS -gt 1 ]]; then
        log_run $RUN_INDEX $RUNS "Starting..."
        RUN_OUT_DIR="$OUT_DIR/run-$RUN_INDEX"
        RUN_EXP_ID="${EXP_ID}_run${RUN_INDEX}"
    else
        RUN_OUT_DIR="$OUT_DIR"
        RUN_EXP_ID="$EXP_ID"
    fi

    # Compute seed for this run
    if [[ -n "$SEED" ]]; then
        RUN_SEED=$((SEED + RUN_INDEX - 1))
    else
        RUN_SEED=""
    fi

    # Create output directories for this run
    mkdir -p "$RUN_OUT_DIR/raw"
    mkdir -p "$RUN_OUT_DIR/merged"
    mkdir -p "$RUN_OUT_DIR/stats"
    mkdir -p "$RUN_OUT_DIR/figures"
    

# =============================================================================
# Step 5: Deploy Kubernetes resources
# =============================================================================
log_step "Step 5/9: Deploying Kubernetes resources (Run $RUN_INDEX/$RUNS)"

# Determine job name early so cleanup can use it
if [[ "$SCALING_MODE" == "true" ]]; then
    # Generate job name for scaling experiments (same logic as below)
    SANITIZE_K8S_NAME() {
        echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/_/-/g' | sed 's/[^a-z0-9-]/-/g' | sed 's/--*/-/g' | sed 's/^-\|-$//g'
    }
    
    REPLICA_SUFFIX=""
    if [[ "$RUN_EXP_ID" =~ _r([0-9]+)$ ]]; then
        REPLICA_SUFFIX="_r${BASH_REMATCH[1]}"
        BASE_EXP_ID="${RUN_EXP_ID%_r*}"
    else
        BASE_EXP_ID="$RUN_EXP_ID"
        if [[ "$REPLICAS" -gt 1 ]]; then
            REPLICA_SUFFIX="_r${REPLICAS}"
        fi
    fi
    
    SANITIZED_BASE=$(SANITIZE_K8S_NAME "$BASE_EXP_ID" | cut -c1-49)
    SANITIZED_SUFFIX=$(SANITIZE_K8S_NAME "$REPLICA_SUFFIX" | sed 's/^_//')
    JOB_NAME="pqc-bench-${SANITIZED_BASE}${SANITIZED_SUFFIX}"
else
    JOB_NAME="pqc-bench-worker"
fi

# Clean up any existing job (now JOB_NAME is set)
cleanup

# Apply PVC (simpler than hostPath - no permission issues)
log_info "Creating PersistentVolumeClaim..."
kubectl apply --validate=false -f "$SCRIPT_DIR/k8s/results-pvc.yaml" -n "$NAMESPACE" >/dev/null 2>&1 || true

# Wait for PVC to be bound
log_info "Waiting for PVC to be bound..."
kubectl wait --for=jsonpath='{.status.phase}'=Bound pvc/"$PVC_NAME" -n "$NAMESPACE" --timeout=60s >/dev/null 2>&1 || {
    log_warn "PVC may not be bound immediately, continuing..."
}

# Create output directory for final results
log_info "Creating output directory: $RUN_OUT_DIR"
create_output_directories "$RUN_OUT_DIR"

# Create ConfigMap from scenario file
log_info "Creating ConfigMap from scenario: $SCENARIO"

# Determine JSONL output path (scaling mode uses different path)
if [[ "$SCALING_MODE" == "true" ]]; then
    JSONL_OUT_PATH="/results/current/raw/run.jsonl"
else
    JSONL_OUT_PATH="/results/raw/run.jsonl"
fi

# Use unified ConfigMap creation function
CONFIGMAP_NAME=$(create_scenario_configmap \
    "$SCENARIO" \
    "$RUN_EXP_ID" \
    "$NAMESPACE" \
    "$SMOKE_TEST" \
    "${RUN_SEED:-}" \
    "$CONFIGMAP_NAME" \
    "$JSONL_OUT_PATH" \
    "${DURATION:-}") || {
    log_error "Failed to create ConfigMap"
    exit 1
}

# Apply Job (use parallel job for scaling tests)
if [[ "$SCALING_MODE" == "true" ]]; then
    log_info "Creating parallel Job with $REPLICAS replicas..."
    
    # Update scaling config
    kubectl create configmap pqc-scaling-config \
        --from-literal=experiment_id="$RUN_EXP_ID" \
        --from-literal=replica_count="$REPLICAS" \
        --from-literal=duration_sec="30" \
        --dry-run=client -o yaml | kubectl apply --validate=false -f - -n "$NAMESPACE"
    
    # JOB_NAME was already set in Step 5 before cleanup, so we can use it here
    
    # Create the parallel job with dynamic parallelism and unique name
    cat "$SCRIPT_DIR/k8s/worker-parallel-job.yaml" | \
        sed "s/name: pqc-bench-scaling/name: $JOB_NAME/" | \
        sed "s/parallelism: 1/parallelism: $REPLICAS/" | \
        sed "s/completions: 1/completions: $REPLICAS/" | \
        kubectl apply --validate=false -f - -n "$NAMESPACE"
else
    log_info "Creating Job..."
    JOB_NAME="pqc-bench-worker"
    
    # Generate Job YAML using unified generator
    TEMP_JOB=$(mktemp)
    "$SCRIPT_DIR/scripts/lib/k8s-job-generator.py" \
        --environment minikube \
        --job-name "$JOB_NAME" \
        --namespace "$NAMESPACE" \
        --image "$LOCAL_IMAGE" \
        --scenario-configmap "$CONFIGMAP_NAME" \
        --experiment-id "$RUN_EXP_ID" \
        --output "$TEMP_JOB" || {
        log_error "Failed to generate Job YAML"
        rm -f "$TEMP_JOB"
        exit 1
    }
    
    # Apply generated Job YAML
    kubectl apply --validate=false -f "$TEMP_JOB" -n "$NAMESPACE" || {
        log_error "Failed to apply Job YAML"
        rm -f "$TEMP_JOB"
        exit 1
    }
    rm -f "$TEMP_JOB"
fi

log_success "Kubernetes resources deployed"

# =============================================================================
# Step 6: Wait for Job completion
# =============================================================================
log_step "Step 6/9: Waiting for Job completion"

# Use unified job waiting function
if ! wait_for_job "$JOB_NAME" "$NAMESPACE" "$JOB_TIMEOUT" "true"; then
    log_error "Job failed or timed out"
    FAILED_RUNS=$((FAILED_RUNS + 1))
    continue
fi

# =============================================================================
# Step 7: Copy results from PVC
# =============================================================================
log_step "Step 7/9: Copying results from PVC"

# Get pod name for later use (manifest generation)
POD_NAME=$(get_job_pods "$JOB_NAME" "$NAMESPACE" | awk '{print $1}')

# Use unified result retrieval function
if ! copy_results_from_pvc "$JOB_NAME" "$RUN_OUT_DIR" "$NAMESPACE" "$PVC_NAME"; then
    log_error "Failed to copy results from PVC"
    if [[ -n "$POD_NAME" ]]; then
        log_info "Main pod logs (last 20 lines):"
        kubectl logs "$POD_NAME" -n "$NAMESPACE" --tail=20 2>&1 | head -20 || true
    fi
    FAILED_RUNS=$((FAILED_RUNS + 1))
    continue
fi

# Additional validation: check for error messages and JSON validity
RAW_JSONL_FILE="$RUN_OUT_DIR/raw/run.jsonl"
if [[ -f "$RAW_JSONL_FILE" ]]; then
    FIRST_LINE=$(head -1 "$RAW_JSONL_FILE" 2>/dev/null || echo "")
    if [[ -n "$FIRST_LINE" ]] && [[ "$FIRST_LINE" =~ ^error: ]]; then
        log_error "Data collection failed: file contains error message, not JSONL data!"
        log_error "First line: ${FIRST_LINE:0:100}..."
        FAILED_RUNS=$((FAILED_RUNS + 1))
        rm -f "$RAW_JSONL_FILE"
        continue
    fi
    
    if [[ -n "$FIRST_LINE" ]]; then
        if ! echo "$FIRST_LINE" | python3 -m json.tool >/dev/null 2>&1; then
            log_error "Data collection failed: file does not contain valid JSONL!"
            log_error "First line: ${FIRST_LINE:0:100}..."
            FAILED_RUNS=$((FAILED_RUNS + 1))
            rm -f "$RAW_JSONL_FILE"
            continue
        fi
    fi
    
    FILE_SIZE=$(stat -f%z "$RAW_JSONL_FILE" 2>/dev/null || stat -c%s "$RAW_JSONL_FILE" 2>/dev/null || echo 0)
    LINE_COUNT=$(wc -l < "$RAW_JSONL_FILE" 2>/dev/null || echo 0)
    log_success "Verified run.jsonl: $FILE_SIZE bytes, $LINE_COUNT events"
fi

JSONL_COUNT=$(find "$RUN_OUT_DIR/raw" -name "*.jsonl" -type f | wc -l)
log_success "Verified $JSONL_COUNT JSONL file(s) with valid data"

# =============================================================================
# Step 8: Generate manifest
# =============================================================================
log_step "Step 8/9: Generating experiment manifest"

END_TIME=$(date +%s)
END_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
ELAPSED=$((END_TIME - START_TIME))

# Get Kubernetes metadata
NODE_NAME=$(kubectl get pods "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.nodeName}' 2>/dev/null || echo "unknown")
K8S_VERSION=$(kubectl version --short 2>/dev/null | grep Server | awk '{print $3}' || echo "unknown")

# Get Git commit
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

# Get event count
if [[ -f "$RUN_OUT_DIR/raw/run.jsonl" ]]; then
    EVENT_COUNT=$(wc -l < "$RUN_OUT_DIR/raw/run.jsonl")
else
    EVENT_COUNT=$(cat "$RUN_OUT_DIR/raw/"*.jsonl 2>/dev/null | wc -l || echo 0)
fi

# Extract scenario ID from scenario file
SCENARIO_ID=$(grep -E "^id:" "$SCENARIO" | awk '{print $2}' | tr -d '"' || echo "$EXP_ID")
MANIFEST_SEED=${RUN_SEED:-null}

# Generate manifest with Minikube-specific fields
local extra_fields=",
    \"scaling_mode\": $SCALING_MODE,
    \"kubernetes\": {
        \"node_name\": \"$NODE_NAME\",
        \"k8s_version\": \"$K8S_VERSION\",
        \"namespace\": \"$NAMESPACE\",
        \"job_name\": \"$JOB_NAME\",
        \"pod_name\": \"$POD_NAME\"
    },
    \"container\": {
        \"image\": \"$IMAGE_NAME:$IMAGE_TAG\",
        \"runtime\": \"podman\"
    },
    \"minikube\": {
        \"version\": \"$MINIKUBE_VERSION\",
        \"driver\": \"podman\"
    }"
generate_manifest "$RUN_OUT_DIR" "$RUN_EXP_ID" "$SCENARIO" "minikube" "$RUN_INDEX" "$EVENT_COUNT" "$ELAPSED" "$MANIFEST_SEED" "$REPLICAS" "$extra_fields"

# =============================================================================
# Step 9: Run analysis pipeline
# =============================================================================
log_step "Step 9/9: Running analysis pipeline"

run_analysis_pipeline "$RUN_OUT_DIR" "$RUN_EXP_ID" "$SKIP_ANALYSIS"

COMPLETED_RUNS=$((COMPLETED_RUNS + 1))

# End of run loop
done

TOTAL_RUN_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_RUN_END - TOTAL_RUN_START))

# =============================================================================
# Aggregation (for multiple runs)
# =============================================================================
if [[ $RUNS -gt 1 ]] && [[ "$SKIP_AGGREGATION" != "true" ]] && [[ $COMPLETED_RUNS -gt 0 ]]; then
    log_step "Aggregating results across $COMPLETED_RUNS runs"
    
    if [[ -f "$SCRIPT_DIR/analysis/aggregate_runs.py" ]]; then
        python3 "$SCRIPT_DIR/analysis/aggregate_runs.py" \
            --input "$OUT_DIR" \
            --runs "$COMPLETED_RUNS" \
            --output "$OUT_DIR" 2>&1 | while read -r line; do
            echo "  $line"
        done || log_warn "Aggregation completed with warnings"
        
        log_success "Aggregation complete"
    else
        log_warn "aggregate_runs.py not found, skipping aggregation"
    fi
    
    # Create combined figures directory
    mkdir -p "$OUT_DIR/figures"
    if [[ -d "$OUT_DIR/run-1/figures" ]]; then
        cp -r "$OUT_DIR/run-1/figures/"* "$OUT_DIR/figures/" 2>/dev/null || true
    fi
fi

# =============================================================================
# Cleanup
# =============================================================================
if [[ "$KEEP_JOB" != "true" ]]; then
    log_info "Cleaning up Job..."
    kubectl delete job "$JOB_NAME" -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true
fi

# =============================================================================
# Final Summary
# =============================================================================
echo ""
echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    EXPERIMENT COMPLETE                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

log_info "Experiment ID: $EXP_ID"
log_info "Total Duration: ${TOTAL_ELAPSED}s"
log_info "Runs: $COMPLETED_RUNS completed, $FAILED_RUNS failed"
echo ""

if [[ $RUNS -eq 1 ]]; then
    log_info "Output files:"
    echo "  Raw JSONL:     $OUT_DIR/raw/"
    [[ -f "$OUT_DIR/merged/merged.jsonl" ]] && echo "  Merged JSONL:  $OUT_DIR/merged/merged.jsonl"
    [[ -f "$OUT_DIR/stats/summary.json" ]] && echo "  Stats:         $OUT_DIR/stats/summary.json"
    echo "  Manifest:      $OUT_DIR/manifest.json"
else
    log_info "Output files:"
    for ((i = 1; i <= COMPLETED_RUNS; i++)); do
        echo "  Run $i:         $OUT_DIR/run-$i/"
    done
    [[ -f "$OUT_DIR/aggregated_stats.json" ]] && echo "  Aggregated:    $OUT_DIR/aggregated_stats.json"
    [[ -f "$OUT_DIR/stability_report.json" ]] && echo "  Stability:     $OUT_DIR/stability_report.json"
fi
echo ""

if [[ -f "$OUT_DIR/aggregated_stats.json" ]]; then
    log_info "Aggregated Statistics:"
    python3 -c "
import json
with open('$OUT_DIR/aggregated_stats.json') as f:
    data = json.load(f)
lat = data.get('latency', {})
if 'p95' in lat:
    p95 = lat['p95']
    print(f\"  p95 latency: {p95['mean']:.0f} ± {p95['std']:.0f} μs (CV: {p95['cv']:.1%})\")
    print(f\"  95% CI: [{p95['ci_95_low']:.0f}, {p95['ci_95_high']:.0f}] μs\")
" 2>/dev/null || true
fi

log_info "To compare with native run:"
echo "  python analysis/compare_native_vs_minikube.py \\"
echo "    --native results/native_exp/stats/summary.json \\"
echo "    --k8s $OUT_DIR/stats/summary.json"
echo ""

if [[ $FAILED_RUNS -gt 0 ]]; then
    log_error "Done with $FAILED_RUNS failed run(s)"
    exit 1
else
    log_success "Done!"
    exit 0
fi


