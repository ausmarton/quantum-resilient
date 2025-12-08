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

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

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
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

log_run() {
    echo -e "${CYAN}[RUN $1/$2]${NC} $3"
}

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
    kubectl delete job "$JOB_NAME" --ignore-not-found=true -n "$NAMESPACE" 2>/dev/null || true
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

# Clean up any existing job
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
mkdir -p "$RUN_OUT_DIR/raw"
mkdir -p "$RUN_OUT_DIR/merged"
mkdir -p "$RUN_OUT_DIR/stats"
mkdir -p "$RUN_OUT_DIR/figures"

# Create ConfigMap from scenario file
log_info "Creating ConfigMap from scenario: $SCENARIO"

# Update scenario to write to /results/raw/run.jsonl
TEMP_SCENARIO=$(mktemp)
cp "$SCENARIO" "$TEMP_SCENARIO"

# Ensure jsonl_out points to /results/raw/run.jsonl
if grep -q "jsonl_out:" "$TEMP_SCENARIO"; then
    sed -i 's|jsonl_out:.*|jsonl_out: "/results/raw/run.jsonl"|' "$TEMP_SCENARIO"
else
    # Add jsonl_out to metrics section
    if grep -q "metrics:" "$TEMP_SCENARIO"; then
        sed -i '/metrics:/a\  jsonl_out: "/results/raw/run.jsonl"' "$TEMP_SCENARIO"
    else
        echo -e "\nmetrics:\n  jsonl_out: \"/results/raw/run.jsonl\"" >> "$TEMP_SCENARIO"
    fi
fi

# Override duration for smoke-test mode
if [[ "$SMOKE_TEST" == "true" ]]; then
    sed -i "s/duration_sec:.*/duration_sec: 5/" "$TEMP_SCENARIO"
fi

# Set seed for this run if specified
if [[ -n "$RUN_SEED" ]]; then
    if grep -q "rng_seed:" "$TEMP_SCENARIO"; then
        sed -i "s/rng_seed:.*/rng_seed: $RUN_SEED/" "$TEMP_SCENARIO"
    else
        sed -i "/^id:/a rng_seed: $RUN_SEED" "$TEMP_SCENARIO"
    fi
fi

kubectl create configmap "$CONFIGMAP_NAME" \
    --from-file=scenario.yaml="$TEMP_SCENARIO" \
    --dry-run=client -o yaml | kubectl apply --validate=false -f - -n "$NAMESPACE"

rm -f "$TEMP_SCENARIO"
log_success "ConfigMap created"

# Apply Job (use parallel job for scaling tests)
if [[ "$SCALING_MODE" == "true" ]]; then
    log_info "Creating parallel Job with $REPLICAS replicas..."
    
    # Update scaling config
    kubectl create configmap pqc-scaling-config \
        --from-literal=experiment_id="$RUN_EXP_ID" \
        --from-literal=replica_count="$REPLICAS" \
        --from-literal=duration_sec="30" \
        --dry-run=client -o yaml | kubectl apply --validate=false -f - -n "$NAMESPACE"
    
    # Create the parallel job with dynamic parallelism
    cat "$SCRIPT_DIR/k8s/worker-parallel-job.yaml" | \
        sed "s/parallelism: 1/parallelism: $REPLICAS/" | \
        sed "s/completions: 1/completions: $REPLICAS/" | \
        kubectl apply --validate=false -f - -n "$NAMESPACE"
    
    JOB_NAME="pqc-bench-scaling"
else
    log_info "Creating Job..."
    # Use original worker-job.yaml with PVC (simpler, no permission issues)
    kubectl apply --validate=false -f "$SCRIPT_DIR/k8s/worker-job.yaml" -n "$NAMESPACE"
fi

log_success "Kubernetes resources deployed"

# =============================================================================
# Step 6: Wait for Job completion
# =============================================================================
log_step "Step 6/9: Waiting for Job completion"

log_info "Waiting for Job to complete (timeout: $JOB_TIMEOUT)..."

# Get pod name
sleep 5  # Give time for pod to be created
POD_NAME=$(kubectl get pods -l job-name="$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

if [[ -n "$POD_NAME" ]]; then
    log_info "Pod: $POD_NAME"
    
    # Stream logs in background
    (
        sleep 10
        kubectl logs -f "$POD_NAME" -n "$NAMESPACE" 2>/dev/null | while read -r line; do
            echo "  [pod] $line"
        done
    ) &
    LOG_PID=$!
fi

# Wait for completion
if ! kubectl wait --for=condition=complete job/"$JOB_NAME" -n "$NAMESPACE" --timeout="$JOB_TIMEOUT"; then
    # Check if failed
    JOB_STATUS=$(kubectl get job "$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || echo "")
    
    if [[ "$JOB_STATUS" == "True" ]]; then
        log_error "Job failed!"
        kubectl describe job "$JOB_NAME" -n "$NAMESPACE"
        kubectl logs -l job-name="$JOB_NAME" -n "$NAMESPACE" --tail=50
        exit 1
    fi
    
    log_error "Job timed out"
    exit 1
fi

# Kill log streaming
kill $LOG_PID 2>/dev/null || true

log_success "Job completed successfully"

# =============================================================================
# Step 7: Copy results from PVC
# =============================================================================
log_step "Step 7/9: Copying results from PVC"

# Get pod name
POD_NAME=$(kubectl get pods -l job-name="$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
POD_PHASE=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
log_info "Pod: $POD_NAME (phase: $POD_PHASE)"

# Copy results from PVC using a temporary pod and kubectl logs
# This is more reliable than kubectl run -i which has timing issues
log_info "Copying results from PVC..."
READ_POD_NAME="pvc-read-$(date +%s)"

COPY_SUCCESS=false
TEMP_OUTPUT=$(mktemp)

# Create a pod that will read the file
READ_POD_YAML=$(mktemp)
cat > "$READ_POD_YAML" <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${READ_POD_NAME}
  namespace: ${NAMESPACE}
spec:
  restartPolicy: Never
  containers:
  - name: read
    image: busybox:1.36
    command: ["sh", "-c", "cat /results/raw/run.jsonl 2>/dev/null || find /results -name '*.jsonl' -type f -exec cat {} \\;"]
    volumeMounts:
    - name: results
      mountPath: /results
      readOnly: true
  volumes:
  - name: results
    persistentVolumeClaim:
      claimName: ${PVC_NAME}
EOF

if kubectl apply --validate=false -f "$READ_POD_YAML" >/dev/null 2>&1; then
    log_info "Waiting for read pod to complete..."
    
    # Wait for pod to be created
    sleep 2
    
    # Wait for completion (with timeout)
    for i in {1..20}; do
        PHASE=$(kubectl get pod "$READ_POD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
        if [[ "$PHASE" == "Succeeded" ]] || [[ "$PHASE" == "Failed" ]]; then
            break
        fi
        sleep 1
    done
    
    if [[ "$PHASE" == "Succeeded" ]] || [[ "$PHASE" == "Failed" ]]; then
        # Get logs (this is more reliable than kubectl run -i)
        if kubectl logs "$READ_POD_NAME" -n "$NAMESPACE" > "$TEMP_OUTPUT" 2>&1; then
            # Filter out kubectl warning messages and check for valid JSONL
            # Remove lines that are kubectl warnings or prompts
            grep -v "^warning:" "$TEMP_OUTPUT" | \
                grep -v "^If you don't see a command prompt" | \
                grep -v "^Internal error occurred" | \
                grep -v "^unable to upgrade connection" | \
                grep -v "^container.*not found" | \
                grep -v "^falling back to streaming logs" > "$RUN_OUT_DIR/raw/run.jsonl" 2>/dev/null || true
            
            # Check if we have valid JSONL data (at least one line starting with {)
            if [[ -f "$RUN_OUT_DIR/raw/run.jsonl" ]] && [[ -s "$RUN_OUT_DIR/raw/run.jsonl" ]]; then
                # Check if it contains valid JSONL (at least one line starting with {)
                if head -1 "$RUN_OUT_DIR/raw/run.jsonl" | grep -q "^{"; then
                    COPY_SUCCESS=true
                    log_success "Successfully copied run.jsonl from PVC"
                else
                    log_warn "Output doesn't appear to be valid JSONL"
                    log_info "First line: $(head -1 "$RUN_OUT_DIR/raw/run.jsonl" 2>/dev/null || echo 'empty')"
                fi
            else
                log_warn "Read pod output is empty or invalid"
            fi
        else
            log_error "Failed to get logs from read pod"
        fi
    else
        log_error "Read pod did not complete (phase: $PHASE)"
        kubectl describe pod "$READ_POD_NAME" -n "$NAMESPACE" 2>&1 | tail -20 || true
    fi
    
    # Clean up
    kubectl delete pod "$READ_POD_NAME" -n "$NAMESPACE" --ignore-not-found=true >/dev/null 2>&1
    rm -f "$READ_POD_YAML" "$TEMP_OUTPUT"
else
    log_error "Failed to create read pod"
    rm -f "$READ_POD_YAML" "$TEMP_OUTPUT"
fi

if [[ "$COPY_SUCCESS" == "false" ]]; then
    log_error "Failed to copy results from PVC"
    log_info "Main pod logs (last 20 lines):"
    kubectl logs "$POD_NAME" -n "$NAMESPACE" --tail=20 2>&1 | head -20 || true
    log_info "To recover data manually, create a pod to read from PVC:"
    log_info "  kubectl run pvc-read --image=busybox:1.36 --rm -i --restart=Never --overrides='{\"spec\":{\"volumes\":[{\"name\":\"results\",\"persistentVolumeClaim\":{\"claimName\":\"$PVC_NAME\"}}],\"containers\":[{\"name\":\"read\",\"image\":\"busybox:1.36\",\"command\":[\"sh\",\"-c\",\"cat /results/raw/run.jsonl\"],\"volumeMounts\":[{\"name\":\"results\",\"mountPath\":\"/results\"}]}]}}' > output.jsonl"
    FAILED_RUNS=$((FAILED_RUNS + 1))
    continue
fi

# Verify results - check both existence AND content
RAW_JSONL_FILE="$RUN_OUT_DIR/raw/run.jsonl"
if [[ ! -f "$RAW_JSONL_FILE" ]]; then
    RAW_JSONL_FILE=$(find "$RUN_OUT_DIR/raw" -name "*.jsonl" -type f | head -1)
fi

if [[ ! -f "$RAW_JSONL_FILE" ]]; then
    log_error "No JSONL files found in results!"
    log_info "Contents of $RUN_OUT_DIR/raw:"
    ls -la "$RUN_OUT_DIR/raw/" || true
    FAILED_RUNS=$((FAILED_RUNS + 1))
    continue
fi

# Validate file has content (not 0 bytes)
if [[ -f "$RAW_JSONL_FILE" ]]; then
    FILE_SIZE=$(stat -f%z "$RAW_JSONL_FILE" 2>/dev/null || stat -c%s "$RAW_JSONL_FILE" 2>/dev/null || echo 0)
    if [[ $FILE_SIZE -eq 0 ]]; then
        log_error "Data collection failed: run.jsonl is 0 bytes!"
        log_error "This indicates the benchmark didn't write any data."
        log_info "Checking pod logs for errors..."
        kubectl logs "$POD_NAME" -n "$NAMESPACE" --tail=100 || true
        log_info "Checking if benchmark process completed..."
        kubectl exec "$POD_NAME" -n "$NAMESPACE" -- ps aux 2>/dev/null || true
        log_info "Checking if results directory exists in pod..."
        kubectl exec "$POD_NAME" -n "$NAMESPACE" -- ls -la /results/raw/ 2>&1 || true
        FAILED_RUNS=$((FAILED_RUNS + 1))
        # Remove empty file so it will be retried
        rm -f "$RAW_JSONL_FILE"
        continue
    else
        # Validate file has valid JSONL content (at least one line)
        LINE_COUNT=$(wc -l < "$RAW_JSONL_FILE" 2>/dev/null || echo 0)
        if [[ $LINE_COUNT -eq 0 ]]; then
            log_error "Data collection failed: run.jsonl has no lines!"
            log_error "File size: $FILE_SIZE bytes, but no JSONL lines found"
            FAILED_RUNS=$((FAILED_RUNS + 1))
            rm -f "$RAW_JSONL_FILE"
            continue
        fi
        # Check if first line is valid JSON (not an error message)
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
        
        log_success "Verified run.jsonl: $FILE_SIZE bytes, $LINE_COUNT events"
    fi
    else
        log_error "run.jsonl file not found"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        continue
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

cat > "$RUN_OUT_DIR/manifest.json" <<EOF
{
    "run_id": "$RUN_EXP_ID",
    "run_index": $RUN_INDEX,
    "scenario_id": "$SCENARIO_ID",
    "scenario_path": "$SCENARIO",
    "environment": "kubernetes",
    "execution_type": "minikube",
    "git_commit": "$GIT_COMMIT",
    "start_time_utc": "$START_ISO",
    "end_time_utc": "$END_ISO",
    "duration_sec": $ELAPSED,
    "events_count": $EVENT_COUNT,
    "rng_seed": $MANIFEST_SEED,
    "replicas": $REPLICAS,
    "scaling_mode": $SCALING_MODE,
    "kubernetes": {
        "node_name": "$NODE_NAME",
        "k8s_version": "$K8S_VERSION",
        "namespace": "$NAMESPACE",
        "job_name": "$JOB_NAME",
        "pod_name": "$POD_NAME"
    },
    "container": {
        "image": "$IMAGE_NAME:$IMAGE_TAG",
        "runtime": "podman"
    },
    "minikube": {
        "version": "$MINIKUBE_VERSION",
        "driver": "podman"
    }
}
EOF

log_success "Manifest written: $RUN_OUT_DIR/manifest.json"

# =============================================================================
# Step 9: Run analysis pipeline
# =============================================================================
log_step "Step 9/9: Running analysis pipeline"

if [[ "$SKIP_ANALYSIS" == "true" ]]; then
    log_warn "Skipping analysis (--skip-analysis)"
else
    log_info "Running analysis pipeline..."
    
    # Run merge
    python3 "$SCRIPT_DIR/analysis/scripts/merge_jsonl.py" \
        --input "$RUN_OUT_DIR/raw" \
        --output "$RUN_OUT_DIR/merged" 2>/dev/null || true
    
    # Run stats
    INPUT_FILE="$RUN_OUT_DIR/merged/merged.parquet"
    [[ ! -f "$INPUT_FILE" ]] && INPUT_FILE="$RUN_OUT_DIR/merged/merged.jsonl"
    [[ ! -f "$INPUT_FILE" ]] && INPUT_FILE="$RUN_OUT_DIR/raw/run.jsonl"
    
    python3 "$SCRIPT_DIR/analysis/scripts/compute_statistics.py" \
        --input "$INPUT_FILE" \
        --output "$RUN_OUT_DIR/stats" \
        --experiment-id "$RUN_EXP_ID" 2>/dev/null || true
    
    log_success "Analysis complete for run $RUN_INDEX"
fi

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

log_success "Done!"

exit 0


