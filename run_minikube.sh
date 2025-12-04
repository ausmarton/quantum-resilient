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
#   - Podman >= 4.0
#   - Minikube >= 1.34
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
SKIP_BUILD=false
SKIP_ANALYSIS=false
KEEP_JOB=false

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

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Run PQC benchmark experiment in Minikube Kubernetes cluster.

OPTIONS:
    --scenario PATH     Path to scenario YAML file (required)
    --out DIR           Output directory for results (required)
    --exp-id ID         Experiment identifier (required)
    --skip-build        Skip container image build
    --skip-analysis     Skip Python analysis after run
    --keep-job          Don't delete Job after completion
    --timeout SEC       Job timeout in seconds (default: 600)
    -h, --help          Show this help message

EXAMPLE:
    $0 --scenario scenarios/hybrid_kyber_dilithium.yaml \\
       --out results/k8s_exp1 --exp-id k8s_exp1

PREREQUISITES:
    1. Start Minikube with Podman driver:
       minikube start --driver=podman

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
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-analysis)
            SKIP_ANALYSIS=true
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
log_info "Started: $START_ISO"

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
log_success "Minikube cluster is running"

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

if [[ "$SKIP_BUILD" == "true" ]]; then
    log_warn "Skipping build (--skip-build)"
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
fi

# =============================================================================
# Step 4: Load image into Minikube
# =============================================================================
log_step "Step 4/9: Loading image into Minikube"

log_info "Loading image into Minikube..."
minikube image load "$IMAGE_NAME:$IMAGE_TAG" 2>&1 | while read -r line; do
    echo "  $line"
done

log_success "Image loaded into Minikube"

# Verify image is available
if ! minikube image ls 2>/dev/null | grep -q "$IMAGE_NAME"; then
    log_warn "Image may not be visible in minikube image ls, continuing..."
fi

# =============================================================================
# Step 5: Deploy Kubernetes resources
# =============================================================================
log_step "Step 5/9: Deploying Kubernetes resources"

# Clean up any existing job
cleanup

# Apply PVC
log_info "Creating PersistentVolumeClaim..."
kubectl apply -f "$SCRIPT_DIR/k8s/results-pvc.yaml" -n "$NAMESPACE"

# Wait for PVC to be bound
log_info "Waiting for PVC to be bound..."
kubectl wait --for=jsonpath='{.status.phase}'=Bound pvc/"$PVC_NAME" -n "$NAMESPACE" --timeout=60s || {
    log_warn "PVC may not be bound immediately, continuing..."
}

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

kubectl create configmap "$CONFIGMAP_NAME" \
    --from-file=scenario.yaml="$TEMP_SCENARIO" \
    --dry-run=client -o yaml | kubectl apply -f - -n "$NAMESPACE"

rm -f "$TEMP_SCENARIO"
log_success "ConfigMap created"

# Apply Job
log_info "Creating Job..."
kubectl apply -f "$SCRIPT_DIR/k8s/worker-job.yaml" -n "$NAMESPACE"

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
# Step 7: Retrieve results from cluster
# =============================================================================
log_step "Step 7/9: Retrieving results from cluster"

# Get pod name (may have changed)
POD_NAME=$(kubectl get pods -l job-name="$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}')
log_info "Copying results from pod: $POD_NAME"

# Copy results
kubectl cp "$NAMESPACE/$POD_NAME:/results/." "$OUT_DIR/raw/" 2>/dev/null || {
    # Try alternative method if cp fails
    log_warn "kubectl cp failed, trying alternative method..."
    kubectl exec "$POD_NAME" -n "$NAMESPACE" -- cat /results/raw/run.jsonl > "$OUT_DIR/raw/run.jsonl" 2>/dev/null || true
    kubectl exec "$POD_NAME" -n "$NAMESPACE" -- cat /results/container_metadata.json > "$OUT_DIR/container_metadata.json" 2>/dev/null || true
}

# Move files to correct locations if needed
if [[ -f "$OUT_DIR/raw/raw/run.jsonl" ]]; then
    mv "$OUT_DIR/raw/raw/"* "$OUT_DIR/raw/" 2>/dev/null || true
    rmdir "$OUT_DIR/raw/raw" 2>/dev/null || true
fi

# Copy container metadata to root
if [[ -f "$OUT_DIR/raw/container_metadata.json" ]]; then
    cp "$OUT_DIR/raw/container_metadata.json" "$OUT_DIR/"
fi

# Verify results
if [[ ! -f "$OUT_DIR/raw/run.jsonl" ]] && [[ $(find "$OUT_DIR/raw" -name "*.jsonl" 2>/dev/null | wc -l) -eq 0 ]]; then
    log_error "No JSONL files found in results!"
    log_info "Contents of $OUT_DIR/raw:"
    ls -la "$OUT_DIR/raw/" || true
    exit 1
fi

JSONL_COUNT=$(find "$OUT_DIR/raw" -name "*.jsonl" | wc -l)
log_success "Retrieved $JSONL_COUNT JSONL file(s)"

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
if [[ -f "$OUT_DIR/raw/run.jsonl" ]]; then
    EVENT_COUNT=$(wc -l < "$OUT_DIR/raw/run.jsonl")
else
    EVENT_COUNT=$(cat "$OUT_DIR/raw/"*.jsonl 2>/dev/null | wc -l || echo 0)
fi

# Extract scenario ID and seed from scenario file
SCENARIO_ID=$(grep -E "^id:" "$SCENARIO" | awk '{print $2}' | tr -d '"' || echo "$EXP_ID")
RNG_SEED=$(grep -E "^rng_seed:" "$SCENARIO" | awk '{print $2}' || echo "null")

cat > "$OUT_DIR/manifest.json" <<EOF
{
    "run_id": "$EXP_ID",
    "scenario_id": "$SCENARIO_ID",
    "scenario_path": "$SCENARIO",
    "environment": "kubernetes",
    "execution_type": "minikube",
    "git_commit": "$GIT_COMMIT",
    "start_time_utc": "$START_ISO",
    "end_time_utc": "$END_ISO",
    "duration_sec": $ELAPSED,
    "events_count": $EVENT_COUNT,
    "rng_seed": $RNG_SEED,
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

log_success "Manifest written: $OUT_DIR/manifest.json"

# =============================================================================
# Step 9: Run analysis pipeline
# =============================================================================
log_step "Step 9/9: Running analysis pipeline"

if [[ "$SKIP_ANALYSIS" == "true" ]]; then
    log_warn "Skipping analysis (--skip-analysis)"
else
    log_info "Running analysis pipeline..."
    
    if [[ -f "$SCRIPT_DIR/analysis/run_full_pipeline.sh" ]]; then
        bash "$SCRIPT_DIR/analysis/run_full_pipeline.sh" "$EXP_ID" "$OUT_DIR/raw" 2>&1 | while read -r line; do
            echo "  $line"
        done || log_warn "Analysis pipeline completed with warnings"
    else
        log_warn "Analysis pipeline script not found, running individual scripts..."
        
        # Run merge
        python3 "$SCRIPT_DIR/analysis/scripts/merge_jsonl.py" \
            --input "$OUT_DIR/raw" \
            --output "$OUT_DIR/merged" 2>/dev/null || true
        
        # Run stats
        INPUT_FILE="$OUT_DIR/merged/merged.parquet"
        [[ ! -f "$INPUT_FILE" ]] && INPUT_FILE="$OUT_DIR/merged/merged.jsonl"
        
        python3 "$SCRIPT_DIR/analysis/scripts/compute_statistics.py" \
            --input "$INPUT_FILE" \
            --output "$OUT_DIR/stats" \
            --experiment-id "$EXP_ID" 2>/dev/null || \
        python3 "$SCRIPT_DIR/analysis/scripts/compute_stats.py" \
            --input "$INPUT_FILE" \
            --output "$OUT_DIR/stats" \
            --experiment-id "$EXP_ID" 2>/dev/null || true
        
        # Run plots
        python3 "$SCRIPT_DIR/analysis/scripts/plot_ecdf.py" \
            --input "$INPUT_FILE" \
            --output "$OUT_DIR/figures" \
            --experiment-id "$EXP_ID" 2>/dev/null || \
        python3 "$SCRIPT_DIR/analysis/scripts/plot_latency.py" \
            --input "$INPUT_FILE" \
            --output "$OUT_DIR/figures" \
            --experiment-id "$EXP_ID" 2>/dev/null || true
        
        python3 "$SCRIPT_DIR/analysis/scripts/plot_throughput.py" \
            --input "$INPUT_FILE" \
            --output "$OUT_DIR/figures" \
            --experiment-id "$EXP_ID" 2>/dev/null || true
    fi
    
    log_success "Analysis complete"
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
log_info "Duration: ${ELAPSED}s"
log_info "Events: $EVENT_COUNT"
echo ""

log_info "Output files:"
echo "  Raw JSONL:     $OUT_DIR/raw/"
[[ -f "$OUT_DIR/merged/merged.jsonl" ]] && echo "  Merged JSONL:  $OUT_DIR/merged/merged.jsonl"
[[ -f "$OUT_DIR/merged/merged.parquet" ]] && echo "  Parquet:       $OUT_DIR/merged/merged.parquet"
[[ -f "$OUT_DIR/stats/summary.json" ]] && echo "  Stats:         $OUT_DIR/stats/summary.json"
[[ -f "$OUT_DIR/figures/latency_cdf.png" ]] && echo "  Latency CDF:   $OUT_DIR/figures/latency_cdf.png"
[[ -f "$OUT_DIR/figures/throughput.png" ]] && echo "  Throughput:    $OUT_DIR/figures/throughput.png"
echo "  Manifest:      $OUT_DIR/manifest.json"
echo ""

log_info "To compare with native run:"
echo "  python analysis/compare_native_vs_minikube.py \\"
echo "    --native results/native_exp/stats/summary.json \\"
echo "    --k8s $OUT_DIR/stats/summary.json"
echo ""

log_success "Done!"

exit 0

