#!/usr/bin/env bash
# =============================================================================
# deploy_gcp.sh - Deploy and run PQC benchmark on GKE
#
# Deploys Terraform infrastructure, builds and pushes container image,
# runs benchmark Job, and uploads results to GCS.
#
# Usage:
#   ./deploy_gcp.sh \
#     --scenario scenarios/hybrid_kyber_dilithium.yaml \
#     --exp-id exp3 \
#     --project my-gcp-project \
#     --region us-central1 \
#     --bucket pqc-bench-results
#
# Requirements:
#   - gcloud CLI authenticated
#   - Terraform >= 1.6
#   - Podman >= 4.0
#   - kubectl
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="$SCRIPT_DIR/terraform/gke"
K8S_GCP_DIR="$SCRIPT_DIR/k8s/gcp"
JOB_NAME="pqc-bench-worker"
JOB_TIMEOUT="900s"

# Default values
SCENARIO=""
EXP_ID=""
PROJECT=""
REGION="us-central1"
BUCKET=""
MACHINE_TYPE="n2-standard-2"
NODE_COUNT=1
RUNS=1
REPLICAS=1
SEED=""
SKIP_TERRAFORM=false
SKIP_BUILD=false
SKIP_AGGREGATION=false
DESTROY_AFTER=false

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

Deploy and run PQC benchmark on GKE.
Supports multiple repeated runs with aggregated statistics.

OPTIONS:
    --scenario PATH       Path to scenario YAML file (required)
    --exp-id ID           Experiment identifier (required)
    --project ID          GCP project ID (required)
    --region REGION       GCP region (default: us-central1)
    --bucket NAME         GCS bucket name (required)
    --runs N              Number of repeated runs (default: 1)
    --seed NUM            Base RNG seed (each run gets seed+run_index)
    --machine-type TYPE   GKE node machine type (default: n2-standard-2)
    --node-count N        Number of nodes (default: 1)
    --skip-terraform      Skip Terraform apply (use existing cluster)
    --skip-build          Skip container image build
    --skip-aggregation    Skip aggregation across runs
    --destroy-after       Destroy infrastructure after experiment
    --timeout SEC         Job timeout in seconds (default: 900)
    -h, --help            Show this help message

EXAMPLES:
    # Single run
    $0 --scenario scenarios/hybrid_kyber_dilithium.yaml \\
       --exp-id exp3 \\
       --project my-project \\
       --region us-central1 \\
       --bucket pqc-results-bucket
EOF
    exit 1
}

cleanup_job() {
    log_info "Cleaning up previous job..."
    kubectl delete job "$JOB_NAME" --ignore-not-found=true 2>/dev/null || true
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
        --exp-id)
            EXP_ID="$2"
            shift 2
            ;;
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --bucket)
            BUCKET="$2"
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
        --machine-type)
            MACHINE_TYPE="$2"
            shift 2
            ;;
        --node-count)
            NODE_COUNT="$2"
            shift 2
            ;;
        --skip-terraform)
            SKIP_TERRAFORM=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-aggregation)
            SKIP_AGGREGATION=true
            shift
            ;;
        --destroy-after)
            DESTROY_AFTER=true
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

if [[ -z "$EXP_ID" ]]; then
    log_error "Missing required argument: --exp-id"
    usage
fi

if [[ -z "$PROJECT" ]]; then
    log_error "Missing required argument: --project"
    usage
fi

if [[ -z "$BUCKET" ]]; then
    log_error "Missing required argument: --bucket"
    usage
fi

if [[ ! -f "$SCENARIO" ]]; then
    log_error "Scenario file not found: $SCENARIO"
    exit 1
fi

# Make paths absolute
SCENARIO="$(cd "$(dirname "$SCENARIO")" && pwd)/$(basename "$SCENARIO")"

# Derived values
CLUSTER_NAME="pqc-bench-gke"
IMAGE_REPO="${REGION}-docker.pkg.dev/${PROJECT}/pqc"
IMAGE_NAME="${IMAGE_REPO}/pqc-bench:latest"
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
START_TIME=$(date +%s)
START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           PQC Benchmark - GCP/GKE Deployment                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

log_info "Experiment ID: $EXP_ID"
log_info "Project: $PROJECT"
log_info "Region: $REGION"
log_info "Bucket: $BUCKET"
log_info "Scenario: $SCENARIO"
log_info "Runs: $RUNS"
[[ -n "$SEED" ]] && log_info "Base RNG seed: $SEED"
log_info "Git Commit: ${GIT_COMMIT:0:8}"
log_info "Started: $START_ISO"

# =============================================================================
# Step 1: Verify prerequisites
# =============================================================================
log_step "Step 1/8: Verifying prerequisites"

# Check gcloud
if ! command -v gcloud &> /dev/null; then
    log_error "gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi
log_success "gcloud: $(gcloud version 2>/dev/null | head -1)"

# Check gcloud authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1 | grep -q "@"; then
    log_error "gcloud not authenticated. Run: gcloud auth login"
    exit 1
fi
GCLOUD_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1)
log_success "gcloud authenticated as: $GCLOUD_ACCOUNT"

# Check Terraform
if ! command -v terraform &> /dev/null; then
    log_error "Terraform not found. Please install Terraform >= 1.6"
    exit 1
fi
TF_VERSION=$(terraform version -json 2>/dev/null | grep -o '"terraform_version":"[^"]*"' | cut -d'"' -f4 || terraform version | head -1)
log_success "Terraform: $TF_VERSION"

# Check Podman
if ! command -v podman &> /dev/null; then
    log_error "Podman not found"
    exit 1
fi
log_success "Podman: $(podman --version | grep -oE '[0-9]+\.[0-9]+' | head -1)"

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    log_error "kubectl not found"
    exit 1
fi
log_success "kubectl available"

# =============================================================================
# Step 2: Run Terraform
# =============================================================================
log_step "Step 2/8: Deploying infrastructure with Terraform"

if [[ "$SKIP_TERRAFORM" == "true" ]]; then
    log_warn "Skipping Terraform (--skip-terraform)"
else
    cd "$TERRAFORM_DIR"
    
    log_info "Initializing Terraform..."
    terraform init -input=false
    
    log_info "Applying Terraform configuration..."
    terraform apply -auto-approve \
        -var="project_id=$PROJECT" \
        -var="region=$REGION" \
        -var="bucket_name=$BUCKET" \
        -var="machine_type=$MACHINE_TYPE" \
        -var="node_count=$NODE_COUNT" \
        -var="cluster_name=$CLUSTER_NAME"
    
    log_success "Infrastructure deployed"
    
    # Get outputs
    AR_REPO=$(terraform output -raw artifact_registry_repository)
    SA_EMAIL=$(terraform output -raw service_account_email)
    
    cd "$SCRIPT_DIR"
fi

# =============================================================================
# Step 3: Configure kubectl
# =============================================================================
log_step "Step 3/8: Configuring kubectl"

log_info "Getting cluster credentials..."
gcloud container clusters get-credentials "$CLUSTER_NAME" \
    --region "$REGION" \
    --project "$PROJECT"

# Verify connection
if ! kubectl cluster-info &> /dev/null; then
    log_error "Failed to connect to GKE cluster"
    exit 1
fi

log_success "kubectl configured for $CLUSTER_NAME"

# Get service account email from Terraform if not already set
if [[ -z "${SA_EMAIL:-}" ]]; then
    SA_EMAIL=$(cd "$TERRAFORM_DIR" && terraform output -raw service_account_email 2>/dev/null || echo "pqc-bench-worker@${PROJECT}.iam.gserviceaccount.com")
fi

# =============================================================================
# Step 4: Build and push container image
# =============================================================================
log_step "Step 4/8: Building and pushing container image"

if [[ "$SKIP_BUILD" == "true" ]]; then
    log_warn "Skipping build (--skip-build)"
else
    # Configure Podman for Artifact Registry
    log_info "Configuring Podman authentication..."
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
    
    # Also configure for Podman specifically
    gcloud auth print-access-token | podman login -u oauth2accesstoken --password-stdin "${REGION}-docker.pkg.dev" 2>/dev/null || true
    
    # Build image
    log_info "Building container image..."
    podman build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Containerfile" "$SCRIPT_DIR"
    
    # Push image
    log_info "Pushing image to Artifact Registry..."
    podman push "$IMAGE_NAME"
    
    log_success "Image pushed: $IMAGE_NAME"
fi

# =============================================================================
# Step 5: Apply Kubernetes manifests
# =============================================================================
log_step "Step 5/8: Deploying Kubernetes resources"

# Clean up any existing job
cleanup_job

# Create scenario ConfigMap
log_info "Creating scenario ConfigMap..."
kubectl create configmap pqc-scenario \
    --from-file=scenario.yaml="$SCENARIO" \
    --dry-run=client -o yaml | kubectl apply -f -

# Create GCP config ConfigMap
log_info "Creating GCP config ConfigMap..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: pqc-gcp-config
  namespace: default
data:
  bucket_name: "$BUCKET"
  experiment_id: "$EXP_ID"
  git_commit: "$GIT_COMMIT"
  container_image: "$IMAGE_NAME"
  region: "$REGION"
  project_id: "$PROJECT"
EOF

# Create/update ServiceAccount with Workload Identity annotation
log_info "Creating ServiceAccount with Workload Identity..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: pqc-bench-sa
  namespace: default
  annotations:
    iam.gke.io/gcp-service-account: "$SA_EMAIL"
EOF

# Apply worker job (with image placeholder replaced)
log_info "Deploying worker Job..."
sed "s|PLACEHOLDER_IMAGE|$IMAGE_NAME|g" "$K8S_GCP_DIR/worker-job.yaml" | \
    sed "s|cloud.google.com/gke-nodepool: pqc-bench-pool|cloud.google.com/gke-nodepool: pqc-bench-pool|g" | \
    kubectl apply -f -

log_success "Kubernetes resources deployed"

# =============================================================================
# Step 6: Wait for Job completion
# =============================================================================
log_step "Step 6/8: Waiting for Job completion"

log_info "Waiting for Job to complete (timeout: $JOB_TIMEOUT)..."

# Wait for pod to be created
sleep 10

# Get pod name
POD_NAME=$(kubectl get pods -l job-name="$JOB_NAME" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

if [[ -n "$POD_NAME" ]]; then
    log_info "Pod: $POD_NAME"
    
    # Stream logs in background
    (
        sleep 30
        kubectl logs -f "$POD_NAME" -c pqc-bench 2>/dev/null | while read -r line; do
            echo "  [pqc-bench] $line"
        done
    ) &
    LOG_PID=$!
fi

# Wait for completion
if ! kubectl wait --for=condition=complete job/"$JOB_NAME" --timeout="$JOB_TIMEOUT"; then
    # Check if failed
    JOB_STATUS=$(kubectl get job "$JOB_NAME" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || echo "")
    
    if [[ "$JOB_STATUS" == "True" ]]; then
        log_error "Job failed!"
        kubectl describe job "$JOB_NAME"
        kubectl logs -l job-name="$JOB_NAME" -c pqc-bench --tail=100
        exit 1
    fi
    
    log_error "Job timed out"
    exit 1
fi

# Kill log streaming
kill $LOG_PID 2>/dev/null || true

log_success "Job completed successfully"

# =============================================================================
# Step 7: Verify GCS artifacts
# =============================================================================
log_step "Step 7/8: Verifying GCS artifacts"

log_info "Checking GCS bucket for results..."

# Wait for upload to complete
sleep 15

# List artifacts
ARTIFACTS=$(gsutil ls "gs://${BUCKET}/experiments/${EXP_ID}/" 2>/dev/null || echo "")

if [[ -z "$ARTIFACTS" ]]; then
    log_error "No artifacts found in GCS!"
    log_info "Checking upload container logs..."
    kubectl logs -l job-name="$JOB_NAME" -c upload-results --tail=50
    exit 1
fi

echo "$ARTIFACTS"

# Verify required files
REQUIRED_FILES=("merged.jsonl" "manifest.json" "provenance.json")
for file in "${REQUIRED_FILES[@]}"; do
    if echo "$ARTIFACTS" | grep -q "$file"; then
        log_success "Found: $file"
    else
        log_warn "Missing: $file"
    fi
done

# =============================================================================
# Step 8: Summary
# =============================================================================
END_TIME=$(date +%s)
END_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  GCP DEPLOYMENT COMPLETE                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

log_info "Experiment ID: $EXP_ID"
log_info "Duration: ${ELAPSED}s"
echo ""

log_info "GCS artifacts location:"
echo "  gs://${BUCKET}/experiments/${EXP_ID}/"
echo ""

log_info "To fetch and analyze results locally:"
echo "  ./fetch_and_analyse_from_gcs.sh \\"
echo "    --exp-id $EXP_ID \\"
echo "    --bucket $BUCKET \\"
echo "    --out results/$EXP_ID"
echo ""

log_info "To compare with other environments:"
echo "  python analysis/compare_all_environments.py \\"
echo "    --native results/exp1/stats/summary.json \\"
echo "    --minikube results/exp2/stats/summary.json \\"
echo "    --gcp results/$EXP_ID/stats/summary.json"
echo ""

# Cleanup if requested
if [[ "$DESTROY_AFTER" == "true" ]]; then
    log_warn "Destroying infrastructure (--destroy-after)..."
    cd "$TERRAFORM_DIR"
    terraform destroy -auto-approve \
        -var="project_id=$PROJECT" \
        -var="region=$REGION" \
        -var="bucket_name=$BUCKET"
    cd "$SCRIPT_DIR"
fi

log_success "Done!"

exit 0

