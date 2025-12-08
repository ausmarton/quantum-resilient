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
SMOKE_TEST=false
EPHEMERAL=false
CREATE_CLUSTER=false
DESTROY_CLUSTER=false

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
    --replicas N          Number of replicas (default: 1)
    --seed NUM            Base RNG seed (each run gets seed+run_index)
    --machine-type TYPE   GKE node machine type (default: n2-standard-2)
    --node-count N        Number of nodes (default: 1)
    --skip-terraform      Skip Terraform apply (use existing cluster)
    --skip-build          Skip container image build
    --skip-aggregation    Skip aggregation across runs
    --destroy-after       Destroy infrastructure after experiment
    --timeout SEC         Job timeout in seconds (default: 900)
    --smoke-test          Enable smoke-test mode (minimal infrastructure)
    --ephemeral           Ephemeral mode: create cluster, run benchmark, destroy all resources
    --create-cluster      Only create the cluster (skip benchmark execution)
    --destroy-cluster     Only destroy the cluster and cleanup resources
    -h, --help            Show this help message

EXAMPLES:
    # Single run
    $0 --scenario scenarios/hybrid_kyber_dilithium.yaml \\
       --exp-id exp3 \\
       --project my-project \\
       --region us-central1 \\
       --bucket pqc-results-bucket
    
    # Ephemeral smoke test (creates, runs, destroys everything)
    $0 --scenario scenarios/hybrid_kyber_dilithium.yaml \\
       --exp-id smoketest \\
       --project my-project \\
       --region us-central1 \\
       --bucket pqc-results-bucket \\
       --smoke-test \\
       --ephemeral
    
    # Create cluster only
    $0 --create-cluster \\
       --project my-project \\
       --region us-central1 \\
       --bucket pqc-results-bucket \\
       --cluster-name my-cluster
    
    # Destroy cluster only
    $0 --destroy-cluster \\
       --project my-project \\
       --region us-central1 \\
       --bucket pqc-results-bucket \\
       --cluster-name my-cluster
EOF
    exit 1
}

cleanup_job() {
    local namespace=${1:-default}
    log_info "Cleaning up previous job..."
    kubectl delete job "$JOB_NAME" --namespace="$namespace" --ignore-not-found=true 2>/dev/null || true
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
        --smoke-test)
            SMOKE_TEST=true
            shift
            ;;
        --ephemeral)
            EPHEMERAL=true
            shift
            ;;
        --create-cluster)
            CREATE_CLUSTER=true
            shift
            ;;
        --destroy-cluster)
            DESTROY_CLUSTER=true
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
# Skip scenario/exp-id validation for create-cluster and destroy-cluster modes
if [[ "$CREATE_CLUSTER" != "true" && "$DESTROY_CLUSTER" != "true" ]]; then
    if [[ -z "$SCENARIO" ]]; then
        log_error "Missing required argument: --scenario"
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
fi

if [[ -z "$PROJECT" ]]; then
    log_error "Missing required argument: --project"
    usage
fi

if [[ -z "$BUCKET" ]]; then
    log_error "Missing required argument: --bucket"
    usage
fi

# Make paths absolute (if scenario provided)
if [[ -n "$SCENARIO" ]]; then
    SCENARIO="$(cd "$(dirname "$SCENARIO")" && pwd)/$(basename "$SCENARIO")"
fi

# Derived values
# CRITICAL: Hardware MUST remain identical between smoke-test and full runs
# Only horizontal scaling (node_count, replicas) may change
if [[ "$SMOKE_TEST" == "true" ]]; then
    CLUSTER_NAME="pqc-smoke-test"
    # DO NOT change MACHINE_TYPE - must stay identical to full runs
    # Only reduce node_count (horizontal scaling)
    NODE_COUNT=1
    RUNS=1
    REPLICAS=1
    JOB_TIMEOUT="300s"  # 5 minutes
    log_info "Smoke-test mode: reduced duration, runs, replicas (hardware identical)"
else
    CLUSTER_NAME="pqc-bench-gke"
fi

# Ephemeral mode overrides
if [[ "$EPHEMERAL" == "true" ]]; then
    log_info "EPHEMERAL MODE: Will create cluster, run benchmark, and destroy all resources"
    SKIP_TERRAFORM=false
    DESTROY_AFTER=true
fi
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

[[ -n "$EXP_ID" ]] && log_info "Experiment ID: $EXP_ID"
log_info "Project: $PROJECT"
log_info "Region: $REGION"
log_info "Bucket: $BUCKET"
[[ -n "$SCENARIO" ]] && log_info "Scenario: $SCENARIO"
[[ -n "$EXP_ID" ]] && log_info "Runs: $RUNS"
[[ -n "$EXP_ID" ]] && log_info "Replicas: $REPLICAS"
[[ "$SMOKE_TEST" == "true" ]] && log_info "Mode: SMOKE-TEST (minimal infrastructure)"
[[ "$EPHEMERAL" == "true" ]] && log_info "Mode: EPHEMERAL (will destroy all resources after completion)"
[[ -n "$SEED" ]] && log_info "Base RNG seed: $SEED"
log_info "Git Commit: ${GIT_COMMIT:0:8}"
log_info "Started: $START_ISO"

# =============================================================================
# Handle create-cluster mode (early exit after creation)
# =============================================================================
if [[ "$CREATE_CLUSTER" == "true" ]]; then
    log_step "Creating GKE cluster only (--create-cluster)"
    
    cd "$TERRAFORM_DIR"
    log_info "Initializing Terraform..."
    terraform init -input=false
    
    # Remove Kubernetes service account from state if it exists
    # (We removed it from the Terraform config because it's created by kubectl)
    log_info "Cleaning up Terraform state (removing Kubernetes resources if present)..."
    if terraform state list 2>/dev/null | grep -q "kubernetes_service_account"; then
        log_info "Removing Kubernetes service account from Terraform state..."
        terraform state rm 'kubernetes_service_account_v1.pqc_bench[0]' 2>/dev/null || \
        terraform state rm kubernetes_service_account_v1.pqc_bench[0] 2>/dev/null || \
        terraform state rm 'module.kubernetes_service_account_v1.pqc_bench[0]' 2>/dev/null || true
        log_success "Kubernetes service account removed from state"
    else
        log_info "No Kubernetes service account in state (this is expected)"
    fi
    
    log_info "Applying Terraform configuration to create cluster..."
    DISK_SIZE_GB="${DISK_SIZE_GB:-50}"
    if terraform apply -auto-approve \
        -var="project_id=$PROJECT" \
        -var="region=$REGION" \
        -var="bucket_name=$BUCKET" \
        -var="machine_type=$MACHINE_TYPE" \
        -var="node_count=$NODE_COUNT" \
        -var="cluster_name=$CLUSTER_NAME" \
        -var="smoke_test=$SMOKE_TEST" \
        -var="disk_size_gb=$DISK_SIZE_GB" \
        -var="ephemeral=$EPHEMERAL"; then
        log_success "Cluster created successfully"
    else
        log_error "Cluster creation failed"
        exit 1
    fi
    
    cd "$SCRIPT_DIR"
    log_success "Cluster creation complete. Use --destroy-cluster to remove it."
    exit 0
fi

# =============================================================================
# Handle destroy-cluster mode (early exit)
# =============================================================================
if [[ "$DESTROY_CLUSTER" == "true" ]]; then
    log_step "Destroying GKE cluster and cleaning up resources"
    
    # First, try Terraform destroy (if Terraform state exists)
    cd "$TERRAFORM_DIR"
    if [[ -f terraform.tfstate ]] || [[ -f .terraform/terraform.tfstate ]]; then
        # CRITICAL: Remove persistent resources from state before destroy
        log_info "Protecting persistent resources from destruction..."
        terraform state rm google_storage_bucket.results 2>/dev/null || true
        terraform state rm google_storage_bucket_object.experiments_marker 2>/dev/null || true
        terraform state rm google_service_account.pqc_bench 2>/dev/null || true
        terraform state rm google_project_iam_member.ar_reader 2>/dev/null || true
        terraform state rm google_storage_bucket_iam_member.gcs_admin 2>/dev/null || true
        terraform state rm google_service_account_iam_member.workload_identity 2>/dev/null || true
        terraform state rm google_artifact_registry_repository.pqc 2>/dev/null || true
        
        log_info "Running Terraform destroy (cluster and node pool only)..."
        if terraform destroy -auto-approve \
            -var="project_id=$PROJECT" \
            -var="region=$REGION" \
            -var="bucket_name=$BUCKET" \
            -var="machine_type=$MACHINE_TYPE" \
            -var="node_count=$NODE_COUNT" \
            -var="cluster_name=$CLUSTER_NAME" \
            -var="smoke_test=$SMOKE_TEST" \
            -var="disk_size_gb=${DISK_SIZE_GB:-50}" \
            -var="ephemeral=true" 2>&1; then
            log_success "Terraform destroy completed (persistent resources preserved)"
        else
            log_warn "Terraform destroy had errors (cluster may not be in Terraform state)"
        fi
    else
        log_info "No Terraform state found, will use gcloud directly"
    fi
    
    cd "$SCRIPT_DIR"
    
    # Run cleanup script to ensure cluster is deleted (even if Terraform didn't work)
    log_info "Running cleanup script to ensure cluster is deleted..."
    if "$SCRIPT_DIR/scripts/cleanup_gcp_resources.sh" \
        --project "$PROJECT" \
        --region "$REGION" \
        --cluster-name "$CLUSTER_NAME"; then
        log_success "Cluster destruction and cleanup complete"
    else
        log_warn "Cleanup script had errors, but cluster deletion may still be in progress"
        log_info "Cluster deletion can take 5-10 minutes. Check status with:"
        echo "  gcloud container clusters list --project $PROJECT --region $REGION"
    fi
    
    exit 0
fi

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
    
    # Optimize: Only run terraform init if not already initialized
    if [[ ! -d ".terraform" ]] || [[ ! -f ".terraform/terraform.tfstate" ]] || [[ -z "$(ls -A .terraform/providers 2>/dev/null)" ]]; then
        log_info "Initializing Terraform..."
        terraform init -input=false
    else
        log_info "Terraform already initialized, skipping init (using cached providers)"
    fi
    
    # Import existing resources into Terraform state (needed for ephemeral mode)
    # Optimize: Only import if not already in state
    log_info "Checking for existing resources to import into Terraform state..."
    
    # Import bucket if it exists and not in state
    if gsutil ls -b "gs://${BUCKET}" &>/dev/null; then
        if ! terraform state show google_storage_bucket.results &>/dev/null; then
            log_info "Bucket exists but not in state, importing..."
            terraform import \
                -var="project_id=$PROJECT" \
                -var="region=$REGION" \
                -var="bucket_name=$BUCKET" \
                -var="machine_type=$MACHINE_TYPE" \
                -var="node_count=$NODE_COUNT" \
                -var="cluster_name=$CLUSTER_NAME" \
                -var="smoke_test=$SMOKE_TEST" \
                -var="disk_size_gb=${DISK_SIZE_GB:-50}" \
                -var="ephemeral=$EPHEMERAL" \
                google_storage_bucket.results "$BUCKET" 2>/dev/null || log_warn "Bucket import failed (may already be in state)"
        else
            log_info "Bucket already in Terraform state, skipping import"
        fi
    fi
    
    # Import service account if it exists and not in state
    SA_EMAIL="pqc-bench-worker@${PROJECT}.iam.gserviceaccount.com"
    if gcloud iam service-accounts describe "$SA_EMAIL" \
        --project "$PROJECT" >/dev/null 2>&1; then
        if ! terraform state show google_service_account.pqc_bench &>/dev/null; then
            log_info "Service account exists but not in state, importing..."
            terraform import \
                -var="project_id=$PROJECT" \
                -var="region=$REGION" \
                -var="bucket_name=$BUCKET" \
                -var="machine_type=$MACHINE_TYPE" \
                -var="node_count=$NODE_COUNT" \
                -var="cluster_name=$CLUSTER_NAME" \
                -var="smoke_test=$SMOKE_TEST" \
                -var="disk_size_gb=${DISK_SIZE_GB:-50}" \
                -var="ephemeral=$EPHEMERAL" \
                google_service_account.pqc_bench "projects/${PROJECT}/serviceAccounts/${SA_EMAIL}" 2>/dev/null || log_warn "Service account import failed (may already be in state)"
        else
            log_info "Service account already in Terraform state, skipping import"
        fi
    fi
    
    # Import Artifact Registry repository if it exists and not in state
    AR_LOCATION="${REGION}"
    AR_REPO="pqc"
    if gcloud artifacts repositories describe "$AR_REPO" \
        --location "$AR_LOCATION" \
        --project "$PROJECT" >/dev/null 2>&1; then
        if ! terraform state show google_artifact_registry_repository.pqc &>/dev/null; then
            log_info "Artifact Registry repository exists but not in state, importing..."
            terraform import \
                -var="project_id=$PROJECT" \
                -var="region=$REGION" \
                -var="bucket_name=$BUCKET" \
                -var="machine_type=$MACHINE_TYPE" \
                -var="node_count=$NODE_COUNT" \
                -var="cluster_name=$CLUSTER_NAME" \
                -var="smoke_test=$SMOKE_TEST" \
                -var="disk_size_gb=${DISK_SIZE_GB:-50}" \
                -var="ephemeral=$EPHEMERAL" \
                google_artifact_registry_repository.pqc "${AR_LOCATION}/${AR_REPO}" 2>/dev/null || log_warn "Artifact Registry import failed (may already be in state)"
        else
            log_info "Artifact Registry repository already in Terraform state, skipping import"
        fi
    fi
    
    # Remove Kubernetes service account from state if it exists
    # (We removed it from the Terraform config because it's created by kubectl)
    log_info "Cleaning up Terraform state (removing Kubernetes resources if present)..."
    if terraform state list 2>/dev/null | grep -q "kubernetes_service_account"; then
        log_info "Removing Kubernetes service account from Terraform state..."
        terraform state rm 'kubernetes_service_account_v1.pqc_bench[0]' 2>/dev/null || \
        terraform state rm kubernetes_service_account_v1.pqc_bench[0] 2>/dev/null || \
        terraform state rm 'module.kubernetes_service_account_v1.pqc_bench[0]' 2>/dev/null || true
        log_success "Kubernetes service account removed from state"
    else
        log_info "No Kubernetes service account in state (this is expected)"
    fi
    
    log_info "Applying Terraform configuration..."
    # CRITICAL: machine_type, disk_size_gb, disk_type MUST stay identical
    # Only node_count may differ (horizontal scaling only)
    # Use consistent disk_size_gb for both smoke-test and full runs
    DISK_SIZE_GB="${DISK_SIZE_GB:-50}"
    if terraform apply -auto-approve \
        -var="project_id=$PROJECT" \
        -var="region=$REGION" \
        -var="bucket_name=$BUCKET" \
        -var="machine_type=$MACHINE_TYPE" \
        -var="node_count=$NODE_COUNT" \
        -var="cluster_name=$CLUSTER_NAME" \
        -var="smoke_test=$SMOKE_TEST" \
        -var="disk_size_gb=$DISK_SIZE_GB" \
        -var="ephemeral=$EPHEMERAL"; then
        log_success "Infrastructure deployed"
    else
        log_error "Terraform apply failed. Checking node pool status..."
        
        # Try to get the actual error from GCP
        if command -v gcloud &> /dev/null; then
            log_info "Fetching node pool status from GCP..."
            gcloud container node-pools describe pqc-bench-pool \
                --cluster "$CLUSTER_NAME" \
                --region "$REGION" \
                --project "$PROJECT" 2>&1 | head -50 || true
            
            log_info "Checking cluster operations..."
            gcloud container operations list \
                --filter="clusterName=$CLUSTER_NAME AND location=$REGION" \
                --project "$PROJECT" \
                --limit 5 \
                --format="table(name,status,operationType)" || true
        fi
        
        log_error "See terraform/gke/DEBUG_NODE_POOL.md for troubleshooting steps"
        exit 1
    fi
    
    # Get outputs
    AR_REPO=$(terraform output -raw artifact_registry_repository)
    SA_EMAIL=$(terraform output -raw service_account_email)
    
    cd "$SCRIPT_DIR"
fi

# =============================================================================
# Step 3: Configure kubectl
# =============================================================================
log_step "Step 3/8: Configuring kubectl"

# Check for gke-gcloud-auth-plugin (required for GKE authentication)
log_info "Checking for gke-gcloud-auth-plugin..."
if ! command -v gke-gcloud-auth-plugin &> /dev/null; then
    log_warn "gke-gcloud-auth-plugin not found. Installing..."
    
    # Try to install via gcloud components
    if gcloud components install gke-gcloud-auth-plugin --quiet 2>/dev/null; then
        log_success "gke-gcloud-auth-plugin installed"
    else
        log_error "Failed to install gke-gcloud-auth-plugin automatically"
        log_info "Please install manually:"
        echo "  gcloud components install gke-gcloud-auth-plugin"
        echo ""
        echo "Or on Fedora/RHEL:"
        echo "  sudo dnf install google-cloud-sdk-gke-gcloud-auth-plugin"
        echo ""
        echo "Or on Ubuntu/Debian:"
        echo "  sudo apt-get install google-cloud-sdk-gke-gcloud-auth-plugin"
        echo ""
        exit 1
    fi
else
    log_success "gke-gcloud-auth-plugin found"
fi

log_info "Getting cluster credentials..."
gcloud container clusters get-credentials "$CLUSTER_NAME" \
    --region "$REGION" \
    --project "$PROJECT"

# Verify connection (with retry)
log_info "Verifying cluster connection..."
RETRY_COUNT=0
MAX_RETRIES=3
while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
    if kubectl cluster-info &> /dev/null; then
        log_success "kubectl configured for $CLUSTER_NAME"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; then
        log_warn "Connection failed, retrying ($RETRY_COUNT/$MAX_RETRIES)..."
        sleep 5
    else
        log_error "Failed to connect to GKE cluster after $MAX_RETRIES attempts"
        log_info "Troubleshooting steps:"
        echo "  1. Verify cluster exists:"
        echo "     gcloud container clusters describe $CLUSTER_NAME --region $REGION --project $PROJECT"
        echo ""
        echo "  2. Check gke-gcloud-auth-plugin:"
        echo "     which gke-gcloud-auth-plugin"
        echo ""
        echo "  3. Try manual credential refresh:"
        echo "     gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project $PROJECT"
        echo ""
        exit 1
    fi
done

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

# Set namespace based on smoke-test mode
if [[ "$SMOKE_TEST" == "true" ]]; then
    NAMESPACE="pqc-smoke-test"
    log_info "Creating smoke-test namespace..."
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
else
    NAMESPACE="default"
fi

# Clean up any existing job
cleanup_job "$NAMESPACE"

# Create scenario ConfigMap (with smoke-test overrides if needed)
log_info "Creating scenario ConfigMap..."
TEMP_SCENARIO=$(mktemp)
cp "$SCENARIO" "$TEMP_SCENARIO"

# CRITICAL: Ensure jsonl_out is always set to /results/raw/run.jsonl
# This is required for the upload sidecar to find the results
# Use Python for reliable YAML modification
if command -v python3 &> /dev/null && python3 -c "import yaml" 2>/dev/null; then
    python3 <<PYTHON_EOF
import yaml
import sys

with open('$TEMP_SCENARIO', 'r') as f:
    scenario = yaml.safe_load(f) or {}

# Ensure metrics section exists
if 'metrics' not in scenario:
    scenario['metrics'] = {}

# Always set jsonl_out to the correct path
scenario['metrics']['jsonl_out'] = '/results/raw/run.jsonl'

# Smoke-test duration override
if '$SMOKE_TEST' == 'true' and 'workload' in scenario:
    scenario['workload']['duration_sec'] = 5

# Write back
with open('$TEMP_SCENARIO', 'w') as f:
    yaml.dump(scenario, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
PYTHON_EOF
else
    # Fallback to sed if Python/YAML not available
    log_warn "Python3/PyYAML not available, using sed fallback"
    # Update existing jsonl_out
    sed -i 's|jsonl_out:.*|jsonl_out: /results/raw/run.jsonl|g' "$TEMP_SCENARIO"
    # If no jsonl_out found, add it after metrics: (or create metrics section)
    if ! grep -q "jsonl_out:" "$TEMP_SCENARIO"; then
        if grep -q "^metrics:" "$TEMP_SCENARIO"; then
            sed -i '/^metrics:/a\  jsonl_out: /results/raw/run.jsonl' "$TEMP_SCENARIO"
        else
            # Add metrics section before execution or at end
            if grep -q "^execution:" "$TEMP_SCENARIO"; then
                sed -i '/^execution:/i\metrics:\n  jsonl_out: /results/raw/run.jsonl\n' "$TEMP_SCENARIO"
            else
                echo "metrics:" >> "$TEMP_SCENARIO"
                echo "  jsonl_out: /results/raw/run.jsonl" >> "$TEMP_SCENARIO"
            fi
        fi
    fi
    
    if [[ "$SMOKE_TEST" == "true" ]]; then
        sed -i "s/duration_sec:.*/duration_sec: 5/" "$TEMP_SCENARIO"
    fi
fi

kubectl create configmap pqc-scenario \
    --from-file=scenario.yaml="$TEMP_SCENARIO" \
    --namespace="$NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f -
rm -f "$TEMP_SCENARIO"

# Create GCP config ConfigMap
log_info "Creating GCP config ConfigMap..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: pqc-gcp-config
  namespace: $NAMESPACE
data:
  bucket_name: "$BUCKET"
  experiment_id: "$EXP_ID"
  git_commit: "$GIT_COMMIT"
  container_image: "$IMAGE_NAME"
  region: "$REGION"
  project_id: "$PROJECT"
  smoke_test: "$([ "$SMOKE_TEST" == "true" ] && echo "true" || echo "false")"
EOF

# Create/update ServiceAccount with Workload Identity annotation
log_info "Creating ServiceAccount with Workload Identity..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: pqc-bench-sa
  namespace: $NAMESPACE
  annotations:
    iam.gke.io/gcp-service-account: "$SA_EMAIL"
EOF

# CRITICAL: Create IAM binding for Workload Identity
# The Terraform binding only covers 'default' namespace, so we need to create
# bindings for other namespaces (like 'pqc-smoke-test') dynamically
log_info "Creating Workload Identity IAM binding for namespace '$NAMESPACE'..."
K8S_SA="${PROJECT}.svc.id.goog[${NAMESPACE}/pqc-bench-sa]"

# Check if binding already exists
if gcloud iam service-accounts get-iam-policy "$SA_EMAIL" \
    --project="$PROJECT" \
    --format="json" 2>/dev/null | grep -q "$K8S_SA"; then
    log_success "Workload Identity binding already exists for $K8S_SA"
else
    log_info "Creating new Workload Identity binding..."
    if gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
        --project="$PROJECT" \
        --role="roles/iam.workloadIdentityUser" \
        --member="serviceAccount:${K8S_SA}"; then
        log_success "Workload Identity binding created for $K8S_SA"
    else
        log_error "Failed to create Workload Identity binding"
        log_info "This may be because:"
        log_info "  1. The GCP service account doesn't exist"
        log_info "  2. You don't have permission to modify IAM policies"
        log_info "  3. The namespace '$NAMESPACE' is not recognized by GKE"
        exit 1
    fi
fi

# CRITICAL: Ensure service account has GCS permissions on the bucket
# This is needed even if the bucket exists and isn't managed by Terraform
log_info "Ensuring service account has GCS permissions on bucket..."
if gcloud storage buckets get-iam-policy "gs://${BUCKET}" \
    --project="$PROJECT" \
    --format="json" 2>/dev/null | grep -q "$SA_EMAIL"; then
    log_success "Service account already has permissions on bucket"
else
    log_info "Granting storage.objectAdmin role to service account on bucket..."
    if gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
        --project="$PROJECT" \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/storage.objectAdmin" 2>/dev/null; then
        log_success "GCS permissions granted"
    else
        log_warn "Failed to grant GCS permissions via gcloud (Terraform should handle this)"
        log_info "If uploads fail, manually grant permissions with:"
        echo "  gcloud storage buckets add-iam-policy-binding gs://${BUCKET} \\"
        echo "    --member=serviceAccount:${SA_EMAIL} \\"
        echo "    --role=roles/storage.objectAdmin"
    fi
fi

# Apply worker job (with image placeholder replaced)
# CRITICAL: Resource requests/limits MUST stay identical between smoke-test and full runs
# Only ttlSecondsAfterFinished and namespace may differ
log_info "Deploying worker Job..."
TEMP_JOB=$(mktemp)
sed "s|PLACEHOLDER_IMAGE|$IMAGE_NAME|g" "$K8S_GCP_DIR/worker-job.yaml" | \
    sed "s|namespace: default|namespace: $NAMESPACE|g" | \
    sed "s|cloud.google.com/gke-nodepool: pqc-bench-pool|cloud.google.com/gke-nodepool: pqc-bench-pool|g" | \
    sed "s|ttlSecondsAfterFinished: 7200|ttlSecondsAfterFinished: $([ "$SMOKE_TEST" == "true" ] && echo "300" || echo "7200")|g" > "$TEMP_JOB"

kubectl apply -f "$TEMP_JOB"
rm -f "$TEMP_JOB"

log_success "Kubernetes resources deployed"

# =============================================================================
# Step 6: Wait for Job completion
# =============================================================================
log_step "Step 6/8: Waiting for Job completion"

log_info "Waiting for Job to complete (timeout: $JOB_TIMEOUT)..."

# Wait for pod to be created
sleep 10

# Get pod name
POD_NAME=$(kubectl get pods -l job-name="$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

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

# Wait for completion or failure
# Use a loop to check both conditions since kubectl wait only checks one at a time
log_info "Monitoring job status..."
START_TIME=$(date +%s)
TIMEOUT_SECONDS=$(echo "$JOB_TIMEOUT" | sed 's/s$//' || echo "300")
JOB_COMPLETE=false
JOB_FAILED=false

# Check if jq is available for JSON parsing
HAS_JQ=false
if command -v jq &> /dev/null; then
    HAS_JQ=true
fi

while true; do
    ELAPSED=$(($(date +%s) - START_TIME))
    
    if [[ $ELAPSED -gt $TIMEOUT_SECONDS ]]; then
        log_error "Job timed out after ${TIMEOUT_SECONDS}s"
        JOB_FAILED=true
        break
    fi
    
    # Check job status
    if [[ "$HAS_JQ" == "true" ]]; then
        # Use jq for more reliable JSON parsing
        JOB_JSON=$(kubectl get job "$JOB_NAME" -n "$NAMESPACE" -o json 2>/dev/null || echo "{}")
        
        # Check for Failed condition
        FAILED_STATUS=$(echo "$JOB_JSON" | jq -r '.status.conditions[]? | select(.type=="Failed") | .status' 2>/dev/null || echo "")
        if [[ "$FAILED_STATUS" == "True" ]]; then
            log_error "Job has failed!"
            JOB_FAILED=true
            break
        fi
        
        # Check for Complete condition
        COMPLETE_STATUS=$(echo "$JOB_JSON" | jq -r '.status.conditions[]? | select(.type=="Complete") | .status' 2>/dev/null || echo "")
        if [[ "$COMPLETE_STATUS" == "True" ]]; then
            log_success "Job completed successfully"
            JOB_COMPLETE=true
            break
        fi
        
        # Check if backoff limit is exceeded
        FAILED_COUNT=$(echo "$JOB_JSON" | jq -r '.status.failed // 0' 2>/dev/null || echo "0")
        BACKOFF_LIMIT=$(echo "$JOB_JSON" | jq -r '.spec.backoffLimit // 0' 2>/dev/null || echo "0")
        if [[ "$FAILED_COUNT" -gt 0 ]] && [[ "$FAILED_COUNT" -gt "$BACKOFF_LIMIT" ]]; then
            log_error "Job has exceeded backoff limit (failed: $FAILED_COUNT, limit: $BACKOFF_LIMIT)"
            JOB_FAILED=true
            break
        fi
        
        # Show progress every 30 seconds
        if [[ $((ELAPSED % 30)) -eq 0 ]] && [[ $ELAPSED -gt 0 ]]; then
            log_info "Still waiting... (${ELAPSED}s elapsed)"
            # Show pod status
            PODS=$(kubectl get pods -l job-name="$JOB_NAME" -n "$NAMESPACE" -o json 2>/dev/null || echo "{}")
            POD_COUNT=$(echo "$PODS" | jq -r '.items | length' 2>/dev/null || echo "0")
            if [[ "$POD_COUNT" -gt 0 ]]; then
                POD_NAME=$(echo "$PODS" | jq -r '.items[0].metadata.name' 2>/dev/null || echo "")
                POD_PHASE=$(echo "$PODS" | jq -r '.items[0].status.phase' 2>/dev/null || echo "Unknown")
                log_info "Pod status: $POD_NAME - $POD_PHASE"
            fi
        fi
    else
        # Fallback: use kubectl wait and jsonpath
        # Check for failed condition
        FAILED_STATUS=$(kubectl get job "$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || echo "")
        if [[ "$FAILED_STATUS" == "True" ]]; then
            log_error "Job has failed!"
            JOB_FAILED=true
            break
        fi
        
        # Check for complete condition
        COMPLETE_STATUS=$(kubectl get job "$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || echo "")
        if [[ "$COMPLETE_STATUS" == "True" ]]; then
            log_success "Job completed successfully"
            JOB_COMPLETE=true
            break
        fi
        
        # Show progress every 30 seconds
        if [[ $((ELAPSED % 30)) -eq 0 ]] && [[ $ELAPSED -gt 0 ]]; then
            log_info "Still waiting... (${ELAPSED}s elapsed)"
            POD_NAME=$(kubectl get pods -l job-name="$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
            POD_PHASE=$(kubectl get pods -l job-name="$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "Unknown")
            if [[ -n "$POD_NAME" ]]; then
                log_info "Pod status: $POD_NAME - $POD_PHASE"
            fi
        fi
    fi
    
    sleep 5
done

# Kill log streaming
kill $LOG_PID 2>/dev/null || true

# Handle failure
if [[ "$JOB_FAILED" == "true" ]]; then
    log_error "Job failed! Gathering diagnostics..."
    
    # Show job description
    log_info "=== Job Description ==="
    kubectl describe job "$JOB_NAME" -n "$NAMESPACE" || true
    
    # Show all pods for this job
    log_info "=== Pod Status ==="
    kubectl get pods -l job-name="$JOB_NAME" -n "$NAMESPACE" || true
    
    # Get the most recent pod
    RECENT_POD=$(kubectl get pods -l job-name="$JOB_NAME" -n "$NAMESPACE" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -n "$RECENT_POD" ]]; then
        log_info "=== Pod Description: $RECENT_POD ==="
        kubectl describe pod "$RECENT_POD" -n "$NAMESPACE" || true
        
        log_info "=== Main Container Logs (pqc-bench): $RECENT_POD ==="
        kubectl logs "$RECENT_POD" -n "$NAMESPACE" -c pqc-bench --tail=200 || true
        
        log_info "=== Upload Container Logs (upload-results): $RECENT_POD ==="
        kubectl logs "$RECENT_POD" -n "$NAMESPACE" -c upload-results --tail=200 || true
        
        log_info "=== Init Container Logs (gather-metadata): $RECENT_POD ==="
        kubectl logs "$RECENT_POD" -n "$NAMESPACE" -c gather-metadata --tail=100 || true
    else
        log_warn "No pods found for job"
        kubectl logs -l job-name="$JOB_NAME" -n "$NAMESPACE" --tail=200 || true
    fi
    
    exit 1
fi

if [[ "$JOB_COMPLETE" != "true" ]]; then
    log_error "Job did not complete successfully"
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

# Wait for upload to complete (give it more time)
log_info "Waiting for upload sidecar to complete..."
sleep 30

# Check upload container status
UPLOAD_POD=$(kubectl get pods -l job-name="$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
if [[ -n "$UPLOAD_POD" ]]; then
    UPLOAD_STATUS=$(kubectl get pod "$UPLOAD_POD" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[?(@.name=="upload-results")].state.terminated.exitCode}' 2>/dev/null || echo "")
    if [[ "$UPLOAD_STATUS" == "1" ]] || [[ "$UPLOAD_STATUS" == "2" ]]; then
        log_error "Upload container exited with error code: $UPLOAD_STATUS"
        log_info "Upload container logs:"
        kubectl logs "$UPLOAD_POD" -n "$NAMESPACE" -c upload-results --tail=100
        exit 1
    fi
fi

# List artifacts
log_info "Listing artifacts in gs://${BUCKET}/experiments/${EXP_ID}/"
ARTIFACTS=$(gsutil ls "gs://${BUCKET}/experiments/${EXP_ID}/" 2>/dev/null || echo "")

if [[ -z "$ARTIFACTS" ]]; then
    log_error "No artifacts found in GCS!"
    log_info "Checking upload container logs..."
    if [[ -n "$UPLOAD_POD" ]]; then
        kubectl logs "$UPLOAD_POD" -n "$NAMESPACE" -c upload-results --tail=100
    else
        kubectl logs -l job-name="$JOB_NAME" -n "$NAMESPACE" -c upload-results --tail=100
    fi
    log_info "Checking if bucket exists and is accessible..."
    if gsutil ls "gs://${BUCKET}/" >/dev/null 2>&1; then
        log_info "Bucket exists. Listing contents:"
        gsutil ls "gs://${BUCKET}/" | head -20
    else
        log_error "Cannot access bucket gs://${BUCKET}/"
    fi
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
# Step 8: Download results locally (optional but recommended)
# =============================================================================
log_step "Step 8/8: Downloading results locally"

LOCAL_OUTPUT_DIR="$SCRIPT_DIR/results/gcp/${EXP_ID}"
log_info "Downloading results to: $LOCAL_OUTPUT_DIR"

# Create output directory
mkdir -p "$LOCAL_OUTPUT_DIR/raw"
mkdir -p "$LOCAL_OUTPUT_DIR/merged"
mkdir -p "$LOCAL_OUTPUT_DIR/stats"
mkdir -p "$LOCAL_OUTPUT_DIR/figures"

# Download merged JSONL
if gsutil -q cp "gs://${BUCKET}/experiments/${EXP_ID}/merged.jsonl" "$LOCAL_OUTPUT_DIR/merged/merged.jsonl" 2>/dev/null; then
    log_success "Downloaded merged.jsonl"
else
    log_warn "merged.jsonl not found, trying raw data..."
    if ! gsutil -m cp -r "gs://${BUCKET}/experiments/${EXP_ID}/raw/*" "$LOCAL_OUTPUT_DIR/raw/" 2>&1; then
        log_error "Failed to download raw data from GCS"
    fi
fi

# Download manifest and metadata
gsutil -q cp "gs://${BUCKET}/experiments/${EXP_ID}/manifest.json" "$LOCAL_OUTPUT_DIR/manifest.json" 2>/dev/null || log_warn "manifest.json not found"
gsutil -q cp "gs://${BUCKET}/experiments/${EXP_ID}/provenance.json" "$LOCAL_OUTPUT_DIR/provenance.json" 2>/dev/null || log_warn "provenance.json not found"
gsutil -q cp "gs://${BUCKET}/experiments/${EXP_ID}/cloud_metadata.json" "$LOCAL_OUTPUT_DIR/cloud_metadata.json" 2>/dev/null || log_warn "cloud_metadata.json not found"
gsutil -q cp "gs://${BUCKET}/experiments/${EXP_ID}/summary.json" "$LOCAL_OUTPUT_DIR/stats/summary.json" 2>/dev/null || log_warn "summary.json not found"

# Download raw data if available
if ! gsutil -m cp -r "gs://${BUCKET}/experiments/${EXP_ID}/raw/*" "$LOCAL_OUTPUT_DIR/raw/" 2>&1; then
    log_warn "Failed to download some raw data files"
fi

# Validate downloaded data integrity
RAW_JSONL_FILE="$LOCAL_OUTPUT_DIR/raw/run.jsonl"
if [[ ! -f "$RAW_JSONL_FILE" ]]; then
    RAW_JSONL_FILE=$(find "$LOCAL_OUTPUT_DIR/raw" -name "*.jsonl" -type f | head -1)
fi

if [[ -f "$RAW_JSONL_FILE" ]]; then
    FILE_SIZE=$(stat -f%z "$RAW_JSONL_FILE" 2>/dev/null || stat -c%s "$RAW_JSONL_FILE" 2>/dev/null || echo 0)
    if [[ $FILE_SIZE -eq 0 ]]; then
        log_error "Data integrity check failed: run.jsonl is 0 bytes!"
        log_error "This indicates the benchmark didn't write any data or download failed."
        exit 1
    else
        # Check if file contains error messages (not JSONL data)
        FIRST_LINE=$(head -1 "$RAW_JSONL_FILE" 2>/dev/null || echo "")
        if [[ -n "$FIRST_LINE" ]] && [[ "$FIRST_LINE" =~ ^error: ]]; then
            log_error "Data integrity check failed: file contains error message, not JSONL data!"
            log_error "First line: ${FIRST_LINE:0:100}..."
            log_error "This indicates the download or upload process failed."
            exit 1
        fi
        
        # Validate JSONL format
        if [[ -n "$FIRST_LINE" ]]; then
            if ! echo "$FIRST_LINE" | python3 -m json.tool >/dev/null 2>&1; then
                log_error "Data integrity check failed: file does not contain valid JSONL!"
                log_error "First line: ${FIRST_LINE:0:100}..."
                exit 1
            fi
        fi
        
        LINE_COUNT=$(wc -l < "$RAW_JSONL_FILE" 2>/dev/null || echo 0)
        if [[ $LINE_COUNT -eq 0 ]]; then
            log_error "Data integrity check failed: run.jsonl has no lines!"
            log_error "File size: $FILE_SIZE bytes, but no JSONL lines found"
            exit 1
        fi
        log_success "Data integrity validated: $FILE_SIZE bytes, $LINE_COUNT events"
    fi
else
    log_error "Data integrity check failed: run.jsonl not found after download"
    exit 1
fi

log_success "Results downloaded and validated: $LOCAL_OUTPUT_DIR"

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

log_info "Results location:"
echo "  Local:  $LOCAL_OUTPUT_DIR"
echo "  GCS:    gs://${BUCKET}/experiments/${EXP_ID}/"
echo ""

log_info "To analyze results:"
echo "  python3 analysis/scripts/compute_statistics.py \\"
echo "    --input $LOCAL_OUTPUT_DIR/merged/merged.jsonl \\"
echo "    --output $LOCAL_OUTPUT_DIR/stats"
echo ""
echo "  python3 analysis/scripts/plot_latency.py \\"
echo "    --input $LOCAL_OUTPUT_DIR/merged/merged.jsonl \\"
echo "    --output $LOCAL_OUTPUT_DIR/figures"
echo ""

log_info "To re-download results (if needed):"
echo "  ./fetch_and_analyse_from_gcs.sh \\"
echo "    --exp-id $EXP_ID \\"
echo "    --bucket $BUCKET \\"
echo "    --out $LOCAL_OUTPUT_DIR"
echo ""

log_info "To compare with other environments:"
echo "  python analysis/compare_all_environments.py \\"
echo "    --native results/exp1/stats/summary.json \\"
echo "    --minikube results/exp2/stats/summary.json \\"
echo "    --gcp results/$EXP_ID/stats/summary.json"
echo ""

# =============================================================================
# Step 9: Cleanup (if requested)
# =============================================================================
if [[ "$DESTROY_AFTER" == "true" || "$EPHEMERAL" == "true" ]]; then
    log_step "Step 9/9: Destroying infrastructure and cleaning up resources"
    
    # CRITICAL: Remove bucket from state before destroy to prevent deletion
    # The bucket is a persistent resource that should never be destroyed
    cd "$TERRAFORM_DIR"
    log_info "Protecting GCS bucket from destruction..."
    if terraform state list 2>/dev/null | grep -q "google_storage_bucket.results"; then
        log_info "Removing bucket from Terraform state (bucket will persist)"
        terraform state rm google_storage_bucket.results 2>/dev/null || \
        terraform state rm 'google_storage_bucket.results' 2>/dev/null || true
        # Also remove bucket object marker
        terraform state rm google_storage_bucket_object.experiments_marker 2>/dev/null || true
        log_success "Bucket protected from destruction"
    fi
    
    # Also protect service account and Artifact Registry (persistent resources)
    log_info "Protecting persistent resources (service account, Artifact Registry)..."
    terraform state rm google_service_account.pqc_bench 2>/dev/null || true
    terraform state rm google_project_iam_member.ar_reader 2>/dev/null || true
    terraform state rm google_storage_bucket_iam_member.gcs_admin 2>/dev/null || true
    terraform state rm google_service_account_iam_member.workload_identity 2>/dev/null || true
    terraform state rm google_artifact_registry_repository.pqc 2>/dev/null || true
    
    # Run Terraform destroy (will only destroy cluster and node pool now)
    log_info "Running Terraform destroy (cluster and node pool only)..."
    if terraform destroy -auto-approve \
        -var="project_id=$PROJECT" \
        -var="region=$REGION" \
        -var="bucket_name=$BUCKET" \
        -var="machine_type=$MACHINE_TYPE" \
        -var="node_count=$NODE_COUNT" \
        -var="cluster_name=$CLUSTER_NAME" \
        -var="smoke_test=$SMOKE_TEST" \
        -var="disk_size_gb=${DISK_SIZE_GB:-50}" \
        -var="ephemeral=$EPHEMERAL"; then
        log_success "Terraform destroy completed (persistent resources preserved)"
    else
        log_warn "Terraform destroy had errors (some resources may already be deleted or not in state)"
    fi
    
    cd "$SCRIPT_DIR"
    
    # Run cleanup script to catch any orphaned resources
    log_info "Running cleanup script to remove any orphaned resources..."
    if "$SCRIPT_DIR/scripts/cleanup_gcp_resources.sh" \
        --project "$PROJECT" \
        --region "$REGION" \
        --cluster-name "$CLUSTER_NAME"; then
        log_success "Cleanup completed successfully"
    else
        log_warn "Cleanup script had errors (some resources may still be cleaning up)"
    fi
    
    # Verify cleanup (only for ephemeral mode)
    if [[ "$EPHEMERAL" == "true" ]]; then
        log_info "Verifying zero residual cost..."
        sleep 10  # Give resources time to be cleaned up
        
        REMAINING_CLUSTERS=$(gcloud container clusters list \
            --project "$PROJECT" \
            --region "$REGION" \
            --format="value(name)" 2>/dev/null | grep -E "pqc-(bench-gke|smoke-test)" || true)
        
        REMAINING_DISKS=$(gcloud compute disks list \
            --project "$PROJECT" \
            --filter="zone:${REGION}* AND name~'pqc|bench'" \
            --format="value(name)" 2>/dev/null || true)
        
        REMAINING_FORWARDING_RULES=$(gcloud compute forwarding-rules list \
            --project "$PROJECT" \
            --regions "$REGION" \
            --format="value(name)" 2>/dev/null || true)
        
        if [[ -z "$REMAINING_CLUSTERS" && -z "$REMAINING_DISKS" && -z "$REMAINING_FORWARDING_RULES" ]]; then
            log_success "Ephemeral run completed with zero residual cost ✓"
        else
            log_warn "Some resources may still be cleaning up:"
            [[ -n "$REMAINING_CLUSTERS" ]] && echo "  Clusters: $REMAINING_CLUSTERS"
            [[ -n "$REMAINING_DISKS" ]] && echo "  Disks: $REMAINING_DISKS"
            [[ -n "$REMAINING_FORWARDING_RULES" ]] && echo "  Forwarding rules: $REMAINING_FORWARDING_RULES"
            log_info "Note: GKE cluster deletion can take 5-15 minutes. Resources will be automatically cleaned up."
        fi
    fi
fi

log_success "Done!"

exit 0

