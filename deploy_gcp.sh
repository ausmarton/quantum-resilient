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

# Source common libraries
source "$SCRIPT_DIR/scripts/lib/common.sh"
source "$SCRIPT_DIR/scripts/lib/directories.sh"
source "$SCRIPT_DIR/scripts/lib/analysis.sh"
source "$SCRIPT_DIR/scripts/lib/manifest.sh"
source "$SCRIPT_DIR/scripts/lib/k8s-job.sh"

TERRAFORM_DIR="$SCRIPT_DIR/iac/terraform/gcp"
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
SKIP_JOB=false
DESTROY_AFTER=false
SMOKE_TEST=false
EPHEMERAL=false
CREATE_CLUSTER=false
DESTROY_CLUSTER=false

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
    --skip-job            Skip job deployment (only build image)
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
        --skip-job)
            SKIP_JOB=true
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
# Use consistent cluster name for all test types and environments
CLUSTER_NAME="pqc-bench"
if [[ "$SMOKE_TEST" == "true" ]]; then
    # DO NOT change MACHINE_TYPE - must stay identical to full runs
    # Only reduce node_count (horizontal scaling)
    NODE_COUNT=1
    RUNS=1
    REPLICAS=1
    JOB_TIMEOUT="300s"  # 5 minutes
    log_info "Smoke-test mode: reduced duration, runs, replicas (hardware identical)"
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
    
    # When using -target to exclude bucket from plan, we must ensure bucket is NOT in state
    # This prevents Terraform from trying to replace it due to name mismatches or prevent_destroy conflicts
    # We'll grant bucket permissions manually after cluster creation instead
    log_info "Preparing Terraform state for cluster creation (excluding bucket)..."
    
    # Remove bucket and related resources from state if they exist
    # This is safe because we're using -target flags to exclude them from the plan
    if terraform state show google_storage_bucket.results &>/dev/null 2>&1; then
        log_info "Removing bucket from Terraform state (not managing with Terraform for cluster creation)"
        terraform state rm google_storage_bucket.results 2>/dev/null || true
        terraform state rm google_storage_bucket_iam_member.gcs_admin 2>/dev/null || true
        terraform state rm google_storage_bucket_object.experiments_marker 2>/dev/null || true
        log_success "Bucket removed from Terraform state"
    else
        log_info "Bucket not in Terraform state (this is expected)"
    fi
    
    # Verify bucket exists (we'll grant permissions manually)
    # Use timeout to prevent hanging if gsutil is slow/unresponsive
    if timeout 10 gsutil ls -b "gs://${BUCKET}" &>/dev/null 2>&1; then
        log_info "Bucket $BUCKET exists - will grant permissions manually after cluster creation"
    else
        log_warn "Bucket $BUCKET check timed out or bucket does not exist"
        log_info "Bucket will need to be created separately or permissions granted manually"
        log_info "Continuing with cluster creation (bucket permissions can be granted later)"
    fi
    
    log_info "Applying Terraform configuration to create cluster..."
    DISK_SIZE_GB="${DISK_SIZE_GB:-50}"
    # Use -target to exclude bucket from plan (bucket may already exist with different name)
    # This prevents Terraform from trying to replace the bucket when only creating cluster
    # Target only the cluster resource - Terraform will create dependencies automatically
    # Use stdbuf to ensure output is unbuffered so we can see progress
    log_info "Running terraform apply (this may take 5-10 minutes for GKE cluster creation)..."
    # Use stdbuf to ensure output is unbuffered so we can see progress in real-time
    # Pipe through tee to both show output and capture it
    TERRAFORM_OUTPUT=$(mktemp)
    MAX_RETRIES=3
    RETRY_COUNT=0
    TERRAFORM_SUCCESS=false
    
    while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
        if [[ $RETRY_COUNT -gt 0 ]]; then
            log_info "Retrying Terraform apply (attempt $((RETRY_COUNT + 1))/$MAX_RETRIES)..."
            sleep 10  # Wait before retry
        fi
        
        if stdbuf -oL -eL terraform apply -auto-approve \
            -target=google_service_account.worker \
            -target=google_service_account.orchestrator \
            -target=google_container_cluster.primary \
            -var="project_id=$PROJECT" \
            -var="region=$REGION" \
            -var="bucket_name=$BUCKET" \
            -var="gke_node_machine_type=$MACHINE_TYPE" \
            -var="gke_initial_node_count=$NODE_COUNT" \
            -var="gke_node_min_count=$NODE_COUNT" \
            -var="gke_name=$CLUSTER_NAME" \
            -var="gke_disk_size_gb=$DISK_SIZE_GB" 2>&1 | tee "$TERRAFORM_OUTPUT"; then
            rm -f "$TERRAFORM_OUTPUT"
            log_success "Cluster created successfully"
            TERRAFORM_SUCCESS=true
            break
        else
            # Check if error is HTTP2 connection lost (retryable)
            if grep -q "http2: client connection lost" "$TERRAFORM_OUTPUT" 2>/dev/null; then
                RETRY_COUNT=$((RETRY_COUNT + 1))
                if [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; then
                    log_warn "HTTP2 connection lost during cluster creation. Will retry..."
                    continue
                else
                    log_error "HTTP2 connection lost after $MAX_RETRIES attempts"
                fi
            else
                # Non-retryable error
                log_error "Terraform apply failed with non-retryable error"
                break
            fi
        fi
    done
    
    if [[ "$TERRAFORM_SUCCESS" == "true" ]]; then
        
        # Grant bucket permissions manually if bucket IAM binding wasn't created
        # (This happens if bucket wasn't in Terraform state)
        log_info "Ensuring service account has bucket permissions..."
        SERVICE_ACCOUNT_EMAIL="qr-worker@${PROJECT}.iam.gserviceaccount.com"
        if gsutil iam ch "serviceAccount:${SERVICE_ACCOUNT_EMAIL}:roles/storage.objectAdmin" "gs://${BUCKET}" 2>&1; then
            log_success "Bucket permissions granted to service account"
        else
            log_warn "Failed to grant bucket permissions via gsutil (may already be set)"
            log_info "Verifying permissions..."
            if gsutil iam get "gs://${BUCKET}" 2>/dev/null | grep -q "$SERVICE_ACCOUNT_EMAIL"; then
                log_success "Service account already has bucket permissions"
            else
                log_error "Service account does not have bucket permissions!"
                log_info "Please grant manually:"
                log_info "  gsutil iam ch serviceAccount:${SERVICE_ACCOUNT_EMAIL}:roles/storage.objectAdmin gs://${BUCKET}"
            fi
        fi
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
        # CRITICAL: Only protect GCS bucket from destruction
        # All other resources (service account, Artifact Registry, cluster, nodes, etc.) are ephemeral
        log_info "Protecting GCS bucket from destruction (all other resources will be destroyed)..."
        terraform state rm google_storage_bucket.results 2>/dev/null || true
        terraform state rm google_storage_bucket_object.experiments_marker 2>/dev/null || true
        terraform state rm google_storage_bucket_iam_member.gcs_admin 2>/dev/null || true
        # Remove kubernetes_manifest resources from state to prevent REST client errors when cluster is gone
        terraform state rm 'kubernetes_manifest.orchestrator_service_monitor[0]' 2>/dev/null || true
        terraform state rm 'kubernetes_manifest.worker_service_monitor[0]' 2>/dev/null || true
        terraform state rm 'kubernetes_manifest.worker_pod_monitor[0]' 2>/dev/null || true
        # Note: Service account, Artifact Registry, and IAM bindings are NOT protected
        # They will be destroyed along with the cluster to ensure complete cleanup
        
        log_info "Running Terraform destroy (cluster and node pool only)..."
        if terraform destroy -auto-approve \
            -var="project_id=$PROJECT" \
            -var="region=$REGION" \
            -var="bucket_name=$BUCKET" \
            -var="gke_node_machine_type=$MACHINE_TYPE" \
            -var="gke_initial_node_count=$NODE_COUNT" \
            -var="gke_node_min_count=$NODE_COUNT" \
            -var="gke_name=$CLUSTER_NAME" \
            -var="gke_disk_size_gb=${DISK_SIZE_GB:-50}" 2>&1; then
            log_success "Terraform destroy completed (GCS bucket preserved, all other resources destroyed)"
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
                -var="gke_node_machine_type=$MACHINE_TYPE" \
                -var="gke_initial_node_count=$NODE_COUNT" \
                -var="gke_node_min_count=$NODE_COUNT" \
                -var="gke_name=$CLUSTER_NAME" \
                -var="gke_disk_size_gb=${DISK_SIZE_GB:-50}" \
                google_storage_bucket.results "$BUCKET" 2>/dev/null || log_warn "Bucket import failed (may already be in state)"
        else
            log_info "Bucket already in Terraform state, skipping import"
        fi
    fi
    
    # Import service account if it exists and not in state
    SA_EMAIL="qr-worker@${PROJECT}.iam.gserviceaccount.com"
    if gcloud iam service-accounts describe "$SA_EMAIL" \
        --project "$PROJECT" >/dev/null 2>&1; then
        if ! terraform state show google_service_account.worker &>/dev/null; then
            log_info "Service account exists but not in state, importing..."
            terraform import \
                -var="project_id=$PROJECT" \
                -var="region=$REGION" \
                -var="bucket_name=$BUCKET" \
                -var="gke_node_machine_type=$MACHINE_TYPE" \
                -var="gke_initial_node_count=$NODE_COUNT" \
                -var="gke_node_min_count=$NODE_COUNT" \
                -var="gke_name=$CLUSTER_NAME" \
                -var="gke_disk_size_gb=${DISK_SIZE_GB:-50}" \
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
                -var="gke_node_machine_type=$MACHINE_TYPE" \
                -var="gke_initial_node_count=$NODE_COUNT" \
                -var="gke_node_min_count=$NODE_COUNT" \
                -var="gke_name=$CLUSTER_NAME" \
                -var="gke_disk_size_gb=${DISK_SIZE_GB:-50}" \
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
    
    # CRITICAL: For ephemeral mode, remove bucket from state before applying
    # This prevents Terraform from trying to destroy/replace the bucket due to prevent_destroy
    # The bucket is a persistent resource that should never be destroyed
    if [[ "$EPHEMERAL" == "true" ]]; then
        log_info "Ephemeral mode: Removing bucket from Terraform state to prevent prevent_destroy conflicts..."
        if terraform state show google_storage_bucket.results &>/dev/null 2>&1; then
            log_info "Removing bucket from Terraform state (bucket will persist, permissions granted manually)"
            terraform state rm google_storage_bucket.results 2>/dev/null || true
            terraform state rm google_storage_bucket_iam_member.gcs_admin 2>/dev/null || true
            terraform state rm google_storage_bucket_object.experiments_marker 2>/dev/null || true
            log_success "Bucket removed from Terraform state"
        else
            log_info "Bucket not in Terraform state (this is expected)"
        fi
    fi
    
    log_info "Applying Terraform configuration..."
    # CRITICAL: machine_type, disk_size_gb, disk_type MUST stay identical
    # Only node_count may differ (horizontal scaling only)
    # Use consistent disk_size_gb for both smoke-test and full runs
    # Use -target to create cluster first, then configure kubectl, then apply Kubernetes resources
    # Note: Default node pool is now configured in google_container_cluster.primary (no separate node pool resource)
    DISK_SIZE_GB="${DISK_SIZE_GB:-50}"
    # Include service accounts so they're created before cluster
    if terraform apply -auto-approve \
        -target=google_service_account.worker \
        -target=google_service_account.orchestrator \
        -target=google_container_cluster.primary \
        -var="project_id=$PROJECT" \
        -var="region=$REGION" \
        -var="bucket_name=$BUCKET" \
        -var="gke_node_machine_type=$MACHINE_TYPE" \
        -var="gke_initial_node_count=$NODE_COUNT" \
        -var="gke_node_min_count=$NODE_COUNT" \
        -var="gke_name=$CLUSTER_NAME" \
        -var="gke_disk_size_gb=$DISK_SIZE_GB"; then
        log_success "Infrastructure deployed"
        
        # Grant bucket permissions manually if bucket was removed from state
        # (This happens in ephemeral mode to prevent prevent_destroy conflicts)
        if [[ "$EPHEMERAL" == "true" ]]; then
            log_info "Ensuring service account has bucket permissions..."
            SERVICE_ACCOUNT_EMAIL="qr-worker@${PROJECT}.iam.gserviceaccount.com"
            if gsutil iam ch "serviceAccount:${SERVICE_ACCOUNT_EMAIL}:roles/storage.objectAdmin" "gs://${BUCKET}" 2>&1; then
                log_success "Bucket permissions granted to service account"
            else
                log_warn "Failed to grant bucket permissions via gsutil (may already be set)"
                log_info "Verifying permissions..."
                if gsutil iam get "gs://${BUCKET}" 2>/dev/null | grep -q "$SERVICE_ACCOUNT_EMAIL"; then
                    log_success "Service account already has bucket permissions"
                else
                    log_error "Service account does not have bucket permissions!"
                    log_info "Please grant manually:"
                    log_info "  gsutil iam ch serviceAccount:${SERVICE_ACCOUNT_EMAIL}:roles/storage.objectAdmin gs://${BUCKET}"
                fi
            fi
        fi
    else
        log_error "Terraform apply failed."
        TERRAFORM_FAILED=true
        
        # Only check node pool/cluster status if cluster actually exists
        # If Terraform failed, the cluster/node pool may not exist, causing misleading 404 errors
        if command -v gcloud &> /dev/null; then
            # Check if cluster exists before trying to describe node pool
            if gcloud container clusters describe "$CLUSTER_NAME" \
                --region "$REGION" \
                --project "$PROJECT" &>/dev/null 2>&1; then
                log_info "Cluster exists, checking node pool status..."
                # Default pool name is "default-pool" (GKE standard)
                DEFAULT_POOL_NAME=$(gcloud container node-pools list \
                    --cluster "$CLUSTER_NAME" \
                    --region "$REGION" \
                    --project "$PROJECT" \
                    --format="value(name)" 2>/dev/null | head -1 || echo "default-pool")
                gcloud container node-pools describe "$DEFAULT_POOL_NAME" \
                    --cluster "$CLUSTER_NAME" \
                    --region "$REGION" \
                    --project "$PROJECT" 2>&1 | head -50 || true
                
                log_info "Checking cluster operations..."
                gcloud container operations list \
                    --filter="clusterName=$CLUSTER_NAME AND location=$REGION" \
                    --project "$PROJECT" \
                    --limit 5 \
                    --format="table(name,status,operationType)" || true
            else
                log_warn "Cluster $CLUSTER_NAME does not exist (Terraform failed before cluster creation)"
                log_info "Root cause: Terraform apply failed - check Terraform error messages above"
                log_info "Common causes:"
                log_info "  - prevent_destroy conflicts (bucket lifecycle)"
                log_info "  - Insufficient GCP quotas"
                log_info "  - Network/permissions issues"
            fi
        fi
        
        log_error "See docs/troubleshooting/gke-node-pool.md for troubleshooting steps"
        log_error "Root cause: Terraform apply failed - see error messages above"
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
    SA_EMAIL=$(cd "$TERRAFORM_DIR" && terraform output -raw worker_sa_email 2>/dev/null || echo "qr-worker@${PROJECT}.iam.gserviceaccount.com")
fi

# =============================================================================
# Step 4: Build and push container image
# =============================================================================
log_step "Step 4/8: Building and pushing container image"

if [[ "$SKIP_BUILD" == "true" ]]; then
    log_warn "Skipping build (--skip-build)"
else
    # CRITICAL: Ensure Artifact Registry repository exists before pushing
    # Extract repository name from image (e.g., europe-west2-docker.pkg.dev/project/pqc/pqc-bench:latest -> pqc)
    AR_REPO="pqc"
    AR_LOCATION="${REGION}"
    
    log_info "Ensuring Artifact Registry repository exists..."
    if ! gcloud artifacts repositories describe "$AR_REPO" \
        --location "$AR_LOCATION" \
        --project "$PROJECT" &>/dev/null; then
        log_info "Artifact Registry repository '$AR_REPO' does not exist. Creating it..."
        if gcloud artifacts repositories create "$AR_REPO" \
            --repository-format=docker \
            --location="$AR_LOCATION" \
            --project="$PROJECT" \
            --description="PQC benchmark container images" 2>&1; then
            log_success "Artifact Registry repository created: $AR_REPO"
        else
            log_error "Failed to create Artifact Registry repository"
            log_error "Please create it manually or run Terraform to create it"
            exit 1
        fi
    else
        log_info "Artifact Registry repository '$AR_REPO' already exists"
    fi
    
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
    if podman push "$IMAGE_NAME" 2>&1; then
        log_success "Image pushed: $IMAGE_NAME"
    else
        log_error "Failed to push image to Artifact Registry"
        log_error "Image: $IMAGE_NAME"
        log_error "Please verify:"
        log_error "  1. Artifact Registry repository exists: gcloud artifacts repositories describe $AR_REPO --location $AR_LOCATION --project $PROJECT"
        log_error "  2. You have push permissions"
        log_error "  3. Podman is authenticated: podman login ${REGION}-docker.pkg.dev"
        exit 1
    fi
fi

# =============================================================================
# Step 5: Apply Kubernetes manifests
# =============================================================================
if [[ "$SKIP_JOB" == "true" ]]; then
    log_warn "Skipping job deployment (--skip-job), image build complete"
    log_success "Image ready: $IMAGE_NAME"
    exit 0
fi

log_step "Step 5/8: Deploying Kubernetes resources"

# Use consistent namespace for all test types
NAMESPACE="default"

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

# Create benchmark config ConfigMap
log_info "Creating benchmark config ConfigMap..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: pqc-bench-config
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
# Ensure binding exists for the target namespace
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

# Use unified job waiting function
# Note: For GCP, we also need to wait for the upload sidecar container (handled below)
if ! wait_for_job "$JOB_NAME" "$NAMESPACE" "$JOB_TIMEOUT" "true"; then
    # Enhanced diagnostics for GCP failures
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

# Get pod name for upload sidecar check
POD_NAME=$(get_job_pods "$JOB_NAME" "$NAMESPACE" | awk '{print $1}')

# =============================================================================
# Step 7: Verify GCS artifacts
# =============================================================================
log_step "Step 7/8: Verifying GCS artifacts"

log_info "Checking GCS bucket for results..."

# Wait for upload sidecar to complete
log_info "Waiting for upload sidecar to complete..."
UPLOAD_POD=$(kubectl get pods -l job-name="$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

if [[ -n "$UPLOAD_POD" ]]; then
    # Wait for upload container to finish (with timeout)
    log_info "Waiting for upload-results container to complete..."
    MAX_WAIT=120  # 2 minutes max wait
    WAIT_COUNT=0
    UPLOAD_COMPLETE=false
    
    while [[ $WAIT_COUNT -lt $MAX_WAIT ]]; do
        # Check if upload container has terminated
        UPLOAD_PHASE=$(kubectl get pod "$UPLOAD_POD" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[?(@.name=="upload-results")].state.terminated.reason}' 2>/dev/null || echo "")
        UPLOAD_EXIT=$(kubectl get pod "$UPLOAD_POD" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[?(@.name=="upload-results")].state.terminated.exitCode}' 2>/dev/null || echo "")
        
        if [[ -n "$UPLOAD_PHASE" ]]; then
            if [[ "$UPLOAD_EXIT" == "0" ]]; then
                log_success "Upload container completed successfully"
                UPLOAD_COMPLETE=true
                break
            elif [[ "$UPLOAD_EXIT" == "1" ]] || [[ "$UPLOAD_EXIT" == "2" ]]; then
                log_error "Upload container exited with error code: $UPLOAD_EXIT"
                log_info "Upload container logs:"
                kubectl logs "$UPLOAD_POD" -n "$NAMESPACE" -c upload-results --tail=100 2>/dev/null || {
                    log_warn "Could not get upload container logs, trying all containers:"
                    kubectl logs "$UPLOAD_POD" -n "$NAMESPACE" --all-containers=true --tail=100
                }
                exit 1
            fi
        fi
        
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
        if [[ $((WAIT_COUNT % 30)) -eq 0 ]]; then
            log_info "Still waiting for upload... ($WAIT_COUNT/$MAX_WAIT seconds)"
        fi
    done
    
    if [[ "$UPLOAD_COMPLETE" != "true" ]]; then
        log_warn "Upload container did not complete within timeout, checking logs..."
        kubectl logs "$UPLOAD_POD" -n "$NAMESPACE" -c upload-results --tail=100 2>/dev/null || {
            log_warn "Could not get upload container logs, trying all containers:"
            kubectl logs "$UPLOAD_POD" -n "$NAMESPACE" --all-containers=true --tail=100
        }
    fi
else
    log_warn "Could not find pod for job $JOB_NAME"
fi

# Give GCS a moment for eventual consistency
sleep 5

# List artifacts
log_info "Listing artifacts in gs://${BUCKET}/experiments/${EXP_ID}/"
ARTIFACTS=$(gsutil ls "gs://${BUCKET}/experiments/${EXP_ID}/" 2>/dev/null || echo "")

if [[ -z "$ARTIFACTS" ]]; then
    log_error "No artifacts found in GCS!"
    log_info "Checking upload container logs..."
    if [[ -n "$UPLOAD_POD" ]]; then
        kubectl logs "$UPLOAD_POD" -n "$NAMESPACE" -c upload-results --tail=200 2>/dev/null || {
            log_warn "Could not get upload container logs, trying all containers:"
            kubectl logs "$UPLOAD_POD" -n "$NAMESPACE" --all-containers=true --tail=200
        }
    else
        kubectl logs -l job-name="$JOB_NAME" -n "$NAMESPACE" --all-containers=true --tail=200 2>/dev/null || true
    fi
    log_info "Checking if bucket exists and is accessible..."
    if gsutil ls "gs://${BUCKET}/" >/dev/null 2>&1; then
        log_info "Bucket exists. Listing contents:"
        gsutil ls "gs://${BUCKET}/" | head -20
        log_info "Checking experiments directory:"
        gsutil ls "gs://${BUCKET}/experiments/" 2>/dev/null | head -20 || log_warn "experiments/ directory not found"
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
create_output_directories "$LOCAL_OUTPUT_DIR"

# Use unified result retrieval function for raw data
if ! download_results_from_gcs "$EXP_ID" "$BUCKET" "$LOCAL_OUTPUT_DIR"; then
    log_error "Failed to download raw data from GCS"
    exit 1
fi

# Download additional metadata files (not included in download_results_from_gcs)
gsutil -q cp "gs://${BUCKET}/experiments/${EXP_ID}/merged.jsonl" "$LOCAL_OUTPUT_DIR/merged/merged.jsonl" 2>/dev/null || log_warn "merged.jsonl not found"
gsutil -q cp "gs://${BUCKET}/experiments/${EXP_ID}/manifest.json" "$LOCAL_OUTPUT_DIR/manifest.json" 2>/dev/null || log_warn "manifest.json not found"
gsutil -q cp "gs://${BUCKET}/experiments/${EXP_ID}/provenance.json" "$LOCAL_OUTPUT_DIR/provenance.json" 2>/dev/null || log_warn "provenance.json not found"
gsutil -q cp "gs://${BUCKET}/experiments/${EXP_ID}/cloud_metadata.json" "$LOCAL_OUTPUT_DIR/cloud_metadata.json" 2>/dev/null || log_warn "cloud_metadata.json not found"
gsutil -q cp "gs://${BUCKET}/experiments/${EXP_ID}/summary.json" "$LOCAL_OUTPUT_DIR/stats/summary.json" 2>/dev/null || log_warn "summary.json not found"

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
    
    # CRITICAL: Only protect GCS bucket from destruction
    # All other resources (service account, Artifact Registry, cluster, nodes, etc.) are ephemeral
    cd "$TERRAFORM_DIR"
    log_info "Protecting GCS bucket from destruction (all other resources will be destroyed)..."
    if terraform state list 2>/dev/null | grep -q "google_storage_bucket.results"; then
        log_info "Removing bucket from Terraform state (bucket will persist, all other resources will be destroyed)"
        terraform state rm google_storage_bucket.results 2>/dev/null || \
        terraform state rm 'google_storage_bucket.results' 2>/dev/null || true
        # Also remove bucket object marker
        terraform state rm google_storage_bucket_object.experiments_marker 2>/dev/null || true
        terraform state rm google_storage_bucket_iam_member.gcs_admin 2>/dev/null || true
        log_success "Bucket protected from destruction"
    fi
    # Remove kubernetes_manifest resources from state to prevent REST client errors when cluster is gone
    terraform state rm 'kubernetes_manifest.orchestrator_service_monitor[0]' 2>/dev/null || true
    terraform state rm 'kubernetes_manifest.worker_service_monitor[0]' 2>/dev/null || true
    terraform state rm 'kubernetes_manifest.worker_pod_monitor[0]' 2>/dev/null || true
    # Note: Service account, Artifact Registry, and IAM bindings are NOT protected
    # They will be destroyed along with the cluster to ensure complete cleanup
    
    # Run Terraform destroy (will only destroy cluster and node pool now)
    log_info "Running Terraform destroy (cluster and node pool only)..."
    if terraform destroy -auto-approve \
        -var="project_id=$PROJECT" \
        -var="region=$REGION" \
        -var="bucket_name=$BUCKET" \
        -var="gke_node_machine_type=$MACHINE_TYPE" \
        -var="gke_initial_node_count=$NODE_COUNT" \
        -var="gke_node_min_count=$NODE_COUNT" \
        -var="gke_name=$CLUSTER_NAME" \
        -var="gke_disk_size_gb=${DISK_SIZE_GB:-50}"; then
        log_success "Terraform destroy completed (GCS bucket preserved, all other resources destroyed)"
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

