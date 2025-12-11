#!/usr/bin/env bash
# =============================================================================
# submit_gcp_job_parallel.sh - Submit a single GCP experiment as a Kubernetes Job
#
# This is a lightweight wrapper that submits a job directly to Kubernetes
# without going through the full deploy_gcp.sh flow. Used for parallel execution.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
K8S_GCP_DIR="$SCRIPT_DIR/k8s/gcp"

# Source common libraries
source "$SCRIPT_DIR/scripts/lib/common.sh"
source "$SCRIPT_DIR/scripts/lib/k8s-configmap.sh"
source "$SCRIPT_DIR/scripts/lib/k8s-job.sh"
source "$SCRIPT_DIR/scripts/lib/k8s-cluster.sh"

SCENARIO=""
EXP_ID=""
PROJECT=""
BUCKET=""
REGION="us-central1"
IMAGE_NAME=""
NAMESPACE="default"
REPLICAS=1
SMOKE_TEST=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scenario) SCENARIO="$2"; shift 2 ;;
        --exp-id) EXP_ID="$2"; shift 2 ;;
        --project) PROJECT="$2"; shift 2 ;;
        --bucket) BUCKET="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --image) IMAGE_NAME="$2"; shift 2 ;;
        --namespace) NAMESPACE="$2"; shift 2 ;;
        --replicas) REPLICAS="$2"; shift 2 ;;
        --smoke-test) SMOKE_TEST=true; shift ;;
        *) shift ;;
    esac
done

# Validate
[[ -z "$SCENARIO" ]] && { echo "ERROR: Missing --scenario" >&2; exit 1; }
[[ -z "$EXP_ID" ]] && { echo "ERROR: Missing --exp-id" >&2; exit 1; }
[[ -z "$IMAGE_NAME" ]] && { echo "ERROR: Missing --image" >&2; exit 1; }

# Determine cluster name from environment or use default
CLUSTER_NAME="${GCP_CLUSTER_NAME:-pqc-smoke-test}"

# Refresh kubectl credentials before each job submission
# This ensures credentials don't expire during long-running tests
log_info "Refreshing kubectl credentials for cluster $CLUSTER_NAME..." >&2
if ! gcloud container clusters get-credentials "$CLUSTER_NAME" \
    --region "$REGION" \
    --project "$PROJECT" >&2 2>&1; then
    echo "ERROR: Failed to get cluster credentials" >&2
    echo "ERROR: Please ensure cluster $CLUSTER_NAME exists in region $REGION" >&2
    exit 1
fi

# Verify kubectl connectivity
if ! verify_kubectl_connectivity; then
    echo "ERROR: Cannot connect to Kubernetes cluster" >&2
    echo "ERROR: Please ensure kubectl is configured:" >&2
    echo "ERROR:   gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project $PROJECT" >&2
    exit 1
fi

# Verify namespace exists
if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
    echo "Creating namespace '$NAMESPACE'..." >&2
    kubectl create namespace "$NAMESPACE" || {
        echo "ERROR: Failed to create namespace" >&2
        exit 1
    }
fi

# Ensure GCP service account is set up
if ! ensure_gcp_service_account "$PROJECT" "$NAMESPACE"; then
    echo "ERROR: Failed to set up GCP service account" >&2
            exit 1
fi

# Use unified job submission function
JOB_NAME=$(submit_k8s_job \
    "gcp" \
    "$SCENARIO" \
    "$EXP_ID" \
    "$IMAGE_NAME" \
    "$NAMESPACE" \
    "$REPLICAS" \
    "$SMOKE_TEST" \
    "" \
    "" \
    "$PROJECT" \
    "$BUCKET" \
    "$REGION") || {
    echo "ERROR: Failed to submit job" >&2
    exit 1
}

echo "$JOB_NAME"

