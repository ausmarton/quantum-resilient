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

# Verify kubectl connectivity
if ! kubectl cluster-info &>/dev/null; then
    echo "ERROR: Cannot connect to Kubernetes cluster" >&2
    echo "ERROR: Please ensure kubectl is configured:" >&2
    echo "ERROR:   gcloud container clusters get-credentials <cluster-name> --region $REGION --project $PROJECT" >&2
    exit 1
fi

# Verify namespace exists
if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
    echo "ERROR: Namespace '$NAMESPACE' does not exist" >&2
    echo "ERROR: Creating namespace..." >&2
    kubectl create namespace "$NAMESPACE" || {
        echo "ERROR: Failed to create namespace" >&2
        exit 1
    }
fi

# Get GCP service account email (from Terraform output or default)
TERRAFORM_DIR="$SCRIPT_DIR/terraform/gke"
SA_EMAIL=""
if [[ -d "$TERRAFORM_DIR" ]] && [[ -d "$TERRAFORM_DIR/.terraform" ]]; then
    SA_EMAIL=$(cd "$TERRAFORM_DIR" && terraform output -raw service_account_email 2>/dev/null || echo "")
fi
if [[ -z "$SA_EMAIL" ]]; then
    # Default service account email format
    SA_EMAIL="pqc-bench-worker@${PROJECT}.iam.gserviceaccount.com"
fi

# Create/update Kubernetes ServiceAccount with Workload Identity annotation
# This is required for pods to access GCS via Workload Identity
if ! kubectl get serviceaccount pqc-bench-sa -n "$NAMESPACE" &>/dev/null; then
    echo "Creating Kubernetes ServiceAccount 'pqc-bench-sa' in namespace '$NAMESPACE'..." >&2
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: pqc-bench-sa
  namespace: $NAMESPACE
  annotations:
    iam.gke.io/gcp-service-account: "$SA_EMAIL"
EOF
    if [[ $? -ne 0 ]]; then
        echo "ERROR: Failed to create ServiceAccount" >&2
        exit 1
    fi
    echo "ServiceAccount created successfully" >&2
else
    # Update annotation if service account exists but annotation is missing/wrong
    CURRENT_ANNOTATION=$(kubectl get serviceaccount pqc-bench-sa -n "$NAMESPACE" -o jsonpath='{.metadata.annotations.iam\.gke\.io/gcp-service-account}' 2>/dev/null || echo "")
    if [[ "$CURRENT_ANNOTATION" != "$SA_EMAIL" ]]; then
        echo "Updating ServiceAccount annotation to point to $SA_EMAIL..." >&2
        kubectl annotate serviceaccount pqc-bench-sa -n "$NAMESPACE" \
            "iam.gke.io/gcp-service-account=$SA_EMAIL" --overwrite || {
            echo "ERROR: Failed to update ServiceAccount annotation" >&2
            exit 1
        }
    fi
fi

# CRITICAL: Create IAM binding for Workload Identity if it doesn't exist
# The Terraform binding only covers 'default' namespace, so we need to create
# bindings for other namespaces (like 'pqc-smoke-test') dynamically
# Only check/create binding if namespace is not 'default' (Terraform handles default)
# For default namespace, Terraform should have already created the binding
K8S_SA="${PROJECT}.svc.id.goog[${NAMESPACE}/pqc-bench-sa]"
if [[ "$NAMESPACE" != "default" ]]; then
    # For non-default namespaces, check and create binding if needed
    if ! gcloud iam service-accounts get-iam-policy "$SA_EMAIL" \
        --project="$PROJECT" \
        --format="json" 2>/dev/null | grep -q "$K8S_SA"; then
        echo "Creating Workload Identity IAM binding for $K8S_SA..." >&2
        if ! gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
            --project="$PROJECT" \
            --role="roles/iam.workloadIdentityUser" \
            --member="serviceAccount:${K8S_SA}" 2>&1; then
            echo "WARNING: Failed to create Workload Identity binding" >&2
            echo "WARNING: This may cause GCS upload failures" >&2
            echo "WARNING: You may need to create it manually:" >&2
            echo "WARNING:   gcloud iam service-accounts add-iam-policy-binding $SA_EMAIL \\" >&2
            echo "WARNING:     --project=$PROJECT \\" >&2
            echo "WARNING:     --role=roles/iam.workloadIdentityUser \\" >&2
            echo "WARNING:     --member=serviceAccount:${K8S_SA}" >&2
            # Don't exit - the binding might already exist from Terraform for default namespace
        else
            echo "Workload Identity binding created successfully" >&2
        fi
    fi
fi
# For default namespace, assume Terraform has already created the binding

# Sanitize names for Kubernetes (RFC 1123 subdomain: lowercase alphanumeric, hyphens, dots only)
# Must start and end with alphanumeric, no underscores
SANITIZE_K8S_NAME() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/_/-/g' | sed 's/[^a-z0-9-]/-/g' | sed 's/--*/-/g' | sed 's/^-\|-$//g'
}

# Sanitize job name (K8s DNS-1123 subdomain)
# CRITICAL: Preserve replica suffix to avoid job name collisions
# Kubernetes job names have a max length of 63 characters (RFC 1123 subdomain)
# The EXP_ID may already include _r4 or _r8 suffix from run_all_experiments.sh
# We need to ensure the replica suffix is preserved after truncation

# Extract replica suffix if present (e.g., _r4, _r8)
REPLICA_SUFFIX=""
if [[ "$EXP_ID" =~ _r([0-9]+)$ ]]; then
    REPLICA_SUFFIX="_r${BASH_REMATCH[1]}"
    BASE_EXP_ID="${EXP_ID%_r*}"  # Remove suffix for truncation
else
    BASE_EXP_ID="$EXP_ID"
    if [[ "$REPLICAS" -gt 1 ]]; then
        REPLICA_SUFFIX="_r${REPLICAS}"
    fi
fi

# Sanitize base ID and truncate, leaving room for replica suffix
# "pqc-bench-" is 10 chars, replica suffix is max 4 chars (_r8), so we have 49 chars for base ID
SANITIZED_BASE=$(SANITIZE_K8S_NAME "$BASE_EXP_ID" | cut -c1-49)
SANITIZED_SUFFIX=$(SANITIZE_K8S_NAME "$REPLICA_SUFFIX" | sed 's/^_//')  # Remove leading underscore after sanitization
JOB_NAME="pqc-bench-${SANITIZED_BASE}${SANITIZED_SUFFIX}"

# Create scenario ConfigMap (unique per experiment to avoid conflicts)
# Sanitize ConfigMap name to be RFC 1123 compliant (max 253 chars for K8s)
SCENARIO_CM_SANITIZED=$(SANITIZE_K8S_NAME "$EXP_ID" | cut -c1-230)
SCENARIO_CM="pqc-scenario-${SCENARIO_CM_SANITIZED}"
TEMP_SCENARIO=$(mktemp)
cp "$SCENARIO" "$TEMP_SCENARIO"

# Update scenario for containerized environment
python3 <<PYTHON_EOF
import yaml
import sys

with open('$TEMP_SCENARIO', 'r') as f:
    scenario = yaml.safe_load(f) or {}

if 'metrics' not in scenario:
    scenario['metrics'] = {}
scenario['metrics']['jsonl_out'] = '/results/raw/run.jsonl'

if '$SMOKE_TEST' == 'true' and 'workload' in scenario:
    scenario['workload']['duration_sec'] = 5

with open('$TEMP_SCENARIO', 'w') as f:
    yaml.dump(scenario, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
PYTHON_EOF

# Debug: Show ConfigMap name being used (for troubleshooting)
if [[ "${DEBUG:-false}" == "true" ]]; then
    echo "DEBUG: Creating ConfigMap '$SCENARIO_CM' for experiment '$EXP_ID'" >&2
fi

SCENARIO_CM_OUTPUT=$(kubectl create configmap "$SCENARIO_CM" \
    --from-file=scenario.yaml="$TEMP_SCENARIO" \
    --namespace="$NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f - 2>&1)
SCENARIO_CM_EXIT_CODE=$?

if [[ $SCENARIO_CM_EXIT_CODE -ne 0 ]]; then
    echo "ERROR: Failed to create scenario ConfigMap '$SCENARIO_CM'" >&2
    echo "ERROR: Original experiment ID: '$EXP_ID'" >&2
    echo "ERROR: Sanitized ConfigMap name: '$SCENARIO_CM'" >&2
    echo "ERROR: ConfigMap creation output:" >&2
    echo "$SCENARIO_CM_OUTPUT" >&2
    rm -f "$TEMP_SCENARIO"
    exit 1
fi
rm -f "$TEMP_SCENARIO"

# Create GCP config ConfigMap (unique per experiment)
# Sanitize ConfigMap name to be RFC 1123 compliant (max 253 chars for K8s)
GCP_CM_SANITIZED=$(SANITIZE_K8S_NAME "$EXP_ID" | cut -c1-228)
GCP_CM="pqc-gcp-config-${GCP_CM_SANITIZED}"
# Debug: Show ConfigMap name being used (for troubleshooting)
if [[ "${DEBUG:-false}" == "true" ]]; then
    echo "DEBUG: Creating GCP ConfigMap '$GCP_CM' for experiment '$EXP_ID'" >&2
fi

GCP_CM_OUTPUT=$(kubectl create configmap "$GCP_CM" \
    --from-literal=bucket_name="$BUCKET" \
    --from-literal=experiment_id="$EXP_ID" \
    --from-literal=region="$REGION" \
    --from-literal=project_id="$PROJECT" \
    --from-literal=smoke_test="$([ "$SMOKE_TEST" == "true" ] && echo "true" || echo "false")" \
    --namespace="$NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f - 2>&1)
GCP_CM_EXIT_CODE=$?

if [[ $GCP_CM_EXIT_CODE -ne 0 ]]; then
    echo "ERROR: Failed to create GCP config ConfigMap '$GCP_CM'" >&2
    echo "ERROR: Original experiment ID: '$EXP_ID'" >&2
    echo "ERROR: Sanitized ConfigMap name: '$GCP_CM'" >&2
    echo "ERROR: ConfigMap creation output:" >&2
    echo "$GCP_CM_OUTPUT" >&2
    exit 1
fi

# Create Job YAML
TEMP_JOB=$(mktemp)
python3 <<PYTHON_EOF
import yaml
import sys

# Read base job template
with open('$K8S_GCP_DIR/worker-job.yaml', 'r') as f:
    job = yaml.safe_load(f)

# Update metadata
job['metadata']['name'] = '$JOB_NAME'
job['metadata']['namespace'] = '$NAMESPACE'
job['metadata']['labels']['experiment-id'] = '${EXP_ID}'

# Update ConfigMap references in volumes
for vol in job['spec']['template']['spec'].get('volumes', []):
    if vol.get('name') == 'scenario-config':
        if 'configMap' in vol:
            vol['configMap']['name'] = '$SCENARIO_CM'

# Update image and ConfigMap references in containers
for container in job['spec']['template']['spec']['containers']:
    if container['name'] == 'pqc-bench':
        container['image'] = '$IMAGE_NAME'
    # Update ConfigMap references in env vars
    for env_var in container.get('env', []):
        if 'configMapKeyRef' in env_var.get('valueFrom', {}):
            if env_var['valueFrom']['configMapKeyRef'].get('name') == 'pqc-gcp-config':
                env_var['valueFrom']['configMapKeyRef']['name'] = '$GCP_CM'

# Also update in init containers
for container in job['spec']['template']['spec'].get('initContainers', []):
    for env_var in container.get('env', []):
        if 'configMapKeyRef' in env_var.get('valueFrom', {}):
            if env_var['valueFrom']['configMapKeyRef'].get('name') == 'pqc-gcp-config':
                env_var['valueFrom']['configMapKeyRef']['name'] = '$GCP_CM'

# Update replicas if needed (for scaling experiments)
# Note: This is handled by the job itself, not by Kubernetes replicas

# Write job
with open('$TEMP_JOB', 'w') as f:
    yaml.dump(job, f, default_flow_style=False, sort_keys=False)
PYTHON_EOF

# Submit job
JOB_OUTPUT=$(kubectl apply -f "$TEMP_JOB" 2>&1)
JOB_EXIT_CODE=$?
rm -f "$TEMP_JOB"

if [[ $JOB_EXIT_CODE -ne 0 ]]; then
    echo "ERROR: Failed to submit job" >&2
    echo "ERROR: Experiment ID: $EXP_ID" >&2
    echo "ERROR: Job name: $JOB_NAME" >&2
    echo "ERROR: Replicas: $REPLICAS" >&2
    echo "ERROR: kubectl output:" >&2
    echo "$JOB_OUTPUT" >&2
    exit 1
fi

echo "$JOB_NAME"

