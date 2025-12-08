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
[[ -z "$SCENARIO" ]] && { echo "ERROR: Missing --scenario"; exit 1; }
[[ -z "$EXP_ID" ]] && { echo "ERROR: Missing --exp-id"; exit 1; }
[[ -z "$IMAGE_NAME" ]] && { echo "ERROR: Missing --image"; exit 1; }

# Sanitize job name (K8s DNS-1123 subdomain)
JOB_NAME="pqc-bench-$(echo "$EXP_ID" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9-]/-/g' | sed 's/--*/-/g' | cut -c1-50)"

# Create scenario ConfigMap (unique per experiment to avoid conflicts)
SCENARIO_CM="pqc-scenario-${EXP_ID}"
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

kubectl create configmap "$SCENARIO_CM" \
    --from-file=scenario.yaml="$TEMP_SCENARIO" \
    --namespace="$NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1
rm -f "$TEMP_SCENARIO"

# Create GCP config ConfigMap (unique per experiment)
GCP_CM="pqc-gcp-config-${EXP_ID}"
kubectl create configmap "$GCP_CM" \
    --from-literal=bucket_name="$BUCKET" \
    --from-literal=experiment_id="$EXP_ID" \
    --from-literal=region="$REGION" \
    --from-literal=project_id="$PROJECT" \
    --from-literal=smoke_test="$([ "$SMOKE_TEST" == "true" ] && echo "true" || echo "false")" \
    --namespace="$NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1

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
kubectl apply -f "$TEMP_JOB" >/dev/null 2>&1
rm -f "$TEMP_JOB"

echo "$JOB_NAME"

