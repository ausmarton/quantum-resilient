#!/usr/bin/env bash
# =============================================================================
# submit_parallel_gcp_jobs.sh - Submit multiple GCP experiments as parallel Jobs
#
# Creates Kubernetes Jobs for multiple experiments and runs them in parallel.
# Much faster than sequential execution.
#
# Usage:
#   ./scripts/submit_parallel_gcp_jobs.sh \
#     --scenarios manifest.json \
#     --project <project> \
#     --bucket <bucket> \
#     --region <region> \
#     --parallel 20
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
K8S_GCP_DIR="$SCRIPT_DIR/k8s/gcp"

# Default values
SCENARIOS_MANIFEST=""
PROJECT=""
BUCKET=""
REGION="us-central1"
PARALLEL=20
NAMESPACE="default"
IMAGE_NAME=""
SMOKE_TEST=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Submit multiple GCP experiments as parallel Kubernetes Jobs.

OPTIONS:
    --scenarios PATH    Path to scenarios manifest.json (required)
    --project ID        GCP project ID (required)
    --bucket NAME       GCS bucket name (required)
    --region REGION     GCP region (default: us-central1)
    --parallel N        Number of jobs to run in parallel (default: 20)
    --image NAME        Container image name (required)
    --namespace NAME    Kubernetes namespace (default: default)
    --smoke-test        Enable smoke-test mode
    -h, --help          Show this help
EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scenarios) SCENARIOS_MANIFEST="$2"; shift 2 ;;
        --project) PROJECT="$2"; shift 2 ;;
        --bucket) BUCKET="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --image) IMAGE_NAME="$2"; shift 2 ;;
        --namespace) NAMESPACE="$2"; shift 2 ;;
        --smoke-test) SMOKE_TEST=true; shift ;;
        -h|--help) usage ;;
        *) log_error "Unknown option: $1"; usage ;;
    esac
done

# Validate
[[ -z "$SCENARIOS_MANIFEST" ]] && { log_error "Missing --scenarios"; usage; }
[[ -z "$PROJECT" ]] && { log_error "Missing --project"; usage; }
[[ -z "$BUCKET" ]] && { log_error "Missing --bucket"; usage; }
[[ -z "$IMAGE_NAME" ]] && { log_error "Missing --image"; usage; }

log_info "Submitting parallel GCP jobs..."
log_info "  Scenarios: $SCENARIOS_MANIFEST"
log_info "  Parallel: $PARALLEL"
log_info "  Namespace: $NAMESPACE"

# Read scenarios from manifest
SCENARIOS=$(python3 -c "
import json
with open('$SCENARIOS_MANIFEST') as f:
    manifest = json.load(f)
for s in manifest['scenarios']:
    print(f\"{s['id']}|{s['path']}\")
")

# Submit jobs in batches
BATCH_NUM=0
JOB_COUNT=0
declare -a PENDING_JOBS=()

while IFS='|' read -r scenario_id scenario_path; do
    # Create unique job name
    JOB_NAME="pqc-bench-${scenario_id}"
    # Sanitize job name (K8s names must be DNS-1123 subdomain)
    JOB_NAME=$(echo "$JOB_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9-]/-/g' | sed 's/--*/-/g' | cut -c1-63)
    
    # Create job YAML
    TEMP_JOB=$(mktemp)
    python3 <<PYTHON_EOF
import yaml
import sys

# Read base job template
with open('$K8S_GCP_DIR/worker-job.yaml', 'r') as f:
    job = yaml.safe_load(f)

# Update job metadata
job['metadata']['name'] = '$JOB_NAME'
job['metadata']['namespace'] = '$NAMESPACE'
job['metadata']['labels']['experiment-id'] = '${scenario_id}'

# Update scenario ConfigMap name
# We'll create a ConfigMap per job
job['spec']['template']['spec']['volumes'] = job['spec']['template']['spec'].get('volumes', [])
# Find and update configMap volume
for vol in job['spec']['template']['spec']['volumes']:
    if vol.get('name') == 'scenario':
        vol['configMap']['name'] = 'pqc-scenario-${scenario_id}'

# Update image
for container in job['spec']['template']['spec']['containers']:
    if container['name'] == 'pqc-bench':
        container['image'] = '$IMAGE_NAME'

# Write job YAML
with open('$TEMP_JOB', 'w') as f:
    yaml.dump(job, f, default_flow_style=False, sort_keys=False)
PYTHON_EOF
    
    # Create scenario ConfigMap
    SCENARIO_CM="pqc-scenario-${scenario_id}"
    kubectl create configmap "$SCENARIO_CM" \
        --from-file=scenario.yaml="$scenario_path" \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1
    
    # Submit job
    kubectl apply -f "$TEMP_JOB" >/dev/null 2>&1
    PENDING_JOBS+=("$JOB_NAME")
    JOB_COUNT=$((JOB_COUNT + 1))
    rm -f "$TEMP_JOB"
    
    # If we've reached parallel limit, wait for some to complete
    if [[ ${#PENDING_JOBS[@]} -ge $PARALLEL ]]; then
        log_info "Submitted $JOB_COUNT jobs, waiting for some to complete..."
        # Wait for at least one job to complete
        while [[ ${#PENDING_JOBS[@]} -ge $PARALLEL ]]; do
            sleep 5
            # Check which jobs are complete
            COMPLETED=()
            for job in "${PENDING_JOBS[@]}"; do
                STATUS=$(kubectl get job "$job" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || echo "")
                if [[ "$STATUS" == "True" ]]; then
                    COMPLETED+=("$job")
                fi
            done
            # Remove completed jobs from pending list
            for job in "${COMPLETED[@]}"; do
                PENDING_JOBS=("${PENDING_JOBS[@]/$job}")
            done
        done
    fi
done <<< "$SCENARIOS"

log_success "Submitted $JOB_COUNT jobs"
log_info "Waiting for all jobs to complete..."

# Wait for all remaining jobs
while [[ ${#PENDING_JOBS[@]} -gt 0 ]]; do
    sleep 10
    COMPLETED=()
    for job in "${PENDING_JOBS[@]}"; do
        STATUS=$(kubectl get job "$job" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || echo "")
        if [[ "$STATUS" == "True" ]]; then
            COMPLETED+=("$job")
        fi
    done
    # Remove completed jobs
    for job in "${COMPLETED[@]}"; do
        PENDING_JOBS=("${PENDING_JOBS[@]/$job}")
    done
    log_info "Waiting... ${#PENDING_JOBS[@]} jobs remaining"
done

log_success "All jobs completed!"

