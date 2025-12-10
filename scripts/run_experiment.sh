#!/usr/bin/env bash
# =============================================================================
# scripts/run_experiment.sh - Unified Entry Point for PQC Benchmark Experiments
#
# Routes experiment execution to the appropriate environment-specific script
# (native, Minikube, or GCP) based on the --env parameter.
#
# Usage:
#   ./scripts/run_experiment.sh --env native --scenario scenarios/kyber512.yaml --out results/exp1
#   ./scripts/run_experiment.sh --env minikube --scenario scenarios/kyber512.yaml --out results/exp2 --exp-id exp2 --replicas 4
#   ./scripts/run_experiment.sh --env gcp --scenario scenarios/kyber512.yaml --exp-id exp3 --project my-project --bucket my-bucket
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    cat <<EOF
Usage: $0 --env <environment> [OPTIONS]

Unified entry point for running PQC benchmark experiments across different environments.

ENVIRONMENTS:
    native      Run benchmark natively on the local machine
    minikube    Run benchmark in a local Minikube Kubernetes cluster
    gcp         Run benchmark on Google Kubernetes Engine (GKE)

OPTIONS (Common):
    --scenario PATH     Path to scenario YAML file (required)
    --out DIR           Output directory for results (required for native/minikube)
    --runs N            Number of repeated runs (default: 1)
    --replicas N        Number of parallel replicas (default: 1, minikube/gcp only)
    --seed NUM          Base RNG seed (optional)
    --skip-analysis     Skip Python analysis step
    --smoke-test        Enable smoke-test mode (reduced duration/scale)
    -h, --help          Show this help message

OPTIONS (Native):
    --duration SEC      Override duration from scenario
    --timeout SEC       Timeout per run (default: 3600)
    --skip-aggregation  Skip aggregation across runs

OPTIONS (Minikube):
    --exp-id ID         Experiment identifier (required)
    --skip-build        Skip container image build
    --force-build       Force rebuild even if image exists
    --tag-git           Tag image with git commit hash
    --skip-aggregation  Skip aggregation across runs
    --keep-job          Don't delete Job after completion
    --timeout SEC       Job timeout in seconds (default: 600)

OPTIONS (GCP):
    --exp-id ID         Experiment identifier (required)
    --project ID        GCP project ID (required)
    --region REGION     GCP region (default: us-central1)
    --bucket NAME       GCS bucket name (required)
    --machine-type TYPE GKE node machine type (default: n2-standard-2)
    --node-count N      Number of nodes (default: 1)
    --skip-terraform    Skip Terraform apply (use existing cluster)
    --skip-build        Skip container image build
    --skip-aggregation  Skip aggregation across runs
    --skip-job          Skip job deployment (only build image)
    --destroy-after     Destroy infrastructure after experiment
    --timeout SEC       Job timeout in seconds (default: 900)
    --ephemeral         Ephemeral mode: create cluster, run benchmark, destroy all resources
    --create-cluster    Only create the cluster (skip benchmark execution)
    --destroy-cluster   Only destroy the cluster and cleanup resources

EXAMPLES:
    # Native run
    $0 --env native --scenario scenarios/kyber512.yaml --out results/exp1

    # Minikube run with 4 replicas
    $0 --env minikube --scenario scenarios/kyber512.yaml --out results/exp2 --exp-id exp2 --replicas 4

    # GCP run
    $0 --env gcp --scenario scenarios/kyber512.yaml --exp-id exp3 --project my-project --bucket my-bucket

    # Smoke test in Minikube
    $0 --env minikube --scenario scenarios/kyber512.yaml --out results/smoke --exp-id smoke --smoke-test

    # Ephemeral GCP run (creates, runs, destroys)
    $0 --env gcp --scenario scenarios/kyber512.yaml --exp-id ephemeral --project my-project --bucket my-bucket --ephemeral
EOF
    exit 1
}

# Parse --env argument first
ENV=""
ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate environment
if [[ -z "$ENV" ]]; then
    log_error "Missing required argument: --env"
    echo ""
    usage
fi

case "$ENV" in
    native)
        SCRIPT="$SCRIPT_DIR/run_local.sh"
        ;;
    minikube)
        SCRIPT="$SCRIPT_DIR/run_minikube.sh"
        ;;
    gcp)
        SCRIPT="$SCRIPT_DIR/deploy_gcp.sh"
        ;;
    *)
        log_error "Invalid environment: $ENV"
        log_error "Valid environments: native, minikube, gcp"
        exit 1
        ;;
esac

# Check if script exists
if [[ ! -f "$SCRIPT" ]]; then
    log_error "Script not found: $SCRIPT"
    exit 1
fi

# Make script executable
chmod +x "$SCRIPT" 2>/dev/null || true

# Route to appropriate script with all remaining arguments
log_info "Routing to $ENV environment script: $SCRIPT"
exec "$SCRIPT" "${ARGS[@]}"

