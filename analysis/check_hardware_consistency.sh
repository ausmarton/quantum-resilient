#!/bin/bash
# =============================================================================
# check_hardware_consistency.sh - Validate hardware consistency between runs
#
# Ensures that smoke-test and full benchmark runs use identical hardware
# configurations to maintain research validity.
#
# Usage:
#   ./check_hardware_consistency.sh <smoke-test-metadata.json> <full-run-metadata.json>
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_error() {
    echo -e "${RED}✗ ERROR:${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}⚠ WARNING:${NC} $1"
}

# Check if jq is available
if ! command -v jq &> /dev/null; then
    log_error "jq is required but not installed. Install with: sudo apt-get install jq"
    exit 1
fi

# Parse arguments
if [[ $# -lt 2 ]]; then
    log_error "Usage: $0 <smoke-test-metadata.json> <full-run-metadata.json>"
    exit 1
fi

SMOKE_METADATA="$1"
FULL_METADATA="$2"

if [[ ! -f "$SMOKE_METADATA" ]]; then
    log_error "Smoke test metadata file not found: $SMOKE_METADATA"
    exit 1
fi

if [[ ! -f "$FULL_METADATA" ]]; then
    log_error "Full run metadata file not found: $FULL_METADATA"
    exit 1
fi

# Extract hardware characteristics
extract_field() {
    local file="$1"
    local field="$2"
    jq -r "$field // \"MISSING\"" "$file" 2>/dev/null || echo "MISSING"
}

echo "=============================================================================="
echo "Hardware Consistency Validation"
echo "=============================================================================="
echo ""

# Check machine type
SMOKE_MACHINE_TYPE=$(extract_field "$SMOKE_METADATA" ".machine_type // .cluster_config.machine_type")
FULL_MACHINE_TYPE=$(extract_field "$FULL_METADATA" ".machine_type // .cluster_config.machine_type")

if [[ "$SMOKE_MACHINE_TYPE" == "MISSING" ]] || [[ "$FULL_MACHINE_TYPE" == "MISSING" ]]; then
    log_error "Machine type not found in metadata files"
    exit 1
fi

if [[ "$SMOKE_MACHINE_TYPE" == "$FULL_MACHINE_TYPE" ]]; then
    log_success "Machine type identical: $SMOKE_MACHINE_TYPE"
else
    log_error "Machine type mismatch: smoke-test=$SMOKE_MACHINE_TYPE, full-run=$FULL_MACHINE_TYPE"
    exit 1
fi

# Check CPU family (extract from machine type prefix)
SMOKE_CPU_FAMILY=$(echo "$SMOKE_MACHINE_TYPE" | cut -d'-' -f1)
FULL_CPU_FAMILY=$(echo "$FULL_MACHINE_TYPE" | cut -d'-' -f1)

if [[ "$SMOKE_CPU_FAMILY" == "$FULL_CPU_FAMILY" ]]; then
    log_success "CPU family identical: $SMOKE_CPU_FAMILY"
else
    log_error "CPU family mismatch: smoke-test=$SMOKE_CPU_FAMILY, full-run=$FULL_CPU_FAMILY"
    exit 1
fi

# Check region
SMOKE_REGION=$(extract_field "$SMOKE_METADATA" ".region // .cluster_config.region")
FULL_REGION=$(extract_field "$FULL_METADATA" ".region // .cluster_config.region")

if [[ "$SMOKE_REGION" != "MISSING" ]] && [[ "$FULL_REGION" != "MISSING" ]]; then
    if [[ "$SMOKE_REGION" == "$FULL_REGION" ]]; then
        log_success "Region identical: $SMOKE_REGION"
    else
        log_error "Region mismatch: smoke-test=$SMOKE_REGION, full-run=$FULL_REGION"
        exit 1
    fi
fi

# Check cluster version (if available)
SMOKE_K8S_VERSION=$(extract_field "$SMOKE_METADATA" ".kubernetes_version // .cluster_config.kubernetes_version")
FULL_K8S_VERSION=$(extract_field "$FULL_METADATA" ".kubernetes_version // .cluster_config.kubernetes_version")

if [[ "$SMOKE_K8S_VERSION" != "MISSING" ]] && [[ "$FULL_K8S_VERSION" != "MISSING" ]]; then
    if [[ "$SMOKE_K8S_VERSION" == "$FULL_K8S_VERSION" ]]; then
        log_success "Kubernetes version identical: $SMOKE_K8S_VERSION"
    else
        log_warn "Kubernetes version differs: smoke-test=$SMOKE_K8S_VERSION, full-run=$FULL_K8S_VERSION"
        log_warn "This may be acceptable if versions are compatible"
    fi
fi

# Check disk type (if available)
SMOKE_DISK_TYPE=$(extract_field "$SMOKE_METADATA" ".disk_type // .cluster_config.disk_type")
FULL_DISK_TYPE=$(extract_field "$FULL_METADATA" ".disk_type // .cluster_config.disk_type")

if [[ "$SMOKE_DISK_TYPE" != "MISSING" ]] && [[ "$FULL_DISK_TYPE" != "MISSING" ]]; then
    if [[ "$SMOKE_DISK_TYPE" == "$FULL_DISK_TYPE" ]]; then
        log_success "Disk type identical: $SMOKE_DISK_TYPE"
    else
        log_error "Disk type mismatch: smoke-test=$SMOKE_DISK_TYPE, full-run=$FULL_DISK_TYPE"
        exit 1
    fi
fi

# Check CPU model (if available)
SMOKE_CPU_MODEL=$(extract_field "$SMOKE_METADATA" ".cpu_model // .cluster_config.cpu_model")
FULL_CPU_MODEL=$(extract_field "$FULL_METADATA" ".cpu_model // .cluster_config.cpu_model")

if [[ "$SMOKE_CPU_MODEL" != "MISSING" ]] && [[ "$FULL_CPU_MODEL" != "MISSING" ]]; then
    if [[ "$SMOKE_CPU_MODEL" == "$FULL_CPU_MODEL" ]]; then
        log_success "CPU model identical: $SMOKE_CPU_MODEL"
    else
        log_warn "CPU model differs: smoke-test=$SMOKE_CPU_MODEL, full-run=$FULL_CPU_MODEL"
        log_warn "This may be acceptable if models are from the same generation"
    fi
fi

echo ""
echo "=============================================================================="
log_success "Hardware consistency validation passed"
echo "=============================================================================="
exit 0

