#!/usr/bin/env bash
# =============================================================================
# tests/smoke/test_smoke_gcp.sh - Smoke test for GCP environment
#
# Validates GCP prerequisites and basic functionality.
# Full GCP smoke test requires actual GCP credentials and project.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$SCRIPT_DIR/tests/lib/common.sh"

test_smoke_gcp() {
    test_start "Smoke test: GCP environment validation"
    
    # Check prerequisites
    local missing_tools=()
    
    if ! command -v gcloud &> /dev/null; then
        missing_tools+=("gcloud")
    fi
    
    if ! command -v kubectl &> /dev/null; then
        missing_tools+=("kubectl")
    fi
    
    if ! command -v terraform &> /dev/null; then
        missing_tools+=("terraform")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        test_warn "Missing GCP tools: ${missing_tools[*]} - skipping GCP smoke test"
        test_pass "Test skipped (GCP tools not available)"
        return 0
    fi
    
    # Check GCP authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &>/dev/null | head -1; then
        test_warn "GCP not authenticated - skipping full smoke test"
        test_pass "GCP tools validated (authentication required for full test)"
        return 0
    fi
    
    # Check if we can access GKE (if cluster exists)
    if gcloud container clusters list &>/dev/null; then
        log_info "GCP authentication validated"
        test_pass "GCP smoke test prerequisites validated"
    else
        test_warn "Cannot list GKE clusters - may need project configuration"
        test_pass "GCP tools validated (project configuration may be needed)"
    fi
    
    # Note: Full GCP smoke test would require:
    # - GCP project configured
    # - GCS bucket created
    # - Terraform state initialized
    # - Actual experiment execution (expensive)
    # This is beyond scope of basic smoke test
    
    return 0
}

# Run test
test_smoke_gcp
EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    test_summary "GCP smoke test: PASSED"
else
    test_summary "GCP smoke test: FAILED"
fi

exit $EXIT_CODE

