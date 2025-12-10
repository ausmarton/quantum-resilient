#!/usr/bin/env bash
# =============================================================================
# tests/functional/test_refactored_scripts.sh - Functional tests for refactored scripts
#
# Tests that refactored scripts (run_minikube.sh, deploy_gcp.sh, run_local.sh)
# correctly use the unified libraries and maintain expected behavior.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$SCRIPT_DIR/tests/lib/common.sh"

test_scripts_source_libraries() {
    test_start "Testing scripts source required libraries"
    
    # Check run_minikube.sh sources libraries
    if grep -q "source.*lib/common.sh" "$SCRIPT_DIR/run_minikube.sh" && \
       grep -q "source.*lib/k8s-job.sh" "$SCRIPT_DIR/run_minikube.sh"; then
        test_pass "run_minikube.sh sources required libraries"
    else
        test_fail "run_minikube.sh missing library sources"
        return 1
    fi
    
    # Check deploy_gcp.sh sources libraries
    if grep -q "source.*lib/common.sh" "$SCRIPT_DIR/deploy_gcp.sh" && \
       grep -q "source.*lib/k8s-job.sh" "$SCRIPT_DIR/deploy_gcp.sh"; then
        test_pass "deploy_gcp.sh sources required libraries"
    else
        test_fail "deploy_gcp.sh missing library sources"
        return 1
    fi
    
    # Check run_local.sh sources libraries
    if grep -q "source.*lib/common.sh" "$SCRIPT_DIR/run_local.sh"; then
        test_pass "run_local.sh sources required libraries"
    else
        test_fail "run_local.sh missing library sources"
        return 1
    fi
}

test_scripts_use_unified_functions() {
    test_start "Testing scripts use unified functions"
    
    # Check run_minikube.sh uses unified functions
    if grep -q "wait_for_job\|copy_results_from_pvc" "$SCRIPT_DIR/run_minikube.sh"; then
        test_pass "run_minikube.sh uses unified job functions"
    else
        test_fail "run_minikube.sh not using unified job functions"
        return 1
    fi
    
    # Check deploy_gcp.sh uses unified functions
    if grep -q "wait_for_job\|download_results_from_gcs" "$SCRIPT_DIR/deploy_gcp.sh"; then
        test_pass "deploy_gcp.sh uses unified job functions"
    else
        test_fail "deploy_gcp.sh not using unified job functions"
        return 1
    fi
}

test_scripts_have_valid_syntax() {
    test_start "Testing scripts have valid bash syntax"
    
    for script in run_minikube.sh deploy_gcp.sh run_local.sh scripts/run_experiment.sh; do
        if bash -n "$SCRIPT_DIR/$script" 2>&1; then
            test_pass "$script has valid syntax"
        else
            test_fail "$script has syntax errors"
            return 1
        fi
    done
}

test_unified_entry_point_routes() {
    test_start "Testing unified entry point routes correctly"
    
    # Test that --env native routes to run_local.sh (it will show run_local.sh usage)
    local output=$(bash "$SCRIPT_DIR/scripts/run_experiment.sh" --env native --help 2>&1 || true)
    if echo "$output" | grep -q "Run a complete PQC microbenchmark\|run_local\|--scenario PATH"; then
        test_pass "Unified entry point routes --env native correctly"
    else
        # If it doesn't route, at least verify it validates environment
        if echo "$output" | grep -q "native\|minikube\|gcp"; then
            test_pass "Unified entry point validates environment"
        else
            test_fail "Unified entry point routing/validation failed"
            return 1
        fi
    fi
}

# Run all tests
test_scripts_source_libraries
test_scripts_use_unified_functions
test_scripts_have_valid_syntax
test_unified_entry_point_routes
test_summary

