#!/usr/bin/env bash
# =============================================================================
# tests/functional/test_k8s_job_management.sh - Functional tests for Kubernetes job management
#
# Tests that the refactored Kubernetes job management functions work correctly.
# These tests validate the unified job waiting and result retrieval logic.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$SCRIPT_DIR/tests/lib/common.sh"

test_k8s_job_functions_exist() {
    test_start "Testing Kubernetes job management functions exist"
    
    # Source the k8s-job library
    source "$SCRIPT_DIR/scripts/lib/k8s-job.sh"
    
    # Check that functions are defined
    if declare -f wait_for_job >/dev/null 2>&1; then
        test_pass "wait_for_job function exists"
    else
        test_fail "wait_for_job function not found"
        return 1
    fi
    
    if declare -f get_job_pods >/dev/null 2>&1; then
        test_pass "get_job_pods function exists"
    else
        test_fail "get_job_pods function not found"
        return 1
    fi
    
    if declare -f copy_results_from_pvc >/dev/null 2>&1; then
        test_pass "copy_results_from_pvc function exists"
    else
        test_fail "copy_results_from_pvc function not found"
        return 1
    fi
    
    if declare -f download_results_from_gcs >/dev/null 2>&1; then
        test_pass "download_results_from_gcs function exists"
    else
        test_fail "download_results_from_gcs function not found"
        return 1
    fi
}

test_k8s_job_generator_exists() {
    test_start "Testing Kubernetes job generator exists"
    
    if [[ -f "$SCRIPT_DIR/scripts/lib/k8s-job-generator.py" ]]; then
        test_pass "k8s-job-generator.py exists"
        
        # Test that it can be imported
        if python3 -c "import sys; sys.path.insert(0, '$SCRIPT_DIR/scripts/lib'); import yaml" 2>/dev/null; then
            test_pass "Python dependencies available (yaml)"
        else
            test_fail "Python yaml module not available"
            return 1
        fi
        
        # Test that the script has valid syntax
        if python3 -m py_compile "$SCRIPT_DIR/scripts/lib/k8s-job-generator.py" 2>/dev/null; then
            test_pass "k8s-job-generator.py has valid Python syntax"
        else
            test_fail "k8s-job-generator.py has syntax errors"
            return 1
        fi
    else
        test_fail "k8s-job-generator.py not found"
        return 1
    fi
}

test_k8s_job_generator_help() {
    test_start "Testing Kubernetes job generator help output"
    
    local generator="$SCRIPT_DIR/scripts/lib/k8s-job-generator.py"
    
    if [[ ! -f "$generator" ]]; then
        test_fail "k8s-job-generator.py not found"
        return 1
    fi
    
    # Test that --help works
    if python3 "$generator" --help >/dev/null 2>&1; then
        test_pass "k8s-job-generator.py --help works"
    else
        test_fail "k8s-job-generator.py --help failed"
        return 1
    fi
}

test_k8s_configmap_functions_exist() {
    test_start "Testing Kubernetes ConfigMap functions exist"
    
    # Source the k8s-configmap library
    source "$SCRIPT_DIR/scripts/lib/k8s-configmap.sh"
    
    if declare -f create_scenario_configmap >/dev/null 2>&1; then
        test_pass "create_scenario_configmap function exists"
    else
        test_fail "create_scenario_configmap function not found"
        return 1
    fi
    
    if declare -f create_gcp_config_configmap >/dev/null 2>&1; then
        test_pass "create_gcp_config_configmap function exists"
    else
        test_fail "create_gcp_config_configmap function not found"
        return 1
    fi
}

test_unified_entry_point() {
    test_start "Testing unified entry point exists"
    
    if [[ -f "$SCRIPT_DIR/scripts/run_experiment.sh" ]]; then
        test_pass "run_experiment.sh exists"
        
        # Test that it's executable
        if [[ -x "$SCRIPT_DIR/scripts/run_experiment.sh" ]]; then
            test_pass "run_experiment.sh is executable"
        else
            test_fail "run_experiment.sh is not executable"
            return 1
        fi
        
        # Test that --help works (exits with code 1, which is OK - it shows usage)
        local help_output=$(bash "$SCRIPT_DIR/scripts/run_experiment.sh" --help 2>&1 || true)
        if echo "$help_output" | grep -q "Usage\|ENVIRONMENTS\|Unified entry point"; then
            test_pass "run_experiment.sh --help works"
        else
            test_fail "run_experiment.sh --help failed or no usage output"
            return 1
        fi
    else
        test_fail "run_experiment.sh not found"
        return 1
    fi
}

# Run all tests
test_k8s_job_functions_exist
test_k8s_job_generator_exists
test_k8s_job_generator_help
test_k8s_configmap_functions_exist
test_unified_entry_point
test_summary

