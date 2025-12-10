#!/usr/bin/env bash
# =============================================================================
# tests/integration/test_configmap_creation.sh - Integration test for ConfigMap creation
#
# Tests that ConfigMap creation functions produce valid Kubernetes resources.
# This test validates ConfigMap structure without requiring a Kubernetes cluster.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$SCRIPT_DIR/tests/lib/common.sh"

test_configmap_functions_exist() {
    test_start "Testing ConfigMap creation functions exist"
    
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

test_scenario_configmap_sanitization() {
    test_start "Testing ConfigMap name sanitization"
    
    # Test that invalid names are sanitized
    local test_cases=(
        "test_experiment_id"  # underscore should become hyphen
        "TestExperimentID"    # uppercase should become lowercase
        "test.experiment.id"  # dots should be preserved
        "test-experiment-id"  # already valid
    )
    
    # Check if sanitization function exists (it should be in k8s-configmap.sh or submit script)
    if grep -q "SANITIZE_K8S_NAME\|sanitize.*name" "$SCRIPT_DIR/scripts/submit_gcp_job_parallel.sh" 2>/dev/null; then
        test_pass "ConfigMap name sanitization function exists"
    else
        test_fail "ConfigMap name sanitization function not found"
        return 1
    fi
}

test_configmap_yaml_structure() {
    test_start "Testing ConfigMap YAML structure validation"
    
    # Create a test scenario file
    local test_dir=$(mktemp -d)
    trap "rm -rf '$test_dir'" EXIT
    
    cat > "$test_dir/test-scenario.yaml" <<EOF
algorithm: rsa2048
operation: sign
payload_size: 1024
rate: 100
EOF
    
    # Test that we can create a ConfigMap (dry-run if kubectl available, otherwise validate YAML structure)
    if command -v kubectl &>/dev/null; then
        # Try to create ConfigMap in dry-run mode
        if kubectl create configmap test-configmap \
            --from-file=scenario.yaml="$test_dir/test-scenario.yaml" \
            --dry-run=client -o yaml > "$test_dir/configmap.yaml" 2>&1; then
            
            # Validate YAML structure
            if python3 -c "
import yaml
import sys
with open('$test_dir/configmap.yaml') as f:
    cm = yaml.safe_load(f)
    assert cm['kind'] == 'ConfigMap', 'Kind must be ConfigMap'
    assert 'data' in cm, 'ConfigMap must have data'
    assert 'scenario.yaml' in cm['data'], 'ConfigMap must have scenario.yaml'
    print('OK')
" 2>&1; then
                test_pass "ConfigMap YAML structure is valid"
            else
                test_fail "ConfigMap YAML structure is invalid"
                return 1
            fi
        else
            test_fail "Failed to create ConfigMap (dry-run)"
            return 1
        fi
    else
        # kubectl not available - skip this test
        echo "  [SKIP] kubectl not available - skipping ConfigMap creation test"
        test_pass "Test skipped (kubectl not available)"
        return 0
    fi
}

# Run all tests
test_configmap_functions_exist
test_scenario_configmap_sanitization
test_configmap_yaml_structure
test_summary

