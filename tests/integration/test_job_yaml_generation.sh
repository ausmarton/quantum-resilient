#!/usr/bin/env bash
# =============================================================================
# tests/integration/test_job_yaml_generation.sh - Integration test for job YAML generation
#
# Tests that the job generator produces valid, correct YAML for both Minikube and GCP.
# This test validates YAML structure without requiring a Kubernetes cluster.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$SCRIPT_DIR/tests/lib/common.sh"

test_job_generator_minikube_yaml() {
    test_start "Testing job generator produces valid Minikube YAML"
    
    local generator="$SCRIPT_DIR/scripts/lib/k8s-job-generator.py"
    local test_dir=$(mktemp -d)
    trap "rm -rf '$test_dir'" EXIT
    
    # Generate Minikube job YAML
    if python3 "$generator" \
        --environment minikube \
        --job-name test-job \
        --namespace default \
        --image localhost/pqc-bench:latest \
        --scenario-configmap test-scenario \
        --output "$test_dir/minikube-job.yaml" 2>&1; then
        test_pass "Job generator produced Minikube YAML"
        
        # Validate YAML structure
        if python3 -c "
import yaml
import sys
with open('$test_dir/minikube-job.yaml') as f:
    job = yaml.safe_load(f)
    # Check required fields
    assert job['kind'] == 'Job', 'Kind must be Job'
    assert job['metadata']['name'] == 'test-job', 'Job name must match'
    assert 'spec' in job, 'Job must have spec'
    assert 'template' in job['spec'], 'Job spec must have template'
    assert 'containers' in job['spec']['template']['spec'], 'Template must have containers'
    assert len(job['spec']['template']['spec']['containers']) > 0, 'Must have at least one container'
    
    # Check Minikube-specific fields
    container = job['spec']['template']['spec']['containers'][0]
    assert container['image'] == 'localhost/pqc-bench:latest', 'Image must match'
    assert container.get('imagePullPolicy') == 'Never', 'Minikube should use Never pull policy'
    
    # Check volumes
    volumes = job['spec']['template']['spec'].get('volumes', [])
    pvc_volume = [v for v in volumes if v.get('persistentVolumeClaim')]
    assert len(pvc_volume) > 0, 'Minikube should have PVC volume'
    
    print('OK')
" 2>&1; then
            test_pass "Minikube YAML structure is valid"
        else
            test_fail "Minikube YAML structure is invalid"
            return 1
        fi
    else
        test_fail "Job generator failed to produce Minikube YAML"
        return 1
    fi
}

test_job_generator_gcp_yaml() {
    test_start "Testing job generator produces valid GCP YAML"
    
    local generator="$SCRIPT_DIR/scripts/lib/k8s-job-generator.py"
    local test_dir=$(mktemp -d)
    trap "rm -rf '$test_dir'" EXIT
    
    # Generate GCP job YAML
    local gcp_config='{"bucket_name": "test-bucket", "experiment_id": "test-exp"}'
    
    if python3 "$generator" \
        --environment gcp \
        --job-name test-job \
        --namespace default \
        --image gcr.io/test/pqc-bench:latest \
        --scenario-configmap test-scenario \
        --gcp-config "$gcp_config" \
        --output "$test_dir/gcp-job.yaml" 2>&1; then
        test_pass "Job generator produced GCP YAML"
        
        # Validate YAML structure
        if python3 -c "
import yaml
import sys
with open('$test_dir/gcp-job.yaml') as f:
    job = yaml.safe_load(f)
    # Check required fields
    assert job['kind'] == 'Job', 'Kind must be Job'
    assert job['metadata']['name'] == 'test-job', 'Job name must match'
    
    # Check GCP-specific fields
    container = job['spec']['template']['spec']['containers'][0]
    assert container['image'] == 'gcr.io/test/pqc-bench:latest', 'Image must match'
    assert container.get('imagePullPolicy') == 'Always', 'GCP should use Always pull policy'
    
    # Check for sidecar container (upload-results)
    containers = job['spec']['template']['spec']['containers']
    sidecar = [c for c in containers if c['name'] == 'upload-results']
    assert len(sidecar) > 0, 'GCP should have upload-results sidecar'
    
    # Check volumes
    volumes = job['spec']['template']['spec'].get('volumes', [])
    empty_dir_volume = [v for v in volumes if v.get('emptyDir')]
    assert len(empty_dir_volume) > 0, 'GCP should have emptyDir volume'
    
    # Check service account
    assert job['spec']['template']['spec'].get('serviceAccountName') == 'pqc-bench-sa', 'GCP should use service account'
    
    print('OK')
" 2>&1; then
            test_pass "GCP YAML structure is valid"
        else
            test_fail "GCP YAML structure is invalid"
            return 1
        fi
    else
        test_fail "Job generator failed to produce GCP YAML"
        return 1
    fi
}

test_job_generator_yaml_differences() {
    test_start "Testing job generator produces different YAML for Minikube vs GCP"
    
    local generator="$SCRIPT_DIR/scripts/lib/k8s-job-generator.py"
    local test_dir=$(mktemp -d)
    trap "rm -rf '$test_dir'" EXIT
    
    # Generate both Minikube and GCP YAMLs
    local gcp_config='{"bucket_name": "test-bucket", "experiment_id": "test-exp"}'
    
    python3 "$generator" \
        --environment minikube \
        --job-name test-minikube \
        --namespace default \
        --image localhost/pqc-bench:latest \
        --scenario-configmap test-scenario \
        --output "$test_dir/minikube.yaml" 2>&1
    
    python3 "$generator" \
        --environment gcp \
        --job-name test-gcp \
        --namespace default \
        --image gcr.io/test/pqc-bench:latest \
        --scenario-configmap test-scenario \
        --gcp-config-configmap test-gcp-config \
        --output "$test_dir/gcp.yaml" 2>&1
    
    # Validate differences
    if python3 -c "
import yaml
import sys

with open('$test_dir/minikube.yaml') as f:
    minikube_job = yaml.safe_load(f)
with open('$test_dir/gcp.yaml') as f:
    gcp_job = yaml.safe_load(f)

# Check image pull policy differences
minikube_container = minikube_job['spec']['template']['spec']['containers'][0]
gcp_container = gcp_job['spec']['template']['spec']['containers'][0]

assert minikube_container.get('imagePullPolicy') == 'Never', 'Minikube should use Never'
assert gcp_container.get('imagePullPolicy') == 'Always', 'GCP should use Always'

# Check volume differences
minikube_volumes = minikube_job['spec']['template']['spec'].get('volumes', [])
gcp_volumes = gcp_job['spec']['template']['spec'].get('volumes', [])

minikube_has_pvc = any(v.get('persistentVolumeClaim') for v in minikube_volumes)
gcp_has_empty_dir = any(v.get('emptyDir') for v in gcp_volumes)

assert minikube_has_pvc, 'Minikube should have PVC volume'
assert gcp_has_empty_dir, 'GCP should have emptyDir volume'

# Check GCP has sidecar
gcp_containers = gcp_job['spec']['template']['spec']['containers']
gcp_has_sidecar = any(c['name'] == 'upload-results' for c in gcp_containers)
assert gcp_has_sidecar, 'GCP should have upload-results sidecar'

# Check GCP has service account
assert gcp_job['spec']['template']['spec'].get('serviceAccountName') == 'pqc-bench-sa', 'GCP should use service account'

print('OK')
" 2>&1; then
        test_pass "Minikube and GCP YAMLs have correct differences"
    else
        test_fail "Minikube and GCP YAMLs don't have expected differences"
        return 1
    fi
}

# Run all tests
test_job_generator_minikube_yaml
test_job_generator_gcp_yaml
test_job_generator_yaml_differences
test_summary

