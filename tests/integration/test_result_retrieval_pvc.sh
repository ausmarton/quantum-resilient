#!/usr/bin/env bash
# =============================================================================
# tests/integration/test_result_retrieval_pvc.sh - Integration test for PVC result retrieval
#
# Tests that results can be retrieved from a PersistentVolumeClaim in Kubernetes.
# This validates the Minikube data collection workflow.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$SCRIPT_DIR/tests/lib/common.sh"

test_result_retrieval_pvc() {
    test_start "Integration test: PVC result retrieval"
    
    # Check prerequisites
    if ! command -v kubectl &> /dev/null; then
        test_warn "kubectl not found - skipping PVC retrieval test"
        test_pass "Test skipped (kubectl not available)"
        return 0
    fi
    
    # Check if we can access Kubernetes
    if ! kubectl cluster-info &>/dev/null; then
        test_warn "Cannot access Kubernetes cluster - skipping test"
        test_pass "Test skipped (Kubernetes not accessible)"
        return 0
    fi
    
    # Create a test namespace
    local test_namespace="pqc-test-$(date +%s)"
    kubectl create namespace "$test_namespace" &>/dev/null || true
    trap "kubectl delete namespace '$test_namespace' --ignore-not-found=true &>/dev/null" EXIT
    
    # Create a test PVC
    local pvc_name="test-pvc-$(date +%s)"
    cat <<EOF | kubectl apply -n "$test_namespace" -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: $pvc_name
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
EOF
    
    # Wait for PVC to be bound
    log_info "Waiting for PVC to be bound..."
    if ! kubectl wait --for=condition=Bound -n "$test_namespace" "pvc/$pvc_name" --timeout=30s &>/dev/null; then
        test_fail "PVC failed to bind"
        return 1
    fi
    
    # Create a pod that writes test data to PVC
    local pod_name="test-writer-$(date +%s)"
    cat <<EOF | kubectl apply -n "$test_namespace" -f -
apiVersion: v1
kind: Pod
metadata:
  name: $pod_name
spec:
  containers:
  - name: writer
    image: busybox:latest
    command: ["sh", "-c"]
    args:
    - |
      mkdir -p /data/results/raw
      echo '{"event_id": 1, "latency_ns": 1000, "timestamp_utc_iso": "2025-01-01T00:00:00Z"}' > /data/results/raw/run.jsonl
      echo '{"event_id": 2, "latency_ns": 2000, "timestamp_utc_iso": "2025-01-01T00:00:01Z"}' >> /data/results/raw/run.jsonl
      sleep 10
    volumeMounts:
    - name: data
      mountPath: /data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: $pvc_name
  restartPolicy: Never
EOF
    
    # Wait for pod to complete
    log_info "Waiting for test pod to complete..."
    if ! kubectl wait --for=condition=Ready -n "$test_namespace" "pod/$pod_name" --timeout=60s &>/dev/null; then
        test_warn "Pod did not become ready in time"
    fi
    
    # Wait for pod to complete
    kubectl wait --for=jsonpath='{.status.phase}'=Succeeded -n "$test_namespace" "pod/$pod_name" --timeout=60s &>/dev/null || true
    
    # Create a debug pod to read from PVC
    local debug_pod="test-reader-$(date +%s)"
    cat <<EOF | kubectl apply -n "$test_namespace" -f -
apiVersion: v1
kind: Pod
metadata:
  name: $debug_pod
spec:
  containers:
  - name: reader
    image: busybox:latest
    command: ["sh", "-c"]
    args:
    - |
      if [ -f /data/results/raw/run.jsonl ]; then
        echo "File exists"
        cat /data/results/raw/run.jsonl
        exit 0
      else
        echo "File not found"
        exit 1
      fi
    volumeMounts:
    - name: data
      mountPath: /data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: $pvc_name
  restartPolicy: Never
EOF
    
    # Wait for debug pod to complete
    log_info "Waiting for debug pod to complete..."
    kubectl wait --for=jsonpath='{.status.phase}'=Succeeded -n "$test_namespace" "pod/$debug_pod" --timeout=60s &>/dev/null || true
    
    # Check output
    if kubectl logs -n "$test_namespace" "$debug_pod" 2>/dev/null | grep -q "File exists"; then
        test_pass "PVC result retrieval test passed"
        return 0
    else
        test_fail "Failed to retrieve results from PVC"
        return 1
    fi
}

# Run test
test_result_retrieval_pvc
EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    test_summary "PVC retrieval test: PASSED"
else
    test_summary "PVC retrieval test: FAILED"
fi

exit $EXIT_CODE

