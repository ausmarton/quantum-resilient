#!/usr/bin/env bash
# =============================================================================
# scripts/lib/k8s-job.sh - Unified Kubernetes Job Management
#
# Provides unified functions for waiting for Kubernetes jobs and retrieving
# results across Minikube and GCP environments.
#
# Functions:
#   wait_for_job() - Wait for a Kubernetes job to complete
#   get_job_pods() - Get pod names for a job
#   stream_job_logs() - Stream logs from job pods
#   retrieve_job_results() - Retrieve results based on environment
#   copy_results_from_pvc() - Copy results from PVC (Minikube)
#   download_results_from_gcs() - Download results from GCS (GCP)
# =============================================================================

# Source common libraries
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$SCRIPT_DIR/scripts/lib/common.sh"

# =============================================================================
# Job Waiting Functions
# =============================================================================

wait_for_job() {
    # Wait for a Kubernetes job to complete.
    #
    # Args:
    #   job_name: Name of the job
    #   namespace: Kubernetes namespace (default: default)
    #   timeout: Timeout duration (default: 600s)
    #   stream_logs: Whether to stream logs in background (default: true)
    #
    # Returns:
    #   0 on success, 1 on failure or timeout
    local job_name="$1"
    local namespace="${2:-default}"
    local timeout="${3:-600s}"
    local stream_logs="${4:-true}"
    
    if [[ -z "$job_name" ]]; then
        log_error "wait_for_job: job_name is required"
        return 1
    fi
    
    log_info "Waiting for Job '$job_name' to complete (timeout: $timeout)..."
    
    # Get pod names for log streaming
    local log_pid=""
    if [[ "$stream_logs" == "true" ]]; then
        sleep 5  # Give time for pods to be created
        local pod_names=($(kubectl get pods -l job-name="$job_name" -n "$namespace" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo ""))
        
        if [[ ${#pod_names[@]} -gt 0 ]]; then
            if [[ ${#pod_names[@]} -eq 1 ]]; then
                log_info "Pod: ${pod_names[0]}"
            else
                log_info "Pods: ${pod_names[*]} (${#pod_names[@]} replicas)"
            fi
            
            # Stream logs from all pods in background
            (
                sleep 10
                for pod in "${pod_names[@]}"; do
                    (
                        kubectl logs -f "$pod" -n "$namespace" 2>/dev/null | while read -r line; do
                            echo "  [pod:$pod] $line"
                        done
                    ) &
                done
                wait
            ) &
            log_pid=$!
        fi
    fi
    
    # Wait for completion
    if ! kubectl wait --for=condition=complete job/"$job_name" -n "$namespace" --timeout="$timeout" 2>/dev/null; then
        # Check if failed
        local job_status=$(kubectl get job "$job_name" -n "$namespace" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || echo "")
        
        # Kill log streaming if active
        [[ -n "$log_pid" ]] && kill "$log_pid" 2>/dev/null || true
        
        if [[ "$job_status" == "True" ]]; then
            log_error "Job '$job_name' failed!"
            kubectl describe job "$job_name" -n "$namespace" 2>&1 | head -50 || true
            kubectl logs -l job-name="$job_name" -n "$namespace" --tail=50 2>&1 || true
            return 1
        fi
        
        log_error "Job '$job_name' timed out after $timeout"
        return 1
    fi
    
    # Kill log streaming
    [[ -n "$log_pid" ]] && kill "$log_pid" 2>/dev/null || true
    
    log_success "Job '$job_name' completed successfully"
    return 0
}

get_job_pods() {
    # Get pod names for a Kubernetes job.
    #
    # Args:
    #   job_name: Name of the job
    #   namespace: Kubernetes namespace (default: default)
    #
    # Outputs:
    #   Pod names (space-separated) to stdout
    local job_name="$1"
    local namespace="${2:-default}"
    
    kubectl get pods -l job-name="$job_name" -n "$namespace" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo ""
}

get_job_status() {
    # Get the status of a Kubernetes job.
    #
    # Args:
    #   job_name: Name of the job
    #   namespace: Kubernetes namespace (default: default)
    #
    # Outputs:
    #   Status string: "Complete", "Failed", "Running", or "Unknown"
    local job_name="$1"
    local namespace="${2:-default}"
    
    local complete_status=$(kubectl get job "$job_name" -n "$namespace" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || echo "")
    local failed_status=$(kubectl get job "$job_name" -n "$namespace" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || echo "")
    
    if [[ "$complete_status" == "True" ]]; then
        echo "Complete"
    elif [[ "$failed_status" == "True" ]]; then
        echo "Failed"
    else
        # Check if job exists and is running
        local job_exists=$(kubectl get job "$job_name" -n "$namespace" -o name 2>/dev/null || echo "")
        if [[ -n "$job_exists" ]]; then
            echo "Running"
        else
            echo "Unknown"
        fi
    fi
}

# =============================================================================
# Result Retrieval Functions
# =============================================================================

retrieve_job_results() {
    # Retrieve results from a completed Kubernetes job.
    #
    # Args:
    #   job_name: Name of the job
    #   output_dir: Local output directory
    #   environment: "minikube" or "gcp"
    #   namespace: Kubernetes namespace (default: default)
    #   experiment_id: Experiment ID (for GCP GCS path)
    #   bucket: GCS bucket name (for GCP)
    #   pvc_name: PVC name (for Minikube, default: pqc-bench-results)
    #
    # Returns:
    #   0 on success, 1 on failure
    local job_name="$1"
    local output_dir="$2"
    local environment="$3"
    local namespace="${4:-default}"
    local experiment_id="${5:-}"
    local bucket="${6:-}"
    local pvc_name="${7:-pqc-bench-results}"
    
    if [[ -z "$job_name" ]] || [[ -z "$output_dir" ]] || [[ -z "$environment" ]]; then
        log_error "retrieve_job_results: job_name, output_dir, and environment are required"
        return 1
    fi
    
    case "$environment" in
        minikube)
            copy_results_from_pvc "$job_name" "$output_dir" "$namespace" "$pvc_name"
            ;;
        gcp)
            if [[ -z "$experiment_id" ]] || [[ -z "$bucket" ]]; then
                log_error "retrieve_job_results: experiment_id and bucket are required for GCP"
                return 1
            fi
            download_results_from_gcs "$experiment_id" "$bucket" "$output_dir"
            ;;
        *)
            log_error "retrieve_job_results: Unknown environment: $environment"
            return 1
            ;;
    esac
}

copy_results_from_pvc() {
    # Copy results from a PVC using a temporary read pod.
    #
    # Args:
    #   job_name: Name of the job (for pod lookup)
    #   output_dir: Local output directory
    #   namespace: Kubernetes namespace (default: default)
    #   pvc_name: PVC name (default: pqc-bench-results)
    #
    # Returns:
    #   0 on success, 1 on failure
    local job_name="$1"
    local output_dir="$2"
    local namespace="${3:-default}"
    local pvc_name="${4:-pqc-bench-results}"
    
    if [[ -z "$job_name" ]] || [[ -z "$output_dir" ]]; then
        log_error "copy_results_from_pvc: job_name and output_dir are required"
        return 1
    fi
    
    log_info "Copying results from PVC '$pvc_name'..."
    
    # Ensure output directory exists
    mkdir -p "$output_dir/raw"
    
    # Get pod name for reference
    local pod_name=$(kubectl get pods -l job-name="$job_name" -n "$namespace" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    local pod_phase=$(kubectl get pod "$pod_name" -n "$namespace" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
    log_info "Pod: $pod_name (phase: $pod_phase)"
    
    # Create a temporary read pod with unique name (include job name and timestamp)
    local job_suffix=$(echo "$job_name" | tr -cd '[:alnum:]' | cut -c1-10)
    local read_pod_name="pvc-read-${job_suffix}-$(date +%s)-$$"
    local temp_output=$(mktemp)
    local read_pod_yaml=$(mktemp)
    
    # Clean up any existing read pods with similar names (from previous failed attempts)
    kubectl delete pod -n "$namespace" -l "app=pvc-read" --ignore-not-found=true >/dev/null 2>&1 || true
    
    # Generate read pod YAML
    cat > "$read_pod_yaml" <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: $read_pod_name
  namespace: $namespace
  labels:
    app: pvc-read
spec:
  restartPolicy: Never
  containers:
  - name: read
    image: busybox:1.36
    command: ["sh", "-c", "if [ -f /results/raw/run.jsonl ]; then cat /results/raw/run.jsonl; elif [ -d /results/raw ]; then find /results/raw -name '*.jsonl' -type f -exec cat {} +; elif [ -d /results/replica-0 ]; then find /results/replica-* -name '*.jsonl' -type f -exec cat {} +; else find /results -name '*.jsonl' -type f -exec cat {} +; fi"]
    volumeMounts:
    - name: results
      mountPath: /results
      readOnly: true
  volumes:
  - name: results
    persistentVolumeClaim:
      claimName: $pvc_name
EOF
    
    # Create read pod (retry if it fails due to name conflict)
    local retries=0
    local apply_error=""
    while [[ $retries -lt 3 ]]; do
        apply_error=$(kubectl apply --validate=false -f "$read_pod_yaml" 2>&1)
        local apply_exit=$?
        if [[ $apply_exit -eq 0 ]]; then
            break
        fi
        # If it failed, regenerate YAML with a more unique name
        if [[ $retries -lt 2 ]]; then
            read_pod_name="pvc-read-${job_suffix}-$(date +%s)-$$-${retries}"
            # Regenerate YAML with new name (more reliable than sed)
            cat > "$read_pod_yaml" <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: $read_pod_name
  namespace: $namespace
  labels:
    app: pvc-read
spec:
  restartPolicy: Never
  containers:
  - name: read
    image: busybox:1.36
    command: ["sh", "-c", "if [ -f /results/raw/run.jsonl ]; then cat /results/raw/run.jsonl; elif [ -d /results/raw ]; then find /results/raw -name '*.jsonl' -type f -exec cat {} +; elif [ -d /results/replica-0 ]; then find /results/replica-* -name '*.jsonl' -type f -exec cat {} +; else find /results -name '*.jsonl' -type f -exec cat {} +; fi"]
    volumeMounts:
    - name: results
      mountPath: /results
      readOnly: true
  volumes:
  - name: results
    persistentVolumeClaim:
      claimName: $pvc_name
EOF
        fi
        retries=$((retries + 1))
        sleep 1
    done
    
    if [[ $retries -eq 3 ]]; then
        log_error "Failed to create read pod after 3 attempts"
        log_error "Last error: ${apply_error:0:200}"
        rm -f "$read_pod_yaml" "$temp_output"
        return 1
    fi
    
    log_info "Created read pod: $read_pod_name"
    
    # Wait for pod to complete
    local phase=""
    for i in {1..30}; do
        phase=$(kubectl get pod "$read_pod_name" -n "$namespace" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
        if [[ -z "$phase" ]]; then
            sleep 1
            continue
        fi
        if [[ "$phase" == "Succeeded" ]] || [[ "$phase" == "Failed" ]]; then
            break
        fi
        sleep 1
    done
    
    if [[ "$phase" != "Succeeded" ]] && [[ "$phase" != "Failed" ]]; then
        log_error "Read pod did not complete (phase: $phase)"
        kubectl delete pod "$read_pod_name" -n "$namespace" --ignore-not-found=true >/dev/null 2>&1
        rm -f "$read_pod_yaml" "$temp_output"
        return 1
    fi
    
    # Get logs (contains the JSONL data)
    if ! kubectl logs "$read_pod_name" -n "$namespace" > "$temp_output" 2>&1; then
        log_error "Failed to get logs from read pod"
        kubectl delete pod "$read_pod_name" -n "$namespace" --ignore-not-found=true >/dev/null 2>&1
        rm -f "$read_pod_yaml" "$temp_output"
        return 1
    fi
    
    # Extract JSONL lines (lines that start with {)
    grep -E "^{" "$temp_output" > "$output_dir/raw/run.jsonl" 2>/dev/null || true
    
    # Clean up
    kubectl delete pod "$read_pod_name" -n "$namespace" --ignore-not-found=true >/dev/null 2>&1
    rm -f "$read_pod_yaml" "$temp_output"
    
    # Validate results
    if [[ ! -f "$output_dir/raw/run.jsonl" ]] || [[ ! -s "$output_dir/raw/run.jsonl" ]]; then
        log_error "Failed to copy results from PVC (file is missing or empty)"
        return 1
    fi
    
    local file_size=$(stat -f%z "$output_dir/raw/run.jsonl" 2>/dev/null || stat -c%s "$output_dir/raw/run.jsonl" 2>/dev/null || echo 0)
    if [[ $file_size -eq 0 ]]; then
        log_error "Failed to copy results from PVC (file is 0 bytes)"
        rm -f "$output_dir/raw/run.jsonl"
        return 1
    fi
    
    log_success "Successfully copied run.jsonl from PVC ($file_size bytes)"
    return 0
}

download_results_from_gcs() {
    # Download results from GCS bucket.
    #
    # Args:
    #   experiment_id: Experiment ID
    #   bucket: GCS bucket name
    #   output_dir: Local output directory
    #
    # Returns:
    #   0 on success, 1 on failure
    local experiment_id="$1"
    local bucket="$2"
    local output_dir="$3"
    
    if [[ -z "$experiment_id" ]] || [[ -z "$bucket" ]] || [[ -z "$output_dir" ]]; then
        log_error "download_results_from_gcs: experiment_id, bucket, and output_dir are required"
        return 1
    fi
    
    log_info "Downloading results from GCS bucket '$bucket'..."
    
    # Ensure output directory exists
    mkdir -p "$output_dir/raw"
    
    local gcs_path="gs://$bucket/experiments/$experiment_id"
    
    # Download raw data
    if ! gsutil -m cp "$gcs_path/raw/run.jsonl" "$output_dir/raw/run.jsonl" 2>&1; then
        # Fallback: try with rsync if direct copy fails
        if ! gsutil -m rsync -r "$gcs_path/raw" "$output_dir/raw" 2>&1; then
            log_error "Failed to download raw data from GCS"
            return 1
        fi
    fi
    
    # Validate results
    if [[ ! -f "$output_dir/raw/run.jsonl" ]] || [[ ! -s "$output_dir/raw/run.jsonl" ]]; then
        log_error "Failed to download results from GCS (file is missing or empty)"
        return 1
    fi
    
    local file_size=$(stat -f%z "$output_dir/raw/run.jsonl" 2>/dev/null || stat -c%s "$output_dir/raw/run.jsonl" 2>/dev/null || echo 0)
    if [[ $file_size -eq 0 ]]; then
        log_error "Failed to download results from GCS (file is 0 bytes)"
        rm -f "$output_dir/raw/run.jsonl"
        return 1
    fi
    
    log_success "Successfully downloaded run.jsonl from GCS ($file_size bytes)"
    return 0
}

# =============================================================================
# Parallel Job Management
# =============================================================================

wait_for_multiple_jobs() {
    # Wait for multiple Kubernetes jobs to complete (for parallel execution).
    #
    # Args:
    #   job_names: Array of job names (passed by reference)
    #   namespace: Kubernetes namespace (default: default)
    #   poll_interval: Polling interval in seconds (default: 5)
    #
    # Returns:
    #   0 if all jobs completed successfully, 1 if any failed
    local -n job_names_ref="$1"
    local namespace="${2:-default}"
    local poll_interval="${3:-5}"
    
    local completed_jobs=()
    local failed_jobs=()
    local pending_jobs=("${job_names_ref[@]}")
    
    log_info "Waiting for ${#pending_jobs[@]} jobs to complete..."
    
    while [[ ${#pending_jobs[@]} -gt 0 ]]; do
        sleep "$poll_interval"
        
        local newly_completed=()
        local newly_failed=()
        
        for job in "${pending_jobs[@]}"; do
            local status=$(get_job_status "$job" "$namespace")
            
            case "$status" in
                Complete)
                    newly_completed+=("$job")
                    ;;
                Failed)
                    newly_failed+=("$job")
                    ;;
                Running|Unknown)
                    # Still pending
                    ;;
            esac
        done
        
        # Update lists
        for job in "${newly_completed[@]}"; do
            completed_jobs+=("$job")
            # Remove from pending
            local new_pending=()
            for pjob in "${pending_jobs[@]}"; do
                [[ "$pjob" != "$job" ]] && new_pending+=("$pjob")
            done
            pending_jobs=("${new_pending[@]}")
        done
        
        for job in "${newly_failed[@]}"; do
            failed_jobs+=("$job")
            # Remove from pending
            local new_pending=()
            for pjob in "${pending_jobs[@]}"; do
                [[ "$pjob" != "$job" ]] && new_pending+=("$pjob")
            done
            pending_jobs=("${new_pending[@]}")
        done
        
        if [[ ${#newly_completed[@]} -gt 0 ]] || [[ ${#newly_failed[@]} -gt 0 ]]; then
            log_info "Progress: ${#completed_jobs[@]} completed, ${#failed_jobs[@]} failed, ${#pending_jobs[@]} pending"
        fi
    done
    
    if [[ ${#failed_jobs[@]} -gt 0 ]]; then
        log_error "${#failed_jobs[@]} job(s) failed: ${failed_jobs[*]}"
        return 1
    fi
    
    log_success "All ${#completed_jobs[@]} jobs completed successfully"
    return 0
}

