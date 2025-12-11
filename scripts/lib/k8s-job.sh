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
        
        # If no pods after 5 seconds, wait a bit more and check again
        if [[ ${#pod_names[@]} -eq 0 ]]; then
            sleep 10  # Wait additional time for pods to be created
            pod_names=($(kubectl get pods -l job-name="$job_name" -n "$namespace" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo ""))
            
            # If still no pods, check job status and provide diagnostics
            if [[ ${#pod_names[@]} -eq 0 ]]; then
                log_warn "No pods found for job '$job_name' after 15 seconds"
                log_info "Job status:"
                kubectl get job "$job_name" -n "$namespace" -o yaml 2>&1 | grep -A 10 "status:" || true
                log_info "Checking for scheduling issues..."
                kubectl describe job "$job_name" -n "$namespace" 2>&1 | grep -A 20 "Events:" || true
            fi
        fi
        
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

# =============================================================================
# Unified Job Submission Function
# =============================================================================

submit_k8s_job() {
    # Submit a Kubernetes job for either Minikube or GKE.
    #
    # Args:
    #   environment: "minikube" or "gcp"
    #   scenario: Path to scenario YAML file
    #   exp_id: Experiment identifier
    #   image: Container image name (full path for GCP, localhost/ prefix for Minikube)
    #   namespace: Kubernetes namespace (default: default)
    #   replicas: Number of replicas (default: 1)
    #   smoke_test: Enable smoke-test mode (default: false)
    #   seed: RNG seed (optional)
    #   duration: Override duration (optional)
    #   project: GCP project (required for GCP)
    #   bucket: GCS bucket (required for GCP)
    #   region: GCP region (required for GCP)
    #   job_name: Custom job name (optional, will be generated if not provided)
    #
    # Returns:
    #   0 on success, 1 on failure
    #   Outputs job name to stdout on success
    local environment="$1"
    local scenario="$2"
    local exp_id="$3"
    local image="$4"
    local namespace="${5:-default}"
    local replicas="${6:-1}"
    local smoke_test="${7:-false}"
    local seed="${8:-}"
    local duration="${9:-}"
    local project="${10:-}"
    local bucket="${11:-}"
    local region="${12:-}"
    local job_name="${13:-}"
    
    if [[ -z "$environment" ]] || [[ -z "$scenario" ]] || [[ -z "$exp_id" ]] || [[ -z "$image" ]]; then
        log_error "submit_k8s_job: environment, scenario, exp_id, and image are required"
        return 1
    fi
    
    if [[ "$environment" != "minikube" ]] && [[ "$environment" != "gcp" ]]; then
        log_error "submit_k8s_job: environment must be 'minikube' or 'gcp'"
        return 1
    fi
    
    if [[ "$environment" == "gcp" ]]; then
        if [[ -z "$project" ]] || [[ -z "$bucket" ]] || [[ -z "$region" ]]; then
            log_error "submit_k8s_job: project, bucket, and region are required for GCP"
            return 1
        fi
    fi
    
    # CRITICAL: Switch kubectl context based on environment
    # This ensures jobs are created in the correct cluster
    if [[ "$environment" == "minikube" ]]; then
        log_info "Switching kubectl context to Minikube..." >&2
        if ! kubectl config use-context minikube &>/dev/null; then
            log_error "Failed to switch to Minikube context. Is Minikube running?" >&2
            return 1
        fi
    elif [[ "$environment" == "gcp" ]]; then
        # For GCP, context should already be set by gcloud get-credentials
        # But verify it's a GCP context (starts with gke_)
        local current_context=$(kubectl config current-context 2>/dev/null || echo "")
        if [[ ! "$current_context" =~ ^gke_ ]]; then
            log_warn "Current kubectl context '$current_context' doesn't appear to be GCP. Attempting to configure..." >&2
            # Try to get credentials (this should have been done earlier, but try anyway)
            if ! gcloud container clusters get-credentials "$(echo "$current_context" | cut -d'_' -f3)" \
                --region "$region" \
                --project "$project" &>/dev/null; then
                log_error "Failed to configure GCP kubectl context" >&2
                return 1
            fi
        fi
    fi
    
    # Source k8s-configmap.sh for ConfigMap creation
    local script_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
    source "$script_dir/scripts/lib/k8s-configmap.sh" 2>/dev/null || {
        log_error "Failed to source k8s-configmap.sh"
        return 1
    }
    
    # Generate job name if not provided
    if [[ -z "$job_name" ]]; then
        # Source common.sh for sanitize_k8s_name function
        source "$script_dir/scripts/lib/common.sh" 2>/dev/null || {
            log_error "Failed to source common.sh"
            return 1
        }
        
        # Extract replica suffix if present (e.g., _r4, _r8)
        local replica_suffix=""
        local base_exp_id="$exp_id"
        if [[ "$exp_id" =~ _r([0-9]+)$ ]]; then
            replica_suffix="_r${BASH_REMATCH[1]}"
            base_exp_id="${exp_id%_r*}"
        else
            if [[ "$replicas" -gt 1 ]]; then
                replica_suffix="_r${replicas}"
            fi
        fi
        
        # Sanitize and truncate (Kubernetes job names max 63 chars)
        # "pqc-bench-" is 10 chars, replica suffix is max 4 chars (_r8), so we have 49 chars for base ID
        local sanitized_base=$(sanitize_k8s_name "$base_exp_id" | cut -c1-49)
        local sanitized_suffix=$(sanitize_k8s_name "$replica_suffix" | sed 's/^_//')
        job_name="pqc-bench-${sanitized_base}${sanitized_suffix}"
    fi
    
    # Determine JSONL output path (scaling mode uses different path)
    local jsonl_out="/results/raw/run.jsonl"
    if [[ "$replicas" -gt 1 ]]; then
        jsonl_out="/results/current/raw/run.jsonl"
    fi
    
    # Create scenario ConfigMap
    local scenario_cm_sanitized=$(sanitize_k8s_name "$exp_id" | cut -c1-230)
    local scenario_cm="pqc-scenario-${scenario_cm_sanitized}"
    
    log_info "Creating scenario ConfigMap..." >&2
    scenario_cm=$(create_scenario_configmap \
        "$scenario" \
        "$exp_id" \
        "$namespace" \
        "$smoke_test" \
        "$seed" \
        "$scenario_cm" \
        "$jsonl_out" \
        "$duration") || {
        log_error "Failed to create scenario ConfigMap" >&2
        return 1
    }
    
    # Create scaling ConfigMap if replicas > 1
    if [[ "$replicas" -gt 1 ]]; then
        log_info "Creating scaling ConfigMap..." >&2
        if ! kubectl create configmap pqc-scaling-config \
            --from-literal=experiment_id="$exp_id" \
            --from-literal=replica_count="$replicas" \
            --from-literal=duration_sec="${duration:-30}" \
            --dry-run=client -o yaml | kubectl apply --validate=false -f - -n "$namespace" >&2; then
            log_warn "Failed to create scaling ConfigMap (may already exist)" >&2
        fi
    fi
    
    # Create GCP config ConfigMap (only for GCP)
    local gcp_cm=""
    if [[ "$environment" == "gcp" ]]; then
        # CRITICAL: Ensure qr-worker service account exists in the namespace
        # Terraform creates it in 'quantum-resilient' namespace (or var.kubernetes_namespace),
        # but we use 'default' namespace for all test types
        log_info "Ensuring qr-worker service account exists in namespace '$namespace'..." >&2
        GCP_SA_EMAIL="qr-worker@${project}.iam.gserviceaccount.com"
        
        # First, verify the GCP service account exists (created by Terraform)
        if ! gcloud iam service-accounts describe "$GCP_SA_EMAIL" \
            --project="$project" &>/dev/null; then
            log_warn "GCP service account $GCP_SA_EMAIL does not exist" >&2
            log_warn "Terraform should have created it. Continuing anyway..." >&2
            # We'll still try to create the KSA, but Workload Identity won't work without the GSA
        fi
        
        if ! kubectl get serviceaccount qr-worker -n "$namespace" &>/dev/null; then
            log_info "Creating qr-worker Kubernetes service account in namespace '$namespace'..." >&2
            # Create ServiceAccount with Workload Identity annotation
            if ! cat <<EOF | kubectl apply -f - 2>&1; then
apiVersion: v1
kind: ServiceAccount
metadata:
  name: qr-worker
  namespace: $namespace
  annotations:
    iam.gke.io/gcp-service-account: ${GCP_SA_EMAIL}
  labels:
    app: quantum-resilient
    component: worker
EOF
                log_error "Failed to create qr-worker service account" >&2
                return 1
            fi
            log_success "qr-worker service account created in namespace '$namespace'" >&2
            
            # Ensure Workload Identity binding exists for this namespace
            # Terraform creates binding for the Terraform namespace, so we ensure it exists for the target namespace
            log_info "Ensuring Workload Identity binding for qr-worker in namespace '$namespace'..." >&2
            K8S_SA="${project}.svc.id.goog[${namespace}/qr-worker]"
            if ! gcloud iam service-accounts get-iam-policy "$GCP_SA_EMAIL" \
                --project="$project" \
                --format="json" 2>/dev/null | grep -q "$K8S_SA"; then
                log_info "Creating Workload Identity binding for $K8S_SA..." >&2
                if ! gcloud iam service-accounts add-iam-policy-binding "$GCP_SA_EMAIL" \
                    --project="$project" \
                    --role="roles/iam.workloadIdentityUser" \
                    --member="serviceAccount:${K8S_SA}" 2>&1; then
                    log_warn "Failed to create Workload Identity binding (may need permissions or GSA doesn't exist)" >&2
                    log_warn "Jobs may fail to authenticate. Ensure Terraform created the GCP service account." >&2
                else
                    log_success "Workload Identity binding created" >&2
                fi
            else
                log_info "Workload Identity binding already exists" >&2
            fi
        else
            log_info "qr-worker service account already exists in namespace '$namespace'" >&2
        fi
        
        local gcp_cm_sanitized=$(sanitize_k8s_name "$exp_id" | cut -c1-228)
        gcp_cm="pqc-bench-config-${gcp_cm_sanitized}"
        
        log_info "Creating benchmark config ConfigMap..." >&2
        # Pass container image to ConfigMap so upload sidecar can use it
        gcp_cm=$(create_gcp_config_configmap \
            "$exp_id" \
            "$bucket" \
            "$region" \
            "$project" \
            "$namespace" \
            "$smoke_test" \
            "$gcp_cm" \
            "$image") || {
            log_error "Failed to create benchmark config ConfigMap" >&2
            return 1
        }
    fi
    
    # Generate Job YAML using unified generator
    # Create temp file in project directory so container can access it
    local temp_job=$(mktemp -p "$script_dir" -t job-yaml-XXXXXX.yaml)
    local generator_args=(
        --environment "$environment"
        --job-name "$job_name"
        --namespace "$namespace"
        --image "$image"
        --scenario-configmap "$scenario_cm"
        --experiment-id "$exp_id"
        --replicas "$replicas"
        --output "$temp_job"
    )
    
    if [[ "$environment" == "gcp" ]] && [[ -n "$gcp_cm" ]]; then
        generator_args+=(--gcp-config-configmap "$gcp_cm")
    fi
    
    log_info "Generating Job YAML..." >&2
    # Use container wrapper for consistent Python environment
    if ! "$script_dir/scripts/lib/run-python-container.sh" \
        "$script_dir/scripts/lib/k8s-job-generator.py" \
        "${generator_args[@]}" >&2; then
        log_error "Failed to generate Job YAML" >&2
        rm -f "$temp_job"
        return 1
    fi
    
    # Apply Job YAML
    log_info "Submitting Job '$job_name' to Kubernetes..." >&2
    local job_output=$(kubectl apply --validate=false -f "$temp_job" -n "$namespace" 2>&1)
    local job_exit_code=$?
    rm -f "$temp_job"
    
    if [[ $job_exit_code -ne 0 ]]; then
        log_error "Failed to submit job" >&2
        log_error "Experiment ID: $exp_id" >&2
        log_error "Job name: $job_name" >&2
        log_error "Replicas: $replicas" >&2
        log_error "kubectl output:" >&2
        echo "$job_output" >&2
        return 1
    fi
    
    log_success "Job '$job_name' submitted successfully" >&2
    echo "$job_name"
    return 0
}

# =============================================================================
# Service Account Management (GCP)
# =============================================================================

ensure_gcp_service_account() {
    # Ensure GCP service account exists with Workload Identity binding.
    #
    # Args:
    #   project: GCP project ID
    #   namespace: Kubernetes namespace (default: default)
    #   sa_email: Service account email (optional, will be derived if not provided)
    #
    # Returns:
    #   0 on success, 1 on failure
    local project="$1"
    local namespace="${2:-default}"
    local sa_email="${3:-}"
    
    if [[ -z "$project" ]]; then
        log_error "ensure_gcp_service_account: project is required"
        return 1
    fi
    
    # Get service account email from Terraform if not provided
    if [[ -z "$sa_email" ]]; then
        local script_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
        local terraform_dir="$script_dir/terraform/gke"
        if [[ -d "$terraform_dir" ]] && [[ -d "$terraform_dir/.terraform" ]]; then
            sa_email=$(cd "$terraform_dir" && terraform output -raw service_account_email 2>/dev/null || echo "")
        fi
        if [[ -z "$sa_email" ]]; then
            # Default service account email format
            sa_email="qr-worker@${project}.iam.gserviceaccount.com"
        fi
    fi
    
    # Create/update Kubernetes ServiceAccount with Workload Identity annotation
    if ! kubectl get serviceaccount pqc-bench-sa -n "$namespace" &>/dev/null; then
        log_info "Creating Kubernetes ServiceAccount 'pqc-bench-sa' in namespace '$namespace'..."
        if ! cat <<EOF | kubectl apply -f - 2>&1; then
apiVersion: v1
kind: ServiceAccount
metadata:
  name: pqc-bench-sa
  namespace: $namespace
  annotations:
    iam.gke.io/gcp-service-account: "$sa_email"
EOF
            log_error "Failed to create ServiceAccount"
            return 1
        fi
        log_success "ServiceAccount created successfully"
    else
        # Update annotation if service account exists but annotation is missing/wrong
        local current_annotation=$(kubectl get serviceaccount pqc-bench-sa -n "$namespace" -o jsonpath='{.metadata.annotations.iam\.gke\.io/gcp-service-account}' 2>/dev/null || echo "")
        if [[ "$current_annotation" != "$sa_email" ]]; then
            log_info "Updating ServiceAccount annotation to point to $sa_email..."
            if ! kubectl annotate serviceaccount pqc-bench-sa -n "$namespace" \
                "iam.gke.io/gcp-service-account=$sa_email" --overwrite 2>&1; then
                log_error "Failed to update ServiceAccount annotation"
                return 1
            fi
        fi
    fi
    
    # Create IAM binding for Workload Identity if it doesn't exist (for non-default namespaces)
    # For default namespace, Terraform should have already created the binding
    if [[ "$namespace" != "default" ]]; then
        local k8s_sa="${project}.svc.id.goog[${namespace}/pqc-bench-sa]"
        if ! gcloud iam service-accounts get-iam-policy "$sa_email" \
            --project="$project" \
            --format="json" 2>/dev/null | grep -q "$k8s_sa"; then
            log_info "Creating Workload Identity IAM binding for $k8s_sa..."
            if ! gcloud iam service-accounts add-iam-policy-binding "$sa_email" \
                --project="$project" \
                --role="roles/iam.workloadIdentityUser" \
                --member="serviceAccount:${k8s_sa}" 2>&1; then
                log_warn "Failed to create Workload Identity binding"
                log_warn "This may cause GCS upload failures"
                log_warn "You may need to create it manually:"
                echo "  gcloud iam service-accounts add-iam-policy-binding $sa_email \\" >&2
                echo "    --project=$project \\" >&2
                echo "    --role=roles/iam.workloadIdentityUser \\" >&2
                echo "    --member=serviceAccount:${k8s_sa}" >&2
                # Don't exit - the binding might already exist from Terraform for default namespace
            else
                log_success "Workload Identity binding created successfully"
            fi
        fi
    fi
    
    return 0
}

