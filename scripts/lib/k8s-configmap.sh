#!/usr/bin/env bash
# =============================================================================
# scripts/lib/k8s-configmap.sh - Unified Kubernetes ConfigMap creation
#
# Provides functions for creating Kubernetes ConfigMaps consistently across
# Minikube and GCP environments.
#
# Usage:
#   source "$SCRIPT_DIR/scripts/lib/k8s-configmap.sh"
#   create_scenario_configmap "$SCENARIO" "$EXP_ID" "$NAMESPACE" "$SMOKE_TEST" "$SEED"
# =============================================================================

# -----------------------------------------------------------------------------
# ConfigMap Creation Functions
# -----------------------------------------------------------------------------

create_scenario_configmap() {
    local scenario_path="$1"
    local exp_id="$2"
    local namespace="${3:-default}"
    local smoke_test="${4:-false}"
    local seed="${5:-}"
    local cm_name="${6:-}"
    local jsonl_out="${7:-/results/raw/run.jsonl}"
    local duration="${8:-}"
    
    if [[ -z "$scenario_path" ]] || [[ -z "$exp_id" ]]; then
        log_error "Scenario path and experiment ID required for ConfigMap creation"
        return 1
    fi
    
    # Generate ConfigMap name if not provided
    if [[ -z "$cm_name" ]]; then
        # Use sanitized experiment ID
        local sanitized=$(sanitize_k8s_name "$exp_id")
        # Truncate to 230 chars (leaving room for "pqc-scenario-" prefix)
        cm_name="pqc-scenario-$(echo "$sanitized" | cut -c1-230)"
    fi
    
    # Create temporary patched scenario
    local temp_scenario=$(mktemp)
    trap "rm -f '$temp_scenario'" EXIT
    
    # Patch scenario using unified Python script
    local script_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
    local patch_args=(
        --input "$scenario_path"
        --output "$temp_scenario"
        --jsonl-out "$jsonl_out"
    )
    
    if [[ "$smoke_test" == "true" ]]; then
        patch_args+=(--smoke-test)
    fi
    
    if [[ -n "$seed" ]]; then
        patch_args+=(--seed "$seed")
    fi
    
    if [[ -n "$duration" ]]; then
        patch_args+=(--duration "$duration")
    fi
    
    # Use container wrapper for consistent Python environment
    if ! "$script_dir/scripts/lib/run-python-container.sh" \
        "$script_dir/scripts/lib/scenario-patch.py" \
        "${patch_args[@]}" 2>&1; then
        log_error "Failed to patch scenario YAML"
        rm -f "$temp_scenario"
        return 1
    fi
    
    # Create ConfigMap
    local cm_output=$(kubectl create configmap "$cm_name" \
        --from-file=scenario.yaml="$temp_scenario" \
        --namespace="$namespace" \
        --dry-run=client -o yaml | kubectl apply -f - 2>&1)
    local cm_exit_code=$?
    
    rm -f "$temp_scenario"
    
    if [[ $cm_exit_code -ne 0 ]]; then
        log_error "Failed to create scenario ConfigMap '$cm_name'"
        log_error "Original experiment ID: '$exp_id'"
        log_error "Sanitized ConfigMap name: '$cm_name'"
        log_error "ConfigMap creation output:"
        echo "$cm_output" >&2
        return 1
    fi
    
    log_success "ConfigMap created: $cm_name" >&2  # Send log to stderr, not stdout
    echo "$cm_name"  # Only output the name to stdout
    return 0
}

create_gcp_config_configmap() {
    local exp_id="$1"
    local bucket="$2"
    local region="$3"
    local project="$4"
    local namespace="${5:-default}"
    local smoke_test="${6:-false}"
    local cm_name="${7:-}"
    
    if [[ -z "$exp_id" ]] || [[ -z "$bucket" ]] || [[ -z "$region" ]] || [[ -z "$project" ]]; then
        log_error "Experiment ID, bucket, region, and project required for GCP ConfigMap"
        return 1
    fi
    
    # Generate ConfigMap name if not provided
    if [[ -z "$cm_name" ]]; then
        local sanitized=$(sanitize_k8s_name "$exp_id")
        # Truncate to 228 chars (leaving room for "pqc-gcp-config-" prefix)
        cm_name="pqc-gcp-config-$(echo "$sanitized" | cut -c1-228)"
    fi
    
    # Create ConfigMap
    local cm_output=$(kubectl create configmap "$cm_name" \
        --from-literal=bucket_name="$bucket" \
        --from-literal=experiment_id="$exp_id" \
        --from-literal=region="$region" \
        --from-literal=project_id="$project" \
        --from-literal=smoke_test="$([ "$smoke_test" == "true" ] && echo "true" || echo "false")" \
        --namespace="$namespace" \
        --dry-run=client -o yaml | kubectl apply -f - 2>&1)
    local cm_exit_code=$?
    
    if [[ $cm_exit_code -ne 0 ]]; then
        log_error "Failed to create GCP config ConfigMap '$cm_name'"
        log_error "Original experiment ID: '$exp_id'"
        log_error "Sanitized ConfigMap name: '$cm_name'"
        log_error "ConfigMap creation output:"
        echo "$cm_output" >&2
        return 1
    fi
    
    log_success "GCP ConfigMap created: $cm_name" >&2  # Send log to stderr, not stdout
    echo "$cm_name"  # Only output the name to stdout
    return 0
}

