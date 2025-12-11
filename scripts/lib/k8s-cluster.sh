#!/usr/bin/env bash
# =============================================================================
# scripts/lib/k8s-cluster.sh - Kubernetes Cluster Management
#
# Provides functions for managing Kubernetes clusters for both Minikube and GKE.
#
# Functions:
#   ensure_minikube_cluster() - Ensure Minikube cluster is running
#   verify_kubectl_connectivity() - Verify kubectl can connect to cluster
# =============================================================================

# Source common libraries
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$SCRIPT_DIR/scripts/lib/common.sh"

# =============================================================================
# Minikube Cluster Functions
# =============================================================================

ensure_minikube_cluster() {
    # Ensure Minikube cluster is running and kubectl is configured.
    #
    # Args:
    #   None
    #
    # Returns:
    #   0 on success, 1 on failure
    local current_context=$(kubectl config current-context 2>/dev/null || echo "")
    
    if [[ "$current_context" != "minikube" ]]; then
        log_warn "kubectl context is not set to 'minikube' (current: ${current_context:-none})"
        log_info "Attempting to switch to minikube context..."
        if ! kubectl config use-context minikube >/dev/null 2>&1; then
            log_error "Failed to switch to minikube context"
            log_info "Please ensure Minikube is running:"
            echo "  minikube start"
            echo ""
            log_info "Or switch context manually:"
            echo "  kubectl config use-context minikube"
            return 1
        fi
    fi
    
    # Verify cluster is accessible
    if ! kubectl cluster-info &>/dev/null; then
        log_error "Cannot connect to Minikube cluster"
        log_info "Please ensure Minikube is running:"
        echo "  minikube start"
        return 1
    fi
    
    log_success "Minikube cluster is running (context: $current_context)"
    return 0
}

# =============================================================================
# Generic Cluster Functions
# =============================================================================

verify_kubectl_connectivity() {
    # Verify kubectl can connect to the current cluster.
    #
    # Args:
    #   None
    #
    # Returns:
    #   0 on success, 1 on failure
    if ! kubectl cluster-info &>/dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        log_info "Please ensure:"
        echo "  1. Cluster is running"
        echo "  2. kubectl is configured correctly"
        echo "  3. You have appropriate permissions"
        return 1
    fi
    
    local current_context=$(kubectl config current-context 2>/dev/null || echo "unknown")
    log_success "kubectl connectivity verified (context: $current_context)"
    return 0
}
