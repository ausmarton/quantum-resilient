#!/usr/bin/env bash
# =============================================================================
# scripts/lib/k8s-image.sh - Unified Container Image Management
#
# Provides functions for building and managing container images for both
# Minikube and GCP environments.
#
# Functions:
#   build_and_load_image_minikube() - Build and load image into Minikube
#   build_and_push_image_gcp() - Build and push image to GCR/Artifact Registry
#   ensure_image_available() - Unified interface for image availability
# =============================================================================

# Source common libraries
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$SCRIPT_DIR/scripts/lib/common.sh"

# =============================================================================
# Minikube Image Functions
# =============================================================================

build_and_load_image_minikube() {
    # Build container image and load it into Minikube.
    #
    # Args:
    #   image_name: Image name (default: pqc-bench)
    #   image_tag: Image tag (default: latest)
    #   containerfile: Path to Containerfile (default: Containerfile)
    #   skip_build: Skip build if image exists (default: false)
    #   force_build: Force rebuild even if image exists (default: false)
    #
    # Returns:
    #   0 on success, 1 on failure
    #   Outputs image name with localhost/ prefix to stdout
    local image_name="${1:-pqc-bench}"
    local image_tag="${2:-latest}"
    local containerfile="${3:-Containerfile}"
    local skip_build="${4:-false}"
    local force_build="${5:-false}"
    
    local script_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
    local full_image="${image_name}:${image_tag}"
    local local_image="localhost/${image_name}:${image_tag}"
    
    # Check if image already exists
    if podman image exists "$full_image" 2>/dev/null; then
        if [[ "$skip_build" == "true" ]]; then
            log_info "Image $full_image already exists, skipping build (--skip-build)" >&2
        elif [[ "$force_build" == "true" ]]; then
            log_info "Force rebuilding image (--force-build)..." >&2
            cd "$script_dir"
            if ! podman build -t "$full_image" -f "$containerfile" . >&2; then
                log_error "Container build failed" >&2
                return 1
            fi
            log_success "Image rebuilt: $full_image" >&2
        else
            log_success "Using existing image: $full_image" >&2
        fi
    else
        log_info "Building $full_image with Podman..." >&2
        cd "$script_dir"
        if ! podman build -t "$full_image" -f "$containerfile" . >&2; then
            log_error "Container build failed" >&2
            return 1
        fi
        log_success "Image built: $full_image" >&2
    fi
    
    # Tag with localhost/ prefix (Minikube expects this for local images)
    if ! podman image exists "$local_image" 2>/dev/null; then
        log_info "Tagging image as $local_image..." >&2
        if ! podman tag "$full_image" "$local_image" 2>/dev/null; then
            log_error "Failed to tag image" >&2
            return 1
        fi
    fi
    
    # Load into Minikube using podman save/load
    log_info "Loading image into Minikube..." >&2
    local temp_tar=$(mktemp --suffix=.tar)
    if podman save "$local_image" -o "$temp_tar" 2>/dev/null; then
        if minikube image load "$temp_tar" >/dev/null 2>&1; then
            log_success "Image loaded into Minikube" >&2
        else
            log_error "Failed to load image into Minikube" >&2
            rm -f "$temp_tar"
            return 1
        fi
        rm -f "$temp_tar"
    else
        log_error "Failed to save image to tar" >&2
        return 1
    fi
    
    # Output only the image name to stdout (for capture)
    echo "$local_image"
    return 0
}

# =============================================================================
# GCP Image Functions
# =============================================================================

build_and_push_image_gcp() {
    # Build container image and push it to GCR/Artifact Registry.
    #
    # Args:
    #   image_name: Full image name (e.g., us-central1-docker.pkg.dev/project/pqc/pqc-bench:latest)
    #   containerfile: Path to Containerfile (default: Containerfile)
    #   region: GCP region (default: us-central1)
    #   skip_build: Skip build (default: false)
    #
    # Returns:
    #   0 on success, 1 on failure
    #   Outputs image name to stdout
    local image_name="$1"
    local containerfile="${2:-Containerfile}"
    local region="${3:-us-central1}"
    local skip_build="${4:-false}"
    
    if [[ -z "$image_name" ]]; then
        log_error "build_and_push_image_gcp: image_name is required"
        return 1
    fi
    
    local script_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
    
    if [[ "$skip_build" == "true" ]]; then
        log_warn "Skipping build (--skip-build)"
    else
        # Configure Podman for Artifact Registry
        log_info "Configuring Podman authentication..."
        gcloud auth configure-docker "${region}-docker.pkg.dev" --quiet
        
        # Also configure for Podman specifically
        gcloud auth print-access-token | podman login -u oauth2accesstoken --password-stdin "${region}-docker.pkg.dev" 2>/dev/null || true
        
        # Build image
        log_info "Building container image..."
        cd "$script_dir"
        if ! podman build -t "$image_name" -f "$containerfile" . 2>&1; then
            log_error "Container build failed"
            return 1
        fi
        
        # Push image
        log_info "Pushing image to Artifact Registry..."
        if ! podman push "$image_name" 2>&1; then
            log_error "Failed to push image"
            return 1
        fi
        
        log_success "Image pushed: $image_name"
    fi
    
    echo "$image_name"
    return 0
}

# =============================================================================
# Unified Image Interface
# =============================================================================

ensure_image_available() {
    # Ensure container image is available for the specified environment.
    #
    # Args:
    #   environment: "minikube" or "gcp"
    #   image_name: Image name (for Minikube) or full image path (for GCP)
    #   image_tag: Image tag (for Minikube, default: latest)
    #   containerfile: Path to Containerfile (default: Containerfile)
    #   skip_build: Skip build if image exists (default: false)
    #   force_build: Force rebuild (default: false)
    #   region: GCP region (required for GCP)
    #
    # Returns:
    #   0 on success, 1 on failure
    #   Outputs image name to stdout
    local environment="$1"
    local image_name="$2"
    local image_tag="${3:-latest}"
    local containerfile="${4:-Containerfile}"
    local skip_build="${5:-false}"
    local force_build="${6:-false}"
    local region="${7:-us-central1}"
    
    if [[ -z "$environment" ]] || [[ -z "$image_name" ]]; then
        log_error "ensure_image_available: environment and image_name are required"
        return 1
    fi
    
    case "$environment" in
        minikube)
            build_and_load_image_minikube "$image_name" "$image_tag" "$containerfile" "$skip_build" "$force_build"
            ;;
        gcp)
            build_and_push_image_gcp "$image_name" "$containerfile" "$region" "$skip_build"
            ;;
        *)
            log_error "ensure_image_available: Unknown environment: $environment"
            return 1
            ;;
    esac
}
