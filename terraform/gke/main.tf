# =============================================================================
# main.tf - Minimal GKE cluster for PQC benchmarking
#
# Creates:
# - Regional GKE cluster (non-Autopilot for raw benchmarking)
# - Single node pool with configurable machine type
# - Workload Identity enabled
# - Service account with Artifact Registry and GCS permissions
# =============================================================================

terraform {
  required_version = ">= 1.6"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = ">= 5.0"
    }
  }
}

# -----------------------------------------------------------------------------
# Providers
# -----------------------------------------------------------------------------

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# -----------------------------------------------------------------------------
# Local values
# -----------------------------------------------------------------------------

locals {
  zone                = var.zone != "" ? var.zone : "${var.region}-a"
  ar_location         = var.artifact_registry_location != "" ? var.artifact_registry_location : var.region
  workload_identity_pool = "${var.project_id}.svc.id.goog"
}

# -----------------------------------------------------------------------------
# Data sources
# -----------------------------------------------------------------------------

data "google_project" "current" {
  project_id = var.project_id
}

data "google_client_config" "default" {}

# -----------------------------------------------------------------------------
# Service Account for GKE workloads
# -----------------------------------------------------------------------------

resource "google_service_account" "pqc_bench" {
  account_id   = "pqc-bench-worker"
  display_name = "PQC Benchmark Worker Service Account"
  project      = var.project_id

  # Allow importing existing service accounts
  lifecycle {
    create_before_destroy = false
  }
}

# Grant Artifact Registry Reader permission
resource "google_project_iam_member" "ar_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.pqc_bench.email}"
}

# Grant GCS Object Admin for results bucket
resource "google_storage_bucket_iam_member" "gcs_admin" {
  bucket = google_storage_bucket.results.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.pqc_bench.email}"
}

# Allow Kubernetes SA to use this GCP SA (Workload Identity)
# Note: This requires the cluster to be created first (workload identity pool must exist)
resource "google_service_account_iam_member" "workload_identity" {
  count              = var.enable_workload_identity ? 1 : 0
  service_account_id = google_service_account.pqc_bench.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[default/pqc-bench-sa]"
  
  # Ensure cluster is created first so workload identity pool exists
  depends_on = [google_container_cluster.primary]
}

# -----------------------------------------------------------------------------
# Artifact Registry Repository
# -----------------------------------------------------------------------------

resource "google_artifact_registry_repository" "pqc" {
  location      = local.ar_location
  repository_id = var.artifact_registry_repository
  description   = "Container images for PQC benchmark framework"
  format        = "DOCKER"
  project       = var.project_id

  # Allow importing existing repositories
  lifecycle {
    create_before_destroy = false
  }
}

# -----------------------------------------------------------------------------
# GKE Cluster
# -----------------------------------------------------------------------------

resource "google_container_cluster" "primary" {
  provider = google-beta

  name     = var.cluster_name
  location = var.region
  project  = var.project_id

  # We'll manage the default node pool separately (Autopilot is disabled by default)
  remove_default_node_pool = true
  initial_node_count       = 1

  # Networking
  network    = "default"
  subnetwork = "default"

  # Workload Identity (must be enabled for Workload Identity to work)
  workload_identity_config {
    workload_pool = var.enable_workload_identity ? "${var.project_id}.svc.id.goog" : null
  }

  # Release channel for stable K8s versions
  release_channel {
    channel = "REGULAR"
  }

  # Monitoring and logging
  # Always enable full logging so logs are visible in Cloud Logging
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }

  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]
    managed_prometheus {
      enabled = false  # Keep it simple for benchmarking
    }
  }

  # Addons
  addons_config {
    http_load_balancing {
      disabled = true  # Not needed for benchmarking
    }
    horizontal_pod_autoscaling {
      disabled = true  # Manual control for reproducibility
    }
    gcs_fuse_csi_driver_config {
      enabled = true  # Enable GCS Fuse for results
    }
  }

  # Maintenance window (avoid during business hours)
  maintenance_policy {
    daily_maintenance_window {
      start_time = "03:00"
    }
  }

  # Binary Authorization
  binary_authorization {
    evaluation_mode = "DISABLED"
  }

  # Deletion protection (disabled for easy teardown, especially in ephemeral mode)
  deletion_protection = false

  # Lifecycle: allow destruction in ephemeral mode
  lifecycle {
    prevent_destroy = false
  }
}

# -----------------------------------------------------------------------------
# Node Pool
# -----------------------------------------------------------------------------

resource "google_container_node_pool" "primary" {
  name       = "pqc-bench-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  project    = var.project_id
  # In ephemeral mode or smoke test, use minimal node count
  node_count = (var.ephemeral || var.smoke_test) ? 1 : var.node_count

  # Lifecycle: allow destruction in ephemeral mode
  lifecycle {
    prevent_destroy = false
    create_before_destroy = false
  }

  # Node configuration
  # CRITICAL: Hardware MUST remain identical between smoke-test and full runs
  # Only node_count (horizontal scaling) may change, NOT machine_type, disk_type, etc.
  node_config {
    # Machine type MUST stay the same - no conditional changes allowed
    machine_type = var.machine_type
    # Disk size and type MUST stay the same
    disk_size_gb = var.disk_size_gb
    disk_type    = "pd-standard"

    # Service account for nodes
    service_account = google_service_account.pqc_bench.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    # Workload Identity (same for both modes)
    workload_metadata_config {
      mode = var.enable_workload_identity ? "GKE_METADATA" : "MODE_UNSPECIFIED"
    }

    # Labels
    labels = {
      app         = "pqc-bench"
      environment = "benchmark"
    }

    # Metadata
    metadata = {
      disable-legacy-endpoints = "true"
    }

    # Shielded instance config (same for both modes)
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }

  # Management
  management {
    auto_repair  = true
    auto_upgrade = true  # Required when using release_channel
  }

  # Upgrade settings
  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
}

# -----------------------------------------------------------------------------
# Kubernetes Service Account (for Workload Identity)
# -----------------------------------------------------------------------------
# NOTE: The Kubernetes service account is created by deploy_gcp.sh via kubectl
# after the cluster is ready. We don't create it via Terraform to avoid
# provider connection issues during plan phase when the cluster doesn't exist yet.
#
# The service account is created in deploy_gcp.sh around line 768 with:
#   kubectl apply -f - <<EOF
#   apiVersion: v1
#   kind: ServiceAccount
#   metadata:
#     name: pqc-bench-sa
#     namespace: default
#     annotations:
#       iam.gke.io/gcp-service-account: "$SA_EMAIL"
#   EOF
#
# This approach is more reliable because:
# 1. It doesn't require the Kubernetes provider to connect during Terraform plan
# 2. It happens after kubectl is configured (cluster is ready)
# 3. It's idempotent (kubectl apply handles existing resources)

