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

  # Deletion protection (disabled for easy teardown)
  deletion_protection = false
}

# -----------------------------------------------------------------------------
# Node Pool
# -----------------------------------------------------------------------------

resource "google_container_node_pool" "primary" {
  name       = "pqc-bench-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  project    = var.project_id
  node_count = var.smoke_test ? 1 : var.node_count

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

resource "kubernetes_service_account_v1" "pqc_bench" {
  count = var.enable_workload_identity ? 1 : 0

  metadata {
    name      = "pqc-bench-sa"
    namespace = "default"
    annotations = {
      "iam.gke.io/gcp-service-account" = google_service_account.pqc_bench.email
    }
  }

  depends_on = [google_container_node_pool.primary]
}

# Kubernetes provider configuration
provider "kubernetes" {
  host                   = "https://${google_container_cluster.primary.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth[0].cluster_ca_certificate)
}

