# -----------------------------------------------------------------------------
# GKE Cluster
# -----------------------------------------------------------------------------

resource "google_container_cluster" "primary" {
  name     = var.gke_name
  location = var.region
  project  = var.project_id

  # We manage node pools separately
  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name

  # IP allocation policy for VPC-native cluster
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Workload Identity for secure pod-to-GCP service access
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  # Master authorized networks (allow all for simplicity, restrict in production)
  master_authorized_networks_config {
    cidr_blocks {
      cidr_block   = "0.0.0.0/0"
      display_name = "All"
    }
  }

  # Addons configuration
  addons_config {
    http_load_balancing {
      disabled = false
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
    gcs_fuse_csi_driver_config {
      enabled = true
    }
  }

  # Logging and monitoring
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }

  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]
    managed_prometheus {
      enabled = true
    }
  }

  # Maintenance window
  maintenance_policy {
    daily_maintenance_window {
      start_time = "03:00"
    }
  }

  # Release channel
  release_channel {
    channel = "REGULAR"
  }

  resource_labels = var.labels

  # Deletion protection (disable for testing)
  deletion_protection = false

  depends_on = [
    google_compute_network.vpc,
    google_compute_subnetwork.subnet,
  ]
}

# -----------------------------------------------------------------------------
# Primary Node Pool
# -----------------------------------------------------------------------------

resource "google_container_node_pool" "primary" {
  name       = "${var.gke_name}-primary-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  project    = var.project_id

  initial_node_count = var.gke_initial_node_count

  # Autoscaling configuration
  autoscaling {
    min_node_count = var.gke_node_min_count
    max_node_count = var.gke_node_max_count
  }

  # Node management
  management {
    auto_repair  = true
    auto_upgrade = true
  }

  # Node configuration
  node_config {
    machine_type = var.gke_node_machine_type
    disk_size_gb = var.gke_disk_size_gb
    disk_type    = var.gke_disk_type

    # OAuth scopes
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    # Workload Identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    # GCS Fuse capability
    gcfs_config {
      enabled = true
    }

    # Labels
    labels = merge(var.labels, {
      "node-pool" = "primary"
    })

    # Metadata
    metadata = {
      disable-legacy-endpoints = "true"
    }

    # Shielded instance config
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }

  # Upgrade settings
  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
}

# -----------------------------------------------------------------------------
# Optional: Dedicated Worker Node Pool (for high-throughput experiments)
# -----------------------------------------------------------------------------

resource "google_container_node_pool" "workers" {
  count    = var.gke_node_max_count > 10 ? 1 : 0
  name     = "${var.gke_name}-worker-pool"
  location = var.region
  cluster  = google_container_cluster.primary.name
  project  = var.project_id

  initial_node_count = 0

  autoscaling {
    min_node_count = 0
    max_node_count = var.gke_node_max_count
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type = "n2-highcpu-8"
    disk_size_gb = var.gke_disk_size_gb
    disk_type    = "pd-ssd"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    labels = merge(var.labels, {
      "node-pool" = "workers"
      "workload"  = "benchmark"
    })

    taint {
      key    = "workload"
      value  = "benchmark"
      effect = "NO_SCHEDULE"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }
  }
}

