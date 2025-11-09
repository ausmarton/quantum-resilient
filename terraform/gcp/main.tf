// Placeholder Terraform for GKE cluster provisioning

resource "google_container_cluster" "pqc_benchmark" {
  name     = var.cluster_name
  location = var.region

  remove_default_node_pool = true
  initial_node_count       = 1
}

resource "google_container_node_pool" "primary_nodes" {
  name       = "${var.cluster_name}-pool"
  location   = var.region
  cluster    = google_container_cluster.pqc_benchmark.name
  node_count = 1

  node_config {
    machine_type = var.node_machine_type
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }
}


