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

// Enable required APIs
resource "google_project_service" "container_api" {
  project = var.project_id
  service = "container.googleapis.com"
}

resource "google_project_service" "storage_api" {
  project = var.project_id
  service = "storage.googleapis.com"
}

// GCS bucket for results
resource "google_storage_bucket" "results_bucket" {
  name     = var.bucket_name
  project  = var.project_id
  location = var.region
  uniform_bucket_level_access = true
  force_destroy = true
}

// Minimal service account for orchestrator to write results to GCS
resource "google_service_account" "orchestrator" {
  account_id   = var.workload_sa_name
  display_name = "PQC Orchestrator Service Account"
  project      = var.project_id
}

// Grant object admin on the results bucket
resource "google_storage_bucket_iam_member" "orchestrator_bucket_writer" {
  bucket = google_storage_bucket.results_bucket.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.orchestrator.email}"
}


