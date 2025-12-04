# -----------------------------------------------------------------------------
# Orchestrator IAM Roles
# -----------------------------------------------------------------------------

# Storage Object Admin - for uploading/managing experiment results
resource "google_project_iam_member" "orchestrator_storage_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.orchestrator.email}"
}

# Container Admin - for managing GKE workloads (Jobs, Pods, etc.)
resource "google_project_iam_member" "orchestrator_container_admin" {
  project = var.project_id
  role    = "roles/container.admin"
  member  = "serviceAccount:${google_service_account.orchestrator.email}"
}

# Compute Viewer - for viewing node information
resource "google_project_iam_member" "orchestrator_compute_viewer" {
  project = var.project_id
  role    = "roles/compute.viewer"
  member  = "serviceAccount:${google_service_account.orchestrator.email}"
}

# Service Account User - for impersonating other service accounts if needed
resource "google_project_iam_member" "orchestrator_sa_user" {
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${google_service_account.orchestrator.email}"
}

# Logging Writer - for application logs
resource "google_project_iam_member" "orchestrator_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.orchestrator.email}"
}

# Monitoring Writer - for custom metrics
resource "google_project_iam_member" "orchestrator_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.orchestrator.email}"
}

# -----------------------------------------------------------------------------
# Worker IAM Roles
# -----------------------------------------------------------------------------

# Storage Object Creator - for uploading experiment results
resource "google_project_iam_member" "worker_storage_creator" {
  project = var.project_id
  role    = "roles/storage.objectCreator"
  member  = "serviceAccount:${google_service_account.worker.email}"
}

# Logging Writer - for application logs
resource "google_project_iam_member" "worker_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.worker.email}"
}

# Monitoring Writer - for Prometheus metrics
resource "google_project_iam_member" "worker_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.worker.email}"
}

# -----------------------------------------------------------------------------
# BigQuery IAM (conditional)
# -----------------------------------------------------------------------------

# BigQuery Data Editor - for orchestrator to export to BigQuery
resource "google_project_iam_member" "orchestrator_bigquery" {
  count   = var.enable_bigquery ? 1 : 0
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.orchestrator.email}"
}

# BigQuery Job User - for running BigQuery load jobs
resource "google_project_iam_member" "orchestrator_bigquery_jobs" {
  count   = var.enable_bigquery ? 1 : 0
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.orchestrator.email}"
}

