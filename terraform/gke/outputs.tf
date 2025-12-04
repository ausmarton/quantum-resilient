# =============================================================================
# outputs.tf - Terraform outputs for GKE cluster
# =============================================================================

output "cluster_name" {
  description = "Name of the GKE cluster"
  value       = google_container_cluster.primary.name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}

output "cluster_location" {
  description = "Location (region) of the GKE cluster"
  value       = google_container_cluster.primary.location
}

output "project_id" {
  description = "GCP Project ID"
  value       = var.project_id
}

output "bucket_name" {
  description = "GCS bucket for experiment results"
  value       = google_storage_bucket.results.name
}

output "bucket_url" {
  description = "GCS bucket URL"
  value       = google_storage_bucket.results.url
}

output "service_account_email" {
  description = "Service account email for workloads"
  value       = google_service_account.pqc_bench.email
}

output "artifact_registry_repository" {
  description = "Artifact Registry repository URL"
  value       = "${local.ar_location}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.pqc.repository_id}"
}

output "kubeconfig_command" {
  description = "Command to configure kubectl"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.primary.name} --region ${google_container_cluster.primary.location} --project ${var.project_id}"
}

output "image_push_command" {
  description = "Command to push container image"
  value       = "podman push ${local.ar_location}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.pqc.repository_id}/pqc-bench:latest"
}

# Kubeconfig auth block for programmatic access
output "kubeconfig_auth" {
  description = "Kubernetes authentication configuration"
  value = {
    host                   = "https://${google_container_cluster.primary.endpoint}"
    cluster_ca_certificate = google_container_cluster.primary.master_auth[0].cluster_ca_certificate
    token_command          = "gcloud config config-helper --format='value(credential.access_token)'"
  }
  sensitive = true
}

output "workload_identity_pool" {
  description = "Workload Identity pool for the cluster"
  value       = var.enable_workload_identity ? local.workload_identity_pool : null
}

