output "cluster_name" {
  description = "Name of the GKE cluster"
  value       = google_container_cluster.pqc_benchmark.name
}

output "cluster_location" {
  description = "Location of the GKE cluster"
  value       = google_container_cluster.pqc_benchmark.location
}

output "results_bucket" {
  description = "GCS bucket for benchmark results"
  value       = google_storage_bucket.results_bucket.name
}

output "orchestrator_service_account_email" {
  description = "Service account email for orchestrator"
  value       = google_service_account.orchestrator.email
}


