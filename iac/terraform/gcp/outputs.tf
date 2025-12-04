# -----------------------------------------------------------------------------
# GKE Cluster Outputs
# -----------------------------------------------------------------------------

output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.primary.name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "GKE cluster CA certificate"
  value       = google_container_cluster.primary.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "cluster_location" {
  description = "GKE cluster location"
  value       = google_container_cluster.primary.location
}

# -----------------------------------------------------------------------------
# Network Outputs
# -----------------------------------------------------------------------------

output "vpc_name" {
  description = "VPC network name"
  value       = google_compute_network.vpc.name
}

output "vpc_self_link" {
  description = "VPC network self link"
  value       = google_compute_network.vpc.self_link
}

output "subnet_name" {
  description = "Subnet name"
  value       = google_compute_subnetwork.subnet.name
}

# -----------------------------------------------------------------------------
# Storage Outputs
# -----------------------------------------------------------------------------

output "results_bucket_name" {
  description = "GCS bucket name for experiment results"
  value       = google_storage_bucket.results.name
}

output "results_bucket_url" {
  description = "GCS bucket URL"
  value       = google_storage_bucket.results.url
}

output "results_bucket_uri" {
  description = "GCS bucket URI (gs://...)"
  value       = "gs://${google_storage_bucket.results.name}"
}

# -----------------------------------------------------------------------------
# Service Account Outputs
# -----------------------------------------------------------------------------

output "orchestrator_sa_email" {
  description = "Orchestrator service account email"
  value       = google_service_account.orchestrator.email
}

output "worker_sa_email" {
  description = "Worker service account email"
  value       = google_service_account.worker.email
}

# -----------------------------------------------------------------------------
# Kubernetes Outputs
# -----------------------------------------------------------------------------

output "kubernetes_namespace" {
  description = "Kubernetes namespace for workloads"
  value       = kubernetes_namespace.qr.metadata[0].name
}

output "orchestrator_service_name" {
  description = "Orchestrator Kubernetes service name"
  value       = "qr-orchestrator"
}

# -----------------------------------------------------------------------------
# Access Commands
# -----------------------------------------------------------------------------

output "gke_connect_command" {
  description = "Command to configure kubectl"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.primary.name} --region ${var.region} --project ${var.project_id}"
}

output "orchestrator_port_forward_command" {
  description = "Command to port-forward orchestrator"
  value       = "kubectl port-forward -n ${var.kubernetes_namespace} svc/qr-orchestrator 7070:7070"
}

output "grafana_port_forward_command" {
  description = "Command to port-forward Grafana"
  value       = var.enable_prometheus ? "kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80" : "Prometheus not enabled"
}

