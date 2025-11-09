output "cluster_name" {
  description = "Name of the GKE cluster"
  value       = google_container_cluster.pqc_benchmark.name
}

output "cluster_location" {
  description = "Location of the GKE cluster"
  value       = google_container_cluster.pqc_benchmark.location
}


