# =============================================================================
# variables.tf - Input variables for GKE cluster
# =============================================================================

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region for the cluster"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for the node pool (optional, defaults to region-a)"
  type        = string
  default     = ""
}

variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
  default     = "pqc-bench-gke"
}

variable "node_count" {
  description = "Number of nodes in the default node pool"
  type        = number
  default     = 1
}

variable "machine_type" {
  description = "Machine type for GKE nodes"
  type        = string
  default     = "n2-standard-2"
}

variable "bucket_name" {
  description = "GCS bucket name for experiment artifacts"
  type        = string
}

variable "artifact_registry_location" {
  description = "Location for Artifact Registry (defaults to region)"
  type        = string
  default     = ""
}

variable "artifact_registry_repository" {
  description = "Artifact Registry repository name"
  type        = string
  default     = "pqc"
}

variable "enable_workload_identity" {
  description = "Enable Workload Identity for the cluster"
  type        = bool
  default     = true
}

variable "disk_size_gb" {
  description = "Boot disk size in GB for nodes"
  type        = number
  default     = 50
}

variable "kubernetes_version" {
  description = "Kubernetes version (use 'latest' for most recent stable)"
  type        = string
  default     = "latest"
}

