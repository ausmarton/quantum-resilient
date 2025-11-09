variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for GKE"
  type        = string
  default     = "us-central1"
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
  default     = "pqc-benchmark"
}

variable "node_machine_type" {
  description = "Machine type for GKE nodes"
  type        = string
  default     = "e2-standard-2"
}

variable "bucket_name" {
	description = "GCS bucket name for results"
	type        = string
}

variable "workload_sa_name" {
	description = "Service account name for orchestrator workload"
	type        = string
	default     = "pqc-orchestrator-sa"
}


