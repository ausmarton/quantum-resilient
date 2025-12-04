# -----------------------------------------------------------------------------
# Required Variables
# -----------------------------------------------------------------------------

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "bucket_name" {
  description = "Name of the GCS bucket for experiment results"
  type        = string
}

# -----------------------------------------------------------------------------
# GKE Configuration
# -----------------------------------------------------------------------------

variable "gke_name" {
  description = "Name of the GKE cluster"
  type        = string
  default     = "quantum-resilient-cluster"
}

variable "gke_node_machine_type" {
  description = "Machine type for GKE nodes"
  type        = string
  default     = "n2-standard-4"
}

variable "gke_node_min_count" {
  description = "Minimum number of nodes per zone"
  type        = number
  default     = 1
}

variable "gke_node_max_count" {
  description = "Maximum number of nodes per zone"
  type        = number
  default     = 7
}

variable "gke_initial_node_count" {
  description = "Initial number of nodes per zone"
  type        = number
  default     = 1
}

variable "gke_disk_size_gb" {
  description = "Disk size for GKE nodes in GB"
  type        = number
  default     = 100
}

variable "gke_disk_type" {
  description = "Disk type for GKE nodes"
  type        = string
  default     = "pd-standard"
}

# -----------------------------------------------------------------------------
# Networking Configuration
# -----------------------------------------------------------------------------

variable "vpc_name" {
  description = "Name of the VPC network"
  type        = string
  default     = "qr-vpc"
}

variable "subnet_cidr" {
  description = "CIDR range for the primary subnet"
  type        = string
  default     = "10.0.0.0/20"
}

variable "pods_cidr" {
  description = "CIDR range for GKE pods"
  type        = string
  default     = "10.16.0.0/14"
}

variable "services_cidr" {
  description = "CIDR range for GKE services"
  type        = string
  default     = "10.20.0.0/20"
}

# -----------------------------------------------------------------------------
# Application Configuration
# -----------------------------------------------------------------------------

variable "orchestrator_image" {
  description = "Container image for the orchestrator"
  type        = string
  default     = "gcr.io/PROJECT_ID/qr-orchestrator:latest"
}

variable "worker_image" {
  description = "Container image for workers"
  type        = string
  default     = "gcr.io/PROJECT_ID/pqc-bench:latest"
}

variable "kubernetes_namespace" {
  description = "Kubernetes namespace for quantum-resilient workloads"
  type        = string
  default     = "quantum-resilient"
}

# -----------------------------------------------------------------------------
# Prometheus/Monitoring Configuration
# -----------------------------------------------------------------------------

variable "enable_prometheus" {
  description = "Enable Prometheus Operator deployment"
  type        = bool
  default     = true
}

variable "grafana_admin_password" {
  description = "Admin password for Grafana"
  type        = string
  sensitive   = true
  default     = "admin"
}

# -----------------------------------------------------------------------------
# Optional Features
# -----------------------------------------------------------------------------

variable "enable_bigquery" {
  description = "Enable BigQuery export support"
  type        = bool
  default     = false
}

variable "bigquery_dataset_id" {
  description = "BigQuery dataset ID for experiment results"
  type        = string
  default     = "quantum_resilient_results"
}

# -----------------------------------------------------------------------------
# Labels and Tags
# -----------------------------------------------------------------------------

variable "labels" {
  description = "Common labels to apply to all resources"
  type        = map(string)
  default = {
    application = "quantum-resilient"
    managed-by  = "terraform"
  }
}

