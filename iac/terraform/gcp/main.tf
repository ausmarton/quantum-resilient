# -----------------------------------------------------------------------------
# Quantum Resilient - GCP Infrastructure
# -----------------------------------------------------------------------------
#
# This Terraform configuration deploys the complete Quantum Resilient
# benchmark framework on Google Cloud Platform.
#
# Components:
# - GKE regional cluster with autoscaling
# - VPC with private subnets and Cloud NAT
# - GCS bucket for experiment results
# - Service accounts with Workload Identity
# - Prometheus/Grafana monitoring stack
# - Orchestrator and worker Helm deployments
#
# Usage:
#   terraform init
#   terraform plan -var="project_id=your-project" -var="bucket_name=your-bucket"
#   terraform apply
#
# -----------------------------------------------------------------------------

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "container.googleapis.com",
    "compute.googleapis.com",
    "iam.googleapis.com",
    "storage.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "bigquery.googleapis.com",
  ])

  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

# -----------------------------------------------------------------------------
# Locals
# -----------------------------------------------------------------------------

locals {
  cluster_name = var.gke_name
  
  common_labels = merge(var.labels, {
    environment = terraform.workspace
    region      = var.region
  })
}

# -----------------------------------------------------------------------------
# Terraform State Locking (optional - uncomment if using GCS backend)
# -----------------------------------------------------------------------------

# resource "google_storage_bucket" "terraform_state" {
#   name                        = "${var.project_id}-terraform-state"
#   location                    = var.region
#   uniform_bucket_level_access = true
#   
#   versioning {
#     enabled = true
#   }
#   
#   lifecycle_rule {
#     condition {
#       num_newer_versions = 5
#     }
#     action {
#       type = "Delete"
#     }
#   }
# }

