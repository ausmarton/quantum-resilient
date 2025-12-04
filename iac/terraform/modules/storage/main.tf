# Storage Module
# This module can be used to create reusable storage configurations.
# Currently, storage resources are defined directly in the main gcp/ configuration.
# This module is a placeholder for future refactoring.

variable "bucket_name" {
  type        = string
  description = "Name of the GCS bucket"
}

variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "region" {
  type        = string
  description = "GCP region"
}

variable "lifecycle_age_days" {
  type        = number
  description = "Days before objects are deleted"
  default     = 30
}

# Placeholder - actual implementation in gcp/storage.tf
output "bucket_name" {
  value = var.bucket_name
}

