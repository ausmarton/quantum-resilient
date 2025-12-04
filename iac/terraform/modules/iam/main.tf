# IAM Module
# This module can be used to create reusable IAM configurations.
# Currently, IAM resources are defined directly in the main gcp/ configuration.
# This module is a placeholder for future refactoring.

variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "service_account_prefix" {
  type        = string
  description = "Prefix for service account names"
  default     = "qr"
}

# Placeholder - actual implementation in gcp/iam.tf and gcp/service_accounts.tf
output "service_account_prefix" {
  value = var.service_account_prefix
}

