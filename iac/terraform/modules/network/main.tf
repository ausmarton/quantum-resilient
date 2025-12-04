# Network Module
# This module can be used to create reusable VPC configurations.
# Currently, network resources are defined directly in the main gcp/ configuration.
# This module is a placeholder for future refactoring.

variable "vpc_name" {
  type        = string
  description = "Name of the VPC network"
}

variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "region" {
  type        = string
  description = "GCP region"
}

# Placeholder - actual implementation in gcp/vpc.tf
output "vpc_name" {
  value = var.vpc_name
}

