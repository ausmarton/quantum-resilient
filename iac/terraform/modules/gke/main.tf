# GKE Module
# This module can be used to create reusable GKE configurations.
# Currently, GKE resources are defined directly in the main gcp/ configuration.
# This module is a placeholder for future refactoring.

variable "cluster_name" {
  type        = string
  description = "Name of the GKE cluster"
}

variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "region" {
  type        = string
  description = "GCP region"
}

variable "network" {
  type        = string
  description = "VPC network name"
}

variable "subnetwork" {
  type        = string
  description = "VPC subnetwork name"
}

# Placeholder - actual implementation in gcp/gke.tf
output "cluster_name" {
  value = var.cluster_name
}

