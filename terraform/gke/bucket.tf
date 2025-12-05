# =============================================================================
# bucket.tf - GCS bucket for experiment artifacts
# =============================================================================

# Create bucket (or import existing one with: terraform import google_storage_bucket.results <bucket-name>)
resource "google_storage_bucket" "results" {
  name          = var.bucket_name
  location      = var.region
  project       = var.project_id
  force_destroy = true  # Allow deletion with objects (for easy cleanup)

  # Versioning for experiment reproducibility
  versioning {
    enabled = true
  }

  # Uniform bucket-level access (recommended)
  uniform_bucket_level_access = true

  # Lifecycle rules - keep old versions for 30 days
  lifecycle_rule {
    condition {
      num_newer_versions = 3
      with_state         = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      days_since_noncurrent_time = 30
    }
    action {
      type = "Delete"
    }
  }

  # Labels
  labels = {
    app         = "pqc-bench"
    environment = "benchmark"
    purpose     = "experiment-results"
  }
}

# Create folder structure markers
resource "google_storage_bucket_object" "experiments_marker" {
  name    = "experiments/.keep"
  content = "# Experiment results are stored here"
  bucket  = google_storage_bucket.results.name
}

