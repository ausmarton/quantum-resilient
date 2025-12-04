# -----------------------------------------------------------------------------
# GCS Bucket for Experiment Results
# -----------------------------------------------------------------------------

resource "google_storage_bucket" "results" {
  name                        = var.bucket_name
  location                    = var.region
  project                     = var.project_id
  uniform_bucket_level_access = true
  force_destroy               = true

  # Lifecycle rules
  lifecycle_rule {
    condition {
      age = 30 # Delete objects older than 30 days
    }
    action {
      type = "Delete"
    }
  }

  # Optional: Move to nearline after 7 days
  lifecycle_rule {
    condition {
      age = 7
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  # Versioning (optional but recommended)
  versioning {
    enabled = false
  }

  # CORS configuration for browser access (if needed)
  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD"]
    response_header = ["*"]
    max_age_seconds = 3600
  }

  labels = var.labels
}

# -----------------------------------------------------------------------------
# Bucket IAM - Grant orchestrator service account access
# -----------------------------------------------------------------------------

resource "google_storage_bucket_iam_member" "orchestrator_admin" {
  bucket = google_storage_bucket.results.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.orchestrator.email}"
}

resource "google_storage_bucket_iam_member" "worker_creator" {
  bucket = google_storage_bucket.results.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.worker.email}"
}

# -----------------------------------------------------------------------------
# BigQuery Dataset (conditional)
# -----------------------------------------------------------------------------

resource "google_bigquery_dataset" "results" {
  count                       = var.enable_bigquery ? 1 : 0
  dataset_id                  = var.bigquery_dataset_id
  friendly_name               = "Quantum Resilient Results"
  description                 = "Dataset for storing quantum resilient benchmark results"
  location                    = var.region
  project                     = var.project_id
  default_table_expiration_ms = 2592000000 # 30 days

  labels = var.labels
}

# BigQuery Table for experiment events
resource "google_bigquery_table" "events" {
  count      = var.enable_bigquery ? 1 : 0
  dataset_id = google_bigquery_dataset.results[0].dataset_id
  table_id   = "experiment_events"
  project    = var.project_id

  time_partitioning {
    type  = "DAY"
    field = "timestamp_utc"
  }

  clustering = ["adapter", "operation"]

  schema = jsonencode([
    {
      name = "event_id"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "timestamp_utc"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "experiment_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "worker_id"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "latency_us"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "queue_delay_us"
      type = "INT64"
      mode = "NULLABLE"
    },
    {
      name = "adapter"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "operation"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "payload_size_bytes"
      type = "INT64"
      mode = "NULLABLE"
    },
    {
      name = "ciphertext_size_bytes"
      type = "INT64"
      mode = "NULLABLE"
    },
    {
      name = "memory_rss_bytes"
      type = "INT64"
      mode = "NULLABLE"
    }
  ])

  labels = var.labels
}

