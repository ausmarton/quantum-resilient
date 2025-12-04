# -----------------------------------------------------------------------------
# Orchestrator Service Account (GSA)
# -----------------------------------------------------------------------------

resource "google_service_account" "orchestrator" {
  account_id   = "qr-orchestrator"
  display_name = "Quantum Resilient Orchestrator"
  description  = "Service account for the QR orchestrator to manage experiments and upload results"
  project      = var.project_id
}

# -----------------------------------------------------------------------------
# Worker Service Account (GSA)
# -----------------------------------------------------------------------------

resource "google_service_account" "worker" {
  account_id   = "qr-worker"
  display_name = "Quantum Resilient Worker"
  description  = "Service account for QR workers to upload results and metrics"
  project      = var.project_id
}

# -----------------------------------------------------------------------------
# Kubernetes Service Accounts (KSA) with Workload Identity
# -----------------------------------------------------------------------------

# Kubernetes namespace for quantum-resilient workloads
resource "kubernetes_namespace" "qr" {
  metadata {
    name = var.kubernetes_namespace
    labels = {
      name        = var.kubernetes_namespace
      application = "quantum-resilient"
    }
  }

  depends_on = [google_container_node_pool.primary]
}

# Orchestrator KSA
resource "kubernetes_service_account" "orchestrator" {
  metadata {
    name      = "qr-orchestrator"
    namespace = kubernetes_namespace.qr.metadata[0].name
    annotations = {
      "iam.gke.io/gcp-service-account" = google_service_account.orchestrator.email
    }
    labels = {
      app       = "quantum-resilient"
      component = "orchestrator"
    }
  }
}

# Worker KSA
resource "kubernetes_service_account" "worker" {
  metadata {
    name      = "qr-worker"
    namespace = kubernetes_namespace.qr.metadata[0].name
    annotations = {
      "iam.gke.io/gcp-service-account" = google_service_account.worker.email
    }
    labels = {
      app       = "quantum-resilient"
      component = "worker"
    }
  }
}

# -----------------------------------------------------------------------------
# Workload Identity Bindings (GSA <-> KSA)
# -----------------------------------------------------------------------------

# Orchestrator Workload Identity binding
resource "google_service_account_iam_binding" "orchestrator_workload_identity" {
  service_account_id = google_service_account.orchestrator.name
  role               = "roles/iam.workloadIdentityUser"
  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[${var.kubernetes_namespace}/qr-orchestrator]"
  ]
}

# Worker Workload Identity binding
resource "google_service_account_iam_binding" "worker_workload_identity" {
  service_account_id = google_service_account.worker.name
  role               = "roles/iam.workloadIdentityUser"
  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[${var.kubernetes_namespace}/qr-worker]"
  ]
}

