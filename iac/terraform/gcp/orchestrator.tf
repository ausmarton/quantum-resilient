# -----------------------------------------------------------------------------
# Orchestrator Helm Release
# -----------------------------------------------------------------------------

resource "helm_release" "orchestrator" {
  name       = "qr-orchestrator"
  namespace  = kubernetes_namespace.qr.metadata[0].name
  chart      = "${path.module}/../../../helm/quantum-resilient-orchestrator"
  
  # Wait for the chart to be deployed
  wait    = true
  timeout = 600

  # Values override
  values = [
    yamlencode({
      replicaCount = 1
      
      image = {
        repository = var.orchestrator_image
        pullPolicy = "Always"
        tag        = "latest"
      }
      
      serviceAccount = {
        create = false
        name   = kubernetes_service_account.orchestrator.metadata[0].name
      }
      
      orchestrator = {
        listenAddr      = "0.0.0.0:7070"
        workerImage     = var.worker_image
        localResultsDir = "/data/results"
        maxTimeDriftNs  = 5000000
        storageUri      = "gs://${google_storage_bucket.results.name}"
      }
      
      resources = {
        requests = {
          cpu    = "200m"
          memory = "512Mi"
        }
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }
      
      podAnnotations = {
        "prometheus.io/scrape" = "true"
        "prometheus.io/port"   = "7070"
        "prometheus.io/path"   = "/metrics"
      }
      
      # GCP-specific settings
      gcp = {
        enabled             = true
        bucketName          = google_storage_bucket.results.name
        projectId           = var.project_id
        region              = var.region
        useWorkloadIdentity = true
      }
    })
  ]

  depends_on = [
    google_container_node_pool.primary,
    kubernetes_service_account.orchestrator,
    google_storage_bucket.results,
  ]
}

# -----------------------------------------------------------------------------
# Orchestrator RBAC (Kubernetes)
# -----------------------------------------------------------------------------

resource "kubernetes_role" "orchestrator" {
  metadata {
    name      = "qr-orchestrator"
    namespace = kubernetes_namespace.qr.metadata[0].name
    labels = {
      app       = "quantum-resilient"
      component = "orchestrator"
    }
  }

  # Jobs management for experiment workers
  rule {
    api_groups = ["batch"]
    resources  = ["jobs"]
    verbs      = ["create", "delete", "get", "list", "watch", "patch"]
  }

  # Pod management and monitoring
  rule {
    api_groups = [""]
    resources  = ["pods"]
    verbs      = ["get", "list", "watch", "delete"]
  }

  # Pod logs for debugging
  rule {
    api_groups = [""]
    resources  = ["pods/log"]
    verbs      = ["get"]
  }

  # Pod exec for result collection
  rule {
    api_groups = [""]
    resources  = ["pods/exec"]
    verbs      = ["create"]
  }

  # ConfigMap management for scenario distribution
  rule {
    api_groups = [""]
    resources  = ["configmaps"]
    verbs      = ["create", "delete", "get", "list", "watch", "patch"]
  }

  # Services for worker discovery
  rule {
    api_groups = [""]
    resources  = ["services"]
    verbs      = ["get", "list", "watch"]
  }
}

resource "kubernetes_role_binding" "orchestrator" {
  metadata {
    name      = "qr-orchestrator"
    namespace = kubernetes_namespace.qr.metadata[0].name
    labels = {
      app       = "quantum-resilient"
      component = "orchestrator"
    }
  }

  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "Role"
    name      = kubernetes_role.orchestrator.metadata[0].name
  }

  subject {
    kind      = "ServiceAccount"
    name      = kubernetes_service_account.orchestrator.metadata[0].name
    namespace = kubernetes_namespace.qr.metadata[0].name
  }
}

