# -----------------------------------------------------------------------------
# Worker Helm Release (Default Deployment for standalone testing)
# -----------------------------------------------------------------------------

resource "helm_release" "workers" {
  name       = "qr-workers"
  namespace  = kubernetes_namespace.qr.metadata[0].name
  chart      = "${path.module}/../../../helm/quantum-resilient"
  
  wait    = true
  timeout = 600

  values = [
    yamlencode({
      replicaCount = 0 # Workers are spawned by orchestrator, not deployed directly
      
      image = {
        repository = var.worker_image
        pullPolicy = "Always"
        tag        = "latest"
      }
      
      serviceAccount = {
        create = false
        name   = kubernetes_service_account.worker.metadata[0].name
      }
      
      resources = {
        requests = {
          cpu    = "100m"
          memory = "256Mi"
        }
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }
      
      podAnnotations = {
        "prometheus.io/scrape" = "true"
        "prometheus.io/port"   = "9898"
        "prometheus.io/path"   = "/metrics"
      }
      
      # Results storage configuration
      results = {
        storageBackend = "gcs"
        gcsBucket      = google_storage_bucket.results.name
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
    google_container_cluster.primary,
    kubernetes_service_account.worker,
    google_storage_bucket.results,
  ]
}

# -----------------------------------------------------------------------------
# Worker ConfigMap for Default Scenario
# -----------------------------------------------------------------------------

resource "kubernetes_config_map" "default_scenario" {
  metadata {
    name      = "qr-default-scenario"
    namespace = kubernetes_namespace.qr.metadata[0].name
    labels = {
      app       = "quantum-resilient"
      component = "scenario"
    }
  }

  data = {
    "active_scenario.yaml" = yamlencode({
      id          = "default_kyber_benchmark"
      description = "Default Kyber KEM benchmark scenario"
      
      workload = {
        msgs_per_sec   = 100
        msg_size_bytes = 256
        duration_sec   = 60
        pattern        = "constant"
      }
      
      execution = {
        mode           = "fixed_pool"
        workers        = 4
        queue_capacity = 2000
      }
      
      algorithm = {
        adapter   = "kyber"
        operation = "kem_aead_encrypt"
      }
      
      metrics = {
        prometheus_endpoint = "0.0.0.0:9898"
        jsonl_out           = "/app/results/results.jsonl"
      }
    })
  }
}

# -----------------------------------------------------------------------------
# Worker Pod Disruption Budget
# -----------------------------------------------------------------------------

resource "kubernetes_pod_disruption_budget" "workers" {
  metadata {
    name      = "qr-workers-pdb"
    namespace = kubernetes_namespace.qr.metadata[0].name
  }

  spec {
    min_available = "50%"
    selector {
      match_labels = {
        app       = "quantum-resilient"
        component = "worker"
      }
    }
  }
}

# -----------------------------------------------------------------------------
# Network Policy for Workers
# -----------------------------------------------------------------------------

resource "kubernetes_network_policy" "workers" {
  metadata {
    name      = "qr-workers-netpol"
    namespace = kubernetes_namespace.qr.metadata[0].name
  }

  spec {
    pod_selector {
      match_labels = {
        component = "worker"
      }
    }

    ingress {
      # Allow traffic from orchestrator
      from {
        pod_selector {
          match_labels = {
            component = "orchestrator"
          }
        }
      }
      
      # Allow Prometheus scraping
      from {
        namespace_selector {
          match_labels = {
            name = "monitoring"
          }
        }
      }
      
      ports {
        port     = 6060
        protocol = "TCP"
      }
      ports {
        port     = 9898
        protocol = "TCP"
      }
    }

    egress {
      # Allow all egress (for GCS upload, orchestrator communication)
      to {}
    }

    policy_types = ["Ingress", "Egress"]
  }
}

