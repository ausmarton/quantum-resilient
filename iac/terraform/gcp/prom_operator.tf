# -----------------------------------------------------------------------------
# Monitoring Namespace
# -----------------------------------------------------------------------------

resource "kubernetes_namespace" "monitoring" {
  count = var.enable_prometheus ? 1 : 0

  metadata {
    name = "monitoring"
    labels = {
      name = "monitoring"
    }
  }

  depends_on = [google_container_node_pool.primary]
}

# -----------------------------------------------------------------------------
# Prometheus Operator (kube-prometheus-stack)
# -----------------------------------------------------------------------------

resource "helm_release" "prometheus_operator" {
  count = var.enable_prometheus ? 1 : 0

  name       = "kube-prometheus-stack"
  namespace  = kubernetes_namespace.monitoring[0].metadata[0].name
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  version    = "55.5.0"

  wait    = true
  timeout = 900

  values = [
    yamlencode({
      # Grafana configuration
      grafana = {
        enabled       = true
        adminPassword = var.grafana_admin_password
        
        persistence = {
          enabled = true
          size    = "10Gi"
        }
        
        # Enable dashboard provisioning
        sidecar = {
          dashboards = {
            enabled         = true
            searchNamespace = "ALL"
            label           = "grafana_dashboard"
          }
          datasources = {
            enabled = true
          }
        }
        
        # Additional data sources
        additionalDataSources = [
          {
            name      = "Prometheus"
            type      = "prometheus"
            url       = "http://kube-prometheus-stack-prometheus:9090"
            access    = "proxy"
            isDefault = true
          }
        ]
        
        service = {
          type = "ClusterIP"
        }
        
        resources = {
          requests = {
            cpu    = "100m"
            memory = "256Mi"
          }
          limits = {
            cpu    = "500m"
            memory = "512Mi"
          }
        }
      }
      
      # Prometheus configuration
      prometheus = {
        prometheusSpec = {
          retention = "7d"
          
          storageSpec = {
            volumeClaimTemplate = {
              spec = {
                accessModes = ["ReadWriteOnce"]
                resources = {
                  requests = {
                    storage = "50Gi"
                  }
                }
              }
            }
          }
          
          # Service monitor selector - match all namespaces
          serviceMonitorSelectorNilUsesHelmValues = false
          podMonitorSelectorNilUsesHelmValues     = false
          
          resources = {
            requests = {
              cpu    = "200m"
              memory = "1Gi"
            }
            limits = {
              cpu    = "2"
              memory = "4Gi"
            }
          }
        }
      }
      
      # AlertManager configuration
      alertmanager = {
        enabled = true
        alertmanagerSpec = {
          storage = {
            volumeClaimTemplate = {
              spec = {
                accessModes = ["ReadWriteOnce"]
                resources = {
                  requests = {
                    storage = "10Gi"
                  }
                }
              }
            }
          }
        }
      }
      
      # Node exporter
      nodeExporter = {
        enabled = true
      }
      
      # Kube state metrics
      kubeStateMetrics = {
        enabled = true
      }
    })
  ]

  depends_on = [
    google_container_node_pool.primary,
    kubernetes_namespace.monitoring,
  ]
}

# -----------------------------------------------------------------------------
# ServiceMonitor for Orchestrator
# -----------------------------------------------------------------------------

resource "kubernetes_manifest" "orchestrator_service_monitor" {
  count = var.enable_prometheus ? 1 : 0

  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "ServiceMonitor"
    metadata = {
      name      = "qr-orchestrator"
      namespace = kubernetes_namespace.qr.metadata[0].name
      labels = {
        app       = "quantum-resilient"
        component = "orchestrator"
      }
    }
    spec = {
      selector = {
        matchLabels = {
          app       = "quantum-resilient"
          component = "orchestrator"
        }
      }
      endpoints = [
        {
          port     = "api"
          path     = "/metrics"
          interval = "15s"
        }
      ]
      namespaceSelector = {
        matchNames = [kubernetes_namespace.qr.metadata[0].name]
      }
    }
  }

  depends_on = [helm_release.prometheus_operator]
}

# -----------------------------------------------------------------------------
# ServiceMonitor for Workers
# -----------------------------------------------------------------------------

resource "kubernetes_manifest" "worker_service_monitor" {
  count = var.enable_prometheus ? 1 : 0

  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "ServiceMonitor"
    metadata = {
      name      = "qr-workers"
      namespace = kubernetes_namespace.qr.metadata[0].name
      labels = {
        app       = "quantum-resilient"
        component = "worker"
      }
    }
    spec = {
      selector = {
        matchLabels = {
          app       = "quantum-resilient"
          component = "worker"
        }
      }
      endpoints = [
        {
          port     = "prom"
          path     = "/metrics"
          interval = "15s"
        }
      ]
      namespaceSelector = {
        matchNames = [kubernetes_namespace.qr.metadata[0].name]
      }
    }
  }

  depends_on = [helm_release.prometheus_operator]
}

# -----------------------------------------------------------------------------
# PodMonitor for Worker Pods (spawned by Jobs)
# -----------------------------------------------------------------------------

resource "kubernetes_manifest" "worker_pod_monitor" {
  count = var.enable_prometheus ? 1 : 0

  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "PodMonitor"
    metadata = {
      name      = "qr-worker-pods"
      namespace = kubernetes_namespace.qr.metadata[0].name
      labels = {
        app       = "quantum-resilient"
        component = "worker"
      }
    }
    spec = {
      selector = {
        matchLabels = {
          app       = "quantum-resilient"
          component = "worker"
        }
      }
      podMetricsEndpoints = [
        {
          port     = "prom"
          path     = "/metrics"
          interval = "15s"
        }
      ]
      namespaceSelector = {
        matchNames = [kubernetes_namespace.qr.metadata[0].name]
      }
    }
  }

  depends_on = [helm_release.prometheus_operator]
}

