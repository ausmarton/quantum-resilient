# -----------------------------------------------------------------------------
# Grafana Dashboard ConfigMaps
# These dashboards are automatically provisioned via Grafana sidecar
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Cluster Throughput Dashboard
# -----------------------------------------------------------------------------

resource "kubernetes_config_map" "dashboard_throughput" {
  count = var.enable_prometheus ? 1 : 0

  metadata {
    name      = "grafana-dashboard-qr-throughput"
    namespace = kubernetes_namespace.monitoring[0].metadata[0].name
    labels = {
      grafana_dashboard = "1"
      app               = "quantum-resilient"
    }
  }

  data = {
    "qr-throughput.json" = file("${path.module}/../dashboards/cluster-throughput.json")
  }

  depends_on = [helm_release.prometheus_operator]
}

# -----------------------------------------------------------------------------
# Crypto Latency Dashboard
# -----------------------------------------------------------------------------

resource "kubernetes_config_map" "dashboard_latency" {
  count = var.enable_prometheus ? 1 : 0

  metadata {
    name      = "grafana-dashboard-qr-latency"
    namespace = kubernetes_namespace.monitoring[0].metadata[0].name
    labels = {
      grafana_dashboard = "1"
      app               = "quantum-resilient"
    }
  }

  data = {
    "qr-latency.json" = file("${path.module}/../dashboards/crypto-latency.json")
  }

  depends_on = [helm_release.prometheus_operator]
}

# -----------------------------------------------------------------------------
# Queueing & Backpressure Dashboard
# -----------------------------------------------------------------------------

resource "kubernetes_config_map" "dashboard_queue" {
  count = var.enable_prometheus ? 1 : 0

  metadata {
    name      = "grafana-dashboard-qr-queue"
    namespace = kubernetes_namespace.monitoring[0].metadata[0].name
    labels = {
      grafana_dashboard = "1"
      app               = "quantum-resilient"
    }
  }

  data = {
    "qr-queue.json" = file("${path.module}/../dashboards/queue-backpressure.json")
  }

  depends_on = [helm_release.prometheus_operator]
}

# -----------------------------------------------------------------------------
# Worker Health Dashboard
# -----------------------------------------------------------------------------

resource "kubernetes_config_map" "dashboard_worker_health" {
  count = var.enable_prometheus ? 1 : 0

  metadata {
    name      = "grafana-dashboard-qr-worker-health"
    namespace = kubernetes_namespace.monitoring[0].metadata[0].name
    labels = {
      grafana_dashboard = "1"
      app               = "quantum-resilient"
    }
  }

  data = {
    "qr-worker-health.json" = file("${path.module}/../dashboards/worker-health.json")
  }

  depends_on = [helm_release.prometheus_operator]
}

# -----------------------------------------------------------------------------
# Experiment Dashboard
# -----------------------------------------------------------------------------

resource "kubernetes_config_map" "dashboard_experiment" {
  count = var.enable_prometheus ? 1 : 0

  metadata {
    name      = "grafana-dashboard-qr-experiment"
    namespace = kubernetes_namespace.monitoring[0].metadata[0].name
    labels = {
      grafana_dashboard = "1"
      app               = "quantum-resilient"
    }
  }

  data = {
    "qr-experiment.json" = file("${path.module}/../dashboards/experiment.json")
  }

  depends_on = [helm_release.prometheus_operator]
}

