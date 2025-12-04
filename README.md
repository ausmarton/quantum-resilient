# Quantum-Resilient Cryptography Benchmark Framework

A modular benchmark test framework for comparing Post-Quantum Cryptography (PQC) algorithms against classical cryptography in real-time streaming pipelines.

## Overview

This framework provides tools for:
- Benchmarking PQC vs classical cryptographic operations
- Simulating real-time streaming pipeline workloads
- Collecting and analyzing performance telemetry
- Scenario-based test configuration via YAML
- Hybrid KEM→AEAD encryption (Kyber + AES-GCM)

## Project Structure

```
quantum-resilient/
├── Cargo.toml              # Workspace definition
├── rust-core/              # Core library and binary (worker)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs              # Library entry point
│       ├── main.rs             # pqc-bench binary
│       ├── crypto_adapter/     # Cryptographic adapters
│       │   ├── mod.rs          # CryptoAdapter trait
│       │   ├── noop_adapter.rs # NoOp baseline adapter
│       │   ├── rsa_adapter.rs  # RSA-2048 adapter
│       │   ├── ecdsa_adapter.rs # ECDSA P-256 adapter
│       │   ├── kyber_adapter.rs # Kyber PQC KEM adapter
│       │   ├── kem_hybrid.rs   # KEM→AEAD hybrid helpers
│       │   └── registry.rs     # Adapter factory
│       ├── pipeline/           # Async streaming pipeline
│       │   ├── mod.rs
│       │   ├── execution.rs    # Execution models (single/fixed/elastic)
│       │   └── workload.rs     # Workload generators
│       ├── controlplane/       # K8s control plane endpoints
│       │   ├── mod.rs
│       │   └── http.rs         # HTTP handlers (/healthz, /readyz, etc.)
│       ├── scenario.rs         # Scenario loading
│       ├── telemetry/          # Metrics & logging
│       └── workload.rs         # Workload generator
├── orchestrator/           # Distributed experiment orchestrator
│   ├── Cargo.toml
│   ├── Dockerfile
│   └── src/
│       ├── main.rs             # Orchestrator entry point
│       ├── api.rs              # REST API endpoints
│       ├── controller.rs       # Experiment lifecycle management
│       ├── coordinator.rs      # Worker coordination
│       ├── k8s_client.rs       # Kubernetes API integration
│       ├── aggregator.rs       # Result aggregation
│       └── storage.rs          # Object storage (S3/GCS)
├── scenarios/              # Benchmark scenario definitions
│   ├── smoke_noop.yaml
│   ├── rsa_smoke.yaml
│   ├── ecdsa_smoke.yaml
│   ├── kyber_hybrid_encrypt.yaml
│   ├── kyber_hybrid_decrypt.yaml
│   ├── fixed_pool_burst.yaml
│   └── elastic_ramp.yaml
├── k8s/                    # Kubernetes manifests
│   └── base/
│       ├── deployment.yaml
│       ├── service.yaml
│       ├── configmap.yaml
│       ├── pvc.yaml
│       ├── hpa.yaml
│       ├── servicemonitor.yaml
│       ├── serviceaccount.yaml
│       ├── role.yaml
│       ├── rolebinding.yaml
│       ├── ingress.yaml
│       ├── orchestrator-deployment.yaml
│       ├── orchestrator-service.yaml
│       └── orchestrator-rbac.yaml
├── helm/                   # Helm charts
│   ├── quantum-resilient/
│   │   ├── Chart.yaml
│   │   ├── values.yaml
│   │   └── templates/
│   └── quantum-resilient-orchestrator/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
├── Makefile
├── Dockerfile.podman
└── README.md
```

## Prerequisites

- Rust 1.85+ (pinned via `.rust-toolchain.toml`)
- Make
- Podman (optional, for containerized builds)

## Quick Start

### Local Build and Run

1. **Build the project:**

```bash
make build
```

2. **Run a benchmark scenario:**

```bash
# NoOp baseline
make run ARGS="--scenario scenarios/smoke_noop.yaml"

# Classical RSA
make run ARGS="--scenario scenarios/rsa_smoke.yaml"

# Post-Quantum Kyber hybrid encryption
make run ARGS="--scenario scenarios/kyber_hybrid_encrypt.yaml"
```

3. **Run tests:**

```bash
make test
```

## Running PQC Experiments (Kyber)

### Quick Local (Fallback)

Build and run with default features using the pure-Rust `pqcrypto-kyber` fallback (no native liboqs required):

```bash
# Build with default features
cargo build

# Run Kyber hybrid encryption benchmark
cargo run --bin pqc-bench -- --scenario scenarios/kyber_hybrid_encrypt.yaml
```

Expected output:
```
Starting PQC Benchmark Framework...
Loaded scenario: kyber_hybrid_encrypt
Using adapter: kyber
Running operation: kem_aead_encrypt
Generating keypair for KEM operations...
Keypair generated: pk_len=800, sk_len=1632
Starting Prometheus metrics server on 0.0.0.0:9898
...
Run complete: 100 events processed
Average latency: ~500-1000 μs
```

### Container Build (with bundled scenarios)

Build the container (uses pqcrypto fallback by default):

```bash
podman build -f Dockerfile.podman -t pqc-bench:latest .
```

Run the container:

```bash
podman run --rm -p 9898:9898 -v $(pwd)/results:/app/results:Z pqc-bench:latest \
    --scenario /app/scenarios/kyber_hybrid_encrypt.yaml
```

## Supported Adapters

| Adapter | Type | Description | Operations |
|---------|------|-------------|------------|
| `noop` | Baseline | Zero-cost operations | All |
| `rsa2048` | Classical | RSA-2048 OAEP | sign, keygen |
| `ecdsa_p256` | Classical | ECDSA P-256 | sign, verify, keygen |
| `kyber` | PQC | Kyber-512 KEM | keygen, encapsulate, decapsulate, kem_aead_* |

## Supported Operations

| Operation | Description |
|-----------|-------------|
| `sign` | Digital signature generation |
| `verify` | Digital signature verification |
| `encrypt` | KEM encapsulation |
| `decrypt` | KEM decapsulation |
| `keygen` | Key pair generation |
| `kem_aead_encrypt` | Hybrid: Kyber KEM + AES-256-GCM encryption |
| `kem_aead_decrypt` | Hybrid: Kyber KEM + AES-256-GCM decryption |

## Hybrid KEM→AEAD Encryption

The `kem_aead_encrypt` operation performs:
1. **KEM Encapsulation**: Generate shared secret using Kyber-512
2. **Key Derivation**: Derive AES-256 key via HKDF-SHA256
3. **AEAD Encryption**: Encrypt plaintext with AES-256-GCM

### Combined Payload Format

```
[ct_kem_len: u16 BE] [ct_kem: bytes] [nonce: 12 bytes] [ct_aead: bytes+tag]
```

- `ct_kem_len`: 2-byte big-endian KEM ciphertext length
- `ct_kem`: Kyber ciphertext (~768 bytes for Kyber-512)
- `nonce`: 12-byte random AES-GCM nonce
- `ct_aead`: AES-GCM ciphertext + 16-byte authentication tag

## Scenario Configuration

Example Kyber scenario:

```yaml
id: kyber_hybrid_encrypt
description: "Kyber KEM -> AES-GCM hybrid encrypt test"

workload:
  msgs_per_sec: 50
  msg_size_bytes: 256
  duration_sec: 2

algorithm:
  adapter: kyber
  operation: kem_aead_encrypt

metrics:
  prometheus_endpoint: "0.0.0.0:9898"
  jsonl_out: "./results/kyber_hybrid_encrypt.jsonl"
```

## Metrics & Telemetry

### Prometheus Metrics

Query metrics at `http://localhost:9898/metrics`:

```bash
curl http://127.0.0.1:9898/metrics | grep pqc_
```

Available metrics:
- `pqc_operation_latency_us` - Operation latency histogram
- `pqc_ops_total{algorithm,operation,success}` - Operation counter
- `pqc_memory_bytes` - Process memory usage
- `pqc_events_processed_total` - Total events processed

### JSONL Output

Each benchmark writes detailed event logs:

```json
{
  "run_id": "kyber_hybrid_encrypt",
  "event_id": 1,
  "timestamp_utc_iso": "2025-01-01T00:00:00.000Z",
  "operation": "kem_aead_encrypt",
  "algorithm": "kyber",
  "latency_us": 523,
  "payload_size_bytes": 256,
  "ciphertext_size_bytes": 1052,
  "cpu_user_seconds": 0.01,
  "memory_rss_bytes": 15000000
}
```

## Security Notes

- **Secret Key Handling**: Secret keys are kept in memory only during benchmark runs and are not logged or persisted to JSONL files.
- **Zeroization**: Secret key material is zeroized when dropped using the `zeroize` crate.
- **No Raw Keys in Logs**: JSONL output only includes key lengths, never raw key bytes.

## Running in Kubernetes

The framework includes full Kubernetes support with Helm charts, probes, autoscaling, and Prometheus integration.

### Prerequisites

- Minikube or any Kubernetes cluster
- kubectl configured
- Helm 3.x
- Podman/Docker for building images

### Quick Start with Minikube

1. **Start Minikube:**

```bash
minikube start --cpus 6 --memory 12g
```

2. **Enable metrics server (for HPA):**

```bash
minikube addons enable metrics-server
```

3. **Build and load the container image:**

```bash
# Build the image
podman build -f Dockerfile.podman -t pqc-bench:latest .

# Load into Minikube (if using Minikube's Docker daemon)
minikube image load pqc-bench:latest
```

4. **Deploy using Helm:**

```bash
helm install qr ./helm/quantum-resilient
```

Or deploy raw manifests:

```bash
kubectl apply -f k8s/base/
```

5. **Verify deployment:**

```bash
# Check pods
kubectl get pods -l app=quantum-resilient

# Check pod details
kubectl describe pod -l app=quantum-resilient

# View logs
kubectl logs -l app=quantum-resilient -f
```

### Accessing Services

**Port forward for metrics:**

```bash
kubectl port-forward svc/qr-quantum-resilient 9898:9898
curl http://localhost:9898/metrics | grep pqc_
```

**Port forward for control plane:**

```bash
kubectl port-forward svc/qr-quantum-resilient 6060:6060

# Health checks
curl http://localhost:6060/healthz
curl http://localhost:6060/readyz
curl http://localhost:6060/workers
```

### Graceful Shutdown

```bash
# Get pod IP
POD_IP=$(kubectl get pod -l app=quantum-resilient -o jsonpath='{.items[0].status.podIP}')

# Or via port-forward
curl -X POST http://localhost:6060/shutdown
```

### Custom Scenarios

Override the default scenario via Helm values:

```bash
helm install qr ./helm/quantum-resilient --set-file scenario.customScenario=my-scenario.yaml
```

Or create a custom ConfigMap:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: quantum-resilient-scenario
data:
  active_scenario.yaml: |
    id: my_custom_test
    workload:
      msgs_per_sec: 100
      msg_size_bytes: 512
      duration_sec: 60
    algorithm:
      adapter: kyber
      operation: kem_aead_encrypt
    execution:
      mode: elastic
      max_workers: 16
      queue_capacity: 5000
```

### Viewing Results

```bash
# Check JSONL results on PVC
kubectl exec -it $(kubectl get pods -l app=quantum-resilient -o jsonpath='{.items[0].metadata.name}') -- ls -la /app/results

# Copy results locally
kubectl cp $(kubectl get pods -l app=quantum-resilient -o jsonpath='{.items[0].metadata.name}'):/app/results ./results
```

### Prometheus Operator Integration

If using Prometheus Operator, the ServiceMonitor will auto-configure scraping:

```bash
# Check ServiceMonitor
kubectl get servicemonitor

# Verify Prometheus is scraping
kubectl port-forward svc/prometheus 9090:9090
# Visit http://localhost:9090/targets
```

### Horizontal Pod Autoscaler

The HPA scales based on `pqc_queue_length` metric:

```bash
# Watch HPA status
kubectl get hpa -w

# Describe HPA
kubectl describe hpa quantum-resilient-hpa
```

**Note:** Custom metrics require a metrics adapter like Prometheus Adapter.

### Helm Values Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.repository` | Container image | `localhost/pqc-bench` |
| `image.tag` | Image tag | `latest` |
| `replicaCount` | Initial replicas | `1` |
| `resources.requests.cpu` | CPU request | `100m` |
| `resources.requests.memory` | Memory request | `256Mi` |
| `resources.limits.cpu` | CPU limit | `2` |
| `resources.limits.memory` | Memory limit | `2Gi` |
| `autoscaling.enabled` | Enable HPA | `true` |
| `autoscaling.minReplicas` | Min replicas | `1` |
| `autoscaling.maxReplicas` | Max replicas | `10` |
| `persistence.enabled` | Enable PVC for results | `true` |
| `persistence.size` | PVC size | `2Gi` |
| `serviceMonitor.enabled` | Enable ServiceMonitor | `true` |

### Environment Variables

The container supports these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `QR_SCENARIO_PATH` | Path to scenario YAML | `/app/scenarios/active_scenario.yaml` |
| `QR_RESULTS_DIR` | Results directory | `/app/results` |
| `QR_CONTROL_PLANE_PORT` | Control plane port | `6060` |
| `QR_PROM_PORT` | Prometheus metrics port | `9898` |
| `RUST_LOG` | Log level | `info` |

## Distributed Experiments with Orchestrator

The orchestrator enables coordinated, multi-pod benchmark experiments across a Kubernetes cluster.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Orchestrator (qr-orchestrator)              │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │   REST API  │  │  Coordinator │  │  K8s Client             │ │
│  │  :7070      │  │  (barriers)  │  │  (Jobs, ConfigMaps)     │ │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Worker Pod 1   │ │  Worker Pod 2   │ │  Worker Pod N   │
│  (pqc-bench)    │ │  (pqc-bench)    │ │  (pqc-bench)    │
│  :6060 :9898    │ │  :6060 :9898    │ │  :6060 :9898    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### Orchestrator API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /experiment` | Create | Create new experiment with scenario and replicas |
| `GET /experiments` | List | List all experiments |
| `GET /experiment/{id}/status` | Status | Get experiment status and worker counts |
| `POST /experiment/{id}/start` | Start | Signal all workers to begin |
| `POST /experiment/{id}/stop` | Stop | Gracefully stop all workers |
| `POST /experiment/{id}/collect` | Collect | Aggregate results from all workers |
| `DELETE /experiment/{id}` | Delete | Clean up experiment resources |

### Creating a Distributed Experiment

1. **Deploy the orchestrator:**

```bash
kubectl apply -f k8s/base/orchestrator-rbac.yaml
kubectl apply -f k8s/base/orchestrator-deployment.yaml
kubectl apply -f k8s/base/orchestrator-service.yaml
```

Or via Helm:

```bash
helm install qr-orch ./helm/quantum-resilient-orchestrator
```

2. **Create an experiment:**

```bash
# Port-forward to orchestrator
kubectl port-forward svc/qr-orchestrator 7070:7070

# Create experiment with 12 replicas
curl -X POST http://localhost:7070/experiment \
  -H "Content-Type: application/json" \
  -d '{
    "scenarioConfig": "id: distributed_kyber\nworkload:\n  msgs_per_sec: 100\n  msg_size_bytes: 256\n  duration_sec: 60\nalgorithm:\n  adapter: kyber\n  operation: kem_aead_encrypt\nexecution:\n  mode: fixed_pool\n  workers: 4\n  queue_capacity: 2000\nmetrics:\n  prometheus_endpoint: \"0.0.0.0:9898\"\n  jsonl_out: \"./results/results.jsonl\"",
    "replicas": 12,
    "startDelayMs": 5000,
    "experimentId": "exp_2025_0101_001"
  }'
```

3. **Wait for workers to register:**

```bash
curl http://localhost:7070/experiment/exp_2025_0101_001/status
# {"replicas": 12, "ready": 12, "completed": 0, "phase": "waiting"}
```

4. **Start the experiment:**

```bash
curl -X POST http://localhost:7070/experiment/exp_2025_0101_001/start
```

All workers will begin processing at a synchronized global start time.

5. **Monitor progress:**

```bash
watch -n 2 'curl -s http://localhost:7070/experiment/exp_2025_0101_001/status'
```

6. **Collect and aggregate results:**

```bash
curl -X POST http://localhost:7070/experiment/exp_2025_0101_001/collect
# {
#   "artifactUri": "file:///data/results/exp_2025_0101_001/merged_results.jsonl",
#   "events": 520000,
#   "duration_sec": 60,
#   "summary": {
#     "total_events": 520000,
#     "throughput_ops_sec": 8666.67,
#     "latency_p50_us": 523,
#     "latency_p99_us": 1245
#   }
# }
```

### Worker Environment Variables (Orchestrated Mode)

When workers are spawned by the orchestrator, these variables are set:

| Variable | Description |
|----------|-------------|
| `QR_ORCHESTRATOR_ADDRESS` | Orchestrator HTTP endpoint |
| `QR_EXPERIMENT_ID` | Experiment identifier |
| `QR_ENFORCE_TIMESYNC` | Warn on time drift > 5ms |
| `POD_NAME` | Kubernetes pod name |
| `POD_IP` | Kubernetes pod IP |

### Time Synchronization

Workers verify time synchronization with the orchestrator at registration:
- If drift > 5ms, a warning is logged
- With `QR_ENFORCE_TIMESYNC=true`, the warning is more prominent
- All workers wait for a global start timestamp to ensure coordinated execution

### Object Storage Integration

Upload results to S3 or GCS:

```bash
# Deploy orchestrator with storage backend
helm install qr-orch ./helm/quantum-resilient-orchestrator \
  --set orchestrator.storageUri=s3://my-bucket/experiments
```

Enable storage features in build:

```bash
# S3 (AWS/MinIO)
cargo build -p orchestrator --features storage-s3

# GCS
cargo build -p orchestrator --features storage-gcs
```

## Running in GCP (GKE)

The framework includes complete Terraform infrastructure-as-code for deploying to Google Cloud Platform.

### Prerequisites

- GCP account with billing enabled
- `gcloud` CLI installed and configured
- Terraform 1.5+
- kubectl

### Quick Start

1. **Authenticate with GCP:**

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

2. **Deploy infrastructure with Terraform:**

```bash
cd iac/terraform/gcp

# Initialize Terraform
terraform init

# Review the plan
terraform plan \
  -var="project_id=YOUR_PROJECT_ID" \
  -var="bucket_name=qr-results-YOUR_PROJECT_ID" \
  -var="region=us-central1"

# Apply the configuration
terraform apply \
  -var="project_id=YOUR_PROJECT_ID" \
  -var="bucket_name=qr-results-YOUR_PROJECT_ID" \
  -var="region=us-central1"
```

3. **Configure kubectl:**

```bash
# Get credentials for the cluster (command shown in terraform output)
gcloud container clusters get-credentials quantum-resilient-cluster \
  --region us-central1 \
  --project YOUR_PROJECT_ID
```

4. **Verify deployment:**

```bash
# Check cluster nodes
kubectl get nodes

# Check pods
kubectl get pods -n quantum-resilient

# Check services
kubectl get svc -n quantum-resilient
```

### Accessing Services

**Port-forward to orchestrator:**

```bash
kubectl port-forward -n quantum-resilient svc/qr-orchestrator 7070:7070
```

**Port-forward to Grafana:**

```bash
kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80
# Default credentials: admin / admin (or value from grafana_admin_password variable)
```

### Running a Distributed Experiment on GKE

1. **Create an experiment:**

```bash
cat > scenario.json << 'EOF'
{
  "scenarioConfig": "id: gke_kyber_test\nworkload:\n  msgs_per_sec: 100\n  msg_size_bytes: 256\n  duration_sec: 60\nalgorithm:\n  adapter: kyber\n  operation: kem_aead_encrypt\nexecution:\n  mode: fixed_pool\n  workers: 4\n  queue_capacity: 2000\nmetrics:\n  prometheus_endpoint: \"0.0.0.0:9898\"\n  jsonl_out: \"./results/results.jsonl\"",
  "replicas": 12,
  "startDelayMs": 5000,
  "experimentId": "exp_gke_001"
}
EOF

curl -X POST http://localhost:7070/experiment \
  -H "Content-Type: application/json" \
  -d @scenario.json
```

2. **Start the experiment:**

```bash
curl -X POST http://localhost:7070/experiment/exp_gke_001/start
```

3. **Monitor progress:**

```bash
watch -n 2 'curl -s http://localhost:7070/experiment/exp_gke_001/status'
```

4. **Collect and export results:**

```bash
curl -X POST http://localhost:7070/experiment/exp_gke_001/collect
```

5. **Verify results in GCS:**

```bash
gsutil ls gs://qr-results-YOUR_PROJECT_ID/exp_gke_001/
gsutil cat gs://qr-results-YOUR_PROJECT_ID/exp_gke_001/results.jsonl | head
```

### Experiment Scheduling

Schedule recurring experiments using cron expressions:

```bash
# Create a nightly benchmark schedule
curl -X POST http://localhost:7070/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "name": "nightly_kyber_benchmark",
    "cron": "0 3 * * *",
    "scenarioConfig": "id: nightly_kyber\nworkload:\n  msgs_per_sec: 200\n  duration_sec: 300\nalgorithm:\n  adapter: kyber\n  operation: kem_aead_encrypt",
    "replicas": 50
  }'

# List schedules
curl http://localhost:7070/schedules

# Check schedule status
curl http://localhost:7070/schedule/nightly_kyber_benchmark
```

### Grafana Dashboards

The deployment includes pre-configured Grafana dashboards:

| Dashboard | Description |
|-----------|-------------|
| QR - Cluster Throughput | Overall and per-worker ops/sec |
| QR - Crypto Latency | Latency histograms by adapter (Kyber, RSA, ECDSA) |
| QR - Queue & Backpressure | Queue length, utilization, and delay distributions |
| QR - Worker Health | CPU, memory, restarts per worker |
| QR - Experiment Overview | Experiment timeline and aggregated metrics |

Access at `http://localhost:3000` after port-forwarding Grafana.

### Terraform Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `project_id` | GCP project ID | (required) |
| `region` | GCP region | `us-central1` |
| `bucket_name` | GCS bucket name | (required) |
| `gke_name` | GKE cluster name | `quantum-resilient-cluster` |
| `gke_node_machine_type` | Node machine type | `n2-standard-4` |
| `gke_node_min_count` | Min nodes per zone | `1` |
| `gke_node_max_count` | Max nodes per zone | `7` |
| `enable_prometheus` | Deploy Prometheus stack | `true` |
| `enable_bigquery` | Enable BigQuery export | `false` |

### Cleanup

```bash
# Destroy all resources
cd iac/terraform/gcp
terraform destroy \
  -var="project_id=YOUR_PROJECT_ID" \
  -var="bucket_name=qr-results-YOUR_PROJECT_ID"
```

## Development

### Available Make Targets

| Target | Description |
|--------|-------------|
| `build` | Build the project (debug) |
| `release` | Build the project (release) |
| `run` | Run pqc-bench binary (use `ARGS` for options) |
| `smoke` | Run the smoke_noop scenario |
| `test` | Run all tests |
| `fmt` | Format code |
| `clippy` | Run clippy lints |
| `check` | Run all checks (fmt, clippy, test) |
| `clean` | Clean build artifacts |
| `container` | Build Podman container |
| `container-run` | Run Podman container with smoke test |

### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `pqcrypto_fallback` | Use pure-Rust pqcrypto-kyber | ✓ |

## License

[To be determined]
