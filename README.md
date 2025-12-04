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
├── analysis/             # Research analysis environment
│   ├── notebooks/            # Jupyter notebooks
│   │   ├── 00_setup.ipynb
│   │   ├── 01_load_results.ipynb
│   │   └── ...
│   ├── scripts/              # CLI analysis tools
│   │   ├── fetch_results.py
│   │   ├── merge_jsonl.py
│   │   ├── compute_statistics.py
│   │   ├── effect_sizes.py
│   │   ├── plot_latency.py
│   │   └── ...
│   ├── requirements.txt
│   ├── pyproject.toml
│   ├── run_full_pipeline.sh
│   └── README.md
├── iac/                  # Infrastructure as Code
│   └── terraform/
│       ├── gcp/              # GCP/GKE deployment
│       └── modules/          # Reusable modules
├── research/             # Research artifact generation
│   ├── templates/            # Jinja2 report templates
│   ├── scripts/              # Generation scripts
│   │   ├── provenance.py
│   │   ├── version_dataset.py
│   │   ├── generate_tables.py
│   │   ├── generate_figures_bundle.py
│   │   ├── generate_report.py
│   │   └── pipeline_runner.py
│   ├── output/               # Generated artifacts
│   └── README.md
├── packaging/            # Experiment packaging & distribution
│   ├── cli.py                # Typer CLI interface
│   ├── manifest.py           # Generate manifest.json
│   ├── archiver.py           # Create ZIP/TAR.GZ bundles
│   ├── exporter.py           # Publication-ready exports
│   ├── release_notes.py      # Generate release notes
│   ├── publish.py            # Publish to GCS/S3/GitHub
│   ├── templates/            # Jinja2 templates
│   └── output/               # Generated bundles
├── reproducibility/      # Reproducibility test suite
│   ├── runner.py             # Multi-run execution
│   ├── variance.py           # Variance analysis
│   ├── confidence.py         # Confidence intervals
│   ├── stability.py          # Distribution stability
│   ├── regression.py         # Regression detection
│   ├── cluster_scaling.py    # Scaling analysis
│   ├── templates/            # Report templates
│   └── output/               # Generated reports
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

## Running the Analysis Suite

The framework includes a comprehensive Python-based analysis suite for processing benchmark results and generating publication-quality figures.

### Prerequisites

- Python 3.10+
- pip or conda

### Installation

```bash
cd analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Quick Start: Full Pipeline

Run the complete analysis pipeline with a single command:

```bash
# From local results
./run_full_pipeline.sh exp_001 file:///path/to/results

# From GCS
./run_full_pipeline.sh exp_001 gs://qr-results/exp_001

# From S3/MinIO
./run_full_pipeline.sh exp_001 s3://bucket/exp_001
```

The pipeline will:
1. Fetch results from storage
2. Merge JSONL files from all workers
3. Compute statistical summaries
4. Generate plots (latency, throughput, queue delay)
5. Export to Parquet and CSV

### Individual Scripts

```bash
# Fetch results
python scripts/fetch_results.py \
  --experiment-id exp_001 \
  --uri gs://qr-results/exp_001 \
  --out data/exp_001/

# Merge JSONL files
python scripts/merge_jsonl.py \
  --input data/exp_001/raw \
  --output data/exp_001/merged

# Compute statistics
python scripts/compute_statistics.py \
  --input data/exp_001/merged/merged.jsonl \
  --output data/exp_001/stats

# Calculate effect sizes between experiments
python scripts/effect_sizes.py \
  --exp-a data/exp_rsa/merged/merged.jsonl \
  --exp-b data/exp_kyber/merged/merged.jsonl \
  --metric latency_us \
  --out data/comparisons/rsa_vs_kyber.json

# Generate plots
python scripts/plot_latency.py \
  --input data/exp_001/merged/merged.jsonl \
  --output figures/exp_001/

python scripts/plot_throughput.py \
  --input data/exp_001/merged/merged.jsonl \
  --output figures/exp_001/
```

### Jupyter Notebooks

Interactive analysis is available via JupyterLab:

```bash
cd analysis
jupyter lab
```

Available notebooks:

| Notebook | Description |
|----------|-------------|
| `00_setup.ipynb` | Environment verification and GCP authentication |
| `01_load_results.ipynb` | Load and explore experiment data |
| `02_latency_analysis.ipynb` | Latency distributions and comparisons |
| `03_throughput_analysis.ipynb` | Throughput over time analysis |
| `04_queue_delay_analysis.ipynb` | Queue delay correlation with load |
| `05_adapter_comparison.ipynb` | RSA vs ECDSA vs Kyber comparison |
| `06_effect_size.ipynb` | Statistical significance testing |
| `07_cluster_scaling_behavior.ipynb` | Kubernetes autoscaling analysis |
| `99_generate_figures.ipynb` | Publication-quality figure generation |

### Output Structure

After running the pipeline:

```
analysis/
├── data/
│   └── exp_001/
│       ├── raw/              # Original JSONL files
│       ├── merged/
│       │   ├── merged.jsonl  # Combined, sorted events
│       │   └── merged.parquet
│       ├── stats/
│       │   ├── summary.json  # Statistical summary
│       │   ├── latency_hist.png
│       │   ├── queue_hist.png
│       │   └── throughput_curve.png
│       └── exports/
│           ├── exp_001.parquet
│           └── exp_001.csv
└── figures/
    └── exp_001/
        ├── latency_cdf.png
        ├── latency_pdf.png
        ├── latency_tail.png
        ├── throughput_timeseries.png
        └── queue_delay_distribution.png
```

### Effect Size Metrics

The analysis suite computes standard effect size metrics for algorithm comparisons:

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| Cohen's d | Standardized mean difference | <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large |
| Hedge's g | Bias-corrected Cohen's d | Same as Cohen's d |
| Glass's Δ | Uses control group std | Same as Cohen's d |
| Cliff's δ | Non-parametric | <0.147 negligible, 0.147-0.33 small, 0.33-0.474 medium, >0.474 large |
| Wasserstein | Earth mover's distance | In original units |
| KS statistic | Distribution distance | 0-1, with p-value |

## Publishing Experiment Results

The framework includes tools for generating dissertation-ready research artifacts.

### Research Artifact Generation Pipeline

Run the complete documentation pipeline:

```bash
python research/scripts/pipeline_runner.py \
  --exp-id exp_2025_02_01_001 \
  --uri gs://qr-results/exp_2025_02_01_001 \
  --generate-all
```

This generates:
1. **Provenance metadata** (`provenance.json`)
2. **Dataset version** (`dataset_version.json`)
3. **LaTeX/Markdown tables** (`tables/*.tex`, `tables/*.md`)
4. **Figure bundle** (PDF, EPS, high-DPI PNG)
5. **Reports** (`report.tex`, `report.md`)

### Individual Scripts

```bash
# Generate provenance metadata
python research/scripts/provenance.py \
  --exp-id exp_001 \
  --data-dir analysis/data/exp_001 \
  --out research/output/exp_001/

# Version dataset with checksums
python research/scripts/version_dataset.py \
  --exp-id exp_001 \
  --data-dir analysis/data/exp_001 \
  --version 1.0.0 \
  --out research/output/exp_001/

# Generate LaTeX and Markdown tables
python research/scripts/generate_tables.py \
  --exp-id exp_001 \
  --stats-file analysis/data/exp_001/stats/summary.json \
  --out research/output/exp_001/tables/

# Bundle figures for publication
python research/scripts/generate_figures_bundle.py \
  --exp-id exp_001 \
  --figures-dir analysis/figures/exp_001 \
  --out research/output/exp_001/figures/

# Generate reports
python research/scripts/generate_report.py \
  --exp-id exp_001 --format tex \
  --out research/output/exp_001/
```

### Output Structure

```
research/output/exp_001/
├── provenance.json           # Full experiment provenance
├── dataset_version.json      # Dataset checksums and version
├── tables/
│   ├── latency_quantiles.tex
│   ├── latency_quantiles.md
│   ├── throughput_summary.tex
│   ├── adapter_comparison.tex
│   └── effect_sizes.tex
├── figures/
│   ├── png/                  # High-DPI PNG (300 DPI)
│   ├── pdf/                  # Vector PDF
│   ├── eps/                  # Vector EPS (LaTeX)
│   ├── manifest.json         # Figure metadata
│   └── figures_bundle_exp_001.tar.gz
├── report.tex                # LaTeX report
└── report.md                 # Markdown report
```

### Using in Dissertation

Insert tables and figures into LaTeX documents:

```latex
% Include a table
\input{research/output/exp_001/tables/latency_quantiles.tex}

% Include a figure
\includegraphics{research/output/exp_001/figures/pdf/latency_cdf.pdf}
```

### Publishing Workflow

#### Full Pipeline with Packaging

```bash
# Run complete pipeline including packaging
python research/scripts/pipeline_runner.py \
  --exp-id exp_001 \
  --uri gs://qr-results/exp_001 \
  --generate-all \
  --package
```

#### Packaging CLI

The packaging tools provide a user-friendly CLI:

```bash
# Create bundle (manifest + archives)
python -m packaging bundle exp_001

# Create export folder for publication
python -m packaging export exp_001 --lite

# Generate manifest only
python -m packaging manifest exp_001

# Generate release notes
python -m packaging notes exp_001

# Publish to GCS
python -m packaging publish exp_001 \
  --target gcs \
  --uri gs://public-artifacts/quantum-resilient/

# Publish to GitHub Releases
python -m packaging publish exp_001 \
  --target github \
  --uri owner/repo

# Run all packaging steps
python -m packaging all exp_001
```

#### Bundle Contents

The generated bundle includes:

```
exp_001-research-bundle/
├── data/
│   ├── merged.parquet
│   └── summary.json
├── figures/
│   ├── png/
│   └── pdf/
├── tables/
│   ├── *.tex
│   └── *.md
├── report/
│   ├── report.tex
│   └── report.md
└── metadata/
    ├── manifest.json
    ├── provenance.json
    └── dataset_version.json
```

#### Publishing to Cloud Storage

```bash
# Publish to GCS (with verification)
python -m packaging publish exp_001 \
  --target gcs \
  --uri gs://public-artifacts/quantum-resilient/ \
  --public

# Verify upload
gsutil ls gs://public-artifacts/quantum-resilient/exp_001/
```

#### Complete Publishing Workflow

1. **Run full analysis pipeline:**
   ```bash
   python research/scripts/pipeline_runner.py \
     --exp-id exp_001 --uri gs://... --generate-all
   ```

2. **Generate package:**
   ```bash
   python -m packaging bundle exp_001
   ```

3. **Upload to GCS:**
   ```bash
   python -m packaging publish exp_001 \
     --target gcs \
     --uri gs://public-artifacts/quantum-resilient/
   ```

4. **Tag release:**
   ```bash
   git add research/output/exp_001/ packaging/output/exp_001/
   git commit -m "Published experiment exp_001"
   git tag -a exp_001_published -m "Published experiment exp_001"
   git push --tags
   ```

## Running Reproducibility Tests

The framework includes a reproducibility suite for validating experimental stability.

### Execute Multiple Runs

```bash
# Run experiment 20 times for statistical analysis
python reproducibility/runner.py \
  --scenario scenarios/kyber_benchmark.yaml \
  --runs 20 \
  --replicas 30 \
  --exp-prefix kyber_stability
```

### Analyze Reproducibility

```bash
# Variance analysis
python reproducibility/variance.py \
  --input reproducibility/output/kyber_stability_20250201_120000

# Confidence intervals (BCa bootstrap)
python reproducibility/confidence.py \
  --input reproducibility/output/kyber_stability_20250201_120000 \
  --method bca

# Stability testing
python reproducibility/stability.py \
  --input reproducibility/output/kyber_stability_20250201_120000

# Regression detection
python reproducibility/regression.py \
  --current batch_002 \
  --baseline batch_001
```

### Integrated Pipeline

```bash
# Run complete pipeline with reproducibility analysis
python research/scripts/pipeline_runner.py \
  --exp-id exp_001 \
  --generate-all \
  --reproducibility \
  --package
```

### Cluster Scaling Analysis

```bash
# Analyze performance at different cluster sizes
python reproducibility/cluster_scaling.py \
  --input scaling_experiments/ \
  --cluster-sizes 2 5 10 20 40
```

### Output

```
reproducibility/output/batch_001/
├── analysis/
│   ├── variance_summary.json
│   ├── variance_plots.png
│   ├── confidence_intervals.json
│   ├── stability_summary.json
│   ├── stability_matrix.png
│   └── reproducibility_report.md
└── run_*/
    ├── merged/
    └── stats/
```

### Interpretation Guide

| Metric | Good | Acceptable | Investigate |
|--------|------|------------|-------------|
| CV (Latency) | < 10% | 10-25% | > 25% |
| CV (Throughput) | < 10% | 10-25% | > 25% |
| p99 CV | < 15% | 15-30% | > 30% |
| KS p-value | > 0.05 | 0.01-0.05 | < 0.01 |

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
