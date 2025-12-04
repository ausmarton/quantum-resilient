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
├── rust-core/              # Core library and binary
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
│       └── ingress.yaml
├── helm/                   # Helm chart
│   └── quantum-resilient/
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
