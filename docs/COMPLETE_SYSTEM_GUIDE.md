# Complete System Guide: End-to-End Codebase Documentation

**Date**: 2025-12-15  
**Status**: Active  
**Purpose**: Comprehensive, low-level documentation covering every aspect of the codebase from code organization to running benchmarks, data capture, analysis, and report generation.

---

## Table of Contents

1. [Overview](#overview)
2. [Code Organization and Structure](#code-organization-and-structure)
3. [Development Practices](#development-practices)
4. [Core Components Deep Dive](#core-components-deep-dive)
5. [Running Benchmarks](#running-benchmarks)
6. [Data Capture and Telemetry](#data-capture-and-telemetry)
7. [Data Analysis Pipeline](#data-analysis-pipeline)
8. [Report and Graph Generation](#report-and-graph-generation)
9. [Complete End-to-End Workflow](#complete-end-to-end-workflow)
10. [Troubleshooting and Debugging](#troubleshooting-and-debugging)

---

## Overview

This guide provides a comprehensive, low-level understanding of the Quantum-Resilient Cryptography Benchmark Framework. It covers every aspect from code organization to execution, data collection, analysis, and reporting.

### System Architecture

The framework consists of five principal layers:

1. **Configuration Layer**: Defines experimental parameters through declarative YAML configuration
2. **Deployment Layer**: Enables execution across multiple environments (Bare-metal, Local-K8s, Cloud-K8s)
3. **Orchestration and Metrics Layer**: Coordinates experiment execution and collects telemetry data
4. **Cryptographic Execution Layer**: Performs cryptographic operations with uniform measurement instrumentation
5. **Analysis Layer**: Processes raw telemetry into statistical summaries and visualizations

### Key Technologies

- **Rust**: Core benchmarking framework (`rust-core`), orchestrator service
- **Python**: Analysis suite, orchestration scripts, scenario generation
- **Kubernetes**: Container orchestration (Minikube, GKE)
- **Terraform**: Infrastructure as Code (GCP/GKE deployment)
- **Prometheus**: Metrics collection
- **JSONL**: Event-level telemetry format

---

## Code Organization and Structure

### Repository Structure

```
quantum-resilient/
├── rust-core/              # Core Rust library and binary
│   ├── Cargo.toml          # Rust dependencies
│   ├── src/
│   │   ├── main.rs         # Binary entry point (pqc-bench)
│   │   ├── lib.rs          # Library entry point
│   │   ├── scenario.rs     # Scenario loading and validation
│   │   ├── workload.rs     # Workload generation
│   │   ├── crypto_adapter/ # Cryptographic adapters
│   │   │   ├── mod.rs      # CryptoAdapter trait
│   │   │   ├── noop_adapter.rs
│   │   │   ├── rsa_adapter.rs
│   │   │   ├── ecdsa_adapter.rs
│   │   │   ├── ecdhe_adapter.rs
│   │   │   ├── kyber_adapter.rs
│   │   │   ├── dilithium_adapter.rs
│   │   │   ├── kem_hybrid.rs
│   │   │   └── registry.rs
│   │   ├── pipeline/       # Async streaming pipeline
│   │   │   ├── mod.rs
│   │   │   ├── execution.rs
│   │   │   └── workload.rs
│   │   ├── telemetry/     # Metrics and logging
│   │   └── controlplane/  # K8s control plane endpoints
│   └── tests/              # Unit tests
│
├── orchestrator/           # Distributed experiment orchestrator
│   ├── Cargo.toml
│   ├── Dockerfile
│   └── src/
│       ├── main.rs         # Orchestrator entry point
│       ├── api.rs          # REST API endpoints
│       ├── controller.rs   # Experiment lifecycle management
│       ├── coordinator.rs  # Worker coordination
│       ├── k8s_client.rs   # Kubernetes API integration
│       ├── aggregator.rs   # Result aggregation
│       └── storage.rs      # Object storage (S3/GCS)
│
├── scenarios/              # Benchmark scenario definitions
│   ├── smoke_noop.yaml
│   ├── rsa_smoke.yaml
│   ├── ecdsa_smoke.yaml
│   ├── kyber_hybrid_encrypt.yaml
│   ├── kyber_hybrid_decrypt.yaml
│   ├── fixed_pool_burst.yaml
│   ├── elastic_ramp.yaml
│   └── hybrid_kyber_dilithium.yaml
│
├── orchestration/          # Experiment orchestration
│   ├── experiment_matrix.yaml  # Declarative experiment config
│   └── generate_scenarios.py   # Scenario auto-generator
│
├── analysis/              # Research analysis environment
│   ├── notebooks/         # Jupyter notebooks
│   ├── scripts/           # CLI analysis tools
│   │   ├── fetch_results.py
│   │   ├── merge_jsonl.py
│   │   ├── compute_statistics.py
│   │   ├── effect_sizes.py
│   │   ├── plot_latency.py
│   │   └── ...
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── run_full_pipeline.sh
│
├── scripts/               # Orchestration scripts
│   ├── lib/               # Library scripts
│   ├── run_experiment.sh
│   ├── check_progress.sh
│   ├── validate_data_collection.sh
│   └── ...
│
├── iac/                   # Infrastructure as Code
│   └── terraform/
│       └── gcp/           # GCP/GKE deployment
│
├── k8s/                   # Kubernetes manifests
│   ├── base/              # Base K8s manifests
│   └── gcp/               # GCP-specific K8s manifests
│
├── helm/                  # Helm charts
│   ├── quantum-resilient/
│   └── quantum-resilient-orchestrator/
│
├── docs/                  # Documentation
│   ├── guides/            # User guides
│   ├── reference/         # Technical reference
│   ├── analysis/          # Analysis documentation
│   └── ...
│
├── run_local.sh           # Local native experiment runner
├── run_minikube.sh        # Minikube K8s experiment runner
├── deploy_gcp.sh          # GKE deployment script
├── run_all_experiments.sh # Master orchestration script
└── Cargo.toml             # Workspace definition
```

### Workspace Organization

The project uses a **Rust workspace** (`Cargo.toml` at root) to manage multiple Rust crates:

```toml
[workspace]
members = ["rust-core", "orchestrator"]
resolver = "2"
```

This allows:
- Shared dependencies across crates
- Unified build commands
- Consistent dependency versions

### Module Structure

#### Rust Core (`rust-core/`)

**Purpose**: Core benchmarking library and binary

**Key Modules**:

1. **`main.rs`**: Binary entry point
   - Parses CLI arguments
   - Loads scenario configuration
   - Initializes telemetry
   - Creates and runs pipeline
   - Handles orchestration mode

2. **`lib.rs`**: Library entry point
   - Exports public API
   - Re-exports key types and functions

3. **`scenario.rs`**: Scenario loading and validation
   - `Scenario` struct: Complete scenario configuration
   - `WorkloadConfig`: Workload parameters
   - `AlgorithmConfig`: Cryptographic algorithm configuration
   - `ExecutionConfig`: Execution mode configuration
   - `MetricsConfig`: Telemetry configuration
   - Validation functions

4. **`workload.rs`**: Workload generation
   - Deterministic RNG (ChaCha20)
   - Workload patterns (constant, burst, ramp, trace)
   - Payload generation

5. **`crypto_adapter/`**: Cryptographic adapters
   - `CryptoAdapter` trait: Unified interface
   - Adapter implementations (RSA, ECDSA, ECDHE, Kyber, Dilithium)
   - Registry for adapter lookup

6. **`pipeline/`**: Async streaming pipeline
   - `Pipeline`: Main pipeline struct
   - `ExecutionEngine`: Execution modes (single, fixed pool, elastic)
   - Event processing and queuing

7. **`telemetry/`**: Metrics and logging
   - Prometheus metrics
   - JSONL event logging
   - System resource sampling

8. **`controlplane/`**: Kubernetes control plane endpoints
   - Health checks (`/healthz`)
   - Readiness probes (`/readyz`)
   - Metrics endpoint

#### Orchestrator (`orchestrator/`)

**Purpose**: Distributed experiment coordination

**Key Modules**:

1. **`main.rs`**: Orchestrator entry point
   - Initializes HTTP server
   - Sets up Kubernetes client
   - Starts API endpoints

2. **`api.rs`**: REST API endpoints
   - Experiment management
   - Worker registration
   - Status queries

3. **`controller.rs`**: Experiment lifecycle management
   - Experiment creation
   - Worker coordination
   - Result collection

4. **`coordinator.rs`**: Worker coordination
   - Worker registration
   - Start signal synchronization
   - Completion tracking

5. **`k8s_client.rs`**: Kubernetes API integration
   - Job creation
   - Pod management
   - Resource queries

6. **`aggregator.rs`**: Result aggregation
   - Multi-run combination
   - Statistics computation

7. **`storage.rs`**: Object storage (S3/GCS)
   - Result upload
   - Result download

---

## Development Practices

### Code Style and Conventions

#### Rust

- **Formatting**: Use `cargo fmt` (rustfmt)
- **Linting**: Use `cargo clippy` (Clippy)
- **Documentation**: Use `///` for public API documentation
- **Error Handling**: Use `Result<T, E>` for fallible operations
- **Async**: Use `async/await` for I/O operations

#### Python

- **Style**: Follow PEP 8
- **Type Hints**: Use type hints for function signatures
- **Docstrings**: Use Google-style docstrings
- **Formatting**: Use `black` formatter

### Testing Strategy

#### Unit Tests

**Rust**:
- Located in `rust-core/tests/` and inline `#[cfg(test)]` modules
- Test cryptographic adapters
- Test workload generation
- Test scenario loading

**Python**:
- Located in `analysis/tests/` (if exists)
- Test analysis scripts
- Test scenario generation

#### Integration Tests

- End-to-end experiment execution
- Cross-environment validation
- Data pipeline validation

### Development Workflow

1. **Create Feature Branch**: `git checkout -b feature/description`
2. **Implement Changes**: Follow coding standards
3. **Write Tests**: Add unit and integration tests
4. **Run Tests**: `cargo test` (Rust), `pytest` (Python)
5. **Update Documentation**: Update relevant docs
6. **Commit Changes**: Use descriptive commit messages
7. **Create Pull Request**: Include description and test results

### Adding a New Cryptographic Adapter

1. **Create Adapter File**: `rust-core/src/crypto_adapter/new_adapter.rs`
2. **Implement Trait**: Implement `CryptoAdapter` trait
3. **Register Adapter**: Add to `registry.rs`
4. **Add Tests**: Create unit tests
5. **Update Documentation**: Update component documentation
6. **Add Scenario Example**: Create example scenario YAML

### Adding a New Analysis Script

1. **Create Script**: `analysis/scripts/new_script.py`
2. **Add CLI Interface**: Use `argparse` or `typer`
3. **Add Docstring**: Document purpose and usage
4. **Add to Pipeline**: Update `run_full_pipeline.sh` if needed
5. **Update Documentation**: Update analysis workflow docs

---

## Core Components Deep Dive

### Scenario Loading (`rust-core/src/scenario.rs`)

**Purpose**: Load and validate benchmark scenario configurations from YAML files.

**Key Data Structures**:

```rust
pub struct Scenario {
    pub id: String,
    pub description: Option<String>,
    pub workload: WorkloadConfig,
    pub algorithm: AlgorithmConfig,
    pub metrics: MetricsConfig,
    pub execution: ExecutionConfig,
    pub rng_seed: Option<u64>,
}

pub struct WorkloadConfig {
    pub msgs_per_sec: u32,
    pub msg_size_bytes: usize,
    pub duration_sec: u32,
    pub pattern: WorkloadPattern,
    pub burst: Option<BurstConfig>,
    pub ramp: Option<RampConfig>,
    pub trace_file: Option<String>,
}

pub struct AlgorithmConfig {
    pub adapter: String,
    pub operation: String,
}

pub struct ExecutionConfig {
    pub mode: ExecutionMode,
    pub workers: Option<usize>,
    pub max_workers: Option<usize>,
    pub queue_capacity: usize,
}

pub struct MetricsConfig {
    pub prometheus_endpoint: String,
    pub jsonl_out: String,
}
```

**Loading Process**:

1. Read YAML file
2. Deserialize into `Scenario` struct
3. Validate all fields
4. Check adapter and operation are supported
5. Validate workload parameters
6. Return validated scenario

**Validation Rules**:

- `id`: Must be non-empty
- `msgs_per_sec`: Must be > 0
- `msg_size_bytes`: Must be > 0
- `duration_sec`: Must be > 0
- `adapter`: Must be in supported list
- `operation`: Must be in supported list
- `mode`: Must be valid execution mode

### Workload Generation (`rust-core/src/workload.rs`)

**Purpose**: Generate deterministic workloads for reproducible experiments.

**Key Components**:

1. **Deterministic RNG**: ChaCha20 with seed from scenario
2. **Workload Patterns**:
   - `Constant`: Fixed rate
   - `Burst`: Periodic bursts
   - `Ramp`: Linear rate increase
   - `Trace`: From CSV file

**Seed Computation**:

```rust
fn compute_rng_seed(scenario: &Scenario) -> u64 {
    // Deterministic seed from scenario parameters
    let mut hasher = DefaultHasher::new();
    hasher.write(scenario.id.as_bytes());
    hasher.write_u32(scenario.workload.msgs_per_sec);
    hasher.write_usize(scenario.workload.msg_size_bytes);
    hasher.write_u32(scenario.workload.duration_sec);
    hasher.finish()
}
```

**Workload Generation**:

1. Initialize RNG with seed
2. Generate events based on pattern
3. Schedule events at correct timestamps
4. Generate payloads with deterministic content

### Cryptographic Adapters (`rust-core/src/crypto_adapter/`)

**Purpose**: Provide unified interface for cryptographic operations.

**Trait Definition**:

```rust
pub trait CryptoAdapter: Send + Sync {
    fn keygen(&self) -> Result<(Vec<u8>, Vec<u8>), CryptoError>;
    fn sign(&self, secret_key: &[u8], message: &[u8]) -> Result<usize, CryptoError>;
    fn verify(&self, public_key: &[u8], message: &[u8], signature: &[u8]) -> Result<bool, CryptoError>;
    fn encapsulate(&self, public_key: &[u8], message: &[u8]) -> Result<(usize, Option<usize>), CryptoError>;
    fn decapsulate(&self, secret_key: &[u8], ciphertext: &[u8], message: &[u8]) -> Result<usize, CryptoError>;
}
```

**Adapter Implementations**:

1. **NoOp Adapter**: Baseline (no-op operations)
2. **RSA Adapter**: RSA-2048 signing/verification
3. **ECDSA Adapter**: ECDSA P-256 signing/verification
4. **ECDHE Adapter**: ECDHE P-256 key exchange
5. **Kyber Adapter**: Kyber-512 KEM
6. **Dilithium Adapter**: Dilithium-2 signatures

**Registry**:

```rust
pub fn create_adapter(name: &str) -> Result<Box<dyn CryptoAdapter>, CryptoError> {
    match name {
        "noop" => Ok(Box::new(NoOpAdapter::new())),
        "rsa2048" => Ok(Box::new(RSAAdapter::new())),
        "ecdsa_p256" => Ok(Box::new(ECDSAAdapter::new())),
        "ecdhe_p256" => Ok(Box::new(ECDHEAdapter::new())),
        "kyber" => Ok(Box::new(KyberAdapter::new())),
        "dilithium" => Ok(Box::new(DilithiumAdapter::new())),
        _ => Err(CryptoError::UnsupportedAdapter),
    }
}
```

### Pipeline Execution (`rust-core/src/pipeline/`)

**Purpose**: Execute benchmarks with async streaming pipeline.

**Pipeline Structure**:

```
Workload Generator → Event Queue → Worker Pool → Crypto Adapter → Telemetry
```

**Execution Modes**:

1. **Single**: Single-threaded execution
2. **Fixed Pool**: Fixed number of worker threads
3. **Elastic**: Dynamic worker pool based on queue depth

**Event Processing**:

1. **Enqueue**: Workload generator creates events and enqueues
2. **Dequeue**: Worker dequeues event
3. **Measure Queue Delay**: Time in queue
4. **Execute Operation**: Call cryptographic adapter
5. **Measure Latency**: Operation duration
6. **Sample Resources**: CPU, memory, I/O
7. **Log Event**: Write to JSONL
8. **Update Metrics**: Update Prometheus metrics

**Key Code Flow**:

```rust
// In pipeline/execution.rs
async fn process_event(
    event: &QueuedEvent,
    worker_id: usize,
    adapter: &Arc<dyn CryptoAdapter + Send + Sync>,
    metrics: &Metrics,
    jsonl_writer: &JsonlWriter,
    sampler: &SysInfoSampler,
    context: &ExecutionContext,
) -> ProcessedEvent {
    let dequeue_ts = Instant::now();
    let queue_delay_ns = dequeue_ts.duration_since(event.enqueue_ts).as_nanos();
    
    let start = Instant::now();
    let op_result = match context.operation.as_str() {
        "sign" => adapter.sign(&[], &event.payload),
        // ... other operations
    };
    let latency_ns = start.elapsed().as_nanos();
    
    let (cpu_user, memory_rss) = sampler.sample();
    
    metrics.observe_latency(&context.algorithm, &context.operation, latency_ns as f64);
    metrics.observe_queue_delay(queue_delay_ns as f64);
    
    let row = EventRowWithQueueDelay {
        // ... event data
        latency_ns,
        queue_delay_ns,
        cpu_user_seconds: cpu_user,
        memory_rss_bytes: memory_rss,
        // ...
    };
    
    jsonl_writer.write(&row)?;
    
    ProcessedEvent { /* ... */ }
}
```

### Telemetry Collection (`rust-core/src/telemetry/`)

**Purpose**: Collect performance metrics and system resource data.

**Components**:

1. **Prometheus Metrics**: Exposed via HTTP endpoint
2. **JSONL Logger**: Event-level telemetry in JSONL format
3. **System Sampler**: CPU, memory, I/O statistics

**Metrics Collected**:

- **Latency**: Operation duration (nanoseconds)
- **Queue Delay**: Time in queue (nanoseconds)
- **Throughput**: Operations per second
- **CPU Time**: User and system CPU time
- **Memory**: RSS (Resident Set Size)
- **I/O**: Read/write statistics

**JSONL Format**:

```json
{
  "run_id": "run-20251215-001549",
  "scenario_id": "kyber512_kem_aead_encrypt_256B_100msg_s_constant",
  "event_id": 12345,
  "timestamp_utc_iso": "2025-12-15T00:15:49.123456Z",
  "timestamp_monotonic_ns": 1234567890123456,
  "operation": "kem_aead_encrypt",
  "algorithm": "kyber512",
  "latency_ns": 1234567,
  "queue_delay_ns": 12345,
  "worker_id": 0,
  "payload_size_bytes": 256,
  "ciphertext_size_bytes": 768,
  "cpu_user_seconds": 0.001234,
  "memory_rss_bytes": 1048576,
  "rng_seed": 1234567890,
  "error": null
}
```

---

## Running Benchmarks

### Environment Setup

#### Native (Bare-metal)

**Requirements**:
- Rust toolchain (`rustc`, `cargo`)
- Python 3.8+ (for analysis)
- System dependencies (varies by OS)

**Build**:
```bash
cd rust-core
cargo build --release
```

**Run**:
```bash
./run_local.sh --scenario scenarios/kyber_hybrid_encrypt.yaml --out results/native
```

**What Happens**:
1. Builds `pqc-bench` binary
2. Loads scenario YAML
3. Executes benchmark
4. Writes JSONL output
5. Generates statistics

#### Minikube (Local Kubernetes)

**Requirements**:
- Minikube installed
- Docker/Podman
- kubectl

**Setup**:
```bash
minikube start
eval $(minikube docker-env)
```

**Build Container**:
```bash
podman build -f Containerfile -t pqc-bench:latest .
```

**Run**:
```bash
./run_minikube.sh --scenario scenarios/kyber_hybrid_encrypt.yaml --out results/minikube --exp-id test
```

**What Happens**:
1. Builds container image
2. Creates Kubernetes Job
3. Deploys worker pod
4. Executes benchmark in pod
5. Collects results from pod
6. Writes JSONL output

#### GCP (Google Kubernetes Engine)

**Requirements**:
- GCP account
- Terraform installed
- gcloud CLI
- kubectl

**Deploy**:
```bash
./deploy_gcp.sh \
  --scenario scenarios/kyber_hybrid_encrypt.yaml \
  --exp-id gcp-test \
  --project <your-project> \
  --bucket <your-bucket> \
  --region us-central1
```

**What Happens**:
1. Terraform creates GKE cluster
2. Builds and pushes container image to GCR
3. Creates Kubernetes Job
4. Deploys worker pods
5. Executes benchmark
6. Uploads results to GCS
7. Downloads results locally

### Master Orchestration Script

**`run_all_experiments.sh`**: Complete end-to-end experiment orchestration

**Usage**:
```bash
./run_all_experiments.sh \
  --envs native,minikube,gcp \
  --project <gcp-project> \
  --bucket <gcs-bucket> \
  --matrix orchestration/experiment_matrix.yaml
```

**What It Does**:

1. **Scenario Generation**:
   - Reads `experiment_matrix.yaml`
   - Generates scenario YAML files
   - Computes RNG seeds

2. **Experiment Execution**:
   - Runs experiments per environment
   - Manages experiment lifecycle
   - Tracks progress

3. **Data Collection**:
   - Collects JSONL files
   - Generates statistics
   - Creates experiment index

4. **Analysis** (if not `--skip-analysis`):
   - Merges JSONL files
   - Computes statistics
   - Generates visualizations
   - Runs hypothesis tests

5. **Reporting**:
   - Creates run directory
   - Generates summary reports
   - Exports report-ready outputs

### Experiment Matrix

**`orchestration/experiment_matrix.yaml`**: Declarative experiment configuration

**Structure**:
```yaml
defaults:
  runs: 5
  duration_sec: 30
  pattern: constant

scaling:
  enabled: true
  replicas: [1, 2, 4, 8]

environments:
  - name: native
    type: bare_metal
  - name: minikube
    type: kubernetes
  - name: gcp
    type: kubernetes
    region: us-central1

experiments:
  - algorithm: kyber512
    adapter: kyber
    operation: kem_aead_encrypt
    payload_sizes: [256, 1024, 4096, 16384]
    rates: [100, 500, 2000, 10000]
    category: pqc
```

**Scenario Generation**:

`orchestration/generate_scenarios.py` reads the matrix and generates individual scenario YAML files:

```python
def generate_scenarios(matrix_file: str, output_dir: str):
    matrix = load_yaml(matrix_file)
    scenarios = []
    
    for exp in matrix['experiments']:
        for payload_size in exp['payload_sizes']:
            for rate in exp['rates']:
                scenario = {
                    'id': f"{exp['algorithm']}_{exp['operation']}_{payload_size}B_{rate}msg_s_constant",
                    'workload': {
                        'msgs_per_sec': rate,
                        'msg_size_bytes': payload_size,
                        'duration_sec': matrix['defaults']['duration_sec'],
                        'pattern': matrix['defaults']['pattern'],
                    },
                    'algorithm': {
                        'adapter': exp['adapter'],
                        'operation': exp['operation'],
                    },
                    # ... other fields
                }
                scenarios.append(scenario)
    
    write_scenarios(scenarios, output_dir)
```

---

## Data Capture and Telemetry

### Instrumentation Points

**Operation Boundaries**:
- Entry: `Instant::now()` before operation
- Exit: `Instant::now()` after operation
- Latency: `elapsed().as_nanos()`

**Queue Monitoring**:
- Enqueue timestamp: When event created
- Dequeue timestamp: When event processed
- Queue delay: `dequeue_ts - enqueue_ts`

**Resource Sampling**:
- CPU: `getrusage()` (user/system time)
- Memory: `/proc/self/status` (RSS)
- I/O: `/proc/self/io` (read/write bytes)

### Telemetry Collection Flow

```
Event Created
    ↓
Enqueue (timestamp: enqueue_ts)
    ↓
Queue (wait for worker)
    ↓
Dequeue (timestamp: dequeue_ts)
    ↓
Measure Queue Delay (dequeue_ts - enqueue_ts)
    ↓
Operation Start (timestamp: start)
    ↓
Execute Cryptographic Operation
    ↓
Operation End (timestamp: end)
    ↓
Measure Latency (end - start)
    ↓
Sample System Resources (CPU, memory, I/O)
    ↓
Write JSONL Event
    ↓
Update Prometheus Metrics
```

### Precision Implementation

**Nanosecond Precision**: All timing measurements use `Instant::now()` and `as_nanos()` to capture sub-microsecond precision.

**Why Nanoseconds**:
- Many operations complete in <1 microsecond
- `as_micros()` truncates to 0 for sub-microsecond operations
- Nanosecond precision preserves full measurement accuracy

**Storage**:
- JSONL: `latency_ns` (u128)
- Analysis: Convert to microseconds for statistics

### Data Storage Structure

```
results/
├── <environment>/
│   ├── <scenario-id>/
│   │   ├── raw/
│   │   │   └── run-<timestamp>/
│   │   │       ├── events.jsonl
│   │   │       └── metadata.json
│   │   ├── stats/
│   │   │   └── summary.json
│   │   └── plots/
│   │       └── *.png
```

**JSONL Format**: One JSON object per line, representing one cryptographic operation.

**Metadata Format**:
```json
{
  "scenario_id": "kyber512_kem_aead_encrypt_256B_100msg_s_constant",
  "run_id": "run-20251215-001549",
  "environment": "native",
  "algorithm": "kyber512",
  "operation": "kem_aead_encrypt",
  "payload_size_bytes": 256,
  "msgs_per_sec": 100,
  "duration_sec": 30,
  "pattern": "constant",
  "execution_mode": "fixed_pool",
  "workers": 4,
  "rng_seed": 1234567890,
  "start_time": "2025-12-15T00:15:49Z",
  "end_time": "2025-12-15T00:16:19Z",
  "total_events": 3000,
  "events_processed": 3000,
  "hardware": {
    "cpu": "AMD Ryzen AI MAX+ PRO 395",
    "cores": 32,
    "memory_gb": 94
  }
}
```

---

## Data Analysis Pipeline

### Analysis Workflow

```
Raw JSONL Files
    ↓
Merge JSONL (merge_jsonl.py)
    ↓
Compute Statistics (compute_statistics.py)
    ↓
Aggregate Results (aggregate_results.py)
    ↓
Hypothesis Tests (hypothesis_tests.py)
    ↓
Generate Visualizations (plot_*.py)
    ↓
Build Report (build_final_report.py)
```

### Key Analysis Scripts

#### 1. Merge JSONL (`analysis/scripts/merge_jsonl.py`)

**Purpose**: Combine multiple JSONL files from multiple runs.

**Input**: Multiple `events.jsonl` files
**Output**: Single merged `events.jsonl` file

**Process**:
1. Read all JSONL files
2. Combine events
3. Sort by timestamp
4. Write merged file

#### 2. Compute Statistics (`analysis/scripts/compute_statistics.py`)

**Purpose**: Compute run-level and cross-run statistics.

**Input**: Merged `events.jsonl`
**Output**: `summary.json` with statistics

**Statistics Computed**:
- **Percentiles**: p50, p95, p99 latency
- **Mean**: Average latency
- **Std Dev**: Standard deviation
- **Min/Max**: Minimum and maximum latency
- **Throughput**: Events per second
- **Queue Delay**: Queue delay statistics

**Output Format**:
```json
{
  "scenario_id": "kyber512_kem_aead_encrypt_256B_100msg_s_constant",
  "runs": 5,
  "total_events": 15000,
  "latency": {
    "p50_us": 12.5,
    "p95_us": 25.0,
    "p99_us": 50.0,
    "mean_us": 13.2,
    "std_dev_us": 5.8,
    "min_us": 8.1,
    "max_us": 125.0
  },
  "throughput": {
    "events_per_sec": 100.5
  },
  "queue_delay": {
    "p50_ns": 1234,
    "p95_ns": 5678,
    "p99_ns": 12345
  }
}
```

#### 3. Aggregate Results (`analysis/aggregate_results.py`)

**Purpose**: Aggregate statistics across all experiments.

**Input**: Experiment index (`index.json`)
**Output**: Aggregated statistics JSON

**Process**:
1. Read experiment index
2. Load all `summary.json` files
3. Group by algorithm, operation, environment
4. Compute aggregate statistics
5. Write aggregated results

#### 4. Hypothesis Tests (`analysis/hypothesis_tests.py`)

**Purpose**: Statistical hypothesis testing for performance comparisons.

**Input**: Aggregated statistics
**Output**: Hypothesis test results JSON

**Tests Performed**:
- **Welch's t-test**: Parametric test (unequal variances)
- **Mann-Whitney U**: Non-parametric test
- **Effect Size**: Cohen's d
- **Confidence Intervals**: 95% CI

**Output Format**:
```json
{
  "comparison": "kyber512 vs rsa2048",
  "metric": "latency_p50_us",
  "group1": {
    "algorithm": "kyber512",
    "mean": 12.5,
    "std_dev": 5.8,
    "n": 5
  },
  "group2": {
    "algorithm": "rsa2048",
    "mean": 8.2,
    "std_dev": 3.1,
    "n": 5
  },
  "welch_t_test": {
    "t_statistic": 2.34,
    "p_value": 0.045,
    "significant": true
  },
  "mann_whitney_u": {
    "u_statistic": 12,
    "p_value": 0.038,
    "significant": true
  },
  "effect_size": {
    "cohens_d": 0.85,
    "interpretation": "large"
  },
  "confidence_interval": {
    "lower": 1.2,
    "upper": 7.4,
    "level": 0.95
  }
}
```

#### 5. Visualization Scripts (`analysis/scripts/plot_*.py`)

**Purpose**: Generate publication-ready figures.

**Scripts**:
- `plot_latency.py`: Latency CDFs
- `plot_throughput.py`: Throughput comparisons
- `plot_queue_delay.py`: Queue delay analysis
- `plot_combined_cdfs.py`: Combined CDF plots
- `plot_effect_size_forest.py`: Effect size forest plots
- `plot_scaling_curves.py`: Scaling analysis

**Output**: PNG/PDF figures in `figures/` directory

### Complete Analysis Pipeline

**`analysis/run_full_pipeline.sh`**: Runs complete analysis pipeline

**Steps**:
1. Merge JSONL files
2. Compute statistics
3. Aggregate results
4. Run hypothesis tests
5. Generate visualizations
6. Build final report

**Usage**:
```bash
cd analysis
./run_full_pipeline.sh \
  --index ../final-results/index.json \
  --output ../final-results
```

---

## Report and Graph Generation

### Report Generation

**`analysis/build_final_report.py`**: Generates report-ready reports

**Input**:
- Aggregated statistics
- Hypothesis test results
- Experiment metadata

**Output**:
- Markdown report
- LaTeX tables
- Figure references

**Report Sections**:
1. Executive Summary
2. Experimental Setup
3. Results
4. Statistical Analysis
5. Discussion
6. Conclusions

### Graph Generation

**Types of Graphs**:

1. **CDFs (Cumulative Distribution Functions)**:
   - Show latency distributions
   - Compare algorithms
   - Highlight tail behavior

2. **Box Plots**:
   - Show quartiles
   - Compare distributions
   - Identify outliers

3. **Scaling Curves**:
   - Show performance vs. payload size
   - Show performance vs. workload rate
   - Identify scaling characteristics

4. **Effect Size Forest Plots**:
   - Visualize effect sizes
   - Show confidence intervals
   - Compare multiple comparisons

**Graph Generation Process**:

1. **Load Data**: Read aggregated statistics
2. **Filter Data**: Select relevant experiments
3. **Compute Statistics**: Percentiles, means, etc.
4. **Create Plot**: Use matplotlib/seaborn
5. **Style Plot**: Apply publication styling
6. **Save Figure**: Export as PNG/PDF

**Example**:
```python
# plot_latency.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_latency_cdf(data, output_file):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algorithm in data['algorithm'].unique():
        algo_data = data[data['algorithm'] == algorithm]
        sorted_latencies = sorted(algo_data['latency_us'])
        cumulative = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
        ax.plot(sorted_latencies, cumulative, label=algorithm)
    
    ax.set_xlabel('Latency (μs)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Latency CDF Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
```

---

## Complete End-to-End Workflow

### Step-by-Step: From Code to Report

#### Phase 1: Development

1. **Write Code**:
   - Implement new adapter/feature
   - Write tests
   - Update documentation

2. **Test Locally**:
   - Run unit tests
   - Run integration tests
   - Smoke test with small scenario

3. **Commit Changes**:
   - Git commit
   - Push to repository

#### Phase 2: Scenario Generation

1. **Define Experiment Matrix**:
   - Edit `orchestration/experiment_matrix.yaml`
   - Define algorithms, operations, parameters

2. **Generate Scenarios**:
   ```bash
   python3 orchestration/generate_scenarios.py \
     --matrix orchestration/experiment_matrix.yaml \
     --output generated-scenarios/
   ```

3. **Verify Scenarios**:
   - Check generated YAML files
   - Validate scenario structure

#### Phase 3: Data Collection

1. **Run Experiments**:
   ```bash
   ./run_all_experiments.sh \
     --envs native,minikube,gcp \
     --project <gcp-project> \
     --bucket <gcs-bucket> \
     --matrix orchestration/experiment_matrix.yaml
   ```

2. **Monitor Progress**:
   ```bash
   ./scripts/check_progress.sh --envs native,minikube,gcp
   ```

3. **Validate Data**:
   ```bash
   ./scripts/validate_data_collection.sh --envs native,minikube,gcp
   ```

#### Phase 4: Data Processing

1. **Regenerate Index**:
   ```bash
   ./scripts/regenerate_index_from_results.sh \
     --matrix orchestration/experiment_matrix.yaml \
     --output final-results/
   ```

2. **Merge JSONL Files**:
   ```bash
   python3 analysis/scripts/merge_jsonl.py \
     --input results/native/*/raw/*/events.jsonl \
     --output results/native/merged/events.jsonl
   ```

3. **Compute Statistics**:
   ```bash
   python3 analysis/scripts/compute_statistics.py \
     --input results/native/merged/events.jsonl \
     --output results/native/stats/summary.json
   ```

#### Phase 5: Analysis

1. **Aggregate Results**:
   ```bash
   python3 analysis/aggregate_results.py \
     --index final-results/index.json \
     --output final-results/
   ```

2. **Run Hypothesis Tests**:
   ```bash
   python3 analysis/hypothesis_tests.py \
     --input final-results/aggregated_stats.json \
     --output final-results/hypothesis_tests.json
   ```

3. **Generate Visualizations**:
   ```bash
   python3 analysis/scripts/plot_combined_cdfs.py \
     --index final-results/index.json \
     --output final-results/figures/
   ```

#### Phase 6: Reporting

1. **Build Final Report**:
   ```bash
   python3 analysis/build_final_report.py \
     --stats final-results/aggregated_stats.json \
     --tests final-results/hypothesis_tests.json \
     --output final-results/report.md
   ```

2. **Extract Tables**:
   ```bash
   python3 scripts/extract_analysis_tables.py \
     --input final-results/aggregated_stats.json \
     --output final-results/tables/
   ```

3. **Generate Artifacts**:
   ```bash
   python3 research/scripts/generate_report.py \
     --input final-results/ \
     --output research/output/
   ```

---

## Troubleshooting and Debugging

### Common Issues

#### 1. Build Failures

**Rust Compilation Errors**:
```bash
cd rust-core
cargo clean
cargo build --release
```

**Python Import Errors**:
```bash
cd analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. Runtime Errors

**Scenario Loading Errors**:
- Check YAML syntax
- Verify adapter/operation are supported
- Check file paths

**Container Errors**:
- Check container image is built
- Verify Kubernetes cluster is running
- Check pod logs: `kubectl logs <pod-name>`

#### 3. Data Collection Issues

**Missing Data**:
- Check experiment completed successfully
- Verify JSONL files exist
- Check disk space

**Invalid Data**:
- Validate JSONL format
- Check for corrupted files
- Verify scenario configuration

#### 4. Analysis Errors

**Statistics Computation Errors**:
- Check JSONL files are valid
- Verify data format
- Check for missing fields

**Visualization Errors**:
- Check matplotlib backend
- Verify data is loaded correctly
- Check output directory permissions

### Debugging Tools

**Rust Debugging**:
```bash
RUST_LOG=debug cargo run --bin pqc-bench -- --scenario scenarios/test.yaml
```

**Python Debugging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Kubernetes Debugging**:
```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name>
kubectl exec -it <pod-name> -- /bin/sh
```

### Logging

**Rust Logging**:
- Uses `tracing` crate
- Structured logging with spans
- Configurable log levels

**Python Logging**:
- Uses `logging` module
- File and console output
- Configurable log levels

---

## Conclusion

This guide provides a comprehensive, low-level understanding of the entire codebase. For specific details on individual components, refer to:

- **[Component Documentation](reference/component-documentation.md)**: Detailed component documentation
- **[Codebase Inventory](CODEBASE_INVENTORY.md)**: Complete file inventory
- **[Data Collection Guide](guides/data-collection.md)**: Data collection procedures
- **[Analysis Workflow](analysis/workflow.md)**: Analysis procedures
- **[Requirements Specification](REQUIREMENTS_SPECIFICATION.md)**: System requirements

---

**Last Updated**: 2025-12-15  
**Maintainer**: Update when system changes or new features are added
