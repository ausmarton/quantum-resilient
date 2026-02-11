# Component Documentation

**Last Updated**: 2025-12-15  
**Purpose**: Detailed documentation of all major components in the quantum-resilient codebase.

---

## Table of Contents

1. [Core Benchmark Framework (rust-core)](#core-benchmark-framework-rust-core)
2. [Orchestrator](#orchestrator)
3. [Analysis Suite](#analysis-suite)
4. [Orchestration System](#orchestration-system)
5. [Scenario Format](#scenario-format)
6. [Workflow Overview](#workflow-overview)

---

## Core Benchmark Framework (rust-core)

### Overview

The `rust-core` crate provides the core benchmarking functionality for comparing Post-Quantum Cryptography (PQC) algorithms against classical cryptography in real-time streaming pipelines.

### Architecture

```
rust-core/
├── src/
│   ├── main.rs              # Binary entry point (pqc-bench)
│   ├── lib.rs               # Library entry point
│   ├── scenario.rs          # Scenario loading and validation
│   ├── workload.rs          # Workload generation
│   ├── crypto_adapter/      # Cryptographic algorithm adapters
│   ├── pipeline/            # Async streaming pipeline
│   ├── telemetry/           # Metrics and logging
│   └── controlplane/        # Kubernetes control plane endpoints
```

### Entry Point: `main.rs`

**Binary**: `pqc-bench`

**Command-Line Arguments**:
- `--scenario <path>`: Path to scenario YAML file (overrides `QR_SCENARIO_PATH` env var)
- `--control-port <port>`: Control plane server port (default: 6060, or `QR_CONTROL_PLANE_PORT` env var)

**Environment Variables**:
- `QR_SCENARIO_PATH`: Path to scenario YAML file
- `QR_CONTROL_PLANE_PORT`: Control plane server port
- `QR_ORCHESTRATOR_ADDRESS`: Orchestrator HTTP endpoint (for distributed mode)
- `QR_EXPERIMENT_ID`: Experiment identifier (for distributed mode)
- `QR_ENFORCE_TIMESYNC`: Enforce time synchronization (warn on drift > 5ms)
- `POD_NAME`: Kubernetes pod name (for distributed mode)
- `POD_IP`: Kubernetes pod IP (for distributed mode)
- `PQC_SMOKE_TEST`: Enable smoke-test mode (reduced scale validation)

**Execution Flow**:

1. **Parse Arguments**: Read CLI args and environment variables
2. **Load Scenario**: Load and validate scenario YAML file
3. **Initialize Tracing**: Set up structured logging
4. **Validate Adapter**: Verify cryptographic adapter is supported
5. **Validate Operation**: Verify operation is supported
6. **Initialize Metrics**: Create Prometheus metrics collector
7. **Start Metrics Server**: Start Prometheus metrics endpoint
8. **Initialize JSONL Writer**: Create JSONL event logger
9. **Initialize System Sampler**: Create system resource sampler
10. **Create Execution State**: Create shared execution state
11. **Create Orchestration State**: Create orchestration state (for distributed mode)
12. **Start Control Plane Server**: Start HTTP control plane server
13. **Register with Orchestrator** (if orchestrated): Register worker and wait for start signal
14. **Create Pipeline**: Create benchmark pipeline
15. **Run Pipeline**: Execute benchmark with configured scenario
16. **Notify Completion** (if orchestrated): Notify orchestrator of completion
17. **Print Summary**: Display run statistics

**Orchestrated Mode**:

When `QR_ORCHESTRATOR_ADDRESS` and `QR_EXPERIMENT_ID` are set, the worker:
1. Registers with orchestrator and receives worker ID
2. Checks time synchronization (warns if drift > 5ms)
3. Notifies orchestrator when ready
4. Waits for global start signal
5. Synchronizes start time with other workers
6. Executes benchmark
7. Notifies orchestrator when complete

### Scenario Loading: `scenario.rs`

**Purpose**: Load and validate benchmark scenario configurations from YAML files.

**Key Types**:

- `Scenario`: Complete scenario configuration
  - `id`: Unique identifier
  - `description`: Optional description
  - `workload`: Workload configuration
  - `algorithm`: Algorithm configuration
  - `metrics`: Metrics configuration
  - `execution`: Execution model configuration
  - `rng_seed`: Optional RNG seed for reproducibility

- `WorkloadConfig`: Workload configuration
  - `msgs_per_sec`: Target messages per second
  - `msg_size_bytes`: Message size in bytes
  - `duration_sec`: Duration in seconds
  - `pattern`: Workload pattern (constant, burst, ramp, trace)
  - `burst`: Burst pattern configuration (if pattern is burst)
  - `ramp`: Ramp pattern configuration (if pattern is ramp)
  - `trace_file`: Path to trace file (if pattern is trace)

- `WorkloadPattern`: Workload pattern types
  - `Constant`: Constant rate (default)
  - `Burst`: Periodic burst spikes
  - `Ramp`: Gradually ramping load
  - `Trace`: Trace-driven replay from CSV

- `AlgorithmConfig`: Algorithm configuration
  - `adapter`: Adapter name (e.g., "kyber", "rsa2048")
  - `operation`: Operation type (e.g., "sign", "kem_aead_encrypt")

- `ExecutionConfig`: Execution model configuration
  - `mode`: Execution mode (single, fixed_pool, elastic)
  - `workers`: Number of workers (for fixed_pool)
  - `max_workers`: Maximum workers (for elastic)
  - `queue_capacity`: Queue capacity

- `MetricsConfig`: Metrics configuration
  - `prometheus_endpoint`: Prometheus metrics endpoint
  - `jsonl_out`: JSONL output path

**Functions**:

- `load_scenario(path: &str) -> Result<Scenario, String>`: Load scenario from YAML file
- `supported_operations() -> Vec<String>`: Get list of supported operations

### Workload Generation: `workload.rs`

**Purpose**: Generate workload events based on configured pattern.

**Workload Patterns**:

1. **Constant**: Constant rate of operations
   - Generates events at fixed interval: `1.0 / msgs_per_sec` seconds

2. **Burst**: Periodic burst spikes
   - Baseline rate: `msgs_per_sec`
   - Burst rate: `msgs_per_sec * burst.factor`
   - Burst duration: `burst.duration_ms` milliseconds
   - Burst interval: `burst.interval_ms` milliseconds

3. **Ramp**: Gradually ramping load
   - Starts at baseline rate
   - Gradually increases to target rate over duration
   - Ramp configuration: `ramp.start_rate`, `ramp.end_rate`, `ramp.duration_ms`

4. **Trace**: Trace-driven replay
   - Loads CSV file with columns: `timestamp_ms`, `rps`
   - Replays workload from trace file

**Key Types**:

- `WorkloadModel`: Trait for workload models
  - `next_event(&mut self) -> Option<Duration>`: Get delay until next event
  - `description(&self) -> String`: Get workload description

- `create_workload_model(config: &WorkloadConfig) -> Box<dyn WorkloadModel>`: Create workload model from configuration

### Cryptographic Adapters: `crypto_adapter/`

**Purpose**: Provide unified interface for classical and post-quantum cryptographic operations.

**Adapter Trait**: `CryptoAdapter`

**Supported Adapters**:

1. **NoOp** (`noop_adapter.rs`): Zero-cost baseline adapter
   - Operations: All (no-op implementations)
   - Purpose: Baseline for measuring framework overhead

2. **RSA-2048** (`rsa_adapter.rs`): Classical RSA-2048
   - Operations: `sign`, `verify`, `keygen`
   - Purpose: Classical signature baseline

3. **ECDSA P-256** (`ecdsa_adapter.rs`): Classical ECDSA P-256
   - Operations: `sign`, `verify`, `keygen`
   - Purpose: Classical signature baseline

4. **ECDH P-256** (`ecdh_adapter.rs`): Classical ECDH P-256
   - Operations: `keygen`, `encapsulate`, `decapsulate`, `kem_aead_encrypt`, `kem_aead_decrypt`
   - Purpose: Classical key exchange baseline

5. **Kyber-512** (`kyber_adapter.rs`): Post-quantum Kyber-512 KEM
   - Operations: `keygen`, `encapsulate`, `decapsulate`, `kem_aead_encrypt`, `kem_aead_decrypt`
   - Purpose: Post-quantum key exchange

6. **Dilithium-2** (`dilithium_adapter.rs`): Post-quantum Dilithium-2 signature
   - Operations: `sign`, `verify`, `keygen`
   - Purpose: Post-quantum signature

**Hybrid Operations**: `kem_hybrid.rs`

- `hybrid_encrypt()`: Kyber KEM + AES-256-GCM encryption
- `hybrid_decrypt()`: Kyber KEM + AES-256-GCM decryption
- `derive_aead_key()`: HKDF-SHA256 key derivation from KEM shared secret

**Adapter Registry**: `registry.rs`

- `get_adapter(name: &str) -> Result<Arc<dyn CryptoAdapter>, String>`: Get adapter by name
- `supported_adapters() -> Vec<String>`: Get list of supported adapters

### Pipeline: `pipeline/`

**Purpose**: Async streaming pipeline for executing benchmarks.

**Execution Modes** (`execution.rs`):

1. **Single**: Single processor task
   - One worker processes all events sequentially
   - Suitable for low-rate workloads

2. **FixedPool**: Fixed number of processor tasks
   - Fixed number of workers process events in parallel
   - Suitable for predictable workloads

3. **Elastic**: Dynamic worker pool
   - Workers expand/contract based on queue pressure
   - Suitable for variable workloads

**Pipeline Flow**:

1. **Producer Task**: Generates events based on workload model
2. **Event Queue**: Buffers events (bounded by `queue_capacity`)
3. **Worker Pool**: Processes events in parallel (based on execution mode)
4. **Telemetry**: Logs events to JSONL and updates metrics

**Key Types**:

- `Pipeline`: Main pipeline struct
- `PipelineStats`: Statistics from pipeline run
- `ExecutionEngine`: Execution engine for different modes
- `ExecutionState`: Shared execution state
- `QueuedEvent`: Event in queue with timing
- `ProcessedEvent`: Result of processing event

**Functions**:

- `Pipeline::new() -> Pipeline`: Create new pipeline
- `Pipeline::run_async(...) -> Result<PipelineStats, PipelineError>`: Run pipeline asynchronously

### Telemetry: `telemetry/`

**Purpose**: Metrics collection and event logging.

**Components**:

1. **Metrics** (`metrics.rs`): Prometheus metrics collection
   - `pqc_operation_latency_us`: Latency histogram
   - `pqc_ops_total`: Operation counter
   - `pqc_memory_bytes`: Memory usage gauge
   - `pqc_events_processed_total`: Events processed counter
   - `pqc_queue_length`: Queue length gauge
   - `pqc_active_workers`: Active workers gauge

2. **JSONL Logger** (`jsonl_logger.rs`): Event logging to JSONL
   - Logs each event with full telemetry
   - Fields: event_id, timestamp, operation, algorithm, latency, payload_size, etc.

3. **System Sampler** (`sysinfo_sampler.rs`): System resource sampling
   - Samples CPU usage, memory usage
   - Attached to each event

4. **Tracing Setup** (`tracing_setup.rs`): Structured logging setup
   - Configures tracing with appropriate log level

### Control Plane: `controlplane/`

**Purpose**: HTTP endpoints for Kubernetes control plane integration.

**Endpoints** (`http.rs`):

- `GET /healthz`: Health check endpoint
- `GET /readyz`: Readiness check endpoint
- `GET /workers`: Get worker status
- `POST /shutdown`: Gracefully shutdown pipeline

**Key Types**:

- `ControlPlaneState`: Shared state for control plane
- `OrchestrationState`: State for orchestrator coordination

---

## Orchestrator

### Overview

The `orchestrator` crate manages distributed multi-pod benchmark experiments across Kubernetes clusters.

### Architecture

```
orchestrator/
├── src/
│   ├── main.rs          # Orchestrator entry point
│   ├── api.rs           # REST API endpoints
│   ├── controller.rs    # Experiment lifecycle management
│   ├── coordinator.rs   # Worker coordination
│   ├── k8s_client.rs    # Kubernetes API integration
│   ├── aggregator.rs    # Result aggregation
│   ├── storage.rs       # Object storage integration
│   └── scheduler.rs     # Experiment scheduling
```

### Entry Point: `main.rs`

**Binary**: `qr-orchestrator`

**Command-Line Arguments**:
- `--listen-addr <addr>`: HTTP API listen address (default: `0.0.0.0:7070`)
- `--namespace <ns>`: Kubernetes namespace (default: `default`)
- `--worker-image <image>`: Worker container image (default: `localhost/pqc-bench:latest`)
- `--storage-uri <uri>`: Storage backend URI (S3/GCS)
- `--local-results-dir <dir>`: Local results directory (default: `/tmp/qr-orchestrator`)
- `--max-time-drift-ns <ns>`: Maximum allowed time drift (default: 5000000)

**Execution Flow**:

1. Initialize tracing
2. Parse command-line arguments
3. Create results directory
4. Initialize Kubernetes client
5. Create experiment controller
6. Create experiment scheduler
7. Build API router
8. Start HTTP API server

### REST API: `api.rs`

**Endpoints**:

**Health & Readiness**:
- `GET /healthz`: Health check endpoint
- `GET /readyz`: Readiness check endpoint (includes experiment and schedule counts)
- `GET /metrics`: Metrics endpoint

**Experiment Management**:
- `POST /experiment`: Create new experiment
- `GET /experiments`: List all experiments
- `GET /experiment/{id}`: Get specific experiment
- `GET /experiment/{id}/status`: Get experiment status
- `POST /experiment/{id}/start`: Start experiment
- `POST /experiment/{id}/stop`: Stop experiment
- `POST /experiment/{id}/collect`: Collect and aggregate results
- `DELETE /experiment/{id}`: Delete experiment

**Worker Registration** (internal, called by workers):
- `POST /experiment/{id}/register`: Register worker with orchestrator
- `POST /experiment/{id}/ready`: Mark worker as ready
- `POST /experiment/{id}/completed`: Mark worker as completed

**Scheduling**:
- `POST /schedule`: Create new schedule (cron-based)
- `GET /schedules`: List all schedules
- `GET /schedule/{name}`: Get specific schedule
- `DELETE /schedule/{name}`: Delete schedule
- `POST /schedule/{name}/enable`: Enable schedule
- `POST /schedule/{name}/disable`: Disable schedule

### Experiment Controller: `controller.rs`

**Purpose**: Manages experiment lifecycle.

**Functions**:

- Create experiment: Spawn worker pods via Kubernetes Jobs
- Monitor experiment: Track worker status
- Collect results: Aggregate results from all workers
- Cleanup: Delete experiment resources

### Worker Coordinator: `coordinator.rs`

**Purpose**: Coordinates worker synchronization.

**Functions**:

- Register worker: Assign worker ID
- Time synchronization: Check time drift between workers
- Start signal: Broadcast global start time
- Barrier: Wait for all workers to be ready

### Kubernetes Client: `k8s_client.rs`

**Purpose**: Kubernetes API integration.

**Functions**:

- Create Job: Spawn worker pods
- Get Job status: Check worker status
- Delete Job: Cleanup worker pods
- Get Pod logs: Retrieve worker logs

### Result Aggregator: `aggregator.rs`

**Purpose**: Aggregate results from multiple workers.

**Functions**:

- Merge JSONL files: Combine results from all workers
- Compute statistics: Aggregate statistics
- Generate summary: Create experiment summary

### Storage: `storage.rs`

**Purpose**: Object storage integration (S3/GCS).

**Functions**:

- Upload results: Upload results to object storage
- Download results: Download results from object storage

### Scheduler: `scheduler.rs`

**Purpose**: Experiment scheduling (cron-based).

**Functions**:

- Schedule experiment: Schedule experiment with cron expression
- List schedules: List all scheduled experiments
- Cancel schedule: Cancel scheduled experiment

---

## Analysis Suite

### Overview

The `analysis/` directory contains Python-based analysis tools for processing benchmark results and generating publication-quality figures.

### Structure

```
analysis/
├── scripts/              # CLI analysis tools
├── notebooks/            # Jupyter notebooks
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Python project configuration
└── README.md             # Analysis suite documentation
```

### Key Scripts

**Core Analysis Scripts** (`analysis/`):
1. **`aggregate_results.py`**: Aggregate statistics across experiments, compute mean/std/CI for p50/p95/p99, effect sizes, environment deltas
2. **`aggregate_runs.py`**: Aggregate multiple runs of same experiment
3. **`build_final_report.py`**: Generate dissertation-ready PDF reports with executive summary, figures, tables, and interpretive paragraphs
4. **`compare_all_environments.py`**: Cross-environment comparison (native vs minikube vs GCP)
5. **`compare_native_vs_minikube.py`**: Native vs Minikube comparison
6. **`hypothesis_tests.py`**: Statistical hypothesis testing (KS test, Mann-Whitney U, Welch's t-test, Cohen's d, Holm-Bonferroni correction)
7. **`plot_combined_cdfs.py`**: Combined CDF plots for all algorithms
8. **`plot_scaling_curves.py`**: Scaling curve visualization (throughput/latency vs replica count)
9. **`plot_effect_size_forest.py`**: Effect size forest plots with confidence intervals
10. **`plot_payload_scaling_loglog.py`**: Payload scaling analysis (log-log plots)
11. **`plot_pqc_vs_classical_distribution.py`**: PQC vs classical distribution comparisons
12. **`plot_replica_scaling.py`**: Replica scaling analysis

**Scripts in `analysis/scripts/`**:
1. **`compute_statistics.py`**: Generate summary statistics from JSONL/Parquet data (percentiles, mean, std, plots)
2. **`merge_jsonl.py`**: Merge multiple JSONL files into single sorted file
3. **`compute_stats.py`**: Alternative statistics computation script
4. **`plot_ecdf.py`**: Generate ECDF plots
5. **`plot_latency.py`**: Generate latency plots
6. **`plot_throughput.py`**: Generate throughput plots
7. **`plot_queue_delay.py`**: Generate queue delay plots
8. **`effect_sizes.py`**: Compute effect sizes (Cohen's d, etc.)
9. **`fetch_results.py`**: Fetch results from GCS/S3/local storage
10. **`export_dataset.py`**: Export dataset for dissertation (CSV, Parquet, Markdown tables)

### Analysis Pipeline

1. **Data Collection**: Raw JSONL files from experiments
2. **Merge**: Merge JSONL files from multiple runs
3. **Statistics**: Compute statistical summaries
4. **Visualization**: Generate publication-quality figures
5. **Hypothesis Testing**: Statistical significance tests
6. **Reporting**: Generate dissertation-ready reports

---

## Orchestration System

### Overview

The `orchestration/` directory contains tools for declarative experiment configuration and scenario generation.

### Experiment Matrix: `experiment_matrix.yaml`

**Purpose**: Declarative definition of all experiments.

**Structure**:

- `defaults`: Global defaults for all experiments
- `scaling`: Scaling experiment configuration
- `environments`: Environment configurations (native, minikube, GCP)
- `experiments`: Experiment definitions
- `comparisons`: Comparison groups for analysis
- `output`: Output configuration

**Experiment Definition**:

```yaml
- algorithm: kyber512
  description: "Kyber-512 KEM + AES-GCM encryption"
  adapter: kyber
  operation: kem_aead_encrypt
  payload_sizes: [256, 1024, 4096, 16384]
  rates: [100, 500, 2000]
  runs: 5
  category: pqc
```

### Scenario Generator: `generate_scenarios.py`

**Purpose**: Generate scenario YAML files from experiment matrix.

**Functions**:

- Read experiment matrix
- Generate scenarios for each experiment configuration
- Write scenario YAML files to `generated-scenarios/`

---

## Scenario Format

### YAML Structure

```yaml
id: scenario_id
description: "Scenario description"
rng_seed: 1234  # Optional: RNG seed for reproducibility

workload:
  msgs_per_sec: 500
  msg_size_bytes: 1024
  duration_sec: 30
  pattern: constant  # constant, burst, ramp, trace
  burst:  # Optional: if pattern is burst
    factor: 5
    duration_ms: 5000
    interval_ms: 30000
  ramp:  # Optional: if pattern is ramp
    start_rate: 100
    end_rate: 2000
    duration_ms: 10000
  trace_file: "path/to/trace.csv"  # Optional: if pattern is trace

algorithm:
  adapter: kyber  # noop, rsa2048, ecdsa_p256, ecdhe_p256, kyber, dilithium
  operation: kem_aead_encrypt  # sign, verify, keygen, encapsulate, decapsulate, kem_aead_encrypt, kem_aead_decrypt

execution:
  mode: fixed_pool  # single, fixed_pool, elastic
  workers: 4  # For fixed_pool
  max_workers: 8  # For elastic
  queue_capacity: 2000

metrics:
  prometheus_endpoint: "0.0.0.0:9898"
  jsonl_out: "./results/scenario.jsonl"
```

### Supported Adapters

- `noop`: NoOp baseline
- `rsa2048`: RSA-2048
- `ecdsa_p256`: ECDSA P-256
- `ecdh_p256`: ECDH P-256
- `kyber`: Kyber-512
- `dilithium`: Dilithium-2

### Supported Operations

The following operations are supported (8 total):

- `sign`: Digital signature generation
- `verify`: Digital signature verification
- `keygen`: Key pair generation
- `encrypt`: KEM encapsulation (alias for encapsulate)
- `decrypt`: KEM decapsulation (alias for decapsulate)
- `encapsulate`: KEM encapsulation
- `decapsulate`: KEM decapsulation
- `kem_aead_encrypt`: Hybrid KEM + AEAD encryption
- `kem_aead_decrypt`: Hybrid KEM + AEAD decryption
- `kem_aead_sign`: Hybrid KEM + AEAD + signature (Kyber KEM + AES-GCM + Dilithium sign)

**Note**: `encrypt`/`decrypt` are aliases for `encapsulate`/`decapsulate` for compatibility.

### Execution Modes

- `single`: Single processor task
- `fixed_pool`: Fixed number of workers
- `elastic`: Dynamic worker pool

---

## Workflow Overview

### Complete Workflow

1. **Scenario Generation**: Generate scenarios from experiment matrix
2. **Data Collection**: Run experiments across environments (native, minikube, GCP)
3. **Data Processing**: Merge JSONL files, compute statistics
4. **Analysis**: Generate visualizations, run hypothesis tests
5. **Reporting**: Generate dissertation-ready reports

### Data Flow

```
Experiment Matrix (YAML)
    ↓
Scenario Generator
    ↓
Scenario Files (YAML)
    ↓
Benchmark Execution (rust-core)
    ↓
Raw JSONL Files
    ↓
Merge & Statistics
    ↓
Aggregated Statistics
    ↓
Visualization & Analysis
    ↓
Dissertation-Ready Reports
```

---

**Last Updated**: 2025-12-15

