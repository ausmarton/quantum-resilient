# Detailed Methodology Documentation

**Last Updated**: 2025-12-15  
**Purpose**: Comprehensive methodology documentation for dissertation Chapter 3 (Methodology).

---

## Table of Contents

1. [Research Methodology Overview](#research-methodology-overview)
2. [Experimental Framework Architecture](#experimental-framework-architecture)
3. [Data Collection Methodology](#data-collection-methodology)
4. [Analysis Methodology](#analysis-methodology)
5. [Reproducibility and Validation](#reproducibility-and-validation)

---

## Research Methodology Overview

### Research Questions

1. **Performance Comparison**: How do post-quantum cryptographic algorithms compare to classical algorithms in terms of latency and throughput in real-time streaming applications?

2. **Scalability Analysis**: How do post-quantum algorithms scale horizontally compared to classical algorithms?

3. **Environment Impact**: What is the performance impact of different deployment environments (native, containerized, cloud) on cryptographic operations?

4. **Enterprise Representativeness**: How representative are the experimental workloads of real-world enterprise applications?

### Experimental Design

**Independent Variables**:
- Cryptographic algorithm (RSA-2048, ECDSA P-256, ECDH P-256, Kyber-512, Dilithium-2, Hybrid)
- Payload size (256B, 1KB, 4KB, 16KB)
- Workload rate (100, 500, 2000, 10000 msg/s)
- Workload pattern (constant, burst, ramp)
- Execution environment (native, minikube, GCP)
- Horizontal scaling (1, 2, 4, 8 replicas)

**Dependent Variables**:
- Latency (p50, p95, p99, mean, std)
- Throughput (ops/sec)
- Queue delay
- Resource utilization (CPU, memory)

**Control Variables**:
- RNG seed (deterministic for reproducibility)
- Hardware specifications (documented per environment)
- System load (minimized for native/minikube)

### Experimental Matrix

The experiment matrix (`orchestration/experiment_matrix.yaml`) defines:
- **459 baseline experiments**: All algorithm × payload × rate combinations
- **27 scaling experiments**: Horizontal scaling tests (1, 2, 4, 8 replicas)
- **5 runs per experiment**: For statistical rigor (3 runs for scaling and 5-minute duration experiments)

**Total Experiments**: 486 experiments × 5 runs = 2,430 individual runs

---

## Experimental Framework Architecture

### Three-Layer Architecture

#### 1. Configuration Layer

**Purpose**: Declarative experiment definition and scenario generation.

**Components**:
- **Experiment Matrix** (`experiment_matrix.yaml`): Declarative YAML definition of all experiments
- **Scenario Generator** (`generate_scenarios.py`): Python script that generates individual scenario YAML files
- **Deterministic RNG**: RNG seed computed from experiment parameters for reproducibility

**Output**: Individual scenario YAML files in `generated-scenarios/`

#### 2. Execution Layer

**Purpose**: Benchmark execution across multiple environments.

**Components**:

**Core Benchmark Framework** (`rust-core/`):
- **Pipeline**: Async streaming pipeline for event processing
- **Execution Modes**: Single, FixedPool, Elastic
- **Workload Generator**: Constant, Burst, Ramp, Trace patterns
- **Cryptographic Adapters**: Unified interface for PQC and classical algorithms
- **Telemetry Collection**: Nanosecond-precision timing, system resource monitoring

**Environments**:
- **Native**: Bare-metal execution on local machine
- **Minikube**: Local Kubernetes cluster (containerized)
- **GCP**: Google Cloud Platform GKE (cloud deployment)

**Orchestrator** (`orchestrator/`):
- **Worker Coordination**: Synchronizes distributed experiments
- **Result Aggregation**: Combines results from multiple workers
- **Storage Integration**: Uploads results to object storage (S3/GCS)

**Output**: Raw JSONL event logs with full telemetry

#### 3. Analysis Layer

**Purpose**: Statistical analysis and visualization.

**Components**:
- **Data Aggregation**: Merge and aggregate statistics across experiments
- **Hypothesis Testing**: Statistical significance tests (t-test, Mann-Whitney U)
- **Effect Size Computation**: Cohen's d, confidence intervals
- **Visualization**: CDF plots, scaling curves, comparison charts
- **Reporting**: Dissertation-ready reports and tables

**Output**: Aggregated statistics, hypothesis test results, publication-quality figures

### Framework Components

#### Cryptographic Adapters

**Unified Interface**: All adapters implement the `CryptoAdapter` trait, providing:
- `sign()`: Digital signature generation
- `verify()`: Digital signature verification
- `keygen()`: Key pair generation
- `encapsulate()`: KEM encapsulation
- `decapsulate()`: KEM decapsulation
- `kem_aead_encrypt()`: Hybrid KEM + AEAD encryption
- `kem_aead_decrypt()`: Hybrid KEM + AEAD decryption

**Supported Algorithms**:
- **Classical**: RSA-2048, ECDSA P-256, ECDH P-256
- **Post-Quantum**: Kyber-512, Dilithium-2
- **Hybrid**: Kyber-512 + AES-256-GCM + Dilithium-2

#### Execution Modes

1. **Single**: Single processor task (sequential processing)
2. **FixedPool**: Fixed number of parallel workers
3. **Elastic**: Dynamic worker pool that expands/contracts based on queue pressure

#### Workload Patterns

1. **Constant**: Constant rate of operations
2. **Burst**: Periodic burst spikes (simulates enterprise load patterns)
3. **Ramp**: Gradually ramping load
4. **Trace**: Trace-driven replay from CSV file

#### Telemetry Collection

**Precision**: Nanosecond-precision timing using `std::time::Instant`

**Metrics Collected**:
- **Per-Event**: Event ID, timestamp (UTC ISO, monotonic), operation, algorithm, latency (ns), payload size, ciphertext size, CPU usage, memory usage, RNG seed, error status
- **System-Level**: CPU usage, memory usage (via `sysinfo` crate)
- **Prometheus Metrics**: Latency histograms, operation counters, queue length, active workers

**Output Format**: JSONL (JSON Lines) for efficient streaming and processing

---

## Data Collection Methodology

### Environment Setup

#### Native Environment

**Hardware**: Documented per run (CPU, memory, OS version)
**System Load**: Minimized (close heavy applications, check system load before runs)
**Multiple Runs**: 5 runs per configuration to account for residual variability

#### Minikube Environment

**Setup**: Local Kubernetes cluster using Minikube with Podman
**Containerization**: Multi-stage container build for consistent execution
**Hardware**: Same as native (shared hardware)
**Multiple Runs**: 5 runs per configuration

#### GCP Environment

**Infrastructure**: GKE cluster provisioned via Terraform
**Ephemeral Mode**: Automatically creates and destroys cluster (zero ongoing cost)
**Hardware**: n2-standard-4 VMs (4 vCPUs, 16GB RAM)
**Isolation**: Isolated cloud VMs (no local system load impact)
**Multiple Runs**: 5 runs per configuration

### Data Collection Process

1. **Scenario Generation**: Generate scenarios from experiment matrix
2. **Environment Execution**: Run experiments in each environment
3. **Data Validation**: Validate data collection completeness
4. **Index Generation**: Generate experiment index for analysis

### Data Structure

**Per Experiment**:
```
results/<env>/<experiment-id>/
├── run-1/
│   ├── raw/
│   │   └── run.jsonl          # Raw event logs
│   ├── merged/
│   │   ├── merged.jsonl        # Merged events
│   │   └── merged.parquet     # Parquet format
│   └── stats/
│       └── summary.json        # Statistical summary
├── run-2/
├── run-3/
├── run-4/
├── run-5/
└── manifest.json               # Experiment metadata
```

**Event Format** (JSONL):
```json
{
  "run_id": "experiment_id",
  "scenario_id": "scenario_id",
  "event_id": 1,
  "timestamp_utc_iso": "2025-12-15T12:34:56.123456Z",
  "timestamp_monotonic_ns": 123456789012345,
  "operation": "kem_aead_encrypt",
  "algorithm": "kyber",
  "latency_us": 512,
  "payload_size_bytes": 1024,
  "ciphertext_size_bytes": 1200,
  "cpu_user_seconds": 0.000123,
  "memory_rss_bytes": 3456784,
  "rng_seed": 1234,
  "error": null
}
```

---

## Analysis Methodology

### Statistical Analysis

#### Summary Statistics

For each experiment, compute:
- **Latency**: p50, p90, p95, p99, p99.9, mean, std, min, max
- **Throughput**: Mean ops/sec, peak ops/sec
- **Queue Delay**: p50, p95, p99, mean, std
- **Resource Utilization**: Mean CPU, mean memory

#### Aggregation Across Runs

For each experiment configuration:
- **Mean Statistics**: Average across 5 runs
- **Standard Deviation**: Variability across runs
- **Confidence Intervals**: 95% CI for key metrics
- **Coefficient of Variation**: CV = std/mean (stability metric)

#### Hypothesis Testing

**Tests Performed**:
1. **Mann-Whitney U Test**: Non-parametric test for distribution differences
2. **Kolmogorov-Smirnov Test**: Distribution shape similarity
3. **Welch's t-test**: Mean differences (unequal variances)
4. **Holm-Bonferroni Correction**: Controls family-wise error rate

**Comparisons**:
- PQC vs Classical (signatures, encryption/KEM)
- Hybrid vs Pure PQC
- Environment comparisons (native vs minikube vs GCP)
- Scaling behavior (1 vs 2 vs 4 vs 8 replicas)

#### Effect Size Computation

**Metrics**:
- **Cohen's d**: Standardized mean difference
- **Hedge's g**: Bias-corrected Cohen's d
- **Glass's Δ**: Uses control group std
- **Cliff's δ**: Non-parametric effect size
- **Wasserstein Distance**: Earth mover's distance
- **KS Statistic**: Distribution distance with p-value

**Interpretation**:
- |d| < 0.2: Negligible
- |d| < 0.5: Small
- |d| < 0.8: Medium
- |d| ≥ 0.8: Large

### Visualization

#### CDF Plots

**Purpose**: Compare latency distributions across algorithms and environments

**Types**:
- Combined CDFs: All algorithms on one plot
- Environment comparison: Native vs Minikube vs GCP
- Algorithm comparison: PQC vs Classical

#### Scaling Curves

**Purpose**: Analyze horizontal scaling behavior

**Metrics**: Throughput and latency vs replica count

#### Effect Size Forest Plots

**Purpose**: Visualize effect sizes with confidence intervals

### Reporting

#### Dissertation-Ready Outputs

1. **Statistical Tables**: LaTeX tables with aggregated statistics
2. **Figures**: Publication-quality figures (300 DPI PNG, vector PDF/EPS)
3. **Hypothesis Test Results**: Statistical significance results
4. **Effect Size Summary**: Effect size metrics with interpretations
5. **Final Report**: Comprehensive analysis report

---

## Reproducibility and Validation

### Reproducibility Measures

1. **Deterministic RNG**: RNG seed computed from experiment parameters
2. **Version Control**: All code and configurations in Git
3. **Containerization**: Consistent execution environments
4. **Documentation**: Complete documentation of all steps
5. **Provenance**: Full experiment provenance metadata

### Validation

#### Data Quality Validation

- **Completeness**: All expected experiments collected
- **Integrity**: JSONL files valid, statistics computed correctly
- **Consistency**: Hardware metadata consistent across runs

#### Statistical Validation

- **Stability**: Coefficient of variation < 15% for key metrics
- **Reproducibility**: Multiple runs show consistent results
- **Outlier Detection**: Identify and investigate outliers

#### Environment Validation

- **Hardware Consistency**: Document hardware specifications
- **System Load**: Minimize system load for native/minikube
- **Isolation**: GCP provides isolated execution environment

### Validation Scripts

- `validate_data_collection.sh`: Validate data collection completeness
- `validate_data_integrity.sh`: Validate data integrity
- `validate_dissertation_data.sh`: Validate dissertation data requirements
- `verify_experiments.sh`: Verify experiment results

---

## Methodology Summary

### Key Strengths

1. **Comprehensive Coverage**: 459 baseline experiments + 27 scaling experiments
2. **Statistical Rigor**: 5 runs per experiment, hypothesis testing, effect sizes
3. **Multiple Environments**: Native, containerized, cloud deployment
4. **Enterprise Representativeness**: Burst patterns, high rates (10K msg/s), sustained load (5-min)
5. **Reproducibility**: Deterministic RNG, version control, containerization
6. **High Precision**: Nanosecond-precision timing, comprehensive telemetry

### Limitations

1. **Hardware Variability**: Native and Minikube share hardware (residual variability)
2. **System Load**: Native and Minikube subject to local system load
3. **Cost Constraints**: GCP experiments limited by cloud costs
4. **Time Constraints**: Full data collection takes ~15-20 hours

### Mitigation Strategies

1. **Multiple Runs**: 5 runs account for residual variability
2. **System Load Checks**: Automatic system load checks before native/minikube runs
3. **Ephemeral GCP**: Auto-destroy GCP resources to minimize costs
4. **Parallel Execution**: Run environments in parallel when possible

---

**Last Updated**: 2025-12-15

