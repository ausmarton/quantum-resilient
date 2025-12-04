# Implementation Guide for Researchers

This guide provides a concise overview for researchers seeking to replicate, validate, or extend the PQC Performance Benchmarking Framework.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Architecture Summary](#architecture-summary)
3. [Key Implementation Details](#key-implementation-details)
4. [Replication Steps](#replication-steps)
5. [Validation Checklist](#validation-checklist)
6. [Extension Points](#extension-points)

---

## Quick Start

### Prerequisites
```bash
# System requirements
- Linux (for /proc resource sampling)
- Rust 1.70+ (cargo, rustc)
- Python 3.11+ (pip, venv)
- Docker (optional, for containerized runs)
- Kubernetes (optional, for distributed execution)
```

### Installation
```bash
# 1. Clone repository
git clone https://github.com/your-org/quantum-resilient.git
cd quantum-resilient

# 2. Build Rust core
cd src/rust_core
cargo build --release

# 3. Install Python orchestrator
cd ../python_orchestrator
pip install -e .

# 4. Verify installation
pqc-orchestrator --help
```

### Run Benchmark
```bash
# Local execution
pqc-orchestrator --config ./configs/default.yaml

# Results appear in ./results/
ls -lh results/
```

---

## Architecture Summary

### Two-Layer Design

**Layer 1: Rust Core (Performance-Critical)**
- Implements cryptographic algorithms via `CryptoAdapter` trait
- Provides deterministic workload generation (seeded RNG)
- Captures microsecond-precision latency measurements
- Samples system resources (CPU, memory, I/O) via `getrusage()` and `/proc`
- Emits structured metrics to JSONL and Prometheus

**Layer 2: Python Orchestrator (Experiment Management)**
- Loads experiment configuration from YAML
- Invokes Rust core via PyO3 bindings
- Aggregates JSONL → CSV for analysis
- Performs statistical tests (t-test, Mann-Whitney U, Cohen's d)
- Generates visualizations (CDF, boxplot, bar charts)
- Packages complete report (ZIP archive with notebook)

### Data Flow
```
YAML Config → Python Orchestrator → Rust Core (PyO3) →
  → Crypto Adapters (Kyber, Dilithium, RSA, etc.) →
  → Instrumented Execution (timers + resource sampling) →
  → Metrics Collectors (JSONL + Prometheus) →
  → Python Aggregation (JSONL → CSV) →
  → Statistical Analysis (scipy, pandas) →
  → Visualization (matplotlib, seaborn) →
  → Report Package (report.zip)
```

---

## Key Implementation Details

### 1. Algorithm Adapters

Each algorithm implements the `CryptoAdapter` trait:

```rust
pub trait CryptoAdapter: Send + Sync {
    fn name(&self) -> &str;
    fn public_key_size(&self) -> usize;
    fn secret_key_size(&self) -> usize;
    fn signature_size(&self) -> usize;
    
    fn keygen(&self) -> CryptoResult<(Vec<u8>, Vec<u8>)>;
    fn encapsulate(&self, public_key: &[u8]) -> CryptoResult<(Vec<u8>, Vec<u8>)>;
    fn decapsulate(&self, secret_key: &[u8], ciphertext: &[u8]) -> CryptoResult<Vec<u8>>;
    fn sign(&self, secret_key: &[u8], message: &[u8]) -> CryptoResult<Vec<u8>>;
    fn verify(&self, public_key: &[u8], message: &[u8], signature: &[u8]) -> CryptoResult<()>;
}
```

**Implemented Algorithms**:
- **PQC**: Kyber512, Kyber768, Dilithium2, Dilithium3
- **Classical**: RSA-2048, ECDSA-P256, ECDHE-P256
- **Symmetric**: AES-GCM-256

**Note**: Current implementations use placeholder/simplified cryptography for deterministic benchmarking. For production use, integrate `liboqs` or equivalent libraries.

### 2. Instrumentation

The `InstrumentedAdapter` wrapper automatically captures metrics for any `CryptoAdapter`:

```rust
pub struct InstrumentedAdapter<A: CryptoAdapter> {
    inner: Box<A>,
    collector: Arc<dyn MetricsCollector>,
}

impl<A: CryptoAdapter> InstrumentedAdapter<A> {
    fn with_metrics<R>(
        &self,
        operation: OperationKind,
        f: impl FnOnce(&A) -> CryptoResult<R>,
    ) -> CryptoResult<R> {
        let start = Instant::now();
        let result = f(&self.inner);
        let elapsed = start.elapsed();
        
        // Sample resources: CPU, memory, disk I/O, network I/O
        let (cpu_user, cpu_system, rss, disk, net_tx, net_rx) = sample_resources();
        
        // Build metrics struct
        let metrics = OperationMetrics {
            timestamp_seconds_utc: Some(chrono::Utc::now()),
            operation,
            latency_micros: elapsed.as_micros() as u64,
            cpu_user_micros: cpu_user,
            cpu_system_micros: cpu_system,
            max_rss_bytes: rss,
            algorithm: Some(self.inner.name().to_string()),
            // ... (other fields)
        };
        
        self.collector.record(&metrics);
        result
    }
}
```

### 3. Resource Sampling

System resource sampling uses POSIX `getrusage()` and Linux `/proc` filesystem:

```rust
fn sample_resources() -> (Option<u64>, Option<u64>, Option<u64>, Option<u64>, Option<u64>, Option<u64>) {
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
            let user_us = (usage.ru_utime.tv_sec as u64) * 1_000_000 
                        + (usage.ru_utime.tv_usec as u64);
            let sys_us = (usage.ru_stime.tv_sec as u64) * 1_000_000 
                       + (usage.ru_stime.tv_usec as u64);
            let rss_bytes = (usage.ru_maxrss as u64) * 1024; // kilobytes → bytes
            
            // Read /proc/self/io for disk I/O
            let disk = read_proc_self_io_bytes();
            
            // Read /proc/net/dev for network I/O
            let (net_rx, net_tx) = read_proc_net_dev_bytes();
            
            (Some(user_us), Some(sys_us), Some(rss_bytes), disk, net_tx, net_rx)
        } else {
            (None, None, None, None, None, None)
        }
    }
}
```

**Portability Note**: Resource sampling is Linux-specific. For other platforms:
- macOS: `getrusage()` works but `/proc` unavailable (use `sysctl` or `task_info`)
- Windows: Use `GetProcessTimes()` and `GetProcessMemoryInfo()`

### 4. Metrics Schema

Each operation emits a JSON record with the following structure:

```json
{
  "timestamp_seconds_utc": "2024-11-10T12:34:56Z",
  "operation": "Keygen",
  "latency_micros": 2533,
  "algorithm": "Kyber512",
  "cpu_user_micros": 320,
  "cpu_system_micros": 487,
  "max_rss_bytes": 2162688,
  "public_key_bytes": 800,
  "secret_key_bytes": 1632,
  "signature_bytes": 0,
  "throughput_ops_per_sec": 394710.3,
  "disk_io_bytes": 0,
  "net_tx_bytes": 0,
  "net_rx_bytes": 0
}
```

Schema validation is performed against `configs/metrics_schema.yaml`.

### 5. Statistical Analysis

Python orchestrator performs:

**Descriptive Statistics**:
- Mean, median, standard deviation
- Percentiles (50th, 95th, 99th)
- Min, max, range

**Inferential Statistics**:
- Independent samples t-test (parametric)
- Mann-Whitney U test (non-parametric)
- Cohen's d effect size
- 95% confidence intervals (t-distribution)

**Sample Size Justification**:
- n=30-60 per operation (varies by operation type)
- Provides >80% statistical power for detecting medium effects (d ≥ 0.5) at α=0.05

**Implementation** (see `src/python_orchestrator/python_orchestrator/statistical_tests.py`):
```python
from scipy import stats
import numpy as np

def compare_algorithms(group_a, group_b):
    # Parametric test
    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    
    # Non-parametric test
    u_stat, p_value_mw = stats.mannwhitneyu(group_a, group_b)
    
    # Effect size
    cohens_d = (np.mean(group_a) - np.mean(group_b)) / np.sqrt(
        (np.std(group_a, ddof=1) ** 2 + np.std(group_b, ddof=1) ** 2) / 2
    )
    
    # Confidence interval
    ci_95 = stats.t.interval(0.95, len(group_a) - 1, 
                              loc=np.mean(group_a), 
                              scale=stats.sem(group_a))
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'mann_whitney_u': u_stat,
        'p_value_mw': p_value_mw,
        'cohens_d': cohens_d,
        'ci_95': ci_95
    }
```

### 6. Visualization

Generated charts:
- **CDF (Cumulative Distribution Function)**: Latency distributions per algorithm
- **Boxplot**: Throughput comparison across algorithms
- **Bar Chart**: Mean CPU/memory usage per algorithm

**Libraries**: matplotlib, seaborn

---

## Replication Steps

### Step 1: Environment Setup

```bash
# Clone repository
git clone https://github.com/your-org/quantum-resilient.git
cd quantum-resilient

# Verify system
uname -a  # Should show Linux
python3 --version  # Should be 3.11+
rustc --version    # Should be 1.70+
```

### Step 2: Configuration

Edit `configs/default.yaml`:

```yaml
algorithms:
  pqc:
    - Kyber512
    - Kyber768
    - Dilithium2
    - Dilithium3
  classical:
    key_exchange:
      - ECDHE-P256
      - RSA-2048
    signature:
      - ECDSA-P256
  symmetric:
    - AES-GCM-256

repetitions: 30  # Number of repetitions per operation
warmup_seconds: 5  # CPU cache warmup

workload:
  tps: 1000  # Transactions per second
  duration_seconds: 60
  chunk_size_bytes: 1024
  backpressure: "Block"  # or "Drop"

output:
  directory: "./results"
  formats:
    - jsonl
    - csv
    - summary
```

### Step 3: Build and Install

```bash
# Build Rust core
cd src/rust_core
cargo build --release
cd ../..

# Install Python orchestrator (editable mode for development)
pip install -e src/python_orchestrator

# Verify installation
pqc-orchestrator --help
```

### Step 4: Run Benchmark

```bash
# Execute benchmark
pqc-orchestrator --config ./configs/default.yaml

# Monitor progress (if Prometheus enabled)
curl http://localhost:9100/metrics

# Check outputs
ls -lh results/
# Expected files:
# - metrics.jsonl (raw events)
# - metrics.csv (aggregated)
# - raw_events.csv (schema-aligned)
# - summary.csv/json/md (statistics)
# - charts/*.png (visualizations)
# - analysis.ipynb (notebook)
# - report.zip (complete archive)
# - environment.json (system snapshot)
# - metrics_validation.json (schema compliance)
```

### Step 5: Validate Results

```bash
# Check metrics schema compliance
cat results/metrics_validation.json

# Verify sample sizes
python3 << EOF
import pandas as pd
df = pd.read_csv('results/metrics.csv')
print(df.groupby(['algorithm', 'operation']).size())
EOF

# Inspect summary statistics
cat results/summary.md
```

---

## Validation Checklist

Use this checklist to validate framework replication:

- [ ] **Environment Snapshot**: `environment.json` contains CPU model, OS version, Python/Rust versions
- [ ] **Deterministic RNG**: Repeated runs with same config produce identical latency distributions (within statistical noise)
- [ ] **Sample Sizes**: Each algorithm+operation combination has n=30-60 samples
- [ ] **Schema Compliance**: `metrics_validation.json` shows 100% valid records
- [ ] **Statistical Power**: Cohen's d effect sizes match reported values (±0.1 tolerance)
- [ ] **Latency Measurements**: Kyber512 keygen mean ~2.5 µs, Kyber768 ~1.2 µs (±20% on different hardware)
- [ ] **Resource Sampling**: CPU/memory/I/O values are non-null on Linux
- [ ] **Visualization**: Charts generated successfully (8 PNG files in `results/charts/`)
- [ ] **Report Archive**: `report.zip` contains all artifacts and is <50 MB

---

## Extension Points

### Adding New Algorithms

1. **Implement CryptoAdapter**:
   ```rust
   // src/rust_core/src/adapters/my_algorithm.rs
   use crate::{CryptoAdapter, CryptoResult};
   
   pub struct MyAlgorithmAdapter;
   
   impl CryptoAdapter for MyAlgorithmAdapter {
       fn name(&self) -> &str { "MyAlgorithm" }
       // ... implement other methods
   }
   ```

2. **Register Adapter**:
   ```rust
   // src/rust_core/src/adapters/mod.rs
   pub mod my_algorithm;
   pub use my_algorithm::MyAlgorithmAdapter;
   ```

3. **Expose via PyO3**:
   ```rust
   // src/rust_core/src/pyo3_mod.rs
   #[pymodule]
   fn pqc_core(_py: Python, m: &PyModule) -> PyResult<()> {
       m.add_class::<MyAlgorithmAdapter>()?;
       // ...
   }
   ```

4. **Update Configuration**:
   ```yaml
   # configs/default.yaml
   algorithms:
     custom:
       - MyAlgorithm
   ```

### Adding New Metrics

1. **Extend OperationMetrics**:
   ```rust
   // src/rust_core/src/lib.rs
   #[derive(Clone, Debug, serde::Serialize)]
   pub struct OperationMetrics {
       // ... existing fields
       pub my_custom_metric: Option<f64>,
   }
   ```

2. **Populate in Instrumentation**:
   ```rust
   // src/rust_core/src/lib.rs
   impl<A: CryptoAdapter> InstrumentedAdapter<A> {
       fn with_metrics(...) -> CryptoResult<R> {
           // ... existing code
           let metrics = OperationMetrics {
               // ... existing fields
               my_custom_metric: Some(calculate_custom_metric()),
           };
       }
   }
   ```

3. **Update Schema**:
   ```yaml
   # configs/metrics_schema.yaml
   fields:
     my_custom_metric:
       type: number
       description: "My custom metric (units)"
   ```

### Adding New Visualizations

```python
# src/python_orchestrator/python_orchestrator/reporting.py

import matplotlib.pyplot as plt
import seaborn as sns

def generate_custom_chart(df, output_dir):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='algorithm', y='my_custom_metric')
    plt.title('Custom Metric Comparison')
    plt.savefig(f'{output_dir}/charts/custom_metric.png', dpi=300)
    plt.close()
```

### Deployment Options

**Docker Compose**:
```bash
docker-compose up
# Access Prometheus: http://localhost:9090
# Results in ./results (volume mount)
```

**Kubernetes (Local)**:
```bash
bash scripts/run_local_k8s.sh --config ./configs/default.yaml
# Results copied to ./results/local_k8s_<timestamp>/
```

**GCP GKE**:
```bash
bash scripts/run_gcp.sh \
  --config ./configs/default.yaml \
  --project my-gcp-project \
  --region us-central1 \
  --cluster pqc-benchmark \
  --bucket gs://my-results-bucket
# Results downloaded to ./results/gcp_<cluster>_<timestamp>/
```

---

## Troubleshooting

### Issue: `No module named 'pqc_core'`

**Cause**: PyO3 bindings not built or not installed.

**Solution**:
```bash
cd src/rust_core
cargo build --release
cp target/release/librust_core.so ../python_orchestrator/python_orchestrator/pqc_core.so
# Or use maturin:
pip install maturin
maturin develop --release
```

### Issue: `Permission denied` accessing `/proc/self/io`

**Cause**: Some Linux distributions restrict `/proc/self/io` to root.

**Solution**: Run with elevated privileges or disable disk I/O sampling.

### Issue: Latency measurements all zero

**Cause**: Operations complete faster than microsecond resolution.

**Solution**: Use higher-precision timing (nanoseconds) or increase workload complexity.

---

## References

- **NIST PQC Standards**: https://www.nist.gov/pqcrypto
- **PyO3 Documentation**: https://pyo3.rs/
- **Prometheus Metrics**: https://prometheus.io/docs/concepts/metric_types/
- **Statistical Power Analysis**: Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
- **Framework Diagrams**: See `docs/framework_diagrams.md`

---

## Contact and Support

For questions regarding framework replication:
- Open an issue on GitHub: https://github.com/your-org/quantum-resilient/issues
- Email: research-support@example.com

---

**Last Updated**: November 10, 2024  
**Framework Version**: 1.0  
**License**: See LICENSE file

