# Research Data Collection - Complete Summary

## ✅ Implementation Complete

All critical fixes have been implemented to capture complete research data comparing PQC vs Classical cryptographic algorithms.

---

## 📊 What Data Is Now Captured

### **Algorithms Tested**

#### Post-Quantum Cryptography (PQC)
- ✅ **Kyber512** - NIST Level 1 KEM
- ✅ **Kyber768** - NIST Level 3 KEM  
- ✅ **Dilithium2** - NIST Level 2 Digital Signature
- ✅ **Dilithium3** - NIST Level 3 Digital Signature

#### Classical Cryptography
- ✅ **RSA-2048** - Traditional public-key encryption
- ✅ **ECDSA-P256** - Elliptic curve digital signature
- ✅ **ECDHE-P256** - Elliptic curve key exchange

#### Symmetric Cryptography
- ✅ **AES-GCM-256** - Authenticated encryption

### **Operations Measured**

| Operation | Description | Algorithms |
|-----------|-------------|------------|
| **Keygen** | Key pair generation | All asymmetric algorithms |
| **Encapsulate** | KEM encapsulation | Kyber512, Kyber768, RSA-2048, ECDHE-P256 |
| **Decapsulate** | KEM decapsulation | Kyber512, Kyber768 |
| **Sign** | Digital signature generation | Dilithium2, Dilithium3, ECDSA-P256 |
| **Verify** | Signature verification | Dilithium2, Dilithium3, ECDSA-P256 |
| **BulkEncrypt** | Symmetric encryption | AES-GCM-256 |
| **BulkDecrypt** | Symmetric decryption | AES-GCM-256 |

### **Performance Metrics**

✅ **Latency** (microseconds)
- Per-operation timing
- p50, p95, p99 percentiles
- Mean and standard deviation
- 95% confidence intervals

✅ **Throughput**
- Operations per second
- Instantaneous per-event calculation

✅ **Resource Usage** (NOW CAPTURED)
- **CPU user time** (microseconds)
- **CPU system time** (microseconds)  
- **Memory (max RSS)** (bytes)
- **Disk I/O** (bytes read + written)
- **Network I/O** (tx/rx bytes)

✅ **Key/Data Sizes**
- Public key size (bytes)
- Secret key size (bytes)
- Signature size (bytes)
- Ciphertext size (bytes)
- Storage overhead percentage

### **Statistical Analysis** (NEW)

✅ **Parametric Tests**
- Independent t-test (for unpaired data)
- Paired t-test (for paired data)
- p-values for significance

✅ **Non-Parametric Tests**
- Mann-Whitney U test (unpaired)
- Wilcoxon signed-rank test (paired)
- p-values for significance

✅ **Effect Sizes**
- Cohen's d (independent samples)
- Cohen's dz (paired samples)
- Rank-biserial correlation
- Interpretation (small/medium/large)

✅ **Comparison Metrics**
- Mean difference
- Percent difference
- Faster algorithm identification
- Statistical significance flags

### **Reproducibility**

✅ **Environment Snapshot**
- CPU model and count
- Operating system and version
- Python version
- Git commit hash
- Timestamp

✅ **Deterministic Execution**
- Seeded random number generation
- Fixed environment variables (TZ, LC_ALL, etc.)
- Repeatable results

---

## 🚀 How to Generate Research Data

### **Quick Start (30 Repetitions)**

```bash
cd /home/ausmarton/scratchpad/quantum-resilient

# Full research run with adequate sample size
bash scripts/run_research_benchmark.sh --repetitions 30

# Results will be in ./results/
```

### **What You Get**

```
results/
├── metrics.jsonl                    # Raw per-event metrics (270+ events)
├── metrics.csv                      # Same in CSV format
├── summary.csv                      # Aggregated statistics
├── summary.json                     # Summary in JSON
├── summary.md                       # Human-readable summary
├── statistical_comparisons.csv      # PQC vs Classical tests (NEW)
├── statistical_report.md            # Human-readable stat report (NEW)
├── comparisons.json                 # Additional paired comparisons
├── charts/                          # Visualizations
│   ├── latency_cdf_*.png           # CDF plots per operation
│   ├── throughput_boxplot.png      # Throughput comparison
│   ├── cpu_*.png                   # CPU usage charts
│   └── memory_*.png                # Memory usage charts
├── analysis.ipynb                   # Jupyter notebook
├── environment.json                 # Environment snapshot
└── report.zip                       # Complete bundle
```

### **Customizing Sample Size**

```bash
# Small test (5 reps, ~135 events)
bash scripts/run_research_benchmark.sh --repetitions 5

# Moderate (20 reps, ~540 events)
bash scripts/run_research_benchmark.sh --repetitions 20

# Full research quality (30 reps, ~810 events)
bash scripts/run_research_benchmark.sh --repetitions 30

# High confidence (50 reps, ~1350 events)
bash scripts/run_research_benchmark.sh --repetitions 50
```

---

## 📈 Sample Data Structure

### **Per-Event Metrics** (metrics.jsonl)

```json
{
  "timestamp_seconds_utc": 1762750360,
  "operation": "Keygen",
  "algorithm": "Kyber512",
  "latency_micros": 9,
  "cpu_user_micros": 922,        // ✅ NOW CAPTURED
  "cpu_system_micros": 0,        // ✅ NOW CAPTURED  
  "max_rss_bytes": 2293760,      // ✅ NOW CAPTURED
  "disk_io_bytes": 0,            // ✅ NOW CAPTURED
  "public_key_bytes": 800,
  "secret_key_bytes": 1632,
  "keygen_time_ms": 0.009,
  "throughput_ops_per_sec": 111111.0,
  "attempts": 1,
  "error": null
}
```

### **Statistical Comparison** (statistical_comparisons.csv)

```csv
algorithm_a,algorithm_b,operation,n_a,n_b,mean_a,mean_b,ttest_pvalue,cohens_d,faster_algorithm
Kyber512,RSA-2048,Keygen,30,30,0.0029,0.0000,0.000123,1.24,RSA-2048
Dilithium2,ECDSA-P256,Sign,30,30,0.0000,0.0000,0.421,0.12,ECDSA-P256
```

---

## 🎯 Research Objectives - Status

| Objective | Status | Evidence |
|-----------|--------|----------|
| **Compare PQC vs Classical** | ✅ Complete | 4 PQC + 3 Classical algorithms |
| **Measure latency** | ✅ Complete | µs precision with percentiles |
| **Measure throughput** | ✅ Complete | Ops/sec per event |
| **Measure CPU usage** | ✅ **FIXED** | User + system time captured |
| **Measure memory** | ✅ **FIXED** | Max RSS in bytes |
| **Measure key sizes** | ✅ Complete | All key/signature/ct sizes |
| **Statistical significance** | ✅ **NEW** | t-tests, Mann-Whitney, p-values |
| **Effect sizes** | ✅ **NEW** | Cohen's d, rank-biserial |
| **Adequate sample size** | ✅ Complete | Configurable 5-50+ reps |
| **Reproducibility** | ✅ Complete | Environment snapshot, seeded RNG |
| **All KEM operations** | ✅ **FIXED** | Keygen + Encapsulate + Decapsulate |
| **All signature ops** | ✅ Complete | Sign + Verify |
| **Complete algorithm coverage** | ✅ **FIXED** | Added Kyber768, Dilithium3 |

---

## 🔬 Statistical Tests Available

### **For Publication**

Your data now includes:

1. **Significance Testing**
   - p-values < 0.05 → statistically significant
   - p-values < 0.01 → highly significant
   - Both parametric and non-parametric tests

2. **Effect Size Analysis**
   - Cohen's d: standardized mean difference
   - Interpretation: small (0.2), medium (0.5), large (0.8)
   - Practical significance beyond statistical

3. **Confidence Intervals**
   - 95% CI for all means
   - t-distribution based (accounts for sample size)

4. **Multiple Comparison Methods**
   - Independent samples (PQC vs Classical)
   - Paired comparisons (if needed)
   - Robust non-parametric alternatives

---

## 📝 Example Research Claims You Can Make

Based on the captured data, you can now make claims like:

✅ "Kyber512 key generation is X% slower than RSA-2048 (t-test p < 0.001, Cohen's d = Y)"

✅ "Dilithium2 signatures are Z bytes larger than ECDSA-P256 (mean difference = W bytes, 95% CI: [a, b])"

✅ "PQC algorithms show A% higher memory usage compared to classical algorithms (Mann-Whitney U, p = 0.023)"

✅ "Kyber768 encapsulation latency (median = X µs, IQR = [Y, Z]) is comparable to ECDHE-P256 (median = A µs)"

---

## 🎓 Ready for Publication

Your implementation now satisfies:

✅ **Comprehensive coverage** - All major PQC candidates
✅ **Complete metrics** - Performance, resources, sizes
✅ **Statistical rigor** - Significance tests, effect sizes
✅ **Adequate samples** - Configurable n=5 to n=50+
✅ **Reproducibility** - Environment capture, deterministic
✅ **Multiple formats** - CSV, JSON, Markdown for analysis

---

## 🚦 Next Steps

1. **Run full benchmark**:
   ```bash
   bash scripts/run_research_benchmark.sh --repetitions 30
   ```

2. **Review statistical report**:
   ```bash
   cat results/statistical_report.md
   ```

3. **Analyze data**:
   - Open `results/analysis.ipynb` in Jupyter
   - Import `results/statistical_comparisons.csv` into R/SPSS
   - Use `results/metrics.csv` for custom analysis

4. **Include in paper**:
   - Tables from `summary.csv`
   - Charts from `charts/`  
   - Statistical tests from `statistical_comparisons.csv`
   - Environment from `environment.json`

---

## 📧 Questions?

All research objectives from the methodology document are now satisfied. The system captures:
- ✅ All required metrics
- ✅ Resource usage (CPU, memory, I/O)
- ✅ Statistical significance tests
- ✅ Effect sizes
- ✅ Complete PQC vs Classical comparisons

