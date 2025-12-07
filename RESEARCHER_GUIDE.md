# Researcher Usage Guide
## Quantum-Resilient Cryptography Benchmark Framework

**Minimal mental overhead. No implementation details. Just what to run, when, and why.**

---

## 📌 1. Project Setup (One-Time Only)

### Clone the repository:
```bash
git clone <repository-url>
cd quantum-resilient
```

### Install required tools:

| Tool | Purpose | Installation |
|------|---------|-------------|
| **Rust** | Build the benchmarking binary | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| **Python 3.10+** | Analysis scripts | `python3 --version` (check) |
| **Podman** | Container builds | `sudo dnf install podman` (Fedora) or `sudo apt install podman` (Ubuntu) |
| **Minikube** (optional) | Local Kubernetes testing | `curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 && sudo install minikube-linux-amd64 /usr/local/bin/minikube` |
| **gcloud SDK** (for GCP) | GCP deployment | `curl https://sdk.cloud.google.com \| bash` |
| **Terraform** (for GCP) | GKE provisioning | `sudo dnf install terraform` or download from hashicorp.com |

### Install Python dependencies:
```bash
cd analysis
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
cd ..
```

### Authenticate to GCP (if using GCP experiments):
```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project <your-project-id>
```

---

## 📌 2. Common Terms

| Term | Meaning |
|------|---------|
| **Scenario** | A YAML file defining which crypto algorithms to test, payload sizes, message rates, duration, number of runs |
| **Smoke test** | Very short version (5 seconds, 1 run) to validate infrastructure before full experiments |
| **Full benchmark** | The real multi-run performance study (30 seconds, 5 runs per configuration) |
| **Environment** | Where the test runs: `native` (local machine), `minikube` (local Kubernetes), `gcp` (Google Cloud GKE) |
| **Algorithm** | Cryptographic primitive: `rsa2048`, `ecdsa_p256`, `kyber512`, `dilithium2`, `hybrid_kyber_dilithium` |
| **Experiment Matrix** | Declarative YAML file (`orchestration/experiment_matrix.yaml`) that defines all combinations to test |

---

## 📌 3. Research Context

**Your dissertation compares:**
- **Classical cryptography**: RSA-2048, ECDSA P-256 (baselines)
- **Post-Quantum cryptography**: Kyber-512 (KEM), Dilithium-2 (signatures)
- **Hybrid schemes**: Kyber-512 KEM → AES-GCM → Dilithium-2 signature

**Across three environments:**
- **Native**: Bare-metal performance (baseline)
- **Minikube**: Containerized local Kubernetes (container overhead)
- **GCP**: Cloud GKE deployment (real-world variability)

**Key metrics:**
- Latency (p50, p95, p99 in microseconds)
- Throughput (operations per second)
- Effect sizes (Cohen's d, statistical significance)

---

## 📌 4. Running Experiments

### Quick Reference: All Experiment Commands

All experiments follow this pattern:

```bash
# Smoke test (5 seconds, validates everything works)
./<script>.sh --scenario <scenario> --smoke-test --out results/<env>-smoke

# Full benchmark (30 seconds, 5 runs, real data)
./<script>.sh --scenario <scenario> --out results/<env>-full
```

---

## 📌 5. Run a Local (Native) Experiment

**Use this for:**
- Fast iteration and debugging
- Baseline performance measurements
- Validating scenarios before cloud deployment

### Smoke test:
```bash
./run_local.sh \
  --scenario scenarios/hybrid_kyber_dilithium.yaml \
  --smoke-test \
  --out results/native-smoke
```

### Full benchmark:
```bash
./run_local.sh \
  --scenario scenarios/hybrid_kyber_dilithium.yaml \
  --out results/native-full \
  --runs 5
```

**Output location:** `results/native-*/`

**What you get:**
- `raw/run.jsonl` - Raw telemetry events
- `merged/merged.jsonl` - Sorted merged events
- `stats/summary.json` - Statistical summary (p50, p95, p99, throughput)
- `figures/latency_cdf.png` - Latency distribution plot
- `figures/throughput.png` - Throughput time series

---

## 📌 6. Run a Minikube (Local Kubernetes) Experiment

**Use this for:**
- Validating Kubernetes manifests
- Comparing containerized vs native performance
- Testing before GCP deployment

### Prerequisites:
```bash
# Start Minikube (one-time setup)
MINIKUBE_ROOTLESS=true minikube start --driver=podman --rootless \
  --kubernetes-version=v1.32.0 \
  --container-runtime=containerd \
  --cni=kindnet \
  --extra-config=controller-manager.cluster-cidr=10.244.0.0/16 \
  --extra-config=kube-proxy.cluster-cidr=10.244.0.0/16 \
  --extra-config=kubelet.pod-cidr=10.244.0.0/16
```

### Smoke test:
```bash
./run_minikube.sh \
  --scenario scenarios/hybrid_kyber_dilithium.yaml \
  --smoke-test \
  --out results/minikube-smoke \
  --exp-id minikube-smoke
```

### Full benchmark:
```bash
./run_minikube.sh \
  --scenario scenarios/hybrid_kyber_dilithium.yaml \
  --out results/minikube-full \
  --exp-id minikube-full \
  --runs 5
```

**Output location:** `results/minikube-*/`

---

## 📌 7. Run a GCP (GKE) Experiment

**Use this for:**
- Real-world cloud performance data
- Production-like environment testing
- Final dissertation results

### Smoke test (minimal cost, ~£0.005) - **RECOMMENDED: Use ephemeral mode**:
```bash
# Ephemeral mode: Creates cluster, runs benchmark, destroys everything automatically
./deploy_gcp.sh \
  --scenario scenarios/hybrid_kyber_dilithium.yaml \
  --exp-id smoketest \
  --smoke-test \
  --ephemeral \
  --project <your-gcp-project> \
  --bucket <your-gcs-bucket> \
  --region us-central1
```

**What happens:**
- Creates minimal GKE cluster (1 node, same machine_type as full runs, same disk size)
- Runs 5-second benchmark
- Uploads results to GCS
- Downloads results locally
- **Automatically destroys cluster and all resources**
- **Verifies zero residual cost**
- **Cost: <£0.01, zero ongoing cost**

**Without `--ephemeral`:** Cluster remains running (costs money until manually destroyed)

### Full benchmark (costs money):
```bash
# For full benchmarks, also use --ephemeral to avoid ongoing costs
./deploy_gcp.sh \
  --scenario scenarios/hybrid_kyber_dilithium.yaml \
  --exp-id gcp-full \
  --project <your-gcp-project> \
  --bucket <your-gcs-bucket> \
  --region us-central1 \
  --runs 5 \
  --ephemeral
```

**What happens:**
- Creates GKE cluster
- Runs full benchmark (30 seconds, 5 runs)
- Uploads results to GCS
- Downloads results locally
- **Automatically destroys cluster and all resources**
- **Verifies zero residual cost**

**Alternative: Manual cluster management**
If you need to keep the cluster running for multiple experiments:
```bash
# Create cluster only
./deploy_gcp.sh --create-cluster --project <p> --bucket <b> --region us-central1

# Run experiments (cluster stays up)
./deploy_gcp.sh --scenario <scenario> --exp-id exp1 --project <p> --bucket <b> --skip-terraform
./deploy_gcp.sh --scenario <scenario> --exp-id exp2 --project <p> --bucket <b> --skip-terraform

# Destroy cluster when done
./deploy_gcp.sh --destroy-cluster --project <p> --bucket <b> --region us-central1
```

**After completion, fetch results:**
```bash
./fetch_and_analyse_from_gcs.sh \
  --exp-id gcp-full \
  --bucket <your-gcs-bucket> \
  --out results/gcp-full
```

**Output location:** `results/gcp-*/`

**Important:** Use `--ephemeral` flag to automatically destroy cluster and avoid costs:
```bash
# Ephemeral mode automatically destroys everything (RECOMMENDED)
./deploy_gcp.sh --scenario <scenario> --exp-id <id> --ephemeral --project <p> --bucket <b>

# Or manually destroy if you didn't use --ephemeral:
./deploy_gcp.sh --destroy-cluster --project <p> --bucket <b> --region us-central1
```

---

## 📌 8. Running Complete Experiment Matrix

**Use this for:**
- Running all algorithm combinations automatically
- Generating dissertation-ready datasets
- Full statistical rigor (5 runs per configuration)

### Generate scenarios from matrix:
```bash
python3 orchestration/generate_scenarios.py \
  --matrix orchestration/experiment_matrix.yaml \
  --output generated-scenarios
```

This creates scenarios for:
- **Algorithms**: RSA-2048, ECDSA P-256, Kyber-512, Dilithium-2, Hybrid
- **Payload sizes**: 256B, 1KB, 4KB
- **Rates**: 100, 500, 2000 msg/s
- **Runs**: 5 per configuration

### Run all experiments (all environments):
```bash
./run_all_experiments.sh \
  --envs native,minikube,gcp \
  --project <your-gcp-project> \
  --bucket <your-gcs-bucket> \
  --matrix orchestration/experiment_matrix.yaml
```

**For smoke-test mode (quick validation):**
```bash
./run_all_experiments.sh \
  --envs gcp \
  --smoke-test \
  --project <your-gcp-project> \
  --bucket <your-gcs-bucket>
```

**Output location:**
- Normal runs: `final-results/`
- Smoke tests: `final-results-smoke/`

**What you get:**
- `index.json` - Master experiment index
- `aggregated_stats.json` - Statistics across all runs
- `hypothesis_tests.json` - Statistical significance tests
- `figures/` - Combined CDFs, scaling curves, comparisons
- `stats/` - Effect sizes, environment deltas
- `tables/` - CSV tables for dissertation

---

## 📌 9. Working With Results

### After any experiment completes, you get:

```
results/<env>/<exp-id>/
├── raw/
│   └── run.jsonl              # Raw telemetry (one JSON object per line)
├── merged/
│   ├── merged.jsonl           # Sorted merged events
│   └── merged.parquet         # Efficient Parquet format
├── stats/
│   └── summary.json           # Statistical summary
├── figures/
│   ├── latency_cdf.png        # Latency ECDF (300 DPI, publication-ready)
│   └── throughput.png         # Throughput time series
└── manifest.json              # Run metadata (git commit, timestamps, etc.)
```

### View raw data:
```bash
# Count events
wc -l results/native-full/raw/run.jsonl

# View first event
head -1 results/native-full/raw/run.jsonl | python3 -m json.tool

# Extract latency values
cat results/native-full/merged/merged.jsonl | jq -r '.latency_us' | head -20
```

### View statistics:
```bash
# Pretty-print summary
cat results/native-full/stats/summary.json | python3 -m json.tool

# Extract specific metric
cat results/native-full/stats/summary.json | jq '.latency.p95'
```

---

## 📌 10. Analyse & Visualise Results

### Single Experiment Analysis

**Automatic analysis (recommended):**
```bash
# For local results
python3 analysis/scripts/compute_statistics.py \
  --input results/native-full/merged/merged.jsonl \
  --output results/native-full/stats \
  --experiment-id native-full

# Generate plots
python3 analysis/scripts/plot_latency.py \
  --input results/native-full/merged/merged.jsonl \
  --output results/native-full/figures

python3 analysis/scripts/plot_throughput.py \
  --input results/native-full/merged/merged.jsonl \
  --output results/native-full/figures
```

### Full Analysis Pipeline (All Experiments)

**After running `run_all_experiments.sh`, analysis is automatic.**

If you need to re-run analysis:
```bash
# Aggregate all results
python3 analysis/aggregate_results.py \
  --index final-results/index.json \
  --output final-results

# Generate combined CDFs (all algorithms together)
python3 analysis/plot_combined_cdfs.py \
  --index final-results/index.json \
  --output final-results/figures

# Generate scaling curves
python3 analysis/plot_scaling_curves.py \
  --index final-results/index.json \
  --output final-results/figures

# Run hypothesis tests (statistical significance)
python3 analysis/hypothesis_tests.py \
  --index final-results/index.json \
  --matrix orchestration/experiment_matrix.yaml \
  --output final-results

# Build final report
python3 analysis/build_final_report.py \
  --results-dir final-results \
  --output final-results/report.pdf
```

**Outputs:**
- `final-results/aggregated_stats.json` - Mean/std/CI for all metrics
- `final-results/hypothesis_tests.json` - P-values, effect sizes
- `final-results/figures/combined_ecdf.png` - All algorithms on one plot
- `final-results/figures/scaling_curves.png` - Throughput vs load
- `final-results/report.pdf` - Dissertation-ready report

---

## 📌 11. Cross-Environment Comparisons

**Compare native vs Minikube vs GCP:**

```bash
python3 analysis/compare_all_environments.py \
  --native results/native-full/stats/summary.json \
  --minikube results/minikube-full/stats/summary.json \
  --gcp results/gcp-full/stats/summary.json \
  --output final-results/comparisons/
```

**Output:**
- Side-by-side CDFs
- Box/violin plots
- Environment impact summary
- Dissertation paragraph text

---

## 📌 12. Full End-to-End Research Workflow

**Here is the complete workflow for your dissertation:**

### 👣 Step 1 — Validate with smoke tests

```bash
# Test all environments work
./run_local.sh --scenario scenarios/hybrid_kyber_dilithium.yaml --smoke-test --out results/native-smoke
./run_minikube.sh --scenario scenarios/hybrid_kyber_dilithium.yaml --smoke-test --out results/minikube-smoke --exp-id smoke
./deploy_gcp.sh --scenario scenarios/hybrid_kyber_dilithium.yaml --exp-id smoketest --smoke-test --project <project> --bucket <bucket>
```

**Purpose:** Confirm infrastructure, containers, telemetry, and analysis all work.

### 👣 Step 2 — Run full benchmarking

**Option A: Individual experiments (more control)**
```bash
# Native baseline
./run_local.sh --scenario scenarios/hybrid_kyber_dilithium.yaml --out results/native-full --runs 5

# Minikube containerized
./run_minikube.sh --scenario scenarios/hybrid_kyber_dilithium.yaml --out results/minikube-full --exp-id minikube-full --runs 5

# GCP cloud
./deploy_gcp.sh --scenario scenarios/hybrid_kyber_dilithium.yaml --exp-id gcp-full --project <project> --bucket <bucket> --runs 5
```

**Option B: Complete matrix (all algorithms, all configs)**
```bash
./run_all_experiments.sh \
  --envs native,minikube,gcp \
  --project <your-gcp-project> \
  --bucket <your-gcs-bucket> \
  --matrix orchestration/experiment_matrix.yaml
```

**Purpose:** Collect all datasets for your Results chapter.

### 👣 Step 3 — Analyse & visualise each dataset

**Individual experiment analysis:**
```bash
python3 analysis/scripts/compute_statistics.py \
  --input results/native-full/merged/merged.jsonl \
  --output results/native-full/stats

python3 analysis/scripts/plot_latency.py \
  --input results/native-full/merged/merged.jsonl \
  --output results/native-full/figures
```

**Full matrix analysis (automatic if using `run_all_experiments.sh`):**
- Aggregated statistics computed automatically
- Combined figures generated automatically
- Hypothesis tests run automatically

**Manual re-run if needed:**
```bash
python3 analysis/aggregate_results.py --index final-results/index.json --output final-results
python3 analysis/plot_combined_cdfs.py --index final-results/index.json --output final-results/figures
python3 analysis/hypothesis_tests.py --index final-results/index.json --matrix orchestration/experiment_matrix.yaml --output final-results
```

**Purpose:** Generate all figures and tables for your dissertation.

### 👣 Step 4 — Cross-environment comparisons

```bash
python3 analysis/compare_all_environments.py \
  --native results/native-full/stats/summary.json \
  --minikube results/minikube-full/stats/summary.json \
  --gcp results/gcp-full/stats/summary.json \
  --output final-results/comparisons/
```

**Purpose:** Quantify container overhead and cloud variability.

### 👣 Step 5 — Generate dissertation-ready artifacts

```bash
# Generate LaTeX tables
python3 research/scripts/generate_tables.py \
  --exp-id full-study \
  --stats-file final-results/aggregated_stats.json \
  --out final-results/tables/

# Bundle figures (PDF, EPS, high-DPI PNG)
python3 research/scripts/generate_figures_bundle.py \
  --exp-id full-study \
  --figures-dir final-results/figures \
  --out final-results/figures-bundle/

# Generate final report
python3 analysis/build_final_report.py \
  --results-dir final-results \
  --output final-results/report.pdf
```

**Purpose:** Create publication-ready tables and figures for your dissertation.

---

## 📌 13. Typical Daily Commands (Minimal Workflow)

**If you forget everything else, remember these:**

### 🟦 For debugging locally:
```bash
./run_local.sh --scenario scenarios/hybrid_kyber_dilithium.yaml --smoke-test --out results/debug
python3 analysis/scripts/compute_statistics.py --input results/debug/merged/merged.jsonl --output results/debug/stats
```

### 🟧 For validating Kubernetes locally:
```bash
./run_minikube.sh --scenario scenarios/hybrid_kyber_dilithium.yaml --smoke-test --out results/minikube-test --exp-id test
```

### 🟥 For validating GCP end-to-end (few pennies):
```bash
./deploy_gcp.sh --scenario scenarios/hybrid_kyber_dilithium.yaml --exp-id smoketest --smoke-test --project <project> --bucket <bucket>
```

### 🟩 For gathering actual research data:
```bash
# Single algorithm, all environments
./run_local.sh --scenario scenarios/kyber_hybrid_encrypt.yaml --out results/native-kyber --runs 5
./run_minikube.sh --scenario scenarios/kyber_hybrid_encrypt.yaml --out results/minikube-kyber --exp-id kyber --runs 5
./deploy_gcp.sh --scenario scenarios/kyber_hybrid_encrypt.yaml --exp-id gcp-kyber --project <project> --bucket <bucket> --runs 5

# OR: Complete matrix (all algorithms)
./run_all_experiments.sh --envs native,minikube,gcp --project <project> --bucket <bucket>
```

### 🟨 For analysis + visualisation:
```bash
# Single experiment
python3 analysis/scripts/compute_statistics.py --input results/native-kyber/merged/merged.jsonl --output results/native-kyber/stats

# Full matrix (if using run_all_experiments.sh, this is automatic)
python3 analysis/aggregate_results.py --index final-results/index.json --output final-results
python3 analysis/plot_combined_cdfs.py --index final-results/index.json --output final-results/figures
```

---

## 📌 14. Understanding Your Results

### Key Metrics Explained

| Metric | Meaning | Dissertation Use |
|--------|---------|------------------|
| **p50 latency** | Median latency (50th percentile) | Typical performance |
| **p95 latency** | 95th percentile latency | Tail latency (worst 5%) |
| **p99 latency** | 99th percentile latency | Extreme tail (worst 1%) |
| **Mean throughput** | Average operations per second | Overall system capacity |
| **Cohen's d** | Effect size (standardized difference) | Magnitude of algorithm differences |
| **p-value** | Statistical significance | Whether differences are real or noise |

### Reading Statistical Outputs

**From `summary.json`:**
```json
{
  "latency": {
    "p50": 498.0,    // Median: 498 microseconds
    "p95": 845.0,    // 95% of operations complete in < 845 μs
    "p99": 1124.0    // 99% complete in < 1124 μs
  },
  "throughput": {
    "mean": 8595.0   // Average: 8595 ops/sec
  }
}
```

**From `hypothesis_tests.json`:**
```json
{
  "comparison_id": "kyber512_vs_rsa2048_native",
  "tests": {
    "welch_t": {
      "p_value": 0.0001,        // Highly significant (p < 0.05)
      "significant": true        // Real difference, not noise
    }
  },
  "effect_size": {
    "cohens_d": 0.85,           // Large effect size
    "interpretation": "large"   // Meaningful practical difference
  }
}
```

### Interpreting Effect Sizes

| Cohen's d | Interpretation | Meaning |
|-----------|----------------|---------|
| < 0.2 | Negligible | No practical difference |
| 0.2 - 0.5 | Small | Minor difference |
| 0.5 - 0.8 | Medium | Moderate difference |
| > 0.8 | Large | Substantial difference |

**Example:** If Kyber-512 has Cohen's d = 0.85 vs RSA-2048, this means Kyber is substantially faster (large effect).

---

## 📌 15. Mental Model For You as a Researcher

**You never have to think about:**
- ❌ Container builds
- ❌ Kubernetes manifests
- ❌ Terraform infrastructure
- ❌ Distributed job orchestration
- ❌ Cryptographic adapter implementations
- ❌ Telemetry collection
- ❌ Logging infrastructure
- ❌ Cluster node sizing

**Your job is only:**
1. ✅ Choose environment (native/minikube/gcp)
2. ✅ Run experiment (with or without smoke-test)
3. ✅ Run analysis (automatic or manual)
4. ✅ Use figures in dissertation

**That's it.**

---

## 📌 16. Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| **"Binary not found"** | Run `cargo build --release` first |
| **"Scenario file not found"** | Check path: `ls scenarios/` |
| **"No results generated"** | Check logs in terminal output |
| **"Analysis script fails"** | Ensure Python venv is activated: `source analysis/venv/bin/activate` |
| **"GCP authentication error"** | Run `gcloud auth application-default login` |
| **"Minikube pod fails"** | Check: `kubectl logs -l job-name=pqc-bench-worker` |
| **"Terraform apply fails"** | Check GCP project permissions and billing |

### Getting Help

1. **Check logs:**
   ```bash
   # Local experiment logs are in terminal
   # Minikube logs
   kubectl logs -l job-name=pqc-bench-worker
   # GCP logs
   kubectl logs -l job-name=pqc-bench-worker -n pqc-smoke-test
   ```

2. **Verify outputs:**
   ```bash
   # Check if results exist
   ls -la results/native-full/
   # Check JSONL is valid
   head -1 results/native-full/merged/merged.jsonl | python3 -m json.tool
   ```

3. **Re-run analysis:**
   ```bash
   # If analysis failed, re-run manually
   python3 analysis/scripts/compute_statistics.py \
     --input results/native-full/merged/merged.jsonl \
     --output results/native-full/stats
   ```

---

## 📌 17. Dissertation Integration

### Using Results in Your Dissertation

**1. Raw Data:**
- Location: `results/<env>/<exp-id>/merged/merged.jsonl`
- Format: One JSON object per line
- Use: For custom analysis, verification, reproducibility

**2. Statistical Summaries:**
- Location: `final-results/aggregated_stats.json`
- Format: JSON with mean/std/CI for all metrics
- Use: Tables in Results chapter

**3. Figures:**
- Location: `final-results/figures/`
- Formats: PNG (300 DPI), PDF, EPS
- Use: Direct insertion into dissertation

**4. Hypothesis Tests:**
- Location: `final-results/hypothesis_tests.json`
- Format: JSON with p-values, effect sizes
- Use: Statistical significance claims in Results chapter

**5. Comparison Tables:**
- Location: `final-results/tables/`
- Formats: CSV, LaTeX
- Use: Tables comparing algorithms/environments

### Example Dissertation Paragraph

After running experiments and analysis, you can write:

> "Across native, Minikube, and GCP execution environments, p95 latency for Kyber-512 increased from 0.48 ms (native) → 0.55 ms (Minikube) → 0.79 ms (GCP). This represents a 14.6% increase from native to containerized execution, and a 64.6% increase from native to cloud execution. Statistical hypothesis testing (Welch's t-test, p < 0.001) confirms these differences are significant, with large effect sizes (Cohen's d = 0.85 for native vs GCP)."

This text is automatically generated by `analysis/compare_all_environments.py`.

---

## 📌 18. Cost Management

### Smoke Tests (Recommended First)
- **Cost:** <£0.01 per run
- **Duration:** ~5 seconds
- **Purpose:** Validate everything works

### Full Benchmarks
- **Native:** Free (local machine)
- **Minikube:** Free (local machine)
- **GCP:** ~£0.50-2.00 per experiment (depending on duration, node count)

### Cost-Saving Tips
1. **Always run smoke tests first** to catch errors early
2. **Use `--ephemeral` flag** - automatically destroys all resources after completion (zero ongoing cost)
3. **Use `--smoke-test` flag** for infrastructure validation
4. **Run native/minikube first** to validate scenarios before GCP
5. **Verify zero residual cost** - ephemeral mode automatically verifies no resources remain

**Ephemeral mode (`--ephemeral`):**
- ✅ Creates cluster only when needed
- ✅ Automatically destroys all resources after completion
- ✅ Cleans up load balancers, IPs, disks, firewall rules
- ✅ Verifies zero residual cost
- ✅ **Recommended for all GCP experiments**

---

## 📌 19. Next Steps

1. **Start with smoke tests** in all environments
2. **Run one full experiment** (single algorithm, single environment) to validate workflow
3. **Run complete matrix** when ready for final data collection
4. **Analyze results** (automatic if using `run_all_experiments.sh`)
5. **Generate dissertation artifacts** from `final-results/`

---

## 📌 20. Quick Reference Card

```bash
# ============================================
# SMOKE TESTS (Validation)
# ============================================
./run_local.sh --scenario scenarios/hybrid_kyber_dilithium.yaml --smoke-test --out results/smoke
./run_minikube.sh --scenario scenarios/hybrid_kyber_dilithium.yaml --smoke-test --out results/smoke --exp-id smoke
./deploy_gcp.sh --scenario scenarios/hybrid_kyber_dilithium.yaml --exp-id smoketest --smoke-test --ephemeral --project <p> --bucket <b>

# ============================================
# FULL EXPERIMENTS (Data Collection)
# ============================================
./run_local.sh --scenario scenarios/hybrid_kyber_dilithium.yaml --out results/native-full --runs 5
./run_minikube.sh --scenario scenarios/hybrid_kyber_dilithium.yaml --out results/minikube-full --exp-id full --runs 5
./deploy_gcp.sh --scenario scenarios/hybrid_kyber_dilithium.yaml --exp-id gcp-full --project <p> --bucket <b> --runs 5

# ============================================
# COMPLETE MATRIX (All Algorithms)
# ============================================
./run_all_experiments.sh --envs native,minikube,gcp --project <p> --bucket <b>

# ============================================
# ANALYSIS
# ============================================
python3 analysis/scripts/compute_statistics.py --input results/*/merged/merged.jsonl --output results/*/stats
python3 analysis/aggregate_results.py --index final-results/index.json --output final-results
python3 analysis/plot_combined_cdfs.py --index final-results/index.json --output final-results/figures
python3 analysis/hypothesis_tests.py --index final-results/index.json --matrix orchestration/experiment_matrix.yaml --output final-results

# ============================================
# COMPARISONS
# ============================================
python3 analysis/compare_all_environments.py --native results/native-full/stats/summary.json --minikube results/minikube-full/stats/summary.json --gcp results/gcp-full/stats/summary.json --output final-results/comparisons/
```

---

**That's everything you need to know. Happy benchmarking! 🚀**

