# Smoke Test Data Flow

## Does the smoke test actually run benchmarks?

**YES.** The smoke test runs **real cryptographic benchmarks** with the same code path as full experiments, just with reduced parameters.

## Ephemeral Mode (Recommended)

**Always use `--ephemeral` flag for smoke tests and full benchmarks** to ensure zero ongoing cost:

```bash
./deploy_gcp.sh \
  --scenario scenarios/hybrid_kyber_dilithium.yaml \
  --exp-id smoketest \
  --smoke-test \
  --ephemeral \
  --project <project> \
  --bucket <bucket> \
  --region us-central1
```

**What ephemeral mode does:**
1. Creates GKE cluster
2. Runs benchmark
3. Collects results
4. **Automatically destroys cluster and all resources**
5. **Cleans up load balancers, IPs, disks, firewall rules**
6. **Verifies zero residual cost**

**Without `--ephemeral`:** Cluster remains running and continues to incur costs until manually destroyed.

## Execution Flow

### 1. Scenario Preparation
```bash
# deploy_gcp.sh modifies the scenario for smoke-test mode:
- duration_sec: 30 → 5 seconds
- runs: 5 → 1
- payload_sizes: [256, 1024, 4096] → [256]
- rates: [100, 500, 2000] → [100]
- replicas: any → 1
```

### 2. Kubernetes Job Execution
The job runs the `pqc-bench` container with:
```bash
pqc-bench --scenario /config/scenario.yaml
```

### 3. Benchmark Execution (Rust Code)
The Rust binary:
1. ✅ Loads the scenario from ConfigMap
2. ✅ Initializes cryptographic adapters (RSA, ECDSA, Kyber, Dilithium, etc.)
3. ✅ Runs **actual cryptographic operations** (sign, encrypt, KEM, etc.)
4. ✅ Generates workload events at the specified rate
5. ✅ Measures latency for each operation
6. ✅ Writes telemetry to `/results/raw/run.jsonl`

**No dummy paths. No bypasses. Full execution.**

### 4. Telemetry Collection
Each event written to JSONL contains:
```json
{
  "event_id": 1,
  "timestamp_utc_iso": "2025-01-15T10:30:45.123456Z",
  "operation": "kem_aead_encrypt",
  "algorithm": "kyber",
  "latency_us": 523,
  "payload_size_bytes": 256,
  "ciphertext_size_bytes": 1052,
  "cpu_user_seconds": 0.000123,
  "memory_rss_bytes": 3456784,
  "error": null
}
```

### 5. Upload to GCS
A sidecar container (`upload-results`) waits for the benchmark to complete, then uploads:
- `merged.jsonl` → `gs://<bucket>/experiments/<exp-id>/merged.jsonl`
- `manifest.json` → `gs://<bucket>/experiments/<exp-id>/manifest.json`
- `provenance.json` → `gs://<bucket>/experiments/<exp-id>/provenance.json`
- `cloud_metadata.json` → `gs://<bucket>/experiments/<exp-id>/cloud_metadata.json`
- `raw/*.jsonl` → `gs://<bucket>/experiments/<exp-id>/raw/`

### 6. Local Download (Automatic)
`deploy_gcp.sh` automatically downloads results to:
```
results/gcp/<exp-id>/
├── merged/
│   └── merged.jsonl          # All telemetry events
├── raw/
│   └── run.jsonl             # Raw telemetry
├── stats/
│   └── summary.json          # Statistical summary (if generated)
├── manifest.json             # Run metadata
├── provenance.json            # Experiment provenance
└── cloud_metadata.json        # GKE cluster metadata
```

## Data Locations

### In GCS (Cloud Storage):
```
gs://<your-bucket>/experiments/<exp-id>/
├── merged.jsonl
├── manifest.json
├── provenance.json
├── cloud_metadata.json
├── summary.json (optional)
└── raw/
    └── run.jsonl
```

### Locally (after deploy_gcp.sh):
```
results/gcp/<exp-id>/
├── merged/merged.jsonl
├── raw/run.jsonl
├── manifest.json
├── provenance.json
└── cloud_metadata.json
```

## Verification

### Check that benchmarks actually ran:

1. **Check pod logs:**
   ```bash
   kubectl logs -l job-name=pqc-bench-worker -n pqc-smoke-test -c pqc-bench
   ```
   Should show:
   - "Starting PQC Benchmark Framework..."
   - "Loaded scenario: ..."
   - "Running operation: ..."
   - "Run complete: X events processed"

2. **Check GCS artifacts:**
   ```bash
   gsutil ls gs://<bucket>/experiments/<exp-id>/
   ```
   Should list: `merged.jsonl`, `manifest.json`, `provenance.json`

3. **Check local results:**
   ```bash
   ls -lh results/gcp/<exp-id>/merged/merged.jsonl
   wc -l results/gcp/<exp-id>/merged/merged.jsonl
   ```
   Should show a file with multiple lines (one per event)

4. **Inspect telemetry:**
   ```bash
   head -1 results/gcp/<exp-id>/merged/merged.jsonl | python3 -m json.tool
   ```
   Should show a valid JSON event with `latency_us`, `operation`, `algorithm`, etc.

## What Gets Measured

Even in smoke-test mode, **everything** is measured:
- ✅ Latency (microseconds) for each cryptographic operation
- ✅ Throughput (operations per second)
- ✅ CPU usage (user/system time)
- ✅ Memory usage (RSS)
- ✅ Queue delays
- ✅ Error rates
- ✅ Ciphertext sizes
- ✅ System metadata (CPU model, kernel version, etc.)

## Analysis

After `deploy_gcp.sh` completes, you can immediately analyze:

```bash
# Compute statistics
python3 analysis/scripts/compute_statistics.py \
  --input results/gcp/<exp-id>/merged/merged.jsonl \
  --output results/gcp/<exp-id>/stats

# Generate plots
python3 analysis/scripts/plot_latency.py \
  --input results/gcp/<exp-id>/merged/merged.jsonl \
  --output results/gcp/<exp-id>/figures

python3 analysis/scripts/plot_throughput.py \
  --input results/gcp/<exp-id>/merged/merged.jsonl \
  --output results/gcp/<exp-id>/figures
```

## Summary

**Smoke test = Real benchmarks, reduced scale:**
- ✅ Same infrastructure (GKE)
- ✅ Same container images
- ✅ Same worker job
- ✅ Same Rust code path
- ✅ Same cryptographic operations
- ✅ Same telemetry collection
- ✅ Same GCS upload flow
- ✅ Same analysis pipeline

**Only differences:**
- Duration: 5 seconds (vs 30)
- Runs: 1 (vs 5)
- Payload: 256B only (vs 256, 1024, 4096)
- Rate: 100 msg/s only (vs 100, 500, 2000)
- Replicas: 1 (vs variable)
- Infrastructure: Same machine_type as full runs, but 1 node (vs multiple nodes)

**Result:** Valid, real data that exercises the full pipeline end-to-end.

