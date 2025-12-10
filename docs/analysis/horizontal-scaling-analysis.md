# Horizontal Scaling Analysis: Impact Assessment

## Executive Summary

**Question**: How would adding horizontal scaling analysis affect the current setup, and does it invalidate existing native data?

**Answer**: 
- ✅ **Native data is SAFE** - Scaling experiments use different scenario IDs
- ⚠️ **Native**: Replicas are **skipped** (not supported)
- ⚠️ **Minikube**: Replicas work but test **orchestration overhead**, not true scaling
- ✅ **GCP**: Replicas test **true horizontal scaling** across nodes
- 📊 **Impact**: Adds ~36-108 additional experiments (separate from your 300 baseline)

---

## How Replicas Work in Each Environment

### 1. Native Environment

**Status**: ❌ **Replicas are NOT supported**

**Code Evidence** (`run_all_experiments.sh:533-535`):
```bash
# For native, only run with 1 replica
if [[ "$env" == "native" ]] && [[ "$replica_count" -gt 1 ]]; then
    continue  # Skip replicas > 1
fi
```

**Why**: Native runs a single-process binary (`pqc-bench`). There's no concept of "replicas" - it's just one process running on your machine.

**What happens if you try**:
- Replicas = 1: ✅ Runs normally (your current 300 experiments)
- Replicas = 2, 4, 8: ❌ **Automatically skipped** - no experiments run

**Conclusion**: Adding scaling experiments **does NOT affect native runs at all**.

---

### 2. Minikube Environment

**Status**: ⚠️ **Replicas work, but limited value**

**How it works**:
- Creates Kubernetes Jobs with `parallelism: N` and `completions: N`
- Multiple pods run **on the same physical machine** (single-node Minikube cluster)
- All pods share the same CPU, memory, and network resources

**What you're actually testing**:
- ✅ Kubernetes orchestration overhead (scheduling, pod creation, etc.)
- ✅ Container runtime overhead (Podman/containerd)
- ✅ Resource contention (multiple processes competing for CPU/memory)
- ❌ **NOT true horizontal scaling** (pods on different nodes)

**Example**:
```
Replicas = 4 on Minikube:
├── Pod 1 ──┐
├── Pod 2 ──┤
├── Pod 3 ──┼── All on same physical machine
└── Pod 4 ──┘
```

**Value for dissertation**:
- **Low-Medium**: Shows orchestration overhead, but not production-like scaling
- Useful if your dissertation addresses "containerization overhead"
- Less useful if you want to show "how algorithms scale in production"

**Conclusion**: Replicas work on Minikube, but you're testing **parallel execution on one machine**, not true horizontal scaling.

---

### 3. GCP Environment

**Status**: ✅ **True horizontal scaling**

**How it works**:
- Creates Kubernetes Jobs in GKE (Google Kubernetes Engine)
- Pods can run on **different nodes** in the cluster
- Each node has its own CPU, memory, and network resources
- Load balancer distributes work across nodes

**What you're actually testing**:
- ✅ True horizontal scaling (pods on different nodes)
- ✅ Network latency between nodes
- ✅ Load distribution across cluster
- ✅ Production-like deployment scenarios

**Example**:
```
Replicas = 4 on GCP:
├── Pod 1 ── Node 1 (us-central1-a)
├── Pod 2 ── Node 2 (us-central1-b)
├── Pod 3 ── Node 3 (us-central1-c)
└── Pod 4 ── Node 4 (us-central1-a)
```

**Value for dissertation**:
- **High**: Shows how algorithms scale in production-like environments
- Essential if your dissertation addresses "production deployment" or "scalability"

**Conclusion**: GCP is where horizontal scaling experiments provide the most value.

---

## Scenario ID Generation and Data Safety

### How Scenario IDs Work

**Base scenario ID** (replicas = 1):
```
{algorithm}_p{payload}_r{rate}_run{N}_{hash}
Example: rsa2048_p256_r100_run1_a1b2c3d4
```

**Scaling scenario ID** (replicas > 1):
```
{algorithm}_p{payload}_r{rate}_run{N}_{hash}_r{replicas}
Example: rsa2048_p256_r100_run1_a1b2c3d4_r4
```

**Code Evidence** (`run_all_experiments.sh:543-549`):
```bash
# Generate unique output dir and ID for scaling experiments
if [[ "$replica_count" -gt 1 ]]; then
    output_dir="$RESULTS_BASE/$env/${scenario_id}_r${replica_count}"
    run_scenario_id="${scenario_id}_r${replica_count}"
else
    output_dir="$RESULTS_BASE/$env/$scenario_id"
    run_scenario_id="$scenario_id"
fi
```

### Data Safety Analysis

**Your existing native data**:
```
results/native/
├── rsa2048_p256_r100_run1_a1b2c3d4/     ← Replicas = 1 (no suffix)
├── rsa2048_p256_r100_run2_b2c3d4e5/     ← Replicas = 1 (no suffix)
└── ... (300 experiments, all replicas = 1)
```

**If you add scaling experiments**:
```
results/minikube/
├── kyber512_p1024_r500_run1_x1y2z3a4/        ← Replicas = 1 (baseline)
├── kyber512_p1024_r500_run1_x1y2z3a4_r2/     ← Replicas = 2 (NEW)
├── kyber512_p1024_r500_run1_x1y2z3a4_r4/     ← Replicas = 4 (NEW)
└── kyber512_p1024_r500_run1_x1y2z3a4_r8/     ← Replicas = 8 (NEW)
```

**Conclusion**: ✅ **Your native data is 100% safe**
- Scaling experiments use **different scenario IDs** (with `_r{count}` suffix)
- They're stored in **separate directories**
- They **do NOT overwrite** your existing data
- They're **additional experiments**, not replacements

---

## Scaling Experiment Configuration

### Current Configuration (`experiment_matrix.yaml`)

```yaml
scaling:
  replicas: [1, 2, 4, 8]
  scaling_algorithms:
    - kyber512
    - dilithium2
    - hybrid_kyber_dilithium
  scaling_payload: 1024
  scaling_rate: 500
  scaling_runs: 3
```

### Experiment Count Calculation

**Per algorithm**:
- 1 payload (1024 bytes)
- 1 rate (500 msg/s)
- 3 runs
- 4 replica counts (1, 2, 4, 8)
- **Total**: 1 × 1 × 3 × 4 = **12 scenarios per algorithm**

**Per environment**:
- 3 algorithms (kyber512, dilithium2, hybrid_kyber_dilithium)
- 12 scenarios per algorithm
- **Total**: 3 × 12 = **36 scenarios per environment**

**Across all environments**:
- Native: 0 (replicas skipped)
- Minikube: 36 scenarios
- GCP: 36 scenarios
- **Total**: **72 additional scenarios**

**Note**: Replicas = 1 are already included in your baseline 300 experiments, so you're only adding replicas 2, 4, 8:
- **Actual new experiments**: 3 algorithms × 1 payload × 1 rate × 3 runs × 3 replica counts (2,4,8) = **27 per environment**
- **Total new experiments**: 27 × 2 (minikube + GCP) = **54 additional experiments**

---

## Impact Assessment

### Time Impact

**Baseline** (current 300 experiments per environment):
- Native: ~4-5 hours
- Minikube: ~5-6 hours
- GCP: ~6-7 hours

**With scaling experiments** (additional):
- Minikube: +27 experiments ≈ **+1-2 hours**
- GCP: +27 experiments ≈ **+1-2 hours**
- **Total additional time**: ~2-4 hours

### Storage Impact

**Baseline**: ~4-5 GB total (300 × 3 environments)
**With scaling**: +54 experiments ≈ **+200-300 MB**
**Total**: ~4.5-5.5 GB

### Analysis Impact

**New analysis outputs**:
- Replica scaling plots (`plot_replica_scaling.py`)
- Throughput vs. replica count curves
- Latency vs. replica count analysis
- Efficiency metrics (throughput per replica)

**Code Evidence** (`run_all_experiments.sh:737-768`):
```bash
# Phase 9: Replica Scaling Analysis
if [[ "$SKIP_SCALING" != "true" ]]; then
    # Check if we have scaling experiments (replicas > 1)
    for r in "${REPLICA_ARRAY[@]}"; do
        if [[ "$r" -gt 1 ]]; then
            HAS_SCALING=true
            break
        fi
    done
    
    if [[ "$HAS_SCALING" == "true" ]]; then
        log_info "Generating replica scaling plots..."
        python3 "$SCRIPT_DIR/analysis/plot_replica_scaling.py" \
            --input "$FINAL_RESULTS_DIR" \
            --output "$FINAL_RESULTS_DIR/figures" \
            --format png
    fi
fi
```

---

## Recommendations

### ✅ **DO Add Scaling Experiments If**:

1. **Your dissertation addresses production deployment**
   - "How do PQC algorithms scale in production?"
   - "What is the overhead of containerization and orchestration?"

2. **You want to show GCP-specific insights**
   - True horizontal scaling across nodes
   - Production-like deployment scenarios

3. **You have time and resources**
   - Additional 2-4 hours of runtime
   - Additional ~300 MB storage

### ❌ **DON'T Add Scaling Experiments If**:

1. **You're focused on algorithm performance only**
   - Your 300 baseline experiments already cover this
   - Scaling adds orchestration overhead, not algorithm insights

2. **You're short on time**
   - The baseline 300 experiments are sufficient for dissertation objectives
   - Scaling is "nice to have," not essential

3. **You only care about native performance**
   - Native doesn't support replicas anyway
   - Minikube scaling is limited (single machine)

### 🎯 **Recommended Approach**:

**Option 1: Minimal Scaling (Recommended)**
- Run scaling experiments **only on GCP** (true horizontal scaling)
- Skip Minikube scaling (limited value, single machine)
- **Additional experiments**: 27 (GCP only)
- **Additional time**: ~1-2 hours

**Option 2: Full Scaling**
- Run scaling on both Minikube and GCP
- **Additional experiments**: 54 (Minikube + GCP)
- **Additional time**: ~2-4 hours

**Option 3: No Scaling**
- Stick with your baseline 300 experiments
- Focus on algorithm performance, not deployment
- **Additional experiments**: 0
- **Additional time**: 0

---

## How to Run Scaling Experiments

### Step 1: Run Baseline Experiments (Current)

```bash
# Native (already done - 300 experiments)
./run_full_scale_data_collection.sh --env native

# Minikube (baseline - 300 experiments)
./run_full_scale_data_collection.sh --env minikube

# GCP (baseline - 300 experiments)
./run_full_scale_data_collection.sh --env gcp --project <project> --bucket <bucket>
```

### Step 2: Run Scaling Experiments (Optional)

**Option A: GCP Only (Recommended)**
```bash
./run_all_experiments.sh \
  --envs gcp \
  --replicas 1,2,4,8 \
  --project <project> \
  --bucket <bucket> \
  --skip-generation \
  --matrix orchestration/experiment_matrix.yaml
```

**Option B: Minikube + GCP**
```bash
./run_all_experiments.sh \
  --envs minikube,gcp \
  --replicas 1,2,4,8 \
  --project <project> \
  --bucket <bucket> \
  --skip-generation \
  --matrix orchestration/experiment_matrix.yaml
```

**Note**: The `--replicas` flag will:
- ✅ Run replicas 2, 4, 8 on Minikube/GCP
- ❌ Skip replicas > 1 on Native (automatic)
- ✅ Use existing scenario generation (if `--skip-generation` is set)

---

## Summary

| Question | Answer |
|----------|--------|
| **Does scaling invalidate native data?** | ❌ **NO** - Different scenario IDs, separate directories |
| **Does scaling work on native?** | ❌ **NO** - Automatically skipped (not supported) |
| **Does scaling work on Minikube?** | ⚠️ **YES, but limited** - Tests orchestration, not true scaling |
| **Does scaling work on GCP?** | ✅ **YES** - True horizontal scaling across nodes |
| **How many additional experiments?** | 27-54 (depending on environments) |
| **How much additional time?** | 1-4 hours (depending on environments) |
| **Should you add scaling?** | ✅ **Only if dissertation addresses production deployment** |

---

## Next Steps

1. **Review your dissertation objectives**
   - Do they mention "scalability" or "production deployment"?
   - If yes → Add scaling experiments (GCP only recommended)
   - If no → Skip scaling, focus on baseline 300 experiments

2. **If adding scaling**:
   - Run baseline experiments first (300 per environment)
   - Then run scaling experiments separately (27-54 additional)
   - Use `--replicas 1,2,4,8` flag

3. **If skipping scaling**:
   - Your baseline 300 experiments are sufficient
   - Focus on algorithm performance analysis
   - No changes needed to current workflow

