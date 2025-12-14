# Cost and Time Analysis: Complete Experiment Suite

## Executive Summary

**Current Experiment Suite**:
- **Native**: 468 experiments (~6.5-8 hours, £0)
- **Minikube**: 495 experiments (~8.5-11 hours, £0)
- **GCP**: 495 experiments (~10.5-12.5 hours, £0.74-0.82)
- **Total**: 1,458 experiments (~25.5-31.5 hours, £0.74-0.82)

---

## Experiment Count Breakdown

### Current Experiment Suite

| Category | Count | Description |
|----------|-------|-------------|
| Baseline (constant, 30s, standard rates) | 360 | 6 algorithms × 4 payloads × 3 rates × 5 runs |
| Burst pattern | 50 | Enterprise workload patterns |
| 10K msg/s rate | 100 | High-throughput scenarios |
| 5-minute duration | 9 | Sustained load tests |
| Scaling baseline (replica=1) | 9 | Horizontal scaling baseline |
| **Total baseline (replica=1)** | **468** | |
| Scaling additional (replicas 2,4,8) | 27 | 9 scenarios × 3 replica counts |

### Per-Environment Breakdown

| Environment | Baseline | Scaling | Total |
|-------------|----------|---------|-------|
| **Native** | 468 | 0 (not supported) | **468** |
| **Minikube** | 468 | 27 | **495** |
| **GCP** | 468 | 27 | **495** |
| **Total** | **1,404** | **54** | **1,458** |

---

## Time Estimates: Detailed Breakdown

### Per-Experiment Time Components

**Standard experiments (30s duration)**:
- Experiment runtime: 30 seconds
- Setup overhead: ~10 seconds (scenario load, worker startup, initialization)
- Teardown overhead: ~5 seconds (result collection, cleanup)
- **Total per experiment**: ~45 seconds

**5-minute duration experiments**:
- Experiment runtime: 300 seconds (5 minutes)
- Setup overhead: ~10 seconds
- Teardown overhead: ~5 seconds
- **Total per experiment**: ~315 seconds (~5.25 minutes)

**Environment-specific overhead**:
- **Native**: Minimal (~0 seconds per experiment)
- **Minikube**: Container overhead (~2-3 seconds per experiment)
- **GCP**: Pod scheduling, image pull, network (~5 seconds per experiment)

**GCP cluster lifecycle** (one-time per run):
- Cluster setup: 10-15 minutes
- Cluster teardown: 3-5 minutes
- **Total overhead**: ~13-20 minutes per GCP run

---

## Native Environment

### Experiment Breakdown

| Category | Count | Duration | Time per Experiment | Total Time |
|----------|-------|----------|---------------------|------------|
| Baseline (30s) | 300 | 30s | ~45s | 3.75 hours |
| Burst (30s) | 50 | 30s | ~45s | 0.625 hours |
| 10K msg/s (30s) | 100 | 30s | ~45s | 1.25 hours |
| 5-minute duration | 9 | 300s | ~315s | 0.788 hours |
| Scaling baseline (30s) | 9 | 30s | ~45s | 0.113 hours |
| **Total** | **468** | | | **6.53 hours** |

### Time Breakdown for Scheduling

| Phase | Time | Description |
|-------|------|-------------|
| **Baseline experiments** | 4.5 hours | Core algorithm comparison (360 core + 216 additional = 576 total, includes ECDHE) |
| **Burst pattern** | 0.625 hours | Enterprise workload patterns (50 experiments) |
| **10K msg/s** | 1.25 hours | High-throughput scenarios (100 experiments) |
| **5-minute duration** | 0.788 hours | Sustained load tests (9 experiments) |
| **Scaling baseline** | 0.113 hours | Scaling baseline (replica=1, 9 experiments) |
| **System variability** | +0.5-1.5 hours | System load, breaks, variability |
| **Total** | **~6.5-8 hours** | |

### Cost

**£0** (runs on local machine, no cloud costs)

### Scheduling Recommendations

- **Can run overnight**: 6.5-8 hours fits in a single night
- **Can pause/resume**: Script supports graceful stop/resume
- **Best time**: When machine is idle (close other applications)
- **Checkpoints**: Progress saved after each experiment

---

## Minikube Environment

### Experiment Breakdown

| Category | Count | Duration | Time per Experiment | Total Time |
|----------|-------|----------|---------------------|------------|
| Baseline (30s) | 300 | 30s | ~47s | 3.92 hours |
| Burst (30s) | 50 | 30s | ~47s | 0.653 hours |
| 10K msg/s (30s) | 100 | 30s | ~47s | 1.306 hours |
| 5-minute duration | 9 | 300s | ~317s | 0.793 hours |
| Scaling baseline (30s) | 9 | 30s | ~47s | 0.118 hours |
| Scaling (replicas 2,4,8) | 27 | 30s | ~47s | 0.353 hours |
| **Total** | **495** | | | **7.14 hours** |

### Time Breakdown for Scheduling

| Phase | Time | Description |
|-------|------|-------------|
| **Baseline experiments** | 3.92 hours | Core algorithm comparison (300 experiments) |
| **Burst pattern** | 0.653 hours | Enterprise workload patterns (50 experiments) |
| **10K msg/s** | 1.306 hours | High-throughput scenarios (100 experiments) |
| **5-minute duration** | 0.793 hours | Sustained load tests (9 experiments) |
| **Scaling baseline** | 0.118 hours | Scaling baseline (replica=1, 9 experiments) |
| **Scaling (replicas 2,4,8)** | 0.353 hours | Horizontal scaling tests (27 experiments) |
| **Container overhead** | +0.5-1 hour | Kubernetes orchestration overhead |
| **System variability** | +0.5-1.5 hours | System load, breaks, variability |
| **Total** | **~8.5-11 hours** | |

### Cost

**£0** (runs on local machine with Minikube, no cloud costs)

### Scheduling Recommendations

- **Can run overnight**: 8.5-11 hours fits in a single night
- **Can pause/resume**: Script supports graceful stop/resume
- **Best time**: When machine is idle (close other applications)
- **Resource requirements**: Ensure sufficient RAM/CPU for Kubernetes
- **Checkpoints**: Progress saved after each experiment

---

## GCP Environment

### Experiment Breakdown

| Category | Count | Duration | Time per Experiment | Total Time |
|----------|-------|----------|---------------------|------------|
| Baseline (30s) | 300 | 30s | ~50s | 4.17 hours |
| Burst (30s) | 50 | 30s | ~50s | 0.694 hours |
| 10K msg/s (30s) | 100 | 30s | ~50s | 1.389 hours |
| 5-minute duration | 9 | 300s | ~320s | 0.8 hours |
| Scaling baseline (30s) | 9 | 30s | ~50s | 0.125 hours |
| Scaling (replicas 2,4,8) | 27 | 30s | ~50s | 0.375 hours |
| **Total runtime** | **495** | | | **7.37 hours** |

### Time Breakdown for Scheduling

| Phase | Time | Description |
|-------|------|-------------|
| **Cluster setup** | 10-15 minutes | One-time: Terraform apply, node provisioning |
| **Baseline experiments** | 4.17 hours | Core algorithm comparison (300 experiments) |
| **Burst pattern** | 0.694 hours | Enterprise workload patterns (50 experiments) |
| **10K msg/s** | 1.389 hours | High-throughput scenarios (100 experiments) |
| **5-minute duration** | 0.8 hours | Sustained load tests (9 experiments) |
| **Scaling baseline** | 0.125 hours | Scaling baseline (replica=1, 9 experiments) |
| **Scaling (replicas 2,4,8)** | 0.375 hours | Horizontal scaling tests (27 experiments) |
| **Cluster teardown** | 3-5 minutes | One-time: Terraform destroy, cleanup |
| **GCP overhead** | +0.5-1 hour | Pod scheduling, image pull, network latency |
| **System variability** | +0.5-1.5 hours | System load, breaks, variability |
| **Total** | **~10.5-12.5 hours** | |

### Cost Analysis (GCP europe-west2, n2-standard-2)

#### GCP Resource Configuration

**From `terraform/gke/variables.tf`**:
- **Machine type**: `n2-standard-2` (2 vCPU, 8GB RAM)
- **Node count**: 1 (for baseline), 2-8 (for scaling experiments)
- **Disk**: 50GB pd-standard
- **Region**: europe-west2 (London)

#### GCP Pricing (europe-west2, as of 2024-2025)

**Compute Engine (GKE nodes)**:
- `n2-standard-2` in europe-west2: **$0.0971 per hour** per node
- **Note**: Pricing is **per hour**, billed in 1-minute increments (minimum 1 minute)

**Storage**:
- `pd-standard`: $0.17 per GB per month
- Pro-rated per hour: $0.17 / (30 days × 24 hours) = **$0.000236 per GB per hour**
- 50GB disk: 50 × $0.000236 = **$0.0118 per hour**

**GKE Cluster Management**:
- **Free** (no additional charge for GKE cluster management)

**Network Egress**:
- First 1GB/month: Free
- 1-10TB: $0.12 per GB
- **Assumption**: Minimal egress (results uploaded to GCS, downloaded once) ≈ **$0.01-0.02**

**GCS Storage**:
- Standard storage: $0.020 per GB per month
- **Assumption**: ~5GB total results = ~$0.10 per month (one-time, minimal)

#### Cost Calculation

**Cluster lifecycle**:
- Setup: 10-15 minutes = 0.167-0.25 hours
- Runtime: 7.37 hours (495 experiments × 50s / 3600)
- Teardown: 3-5 minutes = 0.05-0.083 hours
- **Total cluster time**: ~7.6-7.7 hours

**Compute cost**:
- Baseline experiments (1 node): 7.37 hours × $0.0971/hour = **$0.716**
- Scaling experiments overhead:
  - Replica 2: 9 experiments × 0.014 hours × $0.0971/hour × 2 nodes = $0.002
  - Replica 4: 9 experiments × 0.014 hours × $0.0971/hour × 4 nodes = $0.005
  - Replica 8: 9 experiments × 0.014 hours × $0.0971/hour × 8 nodes = $0.010
  - **Scaling overhead**: **$0.017**
- **Total compute**: $0.716 + $0.017 = **$0.733**

**Storage cost**:
- Disk: 50GB × 7.7 hours × $0.000236/GB/hour = **$0.091**

**Network cost**:
- Egress: **~$0.01-0.02**

**GCS storage**:
- Results: **~$0.10** (one-time, minimal)

**Total GCP cost**: **$0.93 - $0.95** = **£0.74 - £0.76** (assuming £1 = $1.25)

### Cost Breakdown by Experiment Type

| Experiment Type | Count | Time (hours) | Cost (USD) | Cost (GBP) |
|----------------|-------|--------------|------------|------------|
| Baseline (30s) | 300 | 4.17 | $0.41 | £0.33 |
| Burst pattern | 50 | 0.69 | $0.07 | £0.06 |
| 10K msg/s | 100 | 1.39 | $0.14 | £0.11 |
| 5-minute duration | 9 | 0.80 | $0.08 | £0.06 |
| Scaling baseline | 9 | 0.13 | $0.01 | £0.01 |
| Scaling (replica 2) | 9 | 0.13 | $0.02 | £0.02 |
| Scaling (replica 4) | 9 | 0.13 | $0.04 | £0.03 |
| Scaling (replica 8) | 9 | 0.13 | $0.08 | £0.06 |
| Storage/Network | - | - | $0.10-0.12 | £0.08-0.10 |
| **Total** | **495** | **7.6-7.7** | **$0.93-0.95** | **£0.74-0.76** |

**Note**: All costs are **per hour** for compute, with minimum 1-minute billing increments.

### Scheduling Recommendations

- **Can run overnight**: 10.5-12.5 hours fits in a single night
- **Can pause/resume**: Script supports graceful stop/resume
- **Best time**: When you can monitor (for cluster setup/teardown)
- **Cost optimization**: Use ephemeral clusters (current approach) - no ongoing costs
- **Checkpoints**: Progress saved after each experiment

---

## Overall Summary

### Total Time and Cost

| Environment | Experiments | Time | Cost |
|-------------|-------------|------|------|
| **Native** | 468 | 6.5-8 hours | £0 |
| **Minikube** | 495 | 8.5-11 hours | £0 |
| **GCP** | 495 | 10.5-12.5 hours | £0.74-0.76 |
| **Total** | **1,458** | **25.5-31.5 hours** | **£0.74-0.76** |

### Time Breakdown Summary

| Phase | Native | Minikube | GCP |
|-------|--------|----------|-----|
| Baseline experiments | 3.75h | 3.92h | 4.17h |
| Burst pattern | 0.625h | 0.653h | 0.694h |
| 10K msg/s | 1.25h | 1.306h | 1.389h |
| 5-minute duration | 0.788h | 0.793h | 0.8h |
| Scaling baseline | 0.113h | 0.118h | 0.125h |
| Scaling (replicas 2,4,8) | 0h | 0.353h | 0.375h |
| Environment overhead | 0h | 0.5-1h | 0.5-1h |
| Cluster lifecycle | 0h | 0h | 0.25h |
| System variability | 0.5-1.5h | 0.5-1.5h | 0.5-1.5h |
| **Total** | **6.5-8h** | **8.5-11h** | **10.5-12.5h** |

---

## Cost Optimization Strategies

### 1. Use Ephemeral Clusters (Current Approach)

**Benefit**: No ongoing costs, fresh environment each time
**Cost**: Setup/teardown overhead (~13-20 minutes × $0.0971/hour = $0.02-0.03 per run)

**Recommendation**: ✅ Use ephemeral for cost efficiency

### 2. Sequential vs. Parallel Execution

**Sequential** (current):
- **Time**: 10.5-12.5 hours
- **Cost**: £0.74-0.76
- **Benefit**: Lower cost

**Parallel** (10 experiments simultaneously):
- **Time**: ~1-2 hours (10× faster)
- **Cost**: ~£7-8 (10× cluster cost)
- **Trade-off**: 10× faster but 10× more expensive

**Recommendation**: Sequential for cost efficiency, parallel only if time-critical

### 3. Preemptible Instances

**Standard**: $0.0971/hour
**Preemptible**: ~$0.0235/hour (76% discount)
- ⚠️ **Risk**: Can be terminated (experiments may need restart)
- ✅ **Benefit**: 76% cost savings
- **Potential savings**: £0.74-0.76 → **£0.18-0.20** (saves ~£0.56-0.58)

**Recommendation**: Consider preemptible for non-critical runs, standard for final data collection

---

## Scheduling Recommendations

### Recommended Execution Order

1. **Native** (6.5-8 hours, £0)
   - Can run overnight
   - Lowest risk (local machine)
   - Good for initial validation

2. **Minikube** (8.5-11 hours, £0)
   - Can run overnight
   - Tests Kubernetes orchestration
   - Validates containerized workloads

3. **GCP** (10.5-12.5 hours, £0.74-0.76)
   - Can run overnight
   - Most expensive (but still very low cost)
   - Tests cloud-native scaling

### Time Management Tips

- **Sequential execution**: 25.5-31.5 hours total (can run over multiple days)
- **Parallel execution**: 2-3 hours (10× cost, 10× faster)
- **Recommendation**: Sequential for cost efficiency, run over 2-3 days
- **Resume capability**: All scripts support graceful stop/resume
- **Progress tracking**: Use `./scripts/check_progress.sh` to monitor progress

### Cost Management Tips

1. **Use ephemeral clusters** (current approach) - no ongoing costs
2. **Run sequentially** - minimizes cluster time
3. **Consider preemptible** - 76% savings if acceptable risk
4. **Monitor GCS storage** - archive old results to reduce storage costs
5. **Check billing** - Monitor GCP console for unexpected charges

---

## Conclusion

**Complete Experiment Suite**:
- ✅ **1,458 experiments** across 3 environments
- ✅ **25.5-31.5 hours** total time (can be spread over multiple days)
- ✅ **£0.74-0.76 total cost** (GCP only, native/minikube free)
- ✅ **Resume capability** - Can pause and resume at any time
- ✅ **Progress tracking** - Real-time progress indicators

**Recommendation**: ✅ **Proceed with complete experiment suite** - Low cost, comprehensive coverage, flexible scheduling.

---

## Cost Units Clarification

**All costs are per hour for compute resources**:
- n2-standard-2: **$0.0971 per hour** (billed in 1-minute increments, minimum 1 minute)
- Storage: **$0.000236 per GB per hour** (pro-rated from monthly rate)
- Network: **Per GB** (first 1GB free per month)

**Total costs are one-time** for running the complete experiment suite (not recurring).
