# Cost and Time Analysis: Refined and Corrected

## Executive Summary

**Question**: How much existing data is invalidated, and what are the accurate time/cost implications of adding horizontal scaling + quick wins?

**Answer**:
- ✅ **Zero data invalidation** - All new experiments use separate scenario IDs
- ⏱️ **Time**: +8-10 hours total (Native: +2.5h, Minikube: +3h, GCP: +3.5h)
- 💰 **Cost**: +£0.25-0.30 for GCP experiments (europe-west2, n2-standard-2)
- 📊 **New experiments**: 531 total (54 scaling + 477 quick wins)

---

## Data Invalidation Analysis

### ✅ **NO DATA INVALIDATION**

All new experiments use **different scenario IDs** and are stored in separate directories. Your existing 225 native experiments remain 100% valid.

---

## Corrected Experiment Count Breakdown

### Current Baseline

| Environment | Experiments | Calculation |
|-------------|-------------|-------------|
| Native | 300 | 5 algorithms × 4 payloads × 3 rates × 5 runs |
| Minikube | 300 | 5 algorithms × 4 payloads × 3 rates × 5 runs |
| GCP | 300 | 5 algorithms × 4 payloads × 3 rates × 5 runs |
| **Total** | **900** | |

### New Experiments

#### 1. Horizontal Scaling (54 experiments total)

**Configuration**:
- Algorithms: 3 (kyber512, dilithium2, hybrid_kyber_dilithium)
- Payload: 1 (1024 bytes)
- Rate: 1 (500 msg/s)
- Runs: 3
- Replicas: 3 (2, 4, 8 - excluding 1 which is in baseline)
- **Environments**: 2 (Minikube, GCP - **native skipped**)

**Calculation**: 3 × 1 × 1 × 3 × 3 = **27 per environment** × 2 environments = **54 total**

| Environment | Experiments |
|-------------|-------------|
| Minikube | 27 |
| GCP | 27 |
| Native | 0 (not supported) |
| **Total** | **54** |

#### 2. Quick Wins (477 experiments total)

**These run on ALL 3 environments** (Native, Minikube, GCP):

**A. Burst Pattern (150 experiments total)**
- Algorithms: 5
- Payloads: 2 (1KB, 4KB)
- Rate: 1 (2000 msg/s)
- Runs: 5
- **Per environment**: 5 × 2 × 1 × 5 = **50 experiments**
- **Total (3 envs)**: 50 × 3 = **150 experiments**

**B. 10K msg/s Rate (300 experiments total)**
- Algorithms: 5
- Payloads: 4 (256B, 1KB, 4KB, 16KB)
- Rate: 1 (10000 msg/s)
- Runs: 5
- **Per environment**: 5 × 4 × 1 × 5 = **100 experiments**
- **Total (3 envs)**: 100 × 3 = **300 experiments**

**C. 5-minute Duration (27 experiments total)**
- Algorithms: 3 (kyber512, dilithium2, hybrid)
- Payload: 1 (1KB)
- Rate: 1 (2000 msg/s)
- Runs: 3
- Duration: 300s (5 minutes)
- **Per environment**: 3 × 1 × 1 × 3 = **9 experiments**
- **Total (3 envs)**: 9 × 3 = **27 experiments**

**Quick Wins Summary**:
- Per environment: 50 + 100 + 9 = **159 experiments**
- **Total (3 envs)**: 159 × 3 = **477 experiments**

### Combined New Experiments

| Category | Experiments |
|----------|-------------|
| Horizontal scaling | 54 |
| Quick wins | 477 |
| **Total new** | **531** |

### Grand Total (Current + New)

| Environment | Baseline | New | Total |
|-------------|----------|-----|-------|
| Native | 300 | 159 (quick wins only) | 459 |
| Minikube | 300 | 186 (27 scaling + 159 quick wins) | 486 |
| GCP | 300 | 186 (27 scaling + 159 quick wins) | 486 |
| **Total** | **900** | **531** | **1,431** |

---

## Time Estimates (Refined)

### Per-Experiment Time Breakdown

**Baseline experiments (30s duration)**:
- Experiment duration: 30 seconds
- Setup overhead: ~10 seconds (scenario load, worker startup, initialization)
- Teardown overhead: ~5 seconds (result collection, cleanup)
- **Total per experiment**: ~45 seconds

**5-minute duration experiments**:
- Experiment duration: 300 seconds (5 minutes)
- Setup overhead: ~10 seconds
- Teardown overhead: ~5 seconds
- **Total per experiment**: ~315 seconds (~5.25 minutes)

**GCP-specific overhead**:
- **Cluster setup** (one-time per run): ~10-15 minutes
- **Cluster teardown** (one-time per run): ~3-5 minutes
- **Per-experiment GCP overhead**: ~5 seconds (pod scheduling, image pull, network)

### Environment-Specific Time Estimates

#### Native (Local Machine)

**Current baseline**:
- 300 experiments × 45s = 13,500 seconds = **3.75 hours**
- With system variability, breaks: **~4-5 hours**

**New experiments (quick wins only)**:
- Burst: 50 × 45s = 2,250s = 0.625 hours
- 10K msg/s: 100 × 45s = 4,500s = 1.25 hours
- 5-minute: 9 × 315s = 2,835s = 0.788 hours
- **Total new**: 9,585s = **2.66 hours**
- With system variability: **~2.5-3 hours**

**Native total**: 4-5 hours (baseline) + 2.5-3 hours (new) = **~6.5-8 hours**

#### Minikube (Local Machine)

**Current baseline**:
- 300 experiments × 45s = 13,500 seconds = **3.75 hours**
- With container overhead: **~5-6 hours**

**New experiments**:
- Scaling: 27 × 45s = 1,215s = 0.338 hours
- Quick wins: 159 × 45s = 7,155s = 1.988 hours
- **Total new**: 8,370s = **2.33 hours**
- With container overhead: **~2.5-3 hours**

**Minikube total**: 5-6 hours (baseline) + 2.5-3 hours (new) = **~7.5-9 hours**

#### GCP (Cloud - europe-west2)

**Current baseline**:
- Cluster setup: 10-15 minutes (one-time)
- 300 experiments × 50s (45s + 5s GCP overhead) = 15,000s = 4.17 hours
- Cluster teardown: 3-5 minutes (one-time)
- **Total**: 4.17h + 0.25h (setup/teardown) = **~4.5-5 hours**
- With variability: **~6-7 hours**

**New experiments**:
- Cluster setup: 10-15 minutes (one-time, if new run)
- Scaling: 27 × 50s = 1,350s = 0.375 hours
- Quick wins: 159 × 50s = 7,950s = 2.208 hours
- **Total new runtime**: 9,300s = **2.58 hours**
- Cluster teardown: 3-5 minutes (one-time)
- **Total**: 2.58h + 0.25h (setup/teardown) = **~2.8-3 hours**
- With variability: **~3-3.5 hours**

**GCP total**: 6-7 hours (baseline) + 3-3.5 hours (new) = **~9-10.5 hours**

### Total Time Summary

| Environment | Current | New | Total |
|-------------|---------|-----|-------|
| Native | 4-5 hours | 2.5-3 hours | **6.5-8 hours** |
| Minikube | 5-6 hours | 2.5-3 hours | **7.5-9 hours** |
| GCP | 6-7 hours | 3-3.5 hours | **9-10.5 hours** |
| **Total** | **15-18 hours** | **8-9.5 hours** | **23-27.5 hours** |

**Note**: Times assume sequential execution. These can be run over multiple days.

---

## Cost Analysis (GCP europe-west2, n2-standard-2)

### GCP Resource Configuration

**From `terraform/gke/variables.tf`**:
- **Machine type**: `n2-standard-2` (2 vCPU, 8GB RAM)
- **Node count**: 1 (for baseline), 2-8 (for scaling experiments)
- **Disk**: 50GB pd-standard
- **Region**: europe-west2 (London)

### GCP Pricing (europe-west2, as of 2024-2025)

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

### Cost Calculation (Sequential Execution)

#### Current Baseline (300 GCP experiments)

**Cluster lifecycle** (one-time per run):
- Setup: 10-15 minutes = 0.167-0.25 hours
- Runtime: 4.17 hours (300 × 50s / 3600)
- Teardown: 3-5 minutes = 0.05-0.083 hours
- **Total cluster time**: ~4.5-5 hours

**Compute cost**:
- Node hours: 4.5-5 hours × $0.0971/hour = **$0.437 - $0.486**

**Storage cost**:
- Disk: 50GB × 5 hours × $0.000236/GB/hour = **$0.059**

**Network cost**:
- Egress: **~$0.01-0.02**

**GCS storage**:
- Results: **~$0.10** (one-time, minimal)

**Total current GCP cost**: **$0.60 - $0.67** = **£0.48 - £0.54** (assuming £1 = $1.25)

#### New Experiments (531 experiments, GCP portion: 186)

**GCP portion breakdown**:
- Scaling: 27 experiments
- Quick wins: 159 experiments
- **Total GCP**: 186 experiments

**Cluster lifecycle** (one-time per run):
- Setup: 10-15 minutes = 0.167-0.25 hours
- Runtime: 
  - Scaling: 27 × 50s = 1,350s = 0.375 hours
  - Quick wins: 159 × 50s = 7,950s = 2.208 hours
  - **Total runtime**: 2.583 hours
- Teardown: 3-5 minutes = 0.05-0.083 hours
- **Total cluster time**: ~2.8-3 hours

**Compute cost**:
- Node hours: 2.8-3 hours × $0.0971/hour = **$0.272 - $0.291**

**Scaling experiments with multiple replicas** (additional cost):
- Replica 2: 9 experiments × 0.01 hours × $0.0971/hour × 2 nodes = $0.002
- Replica 4: 9 experiments × 0.01 hours × $0.0971/hour × 4 nodes = $0.003
- Replica 8: 9 experiments × 0.01 hours × $0.0971/hour × 8 nodes = $0.007
- **Scaling overhead**: **$0.012**

**Storage cost**:
- Disk: 50GB × 3 hours × $0.000236/GB/hour = **$0.035**

**Network cost**:
- Egress: **~$0.01**

**Total new GCP cost**: **$0.33 - $0.35** = **£0.26 - £0.28**

#### Grand Total GCP Cost

| Category | Cost (USD) | Cost (GBP) |
|----------|------------|------------|
| Current baseline | $0.60 - $0.67 | £0.48 - £0.54 |
| New experiments | $0.33 - $0.35 | £0.26 - £0.28 |
| **Total GCP** | **$0.93 - $1.02** | **£0.74 - £0.82** |

### Cost Breakdown by Experiment Type (GCP)

| Experiment Type | Count | Time (hours) | Cost (USD) | Cost (GBP) |
|----------------|-------|--------------|------------|------------|
| Baseline (30s) | 300 | 4.5-5 | $0.44-0.49 | £0.35-0.39 |
| Scaling (replica 1) | 9 | 0.1 | $0.01 | £0.01 |
| Scaling (replica 2) | 9 | 0.1 | $0.02 | £0.02 |
| Scaling (replica 4) | 9 | 0.1 | $0.04 | £0.03 |
| Scaling (replica 8) | 9 | 0.1 | $0.08 | £0.06 |
| Burst pattern | 50 | 0.6 | $0.06 | £0.05 |
| 10K msg/s | 100 | 1.1 | $0.11 | £0.09 |
| 5-minute duration | 9 | 0.8 | $0.08 | £0.06 |
| Storage/Network | - | - | $0.10-0.12 | £0.08-0.10 |
| **Total** | **486** | **9-10.5** | **$0.93-1.02** | **£0.74-0.82** |

**Note**: All costs are **per hour** for compute, with minimum 1-minute billing increments.

---

## Cost Optimization Strategies

### 1. Use Ephemeral Clusters (Current Approach)

**Benefit**: No ongoing costs, fresh environment each time
**Cost**: Setup/teardown overhead (~13-20 minutes × $0.0971/hour = $0.02-0.03 per run)

**Recommendation**: ✅ Use ephemeral for cost efficiency

### 2. Sequential vs. Parallel Execution

**Sequential** (current):
- **Time**: 9-10.5 hours
- **Cost**: £0.74-0.82
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
- **Potential savings**: £0.74-0.82 → **£0.18-0.20** (saves ~£0.56-0.62)

**Recommendation**: Consider preemptible for non-critical runs, standard for final data collection

---

## Time and Cost Summary

### Current State (Baseline Only)

| Metric | Value |
|--------|-------|
| **Total experiments** | 900 |
| **Total time** | 15-18 hours |
| **GCP cost** | £0.48-0.54 |
| **Native/Minikube cost** | £0 (local) |
| **Total cost** | **£0.48-0.54** |

### Enhanced State (Baseline + New Experiments)

| Metric | Value |
|--------|-------|
| **Total experiments** | 1,431 (+531) |
| **Total time** | 23-27.5 hours (+8-9.5 hours) |
| **GCP cost** | £0.74-0.82 (+£0.26-0.28) |
| **Native/Minikube cost** | £0 (local) |
| **Total cost** | **£0.74-0.82** |

### Incremental Impact

| Metric | Increment |
|--------|-----------|
| **New experiments** | +531 (+59%) |
| **Additional time** | +8-9.5 hours (+47-53%) |
| **Additional cost** | +£0.26-0.28 (+54%) |
| **Data invalidation** | **0 experiments (0%)** |

---

## Per-Environment Breakdown

### Native

| Category | Experiments | Time | Cost |
|----------|-------------|------|------|
| Baseline | 300 | 4-5 hours | £0 |
| Quick wins | 159 | 2.5-3 hours | £0 |
| **Total** | **459** | **6.5-8 hours** | **£0** |

### Minikube

| Category | Experiments | Time | Cost |
|----------|-------------|------|------|
| Baseline | 300 | 5-6 hours | £0 |
| Scaling | 27 | 0.3 hours | £0 |
| Quick wins | 159 | 2.5-3 hours | £0 |
| **Total** | **486** | **7.5-9 hours** | **£0** |

### GCP

| Category | Experiments | Time | Cost |
|----------|-------------|------|------|
| Baseline | 300 | 6-7 hours | £0.48-0.54 |
| Scaling | 27 | 0.3 hours | £0.02 |
| Quick wins | 159 | 2.5-3 hours | £0.24-0.26 |
| **Total** | **486** | **9-10.5 hours** | **£0.74-0.82** |

---

## Recommendations

### ✅ **Proceed with Enhanced Experiments**

**Reasons**:
1. **Zero data invalidation** - All existing data remains valid
2. **Low cost** - Only £0.26-0.28 additional GCP cost
3. **Reasonable time** - +8-9.5 hours total (can run over multiple days)
4. **High value** - Addresses enterprise representativeness gaps
5. **Incremental** - Can run new experiments separately from baseline

### 📋 **Execution Plan**

1. **Complete current baseline** (if not done):
   - Native: 300 experiments (~4-5 hours, £0)
   - Minikube: 300 experiments (~5-6 hours, £0)
   - GCP: 300 experiments (~6-7 hours, £0.48-0.54)

2. **Run new experiments** (separate runs):
   - Native quick wins: 159 experiments (~2.5-3 hours, £0)
   - Minikube: 186 experiments (~2.5-3 hours, £0)
   - GCP: 186 experiments (~3-3.5 hours, £0.26-0.28)
   - **Total new**: 531 experiments (~8-9.5 hours, £0.26-0.28)

3. **Analysis** (same for both):
   - Combined analysis of baseline + new experiments
   - No changes needed to analysis scripts

### 💡 **Cost Optimization Tips**

1. **Use ephemeral clusters** (current approach) - no ongoing costs
2. **Run sequentially** - minimizes cluster time
3. **Consider preemptible** - 76% savings if acceptable risk (£0.74-0.82 → £0.18-0.20)
4. **Monitor GCS storage** - archive old results to reduce storage costs

### ⚠️ **Time Management**

- **Sequential execution**: 23-27.5 hours total (can run over multiple days)
- **Parallel execution**: 2-3 hours (10× cost, 10× faster)
- **Recommendation**: Sequential for cost efficiency, run over 2-3 days

---

## Conclusion

**Adding horizontal scaling + quick wins**:
- ✅ **Zero data invalidation** - All existing data remains valid
- ✅ **Low incremental cost** - Only £0.26-0.28 additional GCP cost
- ✅ **Reasonable time** - +8-9.5 hours total (can be spread over days)
- ✅ **High value** - Addresses enterprise representativeness concerns
- ✅ **Incremental** - Can be run separately from baseline

**Total cost for complete experiment suite**: **£0.74-0.82** (GCP only, native/minikube free)

**Total time for complete experiment suite**: **23-27.5 hours** (can be run over 2-3 days)

**Recommendation**: ✅ **Proceed with enhanced experiments** - Low cost, high value, zero risk to existing data.

---

## Cost Units Clarification

**All costs are per hour for compute resources**:
- n2-standard-2: **$0.0971 per hour** (billed in 1-minute increments, minimum 1 minute)
- Storage: **$0.000236 per GB per hour** (pro-rated from monthly rate)
- Network: **Per GB** (first 1GB free per month)

**Total costs are one-time** for running the complete experiment suite (not recurring).

