# Cluster Sizing Analysis: Resource Requirements vs. Dissertation Claims

## Executive Summary

**Question**: Do we have enough resources to run 20-50 parallel experiments and make meaningful dissertation claims?

**Answer**: 
- ⚠️ **Current sizing is adequate for experiment execution** but has limitations
- ❌ **Resource constraints may limit dissertation claims** about enterprise-scale performance
- ✅ **Isolation is good** (1 job per node with anti-affinity)
- ⚠️ **Node size may be too small** for high-rate experiments
- 🎯 **Recommendation**: Consider larger nodes or adjust claims

---

## Current Resource Requirements

### Per Experiment (Pod)

**Total pod resources**:
- Init container: 100m CPU, 128Mi memory
- Main container: 800m CPU, 1Gi memory  
- Sidecar (upload): 100m CPU, 256Mi memory
- **Total per pod**: ~1000m CPU, ~1.4Gi memory

**Resource limits** (burst capability):
- CPU: 4 cores (can burst if available)
- Memory: 4Gi (can use more if available)

### Per Node (n2-standard-2)

**Node capacity**:
- Total: 2 vCPUs, ~8Gi memory
- Available after system: ~1.5 vCPUs, ~7.5Gi memory
- System overhead: ~0.5 vCPU, ~0.5Gi memory (kubelet, kube-proxy, etc.)

**Capacity per node**:
- Can fit: 1 pod comfortably (1.0 vCPU requested, 1.5 available)
- Can fit: 2 pods if sharing (2.0 vCPU requested, 1.5 available) ❌ **Not recommended**
- Memory: 1 pod uses 1.4Gi, node has 7.5Gi available ✅ **Plenty of headroom**

---

## Current Sizing Formula

```bash
CALCULATED_NODE_COUNT = PARALLEL_JOBS + 1
```

### Analysis for 20 Parallel Jobs

**Cluster size**: 21 nodes
- Total capacity: 21 × 1.5 = 31.5 vCPUs available
- Requested: 20 × 1.0 = 20 vCPUs
- **Utilization**: 63% CPU, 37% memory
- **Headroom**: 11.5 vCPUs (37% unused)

**Isolation**: ✅ Good
- 1 job per node (with anti-affinity)
- No CPU contention
- No memory pressure

**Can it run?**: ✅ Yes, but...

**Issues**:
1. **Low utilization**: 37% of cluster unused (wasteful)
2. **Node size**: n2-standard-2 may be too small for high-rate experiments
3. **Burst capability**: Jobs can burst to 4 CPU, but nodes only have 1.5 available

### Analysis for 50 Parallel Jobs

**Cluster size**: 51 nodes
- Total capacity: 51 × 1.5 = 76.5 vCPUs available
- Requested: 50 × 1.0 = 50 vCPUs
- **Utilization**: 65% CPU, 37% memory
- **Headroom**: 26.5 vCPUs (35% unused)

**Isolation**: ✅ Good
- 1 job per node
- No contention

**Can it run?**: ✅ Yes

**Issues**:
1. **Very low utilization**: 35% of cluster unused
2. **Cost**: 51 nodes × $0.067/hour = $3.42/hour (expensive for low utilization)
3. **Node size**: Still may be too small for high-rate experiments

---

## Resource Constraints for Dissertation Claims

### Claim 1: Horizontal Scaling Analysis

**From HORIZONTAL_SCALING_DISSERTATION_GUIDE.md**:
- Need to answer: "How does throughput scale with replicas?"
- Need to answer: "What is scaling efficiency?"
- Need to answer: "How does latency degrade with scaling?"

**Current setup**:
- ✅ **Can answer**: Scaling experiments run with replicas 1,2,4,8
- ✅ **Isolation**: Good (1 job per node)
- ⚠️ **Limitation**: Each experiment runs independently (not true horizontal scaling)
- ⚠️ **Issue**: Scaling experiments are separate jobs, not replicas of same service

**Problem**: 
- Current setup runs **separate experiments** in parallel (different scenarios)
- Scaling analysis needs **same experiment** with **different replica counts**
- These are **different things**!

**What's actually happening**:
- Experiment 1 (replica 1): kyber512_p1024_r500_run1
- Experiment 2 (replica 2): kyber512_p1024_r500_run1_r2 (different job)
- These are **separate jobs**, not replicas of the same service

**For true scaling analysis**, we need:
- Same service running with N replicas
- All replicas handling the same workload
- Measure aggregate throughput

**Current setup provides**:
- Independent experiments with different replica counts
- Can compare performance, but not true horizontal scaling

**Can we make scaling claims?**: ⚠️ **Partially**
- ✅ Can claim: "Performance with N replicas vs 1 replica"
- ❌ Cannot claim: "True horizontal scaling behavior" (these are separate experiments)
- ✅ Can claim: "Replica count impact on performance"

### Claim 2: Enterprise-Scale Performance

**From ENTERPRISE_REPRESENTATIVENESS_ANALYSIS.md**:
- Enterprise needs: 10K-1M+ msg/s
- Current framework: 100-2000 msg/s
- **Gap**: 50-500× lower

**Current node size (n2-standard-2)**:
- 2 vCPUs per node
- Can handle: ~2000-5000 msg/s per node (estimated)
- **Cannot handle**: 10K+ msg/s per node

**For enterprise-scale claims**, we need:
- Larger nodes (n2-standard-4 or n2-standard-8)
- Higher message rates (10K+ msg/s)
- Longer duration (hours, not seconds)

**Current setup**:
- ❌ Node size too small for enterprise rates
- ❌ Message rates too low (max 2000 msg/s)
- ❌ Duration too short (30 seconds)

**Can we make enterprise claims?**: ❌ **No**
- Cannot claim "enterprise-scale performance"
- Can claim "algorithmic performance characteristics"
- Can claim "relative performance comparison"

### Claim 3: Experiment Integrity

**Requirement**: Experiments must not interfere with each other

**Current setup**:
- ✅ Pod anti-affinity spreads jobs across nodes
- ✅ 1 job per node (with current sizing)
- ✅ Resource requests guarantee allocation
- ✅ No CPU contention (each job gets dedicated node)

**Can we make integrity claims?**: ✅ **Yes**
- Experiments are isolated
- No interference between experiments
- Results are reproducible

---

## Problems with Current Sizing

### Problem 1: Low Resource Utilization

**Current**: 35-37% CPU utilization
- 20 parallel: 20 vCPUs used, 11.5 vCPUs unused
- 50 parallel: 50 vCPUs used, 26.5 vCPUs unused

**Impact**:
- Wasteful (paying for unused resources)
- Could fit more jobs per node
- But would reduce isolation

**Solution options**:
1. **Accept low utilization** (current approach) - better isolation
2. **Allow 2 jobs per node** - better utilization, less isolation
3. **Use larger nodes** - better utilization, more capacity per job

### Problem 2: Node Size May Be Too Small

**n2-standard-2 limitations**:
- 2 vCPUs may not be enough for high-rate experiments
- Jobs can burst to 4 CPU, but node only has 1.5 available
- High-rate experiments (2000 msg/s) may be CPU-bound

**Evidence needed**:
- Check if experiments are CPU-bound
- Monitor actual CPU usage during experiments
- Check if higher rates are achievable

**Solution options**:
1. **Use n2-standard-4** (4 vCPUs) - more headroom
2. **Use n2-standard-8** (8 vCPUs) - even more headroom
3. **Keep n2-standard-2** - if experiments don't need more

### Problem 3: Scaling Experiments Are Separate Jobs

**Current behavior**:
- Replica 1: Separate job with 1 worker
- Replica 2: Separate job with 2 workers (different job)
- Replica 4: Separate job with 4 workers (different job)
- Replica 8: Separate job with 8 workers (different job)

**This is NOT true horizontal scaling**:
- True scaling: Same service, add replicas, measure aggregate throughput
- Current: Different jobs, different worker counts, compare results

**For true scaling**, we need:
- Kubernetes Deployment with replicas: 1, 2, 4, 8
- Same workload, same service
- Measure aggregate performance

**Current setup provides**:
- Worker pool scaling (within single job)
- Not Kubernetes replica scaling

**Can we make scaling claims?**: ⚠️ **Limited**
- ✅ Can claim: "Worker count impact on performance"
- ❌ Cannot claim: "Kubernetes horizontal scaling behavior"
- ⚠️ Can claim: "Scaling characteristics" (with caveats)

---

## Recommendations

### Option 1: Keep Current Sizing (Recommended for Now)

**Pros**:
- ✅ Good isolation (1 job per node)
- ✅ No interference
- ✅ Experiment integrity guaranteed
- ✅ Simple and predictable

**Cons**:
- ❌ Low utilization (35-37%)
- ❌ Higher cost (paying for unused resources)
- ❌ May be too small for high-rate experiments

**Best for**:
- Experiment integrity is priority
- Cost is acceptable
- Current rates are sufficient

### Option 2: Allow 2 Jobs Per Node

**Sizing**: `ceil(PARALLEL_JOBS / 2) + 1`

**For 20 parallel**: 11 nodes (instead of 21)
- 2 jobs per node (mostly)
- Better utilization (~90%)
- Lower cost

**Pros**:
- ✅ Better resource utilization
- ✅ Lower cost
- ✅ Still good isolation (2 jobs per node)

**Cons**:
- ⚠️ Some CPU sharing (may cause interference)
- ⚠️ Less isolation than 1 job per node

**Best for**:
- Cost optimization
- Acceptable to have some sharing
- Experiments are not CPU-intensive

### Option 3: Use Larger Nodes

**Sizing**: Use n2-standard-4 or n2-standard-8
- More CPU per node
- Can handle higher rates
- Better for enterprise-scale claims

**For 20 parallel with n2-standard-4**:
- 11 nodes (2 jobs per node)
- 4 vCPUs per node (2 vCPUs per job)
- Better headroom for high rates

**Pros**:
- ✅ More CPU per job
- ✅ Can handle higher rates
- ✅ Better for enterprise claims
- ✅ Still good isolation

**Cons**:
- ❌ Higher cost per node
- ❌ May still not be enough for enterprise rates

**Best for**:
- Higher message rates needed
- Enterprise-scale claims
- Cost is acceptable

### Option 4: Hybrid Approach

**Use different node sizes for different experiment types**:
- **Baseline experiments**: n2-standard-2 (current)
- **High-rate experiments**: n2-standard-4 or n2-standard-8
- **Scaling experiments**: Larger nodes for true scaling

**Pros**:
- ✅ Right-sized for each experiment type
- ✅ Cost-optimized
- ✅ Supports different claims

**Cons**:
- ⚠️ More complex
- ⚠️ Need to manage multiple node pools

---

## What Claims Can We Actually Make?

### ✅ Can Make These Claims

1. **Algorithmic Performance Comparison**
   - "PQC algorithm X is Y% faster than classical baseline Z"
   - "Relative performance characteristics"
   - "Statistical significance of differences"

2. **Deployment Context Impact**
   - "Containerization adds X% overhead"
   - "Cloud deployment shows Y% variability"
   - "Environment choice impacts performance by Z%"

3. **Worker Count Impact**
   - "Performance with N workers vs 1 worker"
   - "Worker pool scaling characteristics"
   - "Optimal worker count for algorithm X"

4. **Experiment Integrity**
   - "Experiments are isolated (1 job per node)"
   - "No interference between experiments"
   - "Results are reproducible"

### ⚠️ Can Make with Caveats

1. **Scaling Characteristics**
   - "Worker count scaling shows X× speedup with N workers"
   - **Caveat**: "This is worker pool scaling, not Kubernetes replica scaling"
   - **Caveat**: "True horizontal scaling requires Kubernetes Deployment replicas"

2. **Scalability Trends**
   - "Performance scales linearly with load (within tested range)"
   - **Caveat**: "Tested up to 2000 msg/s, enterprise systems may process 10K-1M+ msg/s"
   - **Caveat**: "Results are expected to scale, but require production validation"

### ❌ Cannot Make These Claims

1. **Enterprise-Scale Performance**
   - ❌ "System can handle enterprise-scale loads" (rates too low, nodes too small)
   - ❌ "Production workload representativeness" (duration too short, patterns too simple)

2. **True Horizontal Scaling**
   - ❌ "Kubernetes horizontal scaling behavior" (current setup uses worker pools, not replicas)
   - ❌ "Replica scaling efficiency" (these are separate jobs, not replicas)

3. **Production Validation**
   - ❌ "Production-ready performance" (not validated in production)
   - ❌ "24/7 operation validated" (not tested)

---

## Specific Recommendations for Dissertation

### For Scaling Analysis Claims

**Current setup provides**:
- Worker pool scaling (within single job)
- Comparison across different worker counts

**Reframe claims as**:
- ✅ "Worker pool scaling characteristics"
- ✅ "Impact of worker count on performance"
- ⚠️ "Scaling trends" (with caveat about true horizontal scaling)

**For true horizontal scaling**, need:
- Kubernetes Deployment with replicas
- Same service, different replica counts
- Aggregate throughput measurement

### For Enterprise-Scale Claims

**Current setup limitations**:
- Node size: n2-standard-2 (2 vCPUs) - too small
- Message rates: 100-2000 msg/s - too low
- Duration: 30 seconds - too short

**Reframe claims as**:
- ✅ "Algorithmic performance characteristics"
- ✅ "Relative performance comparison"
- ✅ "Scalability trends (within tested range)"
- ❌ Avoid: "Enterprise-scale", "Production workloads", "Real-world"

### For Resource Utilization

**Current**: 35-37% utilization

**Options**:
1. **Accept low utilization** - prioritize isolation
2. **Allow 2 jobs per node** - better utilization, some sharing
3. **Use larger nodes** - more capacity, better for high rates

**Recommendation**: Keep current (1 job per node) for experiment integrity, but acknowledge low utilization in methodology.

---

## Action Items

**Note**: Implementation-related action items have been moved to `TODO.md` for centralized tracking.

### Dissertation Writing Tasks (External to Codebase)

These are documentation tasks for the dissertation itself:

1. **Clarify scaling experiment design**:
   - Document whether experiments use worker pool scaling or Kubernetes replica scaling
   - Update claims to match actual design
   - Note: Current implementation uses worker pool scaling (not Kubernetes replica scaling)

2. **Document resource utilization**:
   - Acknowledge low utilization in methodology
   - Justify 1 job per node for isolation
   - Consider cost implications

### Verification Tasks (Completed During Data Collection)

1. ✅ **Node size verification**: Completed during GCP data collection
2. ✅ **CPU usage monitoring**: Completed during experiments

### Future Work (Not Tracked in TODO.md)

These are optional enhancements beyond current dissertation scope:

1. **Test with larger nodes**: Optional optimization
2. **Consider 2 jobs per node**: Optional optimization
3. **Add true Kubernetes scaling**: Future research direction

---

## Summary

### Current Sizing Assessment

| Aspect | Status | Notes |
|--------|--------|-------|
| **Can run 20-50 parallel?** | ✅ Yes | Adequate resources |
| **Isolation?** | ✅ Good | 1 job per node |
| **Integrity?** | ✅ Good | No interference |
| **Utilization?** | ⚠️ Low | 35-37% CPU |
| **Node size adequate?** | ⚠️ Maybe | May be too small for high rates |
| **Supports scaling claims?** | ⚠️ Limited | Worker pools, not true replicas |
| **Supports enterprise claims?** | ❌ No | Rates too low, nodes too small |

### Key Findings

1. **Current sizing works** for running experiments
2. **Isolation is good** (1 job per node)
3. **Utilization is low** (35-37% - wasteful but safe)
4. **Node size may be limiting** for high-rate experiments
5. **Scaling experiments are worker pools**, not true Kubernetes replicas
6. **Cannot make enterprise-scale claims** with current setup

### Recommendations

1. **Keep current sizing** for experiment integrity
2. **Reframe dissertation claims** to match actual capabilities
3. **Test node size** to verify adequacy for high rates
4. **Consider larger nodes** if high-rate experiments are CPU-bound
5. **Clarify scaling design** - worker pools vs Kubernetes replicas

