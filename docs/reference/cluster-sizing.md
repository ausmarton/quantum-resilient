# Cluster Sizing Reference

Technical reference for GCP cluster sizing, resource requirements, and capacity planning.

## Overview

This document provides detailed information about:
- Resource requirements per experiment
- Cluster sizing formulas
- Node capacity and utilization
- Cost considerations
- Limitations and recommendations

## Resource Requirements

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

## Sizing Formula

### Automatic Sizing

```bash
# Formula: PARALLEL_JOBS + 1 (for system overhead)
CALCULATED_NODE_COUNT=$((PARALLEL_JOBS + 1))
```

### Sizing Examples

| PARALLEL_JOBS | Node Count | Jobs/Node | CPU Utilization | Memory Utilization |
|---------------|------------|-----------|-----------------|-------------------|
| 1 (sequential) | 2 | 1 | 67% | 19% |
| 10 | 11 | ~1 | 67% | 19% |
| 20 | 21 | ~1 | 67% | 19% |
| 50 | 51 | ~1 | 67% | 19% |

### Why +1 Node?

The extra node provides:
- Buffer for system pods (kube-proxy, monitoring, etc.)
- Headroom for pod scheduling delays
- Safety margin for node failures
- Better pod distribution

## Utilization Analysis

### For 20 Parallel Jobs

**Cluster size**: 21 nodes
- Total capacity: 21 × 1.5 = 31.5 vCPUs available
- Requested: 20 × 1.0 = 20 vCPUs
- **Utilization**: 63% CPU, 37% memory
- **Headroom**: 11.5 vCPUs (37% unused)

**Isolation**: ✅ Good
- 1 job per node (with anti-affinity)
- No CPU contention
- No memory pressure

### For 50 Parallel Jobs

**Cluster size**: 51 nodes
- Total capacity: 51 × 1.5 = 76.5 vCPUs available
- Requested: 50 × 1.0 = 50 vCPUs
- **Utilization**: 65% CPU, 37% memory
- **Headroom**: 26.5 vCPUs (35% unused)

**Isolation**: ✅ Good
- 1 job per node
- No contention

## Limitations

### Node Size Constraints

**n2-standard-2 nodes**:
- ✅ Adequate for most experiments (100-2000 msg/s)
- ⚠️ May be too small for high-rate experiments (10K+ msg/s)
- ⚠️ Burst capability limited (jobs can request 4 CPU, but nodes only have 1.5 available)

### Enterprise-Scale Claims

**Current setup**:
- Node size: n2-standard-2 (2 vCPUs)
- Message rates: 100-2000 msg/s
- Duration: 30 seconds

**For enterprise-scale claims** (10K-1M+ msg/s):
- ❌ Node size too small
- ❌ Message rates too low
- ❌ Duration too short

**Recommendation**: Use larger nodes (n2-standard-4 or n2-standard-8) for enterprise-scale experiments, or adjust claims to match current capabilities.

## Cost Considerations

### Node Costs

- **n2-standard-2**: ~$0.067/hour per node
- **20 parallel jobs**: 21 nodes × $0.067 = ~$1.41/hour
- **50 parallel jobs**: 51 nodes × $0.067 = ~$3.42/hour

### Time Savings

- **Sequential**: 468 experiments × 20 min overhead = ~156 hours overhead
- **Parallel (20)**: ~57 min overhead
- **Savings**: ~155 hours = ~$10.40 at $0.067/hour/node

**Net benefit**: Faster results with same compute cost

## Recommendations

### For Current Experiments

1. **Use n2-standard-2 nodes**: Adequate for 100-2000 msg/s experiments
2. **PARALLEL_JOBS=20**: Good balance of speed and cost
3. **1 job per node**: Maximum isolation
4. **Accept low utilization**: 35% unused capacity is acceptable for isolation

### For Enterprise-Scale Experiments

1. **Use larger nodes**: n2-standard-4 or n2-standard-8
2. **Increase message rates**: 10K+ msg/s
3. **Longer duration**: Hours instead of seconds
4. **Adjust claims**: Match claims to actual capabilities

## Summary

- ✅ **Automatic sizing**: Formula-based cluster sizing
- ✅ **Good isolation**: 1 job per node with anti-affinity
- ⚠️ **Low utilization**: 35% unused capacity (acceptable for isolation)
- ⚠️ **Node size**: May be too small for enterprise-scale claims
- ✅ **Cost effective**: Same compute cost, faster execution

