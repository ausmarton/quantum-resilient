# GCP Isolation and Cluster Sizing Guide

## Overview

This document explains how we ensure experiment isolation and properly size clusters for parallel execution.

## Experiment Isolation

### 1. Pod Anti-Affinity

Each Kubernetes Job uses pod anti-affinity to spread across nodes:

```yaml
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchLabels:
              app: pqc-bench
              component: worker
          topologyKey: kubernetes.io/hostname
```

**Effect**: Kubernetes will prefer to schedule each job on a different node, ensuring:
- No CPU contention between experiments
- No memory contention between experiments
- No network interference
- Maximum isolation for experiment integrity

### 2. Resource Requests and Limits

Each job has explicit resource requests and limits:

```yaml
resources:
  requests:
    cpu: "800m"      # Guaranteed CPU allocation
    memory: "1Gi"    # Guaranteed memory allocation
  limits:
    cpu: "4"         # Maximum CPU (can burst if available)
    memory: "4Gi"    # Maximum memory
```

**Effect**: 
- Kubernetes reserves resources for each job
- Prevents resource starvation
- Ensures consistent performance
- Limits resource usage to prevent interference

### 3. Unique ConfigMaps Per Experiment

Each experiment gets unique ConfigMaps:
- `pqc-scenario-${EXP_ID}`: Scenario configuration
- `pqc-gcp-config-${EXP_ID}`: GCP configuration

**Effect**: No configuration conflicts between parallel experiments.

### 4. Unique Job Names

Each experiment gets a unique Kubernetes Job name:
- Sanitized from experiment ID
- DNS-1123 compliant
- Prevents naming conflicts

## Cluster Sizing

### Automatic Sizing Based on Parallel Jobs

The system automatically calculates node count based on `PARALLEL_JOBS`:

```bash
# Formula: PARALLEL_JOBS + 1 (for system overhead)
CALCULATED_NODE_COUNT=$((PARALLEL_JOBS + 1))
```

### Node Capacity

**n2-standard-2 nodes:**
- 2 vCPUs per node
- ~1.5 vCPUs available after system reservations
- Each job requests 800m CPU
- **Recommendation**: 1 job per node for maximum isolation

### Sizing Examples

| PARALLEL_JOBS | Node Count | Jobs/Node | Isolation Level |
|---------------|------------|-----------|-----------------|
| 1 (sequential) | 1 | 1 | Maximum |
| 10 | 11 | ~1 | Maximum |
| 20 | 21 | ~1 | Maximum |
| 50 | 51 | ~1 | Maximum |

### Why +1 Node?

The extra node provides:
- Buffer for system pods (kube-proxy, monitoring, etc.)
- Headroom for pod scheduling delays
- Safety margin for node failures
- Better pod distribution

## Resource Isolation Guarantees

### CPU Isolation

- **Requests**: 800m CPU guaranteed per job
- **Limits**: 4 CPU maximum (burst capability)
- **Node capacity**: 2 vCPUs per n2-standard-2
- **Isolation**: 1 job per node ensures no CPU contention

### Memory Isolation

- **Requests**: 1Gi memory guaranteed per job
- **Limits**: 4Gi memory maximum
- **Node capacity**: ~7.5Gi available per n2-standard-2
- **Isolation**: Sufficient headroom prevents memory pressure

### Network Isolation

- Each pod gets its own network namespace
- No shared network resources
- GCS uploads are independent
- No network interference between experiments

### Storage Isolation

- Each job uses `emptyDir` volume (ephemeral)
- Results uploaded to GCS immediately
- No shared storage between jobs
- No I/O contention

## Experiment Integrity

### What We Guarantee

1. **No Resource Contention**: Each job has guaranteed resources
2. **Node-Level Isolation**: Pod anti-affinity spreads jobs across nodes
3. **Independent Execution**: No shared state between experiments
4. **Consistent Environment**: Same machine type, same resources for all
5. **Reproducible Results**: Isolated execution ensures consistent performance

### What We Don't Guarantee

1. **Network Latency**: GCS uploads may have variable latency
2. **Node Hardware**: Different nodes may have slight hardware variations
3. **Timing**: Parallel execution means experiments run concurrently (not sequentially)

## Best Practices

### 1. Start Conservative

```bash
# Start with smaller parallel count
--parallel 10
```

### 2. Monitor Cluster

```bash
# Check node utilization
kubectl top nodes

# Check pod distribution
kubectl get pods -o wide
```

### 3. Adjust Based on Results

- If jobs are pending: Increase node count
- If nodes are underutilized: Increase parallel jobs
- If experiments show variance: Ensure proper isolation

### 4. Verify Isolation

```bash
# Check pod distribution across nodes
kubectl get pods -o wide | grep pqc-bench

# Should show pods spread across different nodes
```

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

## Troubleshooting

### Jobs Pending

**Symptom**: Jobs stuck in Pending state

**Cause**: Insufficient nodes or resources

**Solution**: 
- Increase node count: `--node-count N`
- Reduce parallel jobs: `--parallel N`
- Check node capacity: `kubectl describe nodes`

### Jobs on Same Node

**Symptom**: Multiple jobs scheduled on same node

**Cause**: Not enough nodes or anti-affinity not working

**Solution**:
- Increase node count
- Check pod anti-affinity: `kubectl describe pod <pod-name>`
- Verify cluster has enough nodes

### Resource Contention

**Symptom**: Experiments show high variance

**Cause**: Jobs sharing nodes or insufficient resources

**Solution**:
- Ensure 1 job per node (increase node count)
- Verify resource requests are set correctly
- Check for other workloads on cluster

## Summary

- **Isolation**: Pod anti-affinity + resource requests ensure maximum isolation
- **Sizing**: Automatic calculation based on PARALLEL_JOBS
- **Integrity**: Independent execution with guaranteed resources
- **Cost**: Same compute cost, faster execution
- **Flexibility**: PARALLEL_JOBS=1 for sequential, >1 for parallel

