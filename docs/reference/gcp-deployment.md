# GCP Deployment Guide

Complete guide to deploying and running experiments on Google Cloud Platform (GCP) using Google Kubernetes Engine (GKE).

## Overview

The GCP deployment system provides:
- **Automatic cluster management**: Creates and destroys clusters as needed
- **Parallel execution**: Run multiple experiments concurrently
- **Experiment isolation**: Pod anti-affinity ensures no resource contention
- **Cost optimization**: Ephemeral mode for single experiments, persistent clusters for batches
- **Automatic sizing**: Cluster size calculated based on parallelism

## Architecture

### Execution Modes

#### Ephemeral Mode (Single Experiments)
- Creates cluster for each experiment
- Destroys cluster after completion
- **Use case**: Single experiments, cost isolation
- **Overhead**: ~20-22 min per experiment (cluster create/destroy)

#### Persistent Cluster Mode (Batch Runs)
- Creates cluster once for all experiments
- Reuses cluster for multiple experiments
- Destroys cluster at end
- **Use case**: Batch runs (multiple experiments)
- **Overhead**: ~5 min setup + ~5 min teardown total

### Parallel Execution

The system supports two execution patterns:

**Sequential (PARALLEL_JOBS=1)**:
- Submits jobs one at a time
- Waits for each to complete before submitting next
- **Time**: ~5 min setup + (N × experiment_time) + ~5 min teardown

**Parallel (PARALLEL_JOBS > 1)**:
- Submits N jobs simultaneously
- Waits for all to complete
- **Time**: ~5 min setup + max(experiment_times) + ~5 min teardown

## Experiment Isolation

### Pod Anti-Affinity

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

**Effect**: Kubernetes prefers to schedule each job on a different node, ensuring:
- No CPU contention between experiments
- No memory contention between experiments
- No network interference
- Maximum isolation for experiment integrity

### Resource Guarantees

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

### Unique Resources Per Experiment

Each experiment gets:
- Unique Kubernetes Job name (sanitized from experiment ID, RFC 1123 compliant)
- Unique ConfigMaps for scenario and GCP config
- Independent `emptyDir` volume (no shared storage)
- Independent GCS upload path

**Effect**: No resource conflicts between parallel experiments

## Cluster Sizing

### Automatic Sizing

The system automatically calculates node count based on `PARALLEL_JOBS`:

```bash
# Formula: PARALLEL_JOBS + 1 (for system overhead)
CALCULATED_NODE_COUNT=$((PARALLEL_JOBS + 1))
```

### Node Capacity

**n2-standard-2 nodes**:
- 2 vCPUs per node
- ~1.5 vCPUs available after system reservations
- Each job requests 800m CPU
- **Recommendation**: 1 job per node for maximum isolation

### Sizing Examples

| PARALLEL_JOBS | Node Count | Jobs/Node | Isolation Level |
|---------------|------------|-----------|-----------------|
| 1 (sequential) | 2 | 1 | Maximum |
| 10 | 11 | ~1 | Maximum |
| 20 | 21 | ~1 | Maximum |
| 50 | 51 | ~1 | Maximum |

### Why +1 Node?

The extra node provides:
- Buffer for system pods (kube-proxy, monitoring, etc.)
- Headroom for pod scheduling delays
- Safety margin for node failures
- Better pod distribution

## Private Cluster Configuration

**Configuration**: GKE nodes have **no external IP addresses** (private cluster)

**Rationale**:
1. **Enterprise Alignment**: Private clusters are standard in enterprise production environments
2. **Eliminates Network Overhead**: Removes NAT routing overhead that could introduce variability
3. **Security**: Reduces attack surface by removing external IP exposure
4. **Experiment Validity**: Since we measure CPU/crypto performance (not network latency), removing external IPs eliminates unnecessary network overhead

**Access Requirements**:
- Nodes access **Artifact Registry** via Private Google Access (for pulling container images)
- Nodes access **GCS** via Private Google Access (for uploading results)
- Nodes access **GKE control plane** via private endpoint
- **kubectl** access remains via public endpoint (for cluster management)

**Setup Requirement**:
Private Google Access must be enabled on the subnet:

```bash
gcloud compute networks subnets update default \
  --region=<region> \
  --enable-private-ip-google-access
```

**Note**: This is usually enabled by default on the default network, but should be verified.

## Usage

### Basic Usage

```bash
# Single experiment (ephemeral mode)
./deploy_gcp.sh \
  --scenario scenarios/hybrid_kyber_dilithium.yaml \
  --exp-id gcp_exp1 \
  --project YOUR_PROJECT_ID \
  --region us-central1 \
  --bucket pqc-bench-results \
  --ephemeral

# Batch run (persistent cluster mode, automatic)
./run_full_scale_data_collection.sh \
  --env gcp \
  --project YOUR_PROJECT_ID \
  --bucket pqc-bench-results \
  --region us-central1 \
  --parallel 20  # Run 20 experiments in parallel
```

### Recommended Settings

**For 468 experiments:**
- **PARALLEL_JOBS=20**: Good balance of speed and resource usage
- **PARALLEL_JOBS=50**: Maximum speed (requires more nodes)
- **PARALLEL_JOBS=10**: Conservative (fewer nodes needed)

## Cost Analysis

### Sequential (Ephemeral Mode)
- **Overhead**: 468 × 20 min = ~156 hours
- **Compute**: ~15-23 hours
- **Total**: ~171-179 hours

### Parallel (20 per batch, Persistent Cluster)
- **Overhead**: ~5 min setup + ~47 min batches + ~5 min teardown = ~57 min
- **Compute**: ~15-23 hours (same)
- **Total**: ~16-24 hours
- **Savings**: ~155 hours (10x faster!)

### Cost
- **Same compute cost**: You pay for the same total CPU time
- **Faster execution**: Results available much sooner
- **No additional cost**: Parallel execution doesn't increase costs

**Node costs**:
- **n2-standard-2**: ~$0.067/hour per node
- **20 parallel jobs**: 21 nodes × $0.067 = ~$1.41/hour
- **50 parallel jobs**: 51 nodes × $0.067 = ~$3.42/hour

## Implementation Details

### Execution Flow

```
run_all_experiments.sh
  ├─> Persistent cluster mode detected?
  │   ├─> Yes: Create cluster once (sized for PARALLEL_JOBS)
  │   │   ├─> Build image once
  │   │   └─> For each experiment:
  │   │       ├─> Submit Kubernetes Job (via submit_gcp_job_parallel.sh)
  │   │       ├─> If PARALLEL_JOBS=1: Wait for completion, download results
  │   │       └─> If PARALLEL_JOBS>1: Track job, continue
  │   │   └─> If PARALLEL_JOBS>1: Wait for all jobs, download results
  │   └─> No: Use ephemeral mode (deploy_gcp.sh per experiment)
  └─> Destroy cluster (if we created it)
```

### Job Name Generation

Job names are generated from experiment IDs with:
- RFC 1123 compliance (lowercase, alphanumeric, hyphens, max 63 chars)
- Replica suffix preservation (for scaling experiments: `_r2`, `_r4`, `_r8`)
- Base ID truncated to 49 chars (leaving room for "pqc-bench-" prefix and replica suffix)
- Unique per experiment

**Example**:
- Experiment ID: `hybrid_kyber_dilithium_p1024_r500_scaling_run1_91ed938b_r4`
- Base ID (sanitized): `hybrid-kyber-dilithium-p1024-r500-scaling-run1-91ed938b` (truncated to 49 chars)
- Replica suffix: `r4`
- Job name: `pqc-bench-{base}{suffix}` (max 63 chars total, RFC 1123 compliant)

## Troubleshooting

### Node Pool Creation Errors

If node pool creation fails, see the dedicated troubleshooting guide:
- **[GKE Node Pool Troubleshooting](../troubleshooting/gke-node-pool.md)** - Common node pool creation errors and fixes

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

### Cluster Still Being Created Per Experiment

**Issue**: Persistent cluster mode not detected

**Solution**: 
- Check that `ENV_TOTAL_EXPERIMENTS > 1`
- Verify cluster creation logs show "Detected batch run"
- Check that `GCP_USE_PERSISTENT_CLUSTER=true` is exported

## Best Practices

1. **Start Conservative**: Begin with `--parallel 10` to test
2. **Monitor Cluster**: Check node utilization with `kubectl top nodes`
3. **Adjust Based on Results**: Increase parallelism if cluster handles it well
4. **Use Persistent Cluster**: Always use for batch runs
5. **Verify Isolation**: Check pod distribution with `kubectl get pods -o wide`

## Expected Performance

**For 468 experiments with PARALLEL_JOBS=20:**

| Phase | Time |
|-------|------|
| Cluster creation | ~5 min |
| Image build | ~2 min |
| Job submission (468 jobs) | ~2-3 min |
| Job execution (20 parallel batches) | ~47 min (23 batches × ~2 min) |
| Result download | ~5-10 min |
| Cluster destruction | ~5 min |
| **Total** | **~66-72 min** |

**vs Sequential (ephemeral mode):**
- **Total**: ~171-179 hours
- **Speedup**: ~150x faster!

## Summary

- ✅ **Automatic cluster management**: Creates/destroys as needed
- ✅ **Parallel execution**: Run multiple experiments concurrently
- ✅ **Experiment isolation**: Pod anti-affinity + resource guarantees
- ✅ **Automatic sizing**: Cluster sized based on parallelism
- ✅ **Cost optimization**: Same compute cost, faster execution
- ✅ **Flexibility**: Sequential or parallel, automatic detection

