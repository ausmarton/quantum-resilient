# Scaling Experiments Guide

Complete guide to horizontal scaling experiments, including how they work, what they measure, and how to incorporate findings into your dissertation.

## Overview

Scaling experiments test how performance changes when running multiple replicas of the same experiment in parallel. This is important for understanding:
- How throughput scales with replica count
- How latency is affected by parallel execution
- Orchestration overhead in containerized environments
- Production deployment characteristics

## How Scaling Works

### Environment Support

| Environment | Replicas Supported | What It Tests |
|-------------|-------------------|---------------|
| **Native** | ❌ No (only replica 1) | N/A - Single process execution |
| **Minikube** | ✅ Yes (1,2,4,8) | Kubernetes orchestration overhead on single node |
| **GCP** | ✅ Yes (1,2,4,8) | True horizontal scaling across multiple nodes |

### Experiment Configuration

From `orchestration/experiment_matrix.yaml`:
- **Algorithms**: kyber512, dilithium2, hybrid_kyber_dilithium (3)
- **Payload**: 1024 bytes (1)
- **Rate**: 500 msg/s (1)
- **Runs**: 3
- **Replicas**: 1, 2, 4, 8 (4)
- **Total scaling scenarios**: 3 × 1 × 1 × 3 = 9
- **Total scaling experiments per env**: 9 × 4 = 36

### Experiment ID Format

Scaling experiments use special naming:
- **Replica 1 (base)**: `{algorithm}_p{payload}_r{rate}_scaling_run{N}_{hash}`
- **Replica 2+**: `{algorithm}_p{payload}_r{rate}_scaling_run{N}_{hash}_r{replicas}`

**Example**:
- Replica 1: `kyber512_p1024_r500_scaling_run1_2a737cea`
- Replica 2: `kyber512_p1024_r500_scaling_run1_2a737cea_r2`
- Replica 4: `kyber512_p1024_r500_scaling_run1_2a737cea_r4`
- Replica 8: `kyber512_p1024_r500_scaling_run1_2a737cea_r8`

## Implementation Details

### Minikube Scaling

**How it works**:
- Creates Kubernetes Jobs with `parallelism: N` and `completions: N`
- Multiple pods run **on the same physical machine** (single-node Minikube cluster)
- All pods share the same CPU, memory, and network resources

**What you're actually testing**:
- ✅ Kubernetes orchestration overhead (scheduling, pod creation, etc.)
- ✅ Container runtime overhead
- ⚠️ **Not true horizontal scaling** (all pods on same machine)
- ⚠️ Resource contention (shared CPU/memory)

**Limitations**:
- Single-node cluster means no true horizontal scaling
- Resource contention affects results
- Results show orchestration overhead, not scaling efficiency

### GCP Scaling

**How it works**:
- Creates Kubernetes Jobs with `parallelism: N` and `completions: N`
- Multiple pods run **across multiple nodes** (multi-node GKE cluster)
- Each pod gets dedicated node resources (via pod anti-affinity)

**What you're actually testing**:
- ✅ True horizontal scaling across nodes
- ✅ Network overhead between nodes
- ✅ Distributed execution characteristics
- ✅ Production-like deployment

**Advantages**:
- True horizontal scaling
- Better isolation (1 pod per node)
- More representative of production

## Running Scaling Experiments

### Automatic (Recommended)

Scaling experiments run automatically when using `--replicas 1,2,4,8`:

```bash
# Minikube with scaling
./run_all_experiments.sh \
  --envs minikube \
  --replicas 1,2,4,8 \
  --matrix orchestration/experiment_matrix.yaml

# GCP with scaling
./run_all_experiments.sh \
  --envs gcp \
  --replicas 1,2,4,8 \
  --project YOUR_PROJECT \
  --bucket YOUR_BUCKET \
  --matrix orchestration/experiment_matrix.yaml
```

### Expected Experiment Counts

**Native**:
- **Total**: 468 experiments
- **Scaling experiments**: 9 scenarios × 1 replica = 9 (replicas > 1 are skipped)
- **Non-scaling experiments**: 459 scenarios × 1 replica = 459
- **Total**: 9 + 459 = 468 ✅

**Minikube / GCP**:
- **Total**: 495 experiments
- **Scaling experiments**: 9 scenarios × 4 replicas (1,2,4,8) = 36
- **Non-scaling experiments**: 459 scenarios × 1 replica = 459
- **Total**: 36 + 459 = 495 ✅

## Analysis

### Automatic Analysis

When `--replicas 1,2,4,8` is used, the analysis pipeline automatically generates scaling plots:

```bash
# Phase 9: Replica Scaling Analysis
python3 analysis/plot_replica_scaling.py \
    --index final-results/index.json \
    --output final-results/figures/scaling
```

### Generated Outputs

**Plots Generated** (4 total):
1. **Throughput scaling**: Throughput vs replica count
2. **Latency scaling**: Latency (p50, p95, p99) vs replica count
3. **Scaling efficiency**: Actual vs ideal scaling
4. **Per-replica breakdown**: Individual replica performance

**Metrics JSON**:
- Scaling efficiency (actual/ideal)
- Throughput improvement per replica
- Latency degradation per replica
- Statistical significance tests

### Manual Analysis

If you need to run analysis separately:

```bash
# After collecting all data
python3 analysis/plot_replica_scaling.py \
    --index final-results/index.json \
    --output final-results/figures/scaling
```

## Dissertation Integration

### Framing Strategy

Since native doesn't support scaling, frame the analysis as:

**✅ Recommended Framing:**
- **Native**: Baseline single-process performance
- **Minikube**: Container orchestration overhead analysis
- **GCP**: Production horizontal scaling analysis

**Key Claims You Can Make:**
1. **Containerization Impact**: Compare native (baseline) vs Minikube (orchestration overhead)
2. **Scaling Characteristics**: Analyze how performance changes with replica count in GCP
3. **Orchestration Overhead**: Quantify Kubernetes overhead from Minikube results
4. **Production Deployment**: Use GCP results to characterize production behavior

**Claims to Avoid:**
- ❌ "True horizontal scaling comparison across all environments" (native doesn't support it)
- ❌ "Scaling efficiency in native environment" (not applicable)

### Example Dissertation Paragraph

```
Horizontal Scaling Analysis

To understand how PQC algorithms perform in production deployments, we conducted 
scaling experiments using Kubernetes horizontal scaling. Experiments were run 
with 1, 2, 4, and 8 replicas on both Minikube (local containerized environment) 
and GCP (production cloud environment).

Minikube results demonstrate Kubernetes orchestration overhead, showing the 
performance impact of container scheduling and resource management on a single 
physical node. GCP results demonstrate true horizontal scaling across multiple 
nodes, providing insights into production deployment characteristics.

Results show that throughput scales sub-linearly with replica count, with 
scaling efficiency of X% at 8 replicas. Latency increases by Y% per additional 
replica, indicating [interpretation based on results].
```

## Current Status

### GCP Scaling Experiments
✅ **Complete:** 36/36 experiments
- Replica 1 (base): 9 experiments
- Replica 2: 9 experiments  
- Replica 4: 9 experiments
- Replica 8: 9 experiments

### Minikube Scaling Experiments
✅ **Complete:** 36/36 experiments
- Replica 1 (base): 9 experiments
- Replica 2: 9 experiments
- Replica 4: 9 experiments
- Replica 8: 9 experiments

**Note:** All scaling experiments have been successfully collected and validated.

## Troubleshooting

### Scaling Experiments Not Running

**Issue**: Scaling experiments skipped or not generated

**Solution**:
- Verify `--replicas 1,2,4,8` is passed to `run_all_experiments.sh`
- Check that experiment matrix includes scaling scenarios
- Verify environment supports replicas (native skips automatically)

### Job Name Collisions

**Issue**: Kubernetes job name collisions for scaling experiments

**Solution**: 
- Fixed in `scripts/submit_gcp_job_parallel.sh` and `run_minikube.sh`
- Replica suffix (`_r2`, `_r4`, `_r8`) is preserved in job names
- Base experiment ID is truncated to 49 chars (leaving room for "pqc-bench-" prefix and replica suffix)
- Final job name format: `pqc-bench-{base-id}{replica-suffix}` (max 63 chars total)

### Missing Replica Data

**Issue**: Some replicas missing data files

**Solution**:
- Check PVC access for Minikube (read pod creation)
- Verify GCS uploads for GCP
- Check job completion status: `kubectl get jobs`

## Summary

- ✅ **Scaling experiments**: 36 per environment (Minikube, GCP)
- ✅ **Automatic execution**: Runs when `--replicas 1,2,4,8` is specified
- ✅ **Automatic analysis**: Generates scaling plots and metrics
- ✅ **Dissertation ready**: All data collected and validated
- ✅ **Framing strategy**: Native as baseline, Minikube for overhead, GCP for scaling

