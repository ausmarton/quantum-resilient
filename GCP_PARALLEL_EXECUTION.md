# GCP Parallel Execution Strategy

## Problem
- Current: Each experiment creates/destroys cluster (12-13 min create + 8-9 min destroy = ~20-22 min overhead)
- For 468 experiments: ~154-172 hours of overhead alone!

## Solution: Parallel Execution with Single Cluster

### Approach
1. **Create cluster once** (~5 min)
2. **Submit all experiments as Kubernetes Jobs in parallel** (or in batches)
3. **Wait for all to complete**
4. **Destroy cluster once** (~5 min)

### Benefits
- **Time savings**: 20-22 min per experiment → 5 min total overhead
- **Cost**: Same total compute time, just faster execution
- **Scalability**: Can run 10-50+ experiments in parallel (depending on node capacity)

### Implementation Options

#### Option 1: Batch Parallel Execution (Recommended)
- Create cluster once
- Run experiments in batches of N (e.g., 10-20 at a time)
- Each batch runs in parallel
- Destroy cluster at end

**Time**: ~5 min setup + (468 / batch_size) × batch_time + ~5 min teardown
**Example**: 20 experiments/batch, ~2 min per batch = ~5 + 47 + 5 = ~57 min total overhead

#### Option 2: Full Parallel Execution
- Create cluster with enough nodes
- Submit all 468 experiments as Jobs simultaneously
- Wait for all to complete
- Destroy cluster

**Time**: ~5 min setup + max(experiment_times) + ~5 min teardown
**Example**: ~5 + 2-3 min + 5 = ~12-13 min total overhead

**Considerations**:
- Need enough nodes (or larger nodes) to handle all jobs
- GCS upload contention (but GCS handles this well)
- More complex error handling

### Cost Analysis

**Sequential (Current)**:
- 468 experiments × 2-3 min each = ~15-23 hours compute
- 468 × 20 min overhead = ~156 hours overhead
- **Total**: ~171-179 hours

**Parallel (20 per batch)**:
- 468 experiments × 2-3 min each = ~15-23 hours compute (same)
- ~5 min setup + ~47 min batches + ~5 min teardown = ~57 min overhead
- **Total**: ~16-24 hours (10x faster!)

**Cost**: Same compute cost, just faster execution

### Practical Considerations

1. **Node Capacity**: 
   - Each experiment needs ~1-2 CPU cores
   - n2-standard-2 has 2 vCPUs
   - Can run 1-2 experiments per node
   - For 20 parallel: need 10-20 nodes

2. **GCS Upload**:
   - GCS handles concurrent uploads well
   - No significant contention expected

3. **Error Handling**:
   - Track which jobs succeed/fail
   - Retry failed jobs
   - Continue on error

4. **Resource Limits**:
   - Kubernetes default: 110 pods per node
   - GKE default: 100 pods per node
   - Should be fine for 20-50 parallel jobs

### Implementation Plan

1. **Modify `deploy_gcp.sh`** to support batch job submission
2. **Create Kubernetes Job YAML generator** for multiple experiments
3. **Add parallel execution logic** to `run_all_experiments.sh`
4. **Add progress tracking** for parallel jobs
5. **Add error handling** and retry logic

