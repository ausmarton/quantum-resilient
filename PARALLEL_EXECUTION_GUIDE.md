# GCP Parallel Execution Guide

## Overview

We've implemented parallel execution for GCP experiments to dramatically reduce execution time. Instead of running experiments sequentially (with 20-22 min overhead per experiment), we now:

1. **Create cluster once** (~5 min)
2. **Submit multiple experiments as Kubernetes Jobs in parallel** 
3. **Wait for all to complete**
4. **Destroy cluster once** (~5 min)

## How It Works

### Automatic Detection

The system automatically detects when to use parallel execution:
- **Batch runs** (multiple experiments): Uses persistent cluster mode
- **PARALLEL_JOBS > 1**: Submits jobs in parallel instead of sequentially
- **Single experiments**: Still uses ephemeral mode (cost isolation)

### Execution Modes

#### Mode 1: Sequential with Persistent Cluster (PARALLEL_JOBS=1)
- Creates cluster once
- Runs experiments one at a time
- Reuses cluster and image
- **Time savings**: Eliminates 20-22 min overhead per experiment
- **Total overhead**: ~5 min setup + ~5 min teardown = ~10 min total

#### Mode 2: Parallel Execution (PARALLEL_JOBS > 1)
- Creates cluster once
- Submits N jobs in parallel (where N = PARALLEL_JOBS)
- Waits for all to complete
- Downloads results
- **Time savings**: Maximum - runs experiments concurrently
- **Total overhead**: ~5 min setup + max(experiment_times) + ~5 min teardown

## Usage

### Basic Usage (Automatic)

```bash
# Parallel execution is automatic when:
# 1. Running batch (multiple experiments)
# 2. PARALLEL_JOBS > 1

./run_full_scale_data_collection.sh \
  --env gcp \
  --project <project> \
  --bucket <bucket> \
  --region <region> \
  --parallel 20  # Run 20 experiments in parallel
```

### Recommended Settings

**For 468 experiments:**
- **PARALLEL_JOBS=20**: Good balance of speed and resource usage
- **PARALLEL_JOBS=50**: Maximum speed (requires more nodes)
- **PARALLEL_JOBS=10**: Conservative (fewer nodes needed)

**Node requirements:**
- Each experiment needs ~1-2 CPU cores
- n2-standard-2 has 2 vCPUs
- For 20 parallel: need 10-20 nodes
- For 50 parallel: need 25-50 nodes

## Cost Analysis

### Sequential (Before)
- **Overhead**: 468 × 20 min = ~156 hours
- **Compute**: ~15-23 hours
- **Total**: ~171-179 hours

### Parallel (20 per batch)
- **Overhead**: ~5 min setup + ~47 min batches + ~5 min teardown = ~57 min
- **Compute**: ~15-23 hours (same)
- **Total**: ~16-24 hours
- **Savings**: ~155 hours (10x faster!)

### Cost
- **Same compute cost**: You pay for the same total CPU time
- **Faster execution**: Results available much sooner
- **No additional cost**: Parallel execution doesn't increase costs

## Implementation Details

### What Changed

1. **Persistent Cluster Mode**: 
   - Automatically enabled for batch runs
   - Creates cluster once, reuses for all experiments
   - Exports `GCP_USE_PERSISTENT_CLUSTER` variable

2. **Parallel Job Submission**:
   - New script: `scripts/submit_gcp_job_parallel.sh`
   - Submits Kubernetes Jobs directly (bypasses deploy_gcp.sh overhead)
   - Tracks jobs and waits for completion

3. **Job Tracking**:
   - Jobs are tracked in temp file
   - After all submissions, script waits for completion
   - Downloads results for each completed job

### Resource Management

- **ConfigMaps**: Created per experiment (unique names)
- **Jobs**: Each experiment gets unique job name
- **Namespace**: Uses `default` or `pqc-smoke-test` based on mode
- **Cleanup**: Jobs auto-delete after 5 minutes (TTL)

## Troubleshooting

### Cluster Still Being Created Per Experiment

**Issue**: Persistent cluster mode not detected

**Solution**: 
- Check that `ENV_TOTAL_EXPERIMENTS > 1`
- Verify cluster creation logs show "Detected batch run"
- Check that `GCP_USE_PERSISTENT_CLUSTER=true` is exported

### Jobs Not Running in Parallel

**Issue**: Jobs submitted but running sequentially

**Solution**:
- Check `PARALLEL_JOBS` value (should be > 1)
- Verify cluster has enough nodes
- Check Kubernetes resource quotas

### Out of Resources

**Issue**: Jobs pending due to insufficient resources

**Solution**:
- Reduce `PARALLEL_JOBS` value
- Increase node count in cluster
- Use larger node machine type

## Best Practices

1. **Start with PARALLEL_JOBS=10**: Test with smaller parallel count first
2. **Monitor cluster**: Check node utilization
3. **Adjust based on results**: Increase if cluster handles it well
4. **Use persistent cluster**: Always use for batch runs
5. **Clean up**: Cluster is auto-destroyed after batch completion

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

**vs Sequential (before):**
- **Total**: ~171-179 hours
- **Speedup**: ~150x faster!

## Notes

- Parallel execution requires persistent cluster mode
- Single experiments still use ephemeral mode (cost isolation)
- All optimizations are backward compatible
- No breaking changes to existing workflows

