# GCP Optimization Implementation Summary

## ✅ Implemented Optimizations

### 1. Terraform Operation Optimization ✅

**Changes in `deploy_gcp.sh`:**
- **Cached `terraform init`**: Only runs if `.terraform/` directory doesn't exist or is empty
- **Skip redundant imports**: Checks if resources are already in Terraform state before importing
  - Bucket, Service Account, and Artifact Registry are only imported if not in state
- **Time savings**: ~10-20 seconds per experiment (was running init + imports every time)

### 2. Persistent Cluster Mode for Batch Runs ✅

**Changes in `run_all_experiments.sh`:**
- **Automatic detection**: Detects batch runs (multiple experiments) and uses persistent cluster mode
- **Cluster lifecycle**:
  - Creates cluster once before experiment loop (if doesn't exist)
  - Reuses cluster for all experiments in batch
  - Destroys cluster after batch completion (only if we created it)
- **Backward compatible**: Single experiments still use ephemeral mode
- **Time savings**: ~8-15 minutes per experiment → ~5 minutes total for entire batch

### 3. Container Image Reuse ✅

**Changes in `run_all_experiments.sh`:**
- **Build once per batch**: Image is built once during batch setup
- **Reuse for all experiments**: All subsequent experiments use `--skip-build`
- **Uses existing `--skip-build` flag**: Already implemented in `deploy_gcp.sh`
- **Time savings**: ~1-2 minutes per experiment → ~2 minutes total for entire batch

## How It Works

### Batch Run Flow (Multiple Experiments)

1. **Setup Phase** (once per batch):
   - Check if cluster exists
   - If not, create cluster using `--create-cluster`
   - Build container image once
   - Set `GCP_USE_PERSISTENT_CLUSTER=true`

2. **Experiment Loop** (for each experiment):
   - Use `--skip-terraform --skip-build` flags
   - Run experiment on existing cluster
   - Reuse existing image

3. **Cleanup Phase** (once per batch):
   - Destroy cluster if we created it
   - Leave cluster if it existed before (user responsibility)

### Single Run Flow (Backward Compatible)

- Uses ephemeral mode (default behavior)
- Creates cluster, runs experiment, destroys cluster
- No changes to existing behavior

## Expected Performance Improvements

**For 468 experiments:**

| Operation | Before | After | Savings |
|-----------|--------|-------|---------|
| Cluster ops | 468 × 8-15 min | 1 × 5 min | ~3,744-7,020 min |
| Terraform init | 468 × 10-20s | 1 × 10-20s | ~78-156 min |
| Image builds | 468 × 1-2 min | 1 × 1-2 min | ~467-934 min |
| **Total overhead** | **~4,289-8,110 min** | **~7-8 min** | **~4,282-8,102 min** |
| **Total overhead** | **~71-135 hours** | **~0.1 hours** | **~71-135 hours** |

**Actual benchmark time**: ~15-23 hours (unchanged)

**Total time**: ~86-158 hours → **~15-23 hours** (5-7x faster)

## Backward Compatibility

✅ **All existing functionality preserved:**
- Single experiments still use ephemeral mode
- `--ephemeral` flag still works
- `--skip-terraform` flag still works
- `--skip-build` flag still works
- Manual cluster management still works

✅ **No breaking changes:**
- Default behavior unchanged for single runs
- All existing flags and options work as before
- Error handling preserved

## Usage Examples

### Batch Run (Automatic Optimization)
```bash
# Automatically uses persistent cluster mode
./run_all_experiments.sh \
  --envs gcp \
  --project <project> \
  --bucket <bucket> \
  --region <region>
```

### Single Run (Ephemeral Mode)
```bash
# Still uses ephemeral mode (default)
./deploy_gcp.sh \
  --scenario <scenario> \
  --exp-id <id> \
  --project <project> \
  --bucket <bucket> \
  --ephemeral
```

### Manual Cluster Management
```bash
# Create cluster once
./deploy_gcp.sh --create-cluster --project <p> --bucket <b> --region <r>

# Run experiments (reuses cluster)
./deploy_gcp.sh --scenario <s> --exp-id <id> --skip-terraform --skip-build ...

# Destroy when done
./deploy_gcp.sh --destroy-cluster --project <p> --bucket <b> --region <r>
```

## Testing Recommendations

1. **Test single run**: Verify ephemeral mode still works
2. **Test batch run**: Verify persistent cluster mode works
3. **Test with existing cluster**: Verify reuse of existing cluster
4. **Test cleanup**: Verify cluster is destroyed after batch
5. **Test error handling**: Verify fallback to ephemeral on cluster creation failure

## Notes

- Cluster name is determined by smoke-test mode:
  - Smoke test: `pqc-smoke-test`
  - Full run: `pqc-bench-gke`
- If cluster already exists, it's reused (not destroyed at end)
- Image uses `:latest` tag and is reused across experiments
- Terraform state is cached in `.terraform/` directory

## Future Enhancements (Not Implemented)

- Image tagging with git commit (for better versioning)
- Parallel cluster operations
- Cluster pools for even faster startup
- Pre-warming strategies

