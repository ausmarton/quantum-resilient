# Scaling Experiments Fix

## Issue

Scaling experiments were not being run with multiple replicas (1,2,4,8) for minikube and GCP environments. The count showed 468 experiments for all environments, but minikube and GCP should have additional experiments for scaling.

## Root Cause

`run_full_scale_data_collection.sh` was not passing `--replicas 1,2,4,8` to `run_all_experiments.sh`, so:
- Default `REPLICAS="1"` was used
- Scaling experiments only ran with replica 1
- Scaling experiments were counted as 1 experiment each instead of 4

## Fix

Added `--replicas 1,2,4,8` for minikube and GCP environments in `run_full_scale_data_collection.sh`:

```bash
# For minikube and GCP, include scaling replicas (1,2,4,8) for scaling experiments
# Native doesn't support replicas > 1, so only pass for containerized environments
if [[ "$env" == "minikube" ]] || [[ "$env" == "gcp" ]]; then
    CMD+=(--replicas "1,2,4,8")
fi
```

## Expected Experiment Counts

### Native
- **Total**: 468 experiments
- **Scaling experiments**: 9 scenarios × 1 replica = 9 (replicas > 1 are skipped)
- **Non-scaling experiments**: 459 scenarios × 1 replica = 459
- **Total**: 9 + 459 = 468 ✅

### Minikube / GCP
- **Total**: 495 experiments
- **Scaling experiments**: 9 scenarios × 4 replicas (1,2,4,8) = 36
- **Non-scaling experiments**: 459 scenarios × 1 replica = 459
- **Total**: 36 + 459 = 495 ✅

## Scaling Experiment Details

From `experiment_matrix.yaml`:
- **Algorithms**: kyber512, dilithium2, hybrid_kyber_dilithium (3)
- **Payload**: 1024 bytes (1)
- **Rate**: 500 msg/s (1)
- **Runs**: 3
- **Replicas**: 1, 2, 4, 8 (4)
- **Total scaling scenarios**: 3 × 1 × 1 × 3 = 9
- **Total scaling experiments per env**: 9 × 4 = 36

## Verification

After the fix, running:

```bash
./run_full_scale_data_collection.sh --env gcp --project <p> --bucket <b>
```

Should show:
```
[INFO] Total scenarios for gcp: 468
[INFO] Total experiments to run: 495 (accounting for replicas)
```

Instead of:
```
[INFO] Total scenarios for gcp: 468
[INFO] Total experiments to run: 468 (accounting for replicas)  # ❌ Wrong
```

## Impact

- **Before**: Scaling experiments only ran with replica 1 (missing 27 experiments per environment)
- **After**: Scaling experiments run with replicas 1,2,4,8 (complete scaling analysis)
- **Additional experiments**: 27 per environment (9 scenarios × 3 additional replicas)
- **Time impact**: ~1-2 hours additional per environment for scaling experiments

