# Stop and Resume Guide: Safely Interrupting Experiments

## Quick Answer

**To stop safely**: Press `Ctrl+C` once and wait for the script to finish the current experiment.

**To resume**: Simply re-run the exact same command. The script automatically skips completed experiments.

---

## Detailed Instructions

### How to Stop a Running Experiment

#### Method 1: Graceful Stop (Recommended)

1. **Press `Ctrl+C` once** in the terminal where the script is running
2. **Wait** for the current experiment to finish (the script will complete the current experiment before stopping)
3. **The script will display**:
   ```
   [WARN] Received interrupt signal. Saving progress...
   [INFO] Completed experiments are saved and will be skipped on resume.
   [INFO] To resume, simply re-run the same command:
     ./run_full_scale_data_collection.sh --env native
   ```

**What happens**:
- ✅ Current experiment completes (no data corruption)
- ✅ All completed experiments are saved
- ✅ Script exits gracefully (exit code 130)
- ✅ No data is lost

#### Method 2: Force Stop (Emergency Only)

**⚠️ Only use if graceful stop doesn't work**:

1. Press `Ctrl+C` twice quickly (or `Ctrl+Z` to suspend)
2. If the process is stuck, find and kill it:
   ```bash
   # Find the process
   ps aux | grep run_all_experiments
   ps aux | grep run_full_scale
   
   # Kill it (replace PID with actual process ID)
   kill -TERM <PID>
   ```

**What happens**:
- ⚠️ Current experiment may be incomplete (will be cleaned up on resume)
- ✅ All previously completed experiments are safe
- ✅ Script will detect incomplete results and re-run them

---

## How to Resume After Stopping

### Automatic Resume

**Simply re-run the exact same command**:

```bash
# Original command (stopped after 1 hour)
./run_full_scale_data_collection.sh --env native

# Later, resume with the same command
./run_full_scale_data_collection.sh --env native
```

**The script will**:
1. ✅ Check each experiment directory for completion markers
2. ✅ Skip experiments that have `stats/summary.json` or `merged/merged.jsonl`
3. ✅ Continue from where it left off
4. ✅ Show progress: `[cached]` for skipped experiments

### How Resume Detection Works

The script checks for completion markers in each experiment directory:

**Completion markers** (either one indicates completion):
- `results/<env>/<scenario-id>/stats/summary.json` (non-empty)
- `results/<env>/<scenario-id>/merged/merged.jsonl` (non-empty)

**If found**: Experiment is skipped (marked as `[cached]`)

**If missing or empty**: Experiment is re-run

**If incomplete** (has raw data but missing merged/stats):
- Script removes incomplete directory
- Experiment is re-run from scratch

---

## Example Workflow

### Scenario: Need to Use Laptop After 1 Hour

```bash
# 1. Start data collection (9:00 AM)
./run_full_scale_data_collection.sh --env native

# Output shows:
# [INFO] Running experiments for native...
# [ 12%] [native] rsa2048_p256_r100_run1_abc123 | Elapsed: 1h 0m | ETA: 7h 30m | 56/468
# ... running experiment 57 ...

# 2. After 1 hour, need to use laptop (10:00 AM)
# Press Ctrl+C once

# Output shows:
# [WARN] Received interrupt signal. Saving progress...
# [INFO] Completed experiments are saved and will be skipped on resume.
# [INFO] To resume, simply re-run the same command:
#   ./run_full_scale_data_collection.sh --env native

# 3. Use laptop for other work...

# 4. Later, resume (2:00 PM)
./run_full_scale_data_collection.sh --env native

# Output shows:
# [INFO] Running experiments for native...
# [ 12%] [native] rsa2048_p256_r100_run1_abc123 (cached) | Elapsed: 0m | ETA: 7h 30m | 56/468
# [ 12%] [native] rsa2048_p256_r100_run2_def456 (cached) | Elapsed: 0m | ETA: 7h 30m | 57/468
# ... (skips 56 completed experiments) ...
# [ 13%] [native] rsa2048_p256_r100_run3_ghi789 | Elapsed: 0m | ETA: 7h 15m | 57/468
# ... (continues with experiment 57) ...
```

---

## Data Integrity Guarantees

### ✅ What's Safe

1. **Completed experiments**: Fully preserved
   - `stats/summary.json` exists and is non-empty
   - `merged/merged.jsonl` exists and is non-empty
   - These are **never** re-run or overwritten

2. **In-progress experiments**: Handled safely
   - If stopped mid-experiment, the directory is cleaned up
   - Experiment is re-run from scratch on resume
   - No partial data corruption

3. **Raw data**: Preserved if complete
   - `raw/run.jsonl` is kept if experiment completed
   - Incomplete raw data is cleaned up

### ⚠️ What Happens to Incomplete Experiments

**If an experiment is stopped mid-run**:
1. Script detects incomplete results (missing `stats/summary.json` or `merged/merged.jsonl`)
2. Script removes the incomplete directory: `rm -rf "$output_dir"`
3. Experiment is re-run from scratch on resume
4. **No data corruption** - incomplete data is discarded

**If you have raw data but missing merged/stats**:
```bash
# Complete analysis without re-running experiments
./scripts/complete_incomplete_experiments.sh --env native
```

---

## Checking Progress Before Resuming

### Check What's Completed

```bash
# Check progress for all environments
./scripts/check_progress.sh

# Check progress for specific environment
./scripts/check_progress.sh --env native
```

**Output shows**:
- ✅ Completed experiments (with `stats/summary.json` or `merged/merged.jsonl`)
- ⚠️ Incomplete experiments (has raw data, missing merged/stats)
- ❌ Missing experiments (not started)

### Verify Data Integrity

```bash
# Validate all collected data
./scripts/validate_data_collection.sh \
  --matrix orchestration/experiment_matrix.yaml \
  --results-dir results \
  --envs native
```

---

## Best Practices

### 1. Use Graceful Stop

**Always use `Ctrl+C` once** and wait for the script to finish the current experiment. This ensures:
- Current experiment completes successfully
- All data is saved properly
- Clean exit state

### 2. Check Progress Before Resuming

Before resuming, check what's been completed:
```bash
./scripts/check_progress.sh --env native
```

This shows you:
- How many experiments are done
- How many remain
- Estimated time to complete

### 3. Resume with Same Command

**Use the exact same command** you used to start:
- Same environment flags
- Same matrix file
- Same options

The script automatically detects what's been completed.

### 4. Don't Manually Delete Results

**Don't manually delete** experiment directories unless you want to re-run them. The script handles cleanup of incomplete results automatically.

### 5. Complete Incomplete Experiments

If you have raw data but missing merged/stats (e.g., from `--skip-analysis`):
```bash
./scripts/complete_incomplete_experiments.sh --env native
```

This completes the analysis without re-running experiments.

---

## Troubleshooting

### Problem: Script Doesn't Stop on Ctrl+C

**Solution**: Press `Ctrl+C` twice, or use:
```bash
# Find process
ps aux | grep run_all_experiments

# Kill gracefully
kill -TERM <PID>

# If still stuck, force kill (last resort)
kill -9 <PID>
```

### Problem: Resume Shows "Incomplete Results"

**This is normal** if an experiment was stopped mid-run. The script will:
1. Remove incomplete directory
2. Re-run the experiment from scratch
3. Continue with remaining experiments

### Problem: Resume Re-runs Completed Experiments

**Check completion markers**:
```bash
# Check if stats file exists
ls -lh results/native/<scenario-id>/stats/summary.json

# Check if merged file exists
ls -lh results/native/<scenario-id>/merged/merged.jsonl
```

**If files are empty or missing**, the script will re-run. This is expected behavior.

### Problem: Want to Re-run Specific Experiments

**Option 1**: Delete specific experiment directories:
```bash
# Delete specific experiment to force re-run
rm -rf results/native/<scenario-id>
```

**Option 2**: Use cleanup script:
```bash
# Delete all results for an environment
./scripts/cleanup_results.sh --env native --archive
```

---

## For Native Environment

### Stop and Resume

```bash
# Start
./run_full_scale_data_collection.sh --env native

# Stop (Ctrl+C)
# ... use laptop ...

# Resume (same command)
./run_full_scale_data_collection.sh --env native
```

**Time**: 6.5-8 hours total (can be split across multiple sessions)

---

## For Minikube Environment

### Stop and Resume

```bash
# Start
./run_full_scale_data_collection.sh --env minikube

# Stop (Ctrl+C)
# ... use laptop ...

# Resume (same command)
./run_full_scale_data_collection.sh --env minikube
```

**Time**: 8.5-11 hours total (can be split across multiple sessions)

**Note**: Minikube containers may continue running. You can stop them:
```bash
# Stop Minikube (optional, not required)
minikube stop

# Resume later - Minikube will restart automatically
```

---

## For GCP Environment

### Stop and Resume

```bash
# Start
./run_full_scale_data_collection.sh \
  --env gcp \
  --project my-project \
  --bucket my-bucket

# Stop (Ctrl+C)
# ... use laptop ...

# Resume (same command with same parameters)
./run_full_scale_data_collection.sh \
  --env gcp \
  --project my-project \
  --bucket my-bucket
```

**Time**: 10.5-12.5 hours total (can be split across multiple sessions)

**Note**: GCP cluster is ephemeral (destroyed after experiments). Resume will create a new cluster.

---

## Summary

| Action | Command | Result |
|--------|---------|--------|
| **Stop safely** | `Ctrl+C` (once) | Current experiment completes, all data saved |
| **Resume** | Re-run same command | Automatically skips completed, continues from where stopped |
| **Check progress** | `./scripts/check_progress.sh --env native` | Shows completed, incomplete, missing |
| **Complete incomplete** | `./scripts/complete_incomplete_experiments.sh --env native` | Completes analysis for experiments with raw data |

**Key Points**:
- ✅ **Safe to stop anytime** - Completed experiments are preserved
- ✅ **Automatic resume** - Just re-run the same command
- ✅ **No data corruption** - Incomplete experiments are cleaned up and re-run
- ✅ **Progress tracking** - Check progress anytime with `check_progress.sh`

---

## Quick Reference

```bash
# Start data collection
./run_full_scale_data_collection.sh --env native

# Stop (Ctrl+C once, wait for current experiment to finish)

# Check progress
./scripts/check_progress.sh --env native

# Resume (same command)
./run_full_scale_data_collection.sh --env native

# Complete incomplete experiments (if needed)
./scripts/complete_incomplete_experiments.sh --env native
```

