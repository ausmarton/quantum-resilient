# Resume and Validation Guide

This guide explains how the data collection system supports resuming from failures and how to validate that all data is collected before analysis.

## Resume Capability

The data collection scripts **automatically resume** from where they left off. If a run fails partway through, you can simply re-run the same command and it will:

1. **Skip completed experiments** - Checks for `stats/summary.json` or `merged/merged.jsonl`
2. **Only run missing experiments** - Automatically detects what's missing
3. **Continue from failure point** - No need to start from the beginning

### How It Works

The script checks for completed experiments using this logic:

```bash
# Check if already completed
if [[ -f "$output_dir/stats/summary.json" ]] || [[ -f "$output_dir/merged/merged.jsonl" ]]; then
    # Verify file is not empty
    if [[ -s "$output_dir/stats/summary.json" ]] || [[ -s "$output_dir/merged/merged.jsonl" ]]; then
        log_info "  ✓ Skipping (already completed): $scenario_id"
        # Mark as cached and continue
    fi
fi
```

### Example: Resuming After Failure

```bash
# First run - fails after 2 hours due to syntax error
./run_full_scale_data_collection.sh --env native
# ... runs 100 experiments, then fails ...

# Fix the error, then re-run - automatically resumes
./run_full_scale_data_collection.sh --env native
# ... skips 100 completed experiments, continues with remaining 125 ...
```

**Time saved**: Only re-runs the missing experiments, not the entire suite!

### What Gets Checked

The resume logic checks for:
- ✅ `results/<env>/<scenario-id>/stats/summary.json` (statistical summary)
- ✅ `results/<env>/<scenario-id>/merged/merged.jsonl` (merged data)

If either exists and is non-empty, the experiment is considered complete and skipped.

### Incomplete Results

If an experiment directory exists but is incomplete (empty files or missing key files), the script will:
- Detect the incomplete state
- Remove the incomplete directory
- Re-run the experiment

## Validation Before Analysis

Before running analysis, you should validate that all required data is present.

### Quick Validation

```bash
# Basic validation (warns but doesn't fail)
./scripts/validate_data_collection.sh
```

This will:
- Check all expected experiments from the matrix
- Report what's found, missing, or incomplete
- Show completion rates per environment

### Strict Validation

```bash
# Strict mode (exits with error if data is missing)
./scripts/validate_data_collection.sh --strict
```

Use this in scripts or CI/CD to ensure data is complete before analysis.

### Validation Output

The validation script shows:
- **Per-environment status**: Expected vs found experiments
- **Completion rates**: Percentage of experiments complete
- **Missing experiments**: List of experiments that need to be run
- **Incomplete experiments**: Experiments that started but didn't finish

Example output:
```
NATIVE: Checking 225 expected experiments...
  ✓ All 225 experiments complete

MINIKUBE: Checking 225 expected experiments...
  Found: 200/225
  ⚠️  Incomplete: 5
  ✗ Missing: 20

GCP: Checking 225 expected experiments...
  Found: 180/225
  ✗ Missing: 45

======================================================================
VALIDATION SUMMARY
======================================================================
Total expected: 675
Total found: 605 (89.6%)
Incomplete: 5
Missing: 65

⚠️  Data collection incomplete:
   - 65 experiments not found
   - 5 experiments incomplete (missing merged.jsonl or stats)

   Re-run data collection to complete missing experiments.
```

### Save Validation Report

```bash
# Save detailed report to JSON file
./scripts/validate_data_collection.sh --output validation-report.json
```

The JSON report includes:
- Detailed status per environment
- List of all missing experiments
- List of all incomplete experiments
- Completion statistics

## Workflow: Data Collection with Resume

### Step 1: Start Data Collection

```bash
./run_full_scale_data_collection.sh --env native
```

### Step 2: If It Fails

The script will:
- Save all completed experiments to `results/native/`
- Log the failure point
- Exit with error code

### Step 3: Fix the Issue

Fix whatever caused the failure (syntax error, network issue, etc.)

### Step 4: Resume

Simply re-run the same command:

```bash
./run_full_scale_data_collection.sh --env native
```

The script will:
- Detect completed experiments (100 in this example)
- Skip them automatically
- Continue with remaining experiments (125)
- Complete the collection

### Step 5: Validate

After all environments are complete:

```bash
# Validate all environments
./scripts/validate_data_collection.sh \
  --envs native,minikube,gcp \
  --strict
```

If validation passes, proceed to analysis.

## Workflow: Complete Data Collection

### Recommended Approach

1. **Run each environment separately** (to avoid resource conflicts):
   ```bash
   # Day 1: Native
   ./run_full_scale_data_collection.sh --env native
   
   # Day 2: Minikube  
   ./run_full_scale_data_collection.sh --env minikube
   
   # Day 3: GCP
   ./run_full_scale_data_collection.sh --env gcp --project <project> --bucket <bucket>
   ```

2. **After each environment, validate**:
   ```bash
   ./scripts/validate_data_collection.sh --envs native
   ```

3. **Before analysis, validate all**:
   ```bash
   ./scripts/validate_data_collection.sh \
     --envs native,minikube,gcp \
     --strict \
     --output pre-analysis-validation.json
   ```

4. **If validation passes, run analysis**:
   ```bash
   ./run_all_experiments.sh \
     --skip-generation \
     --skip-native --skip-minikube --skip-gcp
   ```

## Troubleshooting Resume Issues

### Experiments Not Being Skipped

If experiments are being re-run when they should be skipped:

1. **Check if files exist**:
   ```bash
   ls -lh results/native/<scenario-id>/stats/summary.json
   ls -lh results/native/<scenario-id>/merged/merged.jsonl
   ```

2. **Check file sizes** (empty files are considered incomplete):
   ```bash
   find results/native -name "summary.json" -size 0
   find results/native -name "merged.jsonl" -size 0
   ```

3. **Remove incomplete results**:
   ```bash
   # Find and remove empty result directories
   find results/native -type d -empty
   ```

### Validation Shows Missing Experiments

If validation reports missing experiments:

1. **Check the scenario IDs**:
   ```bash
   # See what's expected
   ./scripts/validate_data_collection.sh --output report.json
   cat report.json | jq '.missing_experiments[0:5]'
   ```

2. **Check if scenarios were generated**:
   ```bash
   ls generated-scenarios/*/p*/r*/run-*/
   ```

3. **Re-run data collection for missing experiments**:
   ```bash
   # The script will automatically skip completed ones
   ./run_full_scale_data_collection.sh --env native
   ```

## Best Practices

1. **Always validate before analysis** - Don't waste time analyzing incomplete data
2. **Use strict mode in scripts** - Ensures data is complete
3. **Save validation reports** - Useful for documentation and debugging
4. **Check logs after failures** - Understand what went wrong before resuming
5. **Archive completed runs** - Before major changes, backup your results

## Summary

✅ **Resume is automatic** - Just re-run the same command  
✅ **Validation is built-in** - Script validates at the end  
✅ **Strict mode available** - Fail fast if data is incomplete  
✅ **Detailed reporting** - Know exactly what's missing  

The system is designed to be resilient to failures and make it easy to resume and validate data collection.

