# Re-Running All Experiments: Complete Guide

## Quick Answer

**Yes, you can delete all collected data and re-run everything.** The framework supports this, and we've created a cleanup script to make it safe and easy.

---

## Option 1: Using the Cleanup Script (Recommended)

### Delete All Environments (with Archive)

```bash
# Archive and delete all results (safe - preserves data in archive/)
./scripts/cleanup_results.sh --all --archive
```

This will:
- ✅ Archive all data to `archive/` directory with timestamps
- ✅ Delete all experiment results (`results/native/`, `results/minikube/`, `results/gcp/`)
- ✅ Delete analysis outputs (`final-results/`, `final-results-smoke/`)

### Delete Specific Environment

```bash
# Delete only native results (with archive)
./scripts/cleanup_results.sh --env native --archive

# Delete only minikube results
./scripts/cleanup_results.sh --env minikube --archive

# Delete only GCP results
./scripts/cleanup_results.sh --env gcp --archive
```

### Delete Only Analysis Outputs (Keep Raw Data)

If you want to keep the raw experiment data but re-run analysis:

```bash
./scripts/cleanup_results.sh --analysis-only --archive
```

This deletes:
- `final-results/` (analysis outputs)
- `final-results-smoke/` (smoke test analysis)

But keeps:
- `results/native/`, `results/minikube/`, `results/gcp/` (raw data)

### Dry Run (See What Would Be Deleted)

Before actually deleting, see what would be affected:

```bash
./scripts/cleanup_results.sh --all --dry-run
```

### Delete Without Archiving (Dangerous!)

**⚠️ Warning**: This permanently deletes data without backup!

```bash
./scripts/cleanup_results.sh --all --no-archive
```

---

## Option 2: Manual Deletion

If you prefer manual control:

### Step 1: Archive (Recommended)

```bash
# Create archive directory
mkdir -p archive

# Archive with timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
cp -r results archive/results-$TIMESTAMP
cp -r final-results archive/final-results-$TIMESTAMP 2>/dev/null || true
cp -r final-results-smoke archive/final-results-smoke-$TIMESTAMP 2>/dev/null || true
```

### Step 2: Delete

```bash
# Delete all experiment results
rm -rf results/native/* results/minikube/* results/gcp/*

# Delete analysis outputs
rm -rf final-results/* final-results-smoke/*

# Recreate empty directories (optional, but clean)
mkdir -p results/native results/minikube results/gcp
mkdir -p final-results final-results-smoke
```

---

## After Cleanup: Re-Running Experiments

Once data is deleted, re-run experiments:

### 1. Re-run Native

```bash
./run_full_scale_data_collection.sh --env native
```

⏱️ **Time**: ~6.5-8 hours | 📊 **Scenarios**: 468

### 2. Re-run Minikube

```bash
./run_full_scale_data_collection.sh --env minikube
```

⏱️ **Time**: ~7.5-9 hours | 📊 **Scenarios**: 468 (baseline)

Then run horizontal scaling:
```bash
./run_all_experiments.sh \
  --envs minikube \
  --replicas 1,2,4,8 \
  --skip-generation \
  --matrix orchestration/experiment_matrix.yaml
```

⏱️ **Additional time**: ~1-2 hours | 📊 **Additional scenarios**: 27 (replicas 2,4,8)

### 3. Re-run GCP

```bash
./run_full_scale_data_collection.sh \
  --env gcp \
  --project <your-gcp-project> \
  --bucket <your-gcs-bucket>
```

⏱️ **Time**: ~9-10.5 hours | 📊 **Scenarios**: 468 (baseline)

Then run horizontal scaling:
```bash
./run_all_experiments.sh \
  --envs gcp \
  --replicas 1,2,4,8 \
  --project <your-gcp-project> \
  --bucket <your-gcs-bucket> \
  --skip-generation \
  --matrix orchestration/experiment_matrix.yaml
```

⏱️ **Additional time**: ~1.5-2 hours | 📊 **Additional scenarios**: 27 (replicas 2,4,8)

### 4. Verify Data Collection

```bash
./scripts/validate_data_collection.sh --envs native,minikube,gcp
```

### 5. Run Analysis

```bash
./run_all_experiments.sh \
  --skip-generation \
  --skip-native --skip-minikube --skip-gcp \
  --matrix orchestration/experiment_matrix.yaml
```

---

## What Gets Deleted

### Experiment Results (`results/`)
- `results/native/*` - All native experiment directories
- `results/minikube/*` - All Minikube experiment directories
- `results/gcp/*` - All GCP experiment directories

Each directory contains:
- `raw/run.jsonl` - Raw telemetry data
- `merged/merged.jsonl` - Merged and sorted data
- `stats/summary.json` - Statistical summary
- `figures/` - Individual experiment plots

### Analysis Outputs (`final-results/`)
- `final-results/index.json` - Master experiment index
- `final-results/aggregated_stats.json` - Aggregated statistics
- `final-results/hypothesis_tests.json` - Statistical test results
- `final-results/figures/` - Combined figures
- `final-results/stats/` - Additional statistics
- `final-results/report.pdf` - PDF report

### Smoke Test Outputs (`final-results-smoke/`)
- Same structure as `final-results/` but for smoke tests

---

## What Gets Archived (if using `--archive`)

The cleanup script creates timestamped archives in `archive/`:

```
archive/
├── results-native-20251207-224954/
├── results-minikube-20251207-224954/
├── results-gcp-20251207-224954/
├── final-results-20251207-224954/
└── final-results-smoke-20251207-224954/
```

**Archive size**: Approximately same as original data (4.3GB for native in your case)

---

## Complete Re-Run Workflow

### Step-by-Step

1. **Check current data**:
   ```bash
   ./scripts/check_progress.sh
   ```

2. **Archive and delete** (if you want to preserve):
   ```bash
   ./scripts/cleanup_results.sh --all --archive
   ```

   Or delete without archive:
   ```bash
   ./scripts/cleanup_results.sh --all --no-archive
   ```

3. **Re-run native**:
   ```bash
   ./run_full_scale_data_collection.sh --env native
   ```

4. **Re-run minikube**:
   ```bash
   ./run_full_scale_data_collection.sh --env minikube
   ```

5. **Re-run minikube scaling**:
   ```bash
   ./run_all_experiments.sh \
     --envs minikube \
     --replicas 1,2,4,8 \
     --skip-generation \
     --matrix orchestration/experiment_matrix.yaml
   ```

6. **Re-run GCP**:
   ```bash
   ./run_full_scale_data_collection.sh \
     --env gcp \
     --project <project> \
     --bucket <bucket>
   ```

7. **Re-run GCP scaling**:
   ```bash
   ./run_all_experiments.sh \
     --envs gcp \
     --replicas 1,2,4,8 \
     --project <project> \
     --bucket <bucket> \
     --skip-generation \
     --matrix orchestration/experiment_matrix.yaml
   ```

8. **Verify all data collected**:
   ```bash
   ./scripts/validate_data_collection.sh --envs native,minikube,gcp
   ```

9. **Run analysis**:
   ```bash
   ./run_all_experiments.sh \
     --skip-generation \
     --skip-native --skip-minikube --skip-gcp \
     --matrix orchestration/experiment_matrix.yaml
   ```

---

## Time Estimates for Complete Re-Run

| Environment | Baseline | Scaling | Total |
|-------------|----------|---------|-------|
| **Native** | 6.5-8 hours | N/A | 6.5-8 hours |
| **Minikube** | 7.5-9 hours | +1-2 hours | 8.5-11 hours |
| **GCP** | 9-10.5 hours | +1.5-2 hours | 10.5-12.5 hours |
| **Total** | | | **25.5-31.5 hours** |

**Note**: These can be run in parallel (native + minikube on local machine, GCP in cloud), reducing total wall-clock time.

---

## Disk Space Considerations

### Current Data Size
- Native: ~4.3 GB (230 experiments)
- Minikube: ~20 KB (5 experiments)
- GCP: 0 (not run yet)

### Expected Full Data Size
- Native: ~6-8 GB (468 experiments)
- Minikube: ~6-8 GB (495 experiments including scaling)
- GCP: ~6-8 GB (495 experiments including scaling)
- **Total**: ~18-24 GB

### Archive Size
If you archive before deleting, you'll need:
- **Current**: ~4.3 GB for archive
- **After full run**: ~18-24 GB for archive

**Recommendation**: Archive to external storage or delete old archives after verifying new data.

---

## Safety Features

### Built-in Protection

1. **Archive by default**: The cleanup script archives data by default (unless `--no-archive` is used)
2. **Dry run**: Always test with `--dry-run` first
3. **Resume capability**: If re-run is interrupted, you can resume (skips completed experiments)
4. **Validation**: Use `validate_data_collection.sh` to verify all data is collected

### Best Practices

1. **Always archive first** (unless you're absolutely sure you don't need the data)
2. **Use dry-run** to see what will be deleted
3. **Verify archive** before deleting (check `archive/` directory)
4. **Keep archives** until new data collection is complete and verified
5. **Delete archives** only after confirming new data is correct

---

## Troubleshooting

### "Directory not found" errors

This is normal if an environment hasn't been run yet. The script handles this gracefully.

### Archive takes too long

For large datasets (4+ GB), archiving can take several minutes. This is normal.

### Not enough disk space

If you don't have space for both archive and new data:
1. Archive to external storage
2. Or delete without archive (if you're certain you don't need old data)
3. Or delete archives after verifying new data

### Re-run is slow

This is expected - full data collection takes 25-31 hours total. Use the progress indicators:
- `./scripts/check_progress.sh` - Check overall progress
- Progress bars in `run_all_experiments.sh` - Real-time progress

---

## Summary

✅ **Yes, you can delete and re-run everything**

**Recommended approach**:
1. Use `./scripts/cleanup_results.sh --all --archive` (safest)
2. Re-run experiments using `run_full_scale_data_collection.sh`
3. Verify with `validate_data_collection.sh`
4. Run analysis with `run_all_experiments.sh --skip-generation --skip-native --skip-minikube --skip-gcp`

**Total time**: ~25-31 hours for complete re-run (can be parallelized)

**Data safety**: Archive preserves old data, resume capability prevents re-running completed experiments

