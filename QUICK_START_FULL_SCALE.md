# Quick Start: Full-Scale Data Collection

## TL;DR

**Goal**: Collect raw data from full-scale runs separately to avoid resource throttling.

## Step-by-Step

### 1. Run Native (Local Machine)
```bash
./run_full_scale_data_collection.sh --env native
```
⏱️ **Time**: ~3-4 hours  
📊 **Scenarios**: 225 (5 algorithms × 3 payloads × 3 rates × 5 runs)

### 2. Run Minikube (Local Machine)
```bash
./run_full_scale_data_collection.sh --env minikube
```
⏱️ **Time**: ~4-5 hours  
📊 **Scenarios**: 225

### 3. Run GCP (Cloud)
```bash
./run_full_scale_data_collection.sh \
  --env gcp \
  --project <your-gcp-project> \
  --bucket <your-gcs-bucket>
```
⏱️ **Time**: ~5-6 hours  
📊 **Scenarios**: 225

### 4. Verify Data Collection
```bash
# Check all environments
./scripts/verify_experiments.sh

# Or check specific
ls -lh results/native/ | wc -l    # Should be ~225
ls -lh results/minikube/ | wc -l   # Should be ~225
ls -lh results/gcp/ | wc -l        # Should be ~225
```

### 5. Regenerate Combined Index (If Needed)
```bash
# If you ran environments separately, regenerate the combined index
./scripts/regenerate_index_from_results.sh \
  --matrix orchestration/experiment_matrix.yaml \
  --output final-results/
```

### 6. Run Analysis (Later, When Ready)
```bash
./run_all_experiments.sh \
  --skip-generation \
  --skip-native --skip-minikube --skip-gcp \
  --matrix orchestration/experiment_matrix.yaml
```

This generates:
- `final-results/index.json` - Master index
- `final-results/aggregated_stats.json` - Statistics
- `final-results/hypothesis_tests.json` - Statistical tests
- `final-results/figures/` - All figures
- `final-results/report.pdf` - Dissertation report

## What Gets Collected

For each of 225 scenarios per environment:
- ✅ Raw JSONL telemetry (`raw/run.jsonl`)
- ✅ Merged/sorted data (`merged/merged.jsonl`)
- ✅ Statistical summary (`stats/summary.json`)
- ✅ Run metadata (`manifest.json`)

**Total data**: ~3-4 GB per environment

## Key Points

✅ **No analysis during collection** - Saves time, focus on data  
✅ **Run environments separately** - Avoids resource throttling  
✅ **All raw data preserved** - Can re-analyze anytime  
✅ **Academic rigor** - 5 runs per configuration  
✅ **Reproducible** - Deterministic seeds, full metadata

## Troubleshooting

**Experiments skipped?**  
→ Script checks for existing results and skips them. To force re-run, delete the scenario directory.

**Out of disk space?**  
→ Each environment needs ~3-4 GB. Check with `du -sh results/`

**Analysis fails?**  
→ Re-run analysis only: `./run_all_experiments.sh --skip-generation --skip-native --skip-minikube --skip-gcp`

## Full Documentation

See `FULL_SCALE_DATA_COLLECTION_GUIDE.md` for detailed information.

