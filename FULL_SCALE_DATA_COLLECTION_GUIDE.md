# Full-Scale Data Collection Guide

This guide explains how to run full-scale benchmarks separately for each environment to collect all raw data needed for dissertation analysis.

> **Related**: [STORAGE_AND_OUTPUT_GUIDE.md](STORAGE_AND_OUTPUT_GUIDE.md) - Where results are stored and overwrite behavior

## Quick Start

**Goal**: Collect raw data from full-scale runs separately to avoid resource throttling.

### Step-by-Step

1. **Run Native** (Local Machine)
   ```bash
   ./run_full_scale_data_collection.sh --env native
   ```
   ⏱️ **Time**: ~3-4 hours | 📊 **Scenarios**: 225

2. **Run Minikube** (Local Machine)
   ```bash
   ./run_full_scale_data_collection.sh --env minikube
   ```
   ⏱️ **Time**: ~4-5 hours | 📊 **Scenarios**: 225

3. **Run GCP** (Cloud)
   ```bash
   ./run_full_scale_data_collection.sh \
     --env gcp \
     --project <your-gcp-project> \
     --bucket <your-gcs-bucket>
   ```
   ⏱️ **Time**: ~5-6 hours | 📊 **Scenarios**: 225

4. **Verify Data Collection**
   ```bash
   ./scripts/validate_data_collection.sh --envs native,minikube,gcp
   ```

5. **Regenerate Combined Index** (If Needed)
   ```bash
   ./scripts/regenerate_index_from_results.sh \
     --matrix orchestration/experiment_matrix.yaml \
     --output final-results/
   ```

6. **Run Analysis** (Later, When Ready)
   ```bash
   ./run_all_experiments.sh \
     --skip-generation \
     --skip-native --skip-minikube --skip-gcp \
     --matrix orchestration/experiment_matrix.yaml
   ```

**Key Points:**
- ✅ No analysis during collection - Saves time, focus on data
- ✅ Run environments separately - Avoids resource throttling
- ✅ All raw data preserved - Can re-analyze anytime
- ✅ Academic rigor - 5 runs per configuration
- ✅ Reproducible - Deterministic seeds, full metadata

## Overview

**Goal**: Collect all raw benchmark data from full-scale runs without running analysis, allowing you to:
- Run each environment separately (avoid resource throttling)
- Analyze data later offline
- Ensure academic rigor (5 runs per configuration)
- Preserve all raw data for reproducibility

## What Gets Collected

For each experiment scenario, the following data is captured:

```
results/<env>/<scenario-id>/
├── raw/
│   └── run.jsonl              # Raw telemetry events (one per line)
├── merged/
│   ├── merged.jsonl           # Sorted merged events
│   └── merged.parquet         # Parquet format (if generated)
├── stats/
│   └── summary.json           # Statistical summary (p50, p95, p99, throughput)
└── manifest.json              # Run metadata (git commit, timestamps, hardware)
```

**Each event in JSONL contains:**
- `latency_us`: Operation latency in microseconds
- `operation`: Operation type (sign, encrypt, etc.)
- `algorithm`: Algorithm name
- `timestamp_utc_iso`: Event timestamp
- `cpu_user_time_us`: CPU usage
- `memory_rss_bytes`: Memory usage
- `queue_delay_us`: Queue delay (if applicable)
- `ciphertext_size_bytes`: Ciphertext size
- System metadata (CPU model, kernel version, etc.)

## Full-Scale Run Parameters

Based on `orchestration/experiment_matrix.yaml`:

- **Algorithms**: 5 (RSA-2048, ECDSA P-256, Kyber-512, Dilithium-2, Hybrid)
- **Payload sizes**: 3 (256B, 1KB, 4KB)
- **Rates**: 3 (100, 500, 2000 msg/s)
- **Runs per configuration**: 5
- **Duration per run**: 30 seconds
- **Total scenarios per environment**: 5 × 3 × 3 × 5 = **225 scenarios**

**Estimated time per environment:**
- Native: ~3-4 hours (depending on hardware)
- Minikube: ~4-5 hours (container overhead)
- GCP: ~5-6 hours (includes cluster setup/teardown)

## Running Data Collection

### Option 1: Run All Environments Sequentially

```bash
./run_full_scale_data_collection.sh \
  --all \
  --project <your-gcp-project> \
  --bucket <your-gcs-bucket> \
  --matrix orchestration/experiment_matrix.yaml
```

This will:
1. Run native experiments
2. Run minikube experiments
3. Run GCP experiments
4. Generate data collection manifest

### Option 2: Run Environments Separately (Recommended)

**Step 1: Run Native (Local Machine)**
```bash
./run_full_scale_data_collection.sh \
  --env native \
  --matrix orchestration/experiment_matrix.yaml
```

**Step 2: Run Minikube (Local Machine)**
```bash
./run_full_scale_data_collection.sh \
  --env minikube \
  --matrix orchestration/experiment_matrix.yaml
```

**Step 3: Run GCP (Cloud)**
```bash
./run_full_scale_data_collection.sh \
  --env gcp \
  --project <your-gcp-project> \
  --bucket <your-gcs-bucket> \
  --region us-central1 \
  --matrix orchestration/experiment_matrix.yaml
```

### Option 3: Run Specific Environments Only

```bash
# Only native
./run_full_scale_data_collection.sh --env native

# Only minikube
./run_full_scale_data_collection.sh --env minikube

# Only GCP
./run_full_scale_data_collection.sh --env gcp --project <project> --bucket <bucket>
```

## What Happens During Collection

1. **Scenario Generation**: Scenarios are generated from the matrix (if not already done)
2. **Experiment Execution**: Each scenario runs 5 times (for statistical rigor)
3. **Data Collection**: Raw JSONL, merged data, and statistics are saved
4. **Manifest Creation**: A manifest is created listing all collected experiments
5. **Analysis Skipped**: No figures or hypothesis tests are generated (saves time)

## Output Structure

After running, you'll have:

```
quantum-resilient/
├── results/
│   ├── native/
│   │   ├── rsa2048_p256_r100_run1_<hash>/
│   │   ├── rsa2048_p256_r100_run2_<hash>/
│   │   └── ... (225 scenarios)
│   ├── minikube/
│   │   └── ... (225 scenarios)
│   └── gcp/
│       └── ... (225 scenarios)
│
└── data-collection-<timestamp>/
    ├── manifest.json           # Complete list of collected experiments
    ├── summary.txt             # Human-readable summary
    ├── native_run.log          # Native environment run log
    ├── minikube_run.log        # Minikube environment run log
    └── gcp_run.log             # GCP environment run log
```

## Resume Capability

The data collection scripts **automatically resume** from where they left off. If a run fails partway through, simply re-run the same command:

```bash
# First run - fails after 2 hours
./run_full_scale_data_collection.sh --env native
# ... runs 100 experiments, then fails ...

# Fix the error, then re-run - automatically resumes
./run_full_scale_data_collection.sh --env native
# ... skips 100 completed experiments, continues with remaining 125 ...
```

**How it works**: The script checks for `stats/summary.json` or `merged/merged.jsonl`. If either exists and is non-empty, the experiment is skipped.

**Incomplete Results**: If experiments have raw data (`raw/run.jsonl`) but are missing merged/stats files, complete them without re-running:
```bash
./scripts/complete_incomplete_experiments.sh --env native
```

## Validation

Before running analysis, validate that all required data is present:

### Quick Validation
```bash
./scripts/validate_data_collection.sh --envs native,minikube,gcp
```

This shows:
- Per-environment status (expected vs found)
- Completion rates
- Missing experiments
- Incomplete experiments

### Strict Validation
```bash
./scripts/validate_data_collection.sh --envs native,minikube,gcp --strict
```

Exits with error if data is incomplete (useful for scripts/CI).

### Save Validation Report
```bash
./scripts/validate_data_collection.sh --output validation-report.json
```

## Verifying Data Collection

After each environment completes, verify the data:

```bash
# Check what was collected
./scripts/verify_experiments.sh results/

# Or check specific environment
ls -lh results/native/ | wc -l  # Should show ~225 directories
ls -lh results/minikube/ | wc -l
ls -lh results/gcp/ | wc -l
```

## Running Analysis Later

Once all environments have collected data, you can run analysis:

### Step 1: Regenerate Combined Index (If Needed)

If you ran environments separately, regenerate the combined index:

```bash
./scripts/regenerate_index_from_results.sh \
  --matrix orchestration/experiment_matrix.yaml \
  --output final-results/
```

This creates a combined `index.json` from all existing results directories.

### Step 2: Analyze All Environments Together

```bash
./run_all_experiments.sh \
  --skip-generation \
  --skip-native --skip-minikube --skip-gcp \
  --matrix orchestration/experiment_matrix.yaml
```

This will:
- Skip scenario generation (already done)
- Skip experiment execution (data already collected)
- Use existing index.json (or regenerated one)
- Run analysis on all collected data
- Generate figures, hypothesis tests, and reports

### Option 2: Analyze Individual Environment

```bash
# Analyze only native
./run_all_experiments.sh \
  --skip-generation \
  --envs native \
  --skip-native false \
  --skip-minikube true \
  --skip-gcp true \
  --matrix orchestration/experiment_matrix.yaml
```

### Option 3: Custom Analysis

You can also run analysis scripts directly:

```bash
# Aggregate results
python3 analysis/aggregate_results.py \
  --index results/index.json \
  --output final-results/

# Generate figures
python3 analysis/plot_combined_cdfs.py \
  --index results/index.json \
  --output final-results/figures/

# Hypothesis tests
python3 analysis/hypothesis_tests.py \
  --index results/index.json \
  --matrix orchestration/experiment_matrix.yaml \
  --output final-results/
```

## Academic Rigor Checklist

✅ **Multiple runs**: 5 runs per configuration (as per matrix)
✅ **Statistical analysis**: p50, p95, p99 percentiles, confidence intervals
✅ **Hypothesis testing**: Kolmogorov-Smirnov, Mann-Whitney U, Welch's t-test
✅ **Effect sizes**: Cohen's d with 95% confidence intervals
✅ **Multiple comparisons correction**: Holm-Bonferroni correction
✅ **Reproducibility**: Deterministic RNG seeds, full metadata capture
✅ **Raw data preservation**: All JSONL files preserved for re-analysis

## Data Requirements for Dissertation

The collected data supports answering:

1. **Performance comparison**: PQC vs classical (latency, throughput)
2. **Environment comparison**: Native vs Minikube vs GCP
3. **Scaling behavior**: Performance at different rates and payload sizes
4. **Statistical significance**: Which differences are significant?
5. **Effect sizes**: How large are the practical differences?
6. **Distribution analysis**: CDFs, ECDFs, distribution shapes

All of this is captured in the raw JSONL files and can be re-analyzed as needed.

## Troubleshooting

### Experiments are being skipped

The script checks if results already exist and skips them. To force re-run:
```bash
# Remove specific experiment
rm -rf results/native/<scenario-id>/

# Or remove all and start fresh (be careful!)
rm -rf results/native/* results/minikube/* results/gcp/*
```

### Validation shows incomplete experiments

If validation reports experiments as "incomplete" (have raw data but missing merged/stats files), you can complete them without re-running:

```bash
# Complete analysis for all incomplete experiments (fast)
./scripts/complete_incomplete_experiments.sh --env native

# Check what would be processed (dry run)
./scripts/complete_incomplete_experiments.sh --env native --dry-run
```

This is much faster than re-running the experiments since it only processes existing raw data.

### Experiments not being skipped

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

### Out of disk space

Each experiment generates ~1-5 MB of data. For 225 scenarios × 3 environments = ~675 experiments = ~3-4 GB total.

Check disk usage:
```bash
du -sh results/
```

### GCP costs

Always use `--ephemeral` flag (automatically used by the script) to ensure clusters are destroyed after runs.

### Analysis fails later

If analysis fails, you can re-run just the analysis:
```bash
./run_all_experiments.sh \
  --skip-generation \
  --skip-native --skip-minikube --skip-gcp \
  --matrix orchestration/experiment_matrix.yaml
```

## Best Practices

1. **Run environments separately**: Avoids resource throttling
2. **Verify after each environment**: Check that data was collected
3. **Archive after collection**: Backup `results/` directory
4. **Document collection timestamp**: Note when each environment was run
5. **Keep collection manifest**: The manifest.json lists all collected experiments

## Example Workflow

```bash
# Day 1: Collect native data
./run_full_scale_data_collection.sh --env native
# Verify
./scripts/verify_experiments.sh results/native/

# Day 2: Collect minikube data
./run_full_scale_data_collection.sh --env minikube
# Verify
./scripts/verify_experiments.sh results/minikube/

# Day 3: Collect GCP data
./run_full_scale_data_collection.sh --env gcp --project <project> --bucket <bucket>
# Verify
./scripts/verify_experiments.sh results/gcp/

# Day 4: Run analysis on all collected data
./run_all_experiments.sh \
  --skip-generation \
  --skip-native --skip-minikube --skip-gcp \
  --matrix orchestration/experiment_matrix.yaml

# Archive results
tar -czf "full-scale-results-$(date +%Y%m%d).tar.gz" results/ final-results/
```

## Complete Workflow: From Data Collection to Dissertation

This section provides a step-by-step workflow for collecting all data and generating dissertation-ready analysis.

### Phase 1: Data Collection (You Are Here)

**Status**: ✅ Native complete (225 experiments)

**Next Steps**:

1. **Run Minikube Data Collection**
   ```bash
   ./run_full_scale_data_collection.sh --env minikube
   ```
   ⏱️ **Time**: ~4-5 hours | 📊 **Scenarios**: 225
   
   After completion, verify:
   ```bash
   ./scripts/validate_data_collection.sh --envs minikube
   ```

2. **Run GCP Data Collection**
   ```bash
   ./run_full_scale_data_collection.sh \
     --env gcp \
     --project <your-gcp-project> \
     --bucket <your-gcs-bucket> \
     --region us-central1
   ```
   ⏱️ **Time**: ~5-6 hours | 📊 **Scenarios**: 225
   
   After completion, verify:
   ```bash
   ./scripts/validate_data_collection.sh --envs gcp
   ```

3. **Final Validation (All Environments)**
   ```bash
   ./scripts/validate_data_collection.sh --envs native,minikube,gcp --strict
   ```
   
   This should show:
   ```
   NATIVE: Checking 225 expected experiments...
     ✓ All 225 experiments complete
   
   MINIKUBE: Checking 225 expected experiments...
     ✓ All 225 experiments complete
   
   GCP: Checking 225 expected experiments...
     ✓ All 225 experiments complete
   
   ✓ All required data is present - ready for analysis!
   ```

### Phase 2: Prepare for Analysis

4. **Regenerate Combined Index**
   
   Since you ran environments separately, regenerate the combined index:
   ```bash
   ./scripts/regenerate_index_from_results.sh \
     --matrix orchestration/experiment_matrix.yaml \
     --output final-results/
   ```
   
   This creates `final-results/index.json` with all 675 experiments (225 × 3 environments).

### Phase 3: Run Complete Analysis

5. **Generate All Analysis Artifacts**
   
   This single command generates everything needed for your dissertation:
   ```bash
   ./run_all_experiments.sh \
     --skip-generation \
     --skip-native --skip-minikube --skip-gcp \
     --matrix orchestration/experiment_matrix.yaml
   ```
   
   **What this generates:**
   - `final-results/index.json` - Master experiment index
   - `final-results/aggregated_stats.json` - Aggregated statistics across all experiments
   - `final-results/aggregated_stats.csv` - CSV version for spreadsheets
   - `final-results/hypothesis_tests.json` - Statistical test results (KS, Mann-Whitney, Welch's t-test)
   - `final-results/hypothesis_table.csv` - Hypothesis tests in tabular format
   - `final-results/hypothesis_interpretation.txt` - Human-readable interpretation
   - `final-results/figures/` - All visualizations:
     - `combined_ecdf.png` - ECDF across all algorithms/environments
     - `classical_vs_pqc.png` - PQC vs classical comparison
     - `scaling_curves.png` - Throughput/latency scaling
     - `ecdf_by_payload.png` - ECDF by payload size
     - `ecdf_*.png` - Per-algorithm ECDFs
     - `scaling/` - Replica scaling plots (if applicable)
   - `final-results/stats/` - Additional statistics:
     - `effect_sizes.json` - Cohen's d effect sizes
     - `environment_deltas.json` - Environment comparisons
   - `final-results/report.pdf` - Complete dissertation-ready PDF report

### Phase 4: Use Results for Dissertation

6. **Access Your Results**
   
   All dissertation-ready outputs are in `final-results/`:
   ```bash
   cd final-results/
   ls -lh figures/    # All charts and graphs
   ls -lh stats/      # Statistical summaries
   cat hypothesis_interpretation.txt  # Statistical test interpretations
   ```

7. **Key Files for Dissertation**:
   - **Figures**: `final-results/figures/*.png` - Use these in your dissertation
   - **Tables**: `final-results/aggregated_stats.csv` - Import into your document
   - **Statistical Tests**: `final-results/hypothesis_table.csv` - Statistical significance results
   - **Interpretations**: `final-results/hypothesis_interpretation.txt` - Pre-written interpretations
   - **Complete Report**: `final-results/report.pdf` - Full analysis report

### Summary Checklist

- [x] ✅ Native data collection complete (225 experiments)
- [ ] ⏳ Minikube data collection (~4-5 hours)
- [ ] ⏳ GCP data collection (~5-6 hours)
- [ ] ⏳ Validate all environments
- [ ] ⏳ Regenerate combined index
- [ ] ⏳ Run complete analysis
- [ ] ⏳ Review generated figures and statistics
- [ ] ⏳ Use results in dissertation

## Next Steps (After All Data Collected)

After completing all three environments:
1. Verify all data was collected (use validation script)
2. Archive the `results/` directory (backup before analysis)
3. Run analysis (generates all figures and statistics)
4. Review generated visualizations and interpretations
5. Use figures and tables in dissertation
6. Write up results chapter using the statistical interpretations

