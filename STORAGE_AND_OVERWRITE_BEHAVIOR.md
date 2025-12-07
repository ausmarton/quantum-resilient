# Storage and Overwrite Behavior

This document explains how results are stored, organized, and whether different runs overwrite each other.

## Directory Structure Overview

```
quantum-resilient/
├── results/                    # Individual experiment results
│   ├── native/
│   │   └── <scenario-id>/     # Per-scenario results
│   ├── minikube/
│   │   └── <scenario-id>/
│   └── gcp/
│       └── <scenario-id>/
│
├── final-results/             # Full-scale run outputs (dissertation-ready)
│   ├── index.json
│   ├── aggregated_stats.json
│   ├── figures/
│   └── ...
│
└── final-results-smoke/       # Smoke-test run outputs (separate!)
    ├── index.json
    ├── aggregated_stats.json
    ├── figures/
    └── ...
```

## Scenario ID Generation

Scenario IDs are **deterministic** - they're generated from:
- Algorithm name
- Payload size
- Message rate
- Run index (for full runs)
- Smoke-test flag

**Format:**
- **Smoke test**: `{algorithm}-smoketest-p{payload}-r{rate}`
  - Example: `rsa2048-smoketest-p256-r50`
- **Full run**: `{algorithm}_p{payload}_r{rate}_run{N}_{hash}`
  - Example: `rsa2048_p256_r50_run1_a1b2c3d4`

## Overwrite Behavior

### 1. Individual Experiment Results (`results/`)

**Location**: `results/<environment>/<scenario-id>/`

**Behavior**: ⚠️ **WILL OVERWRITE** if same scenario ID is run again

**Why**: Scenario IDs are deterministic. If you run the same scenario twice:
- Same algorithm
- Same payload size  
- Same rate
- Same run index (for full runs)
- Same smoke-test flag

→ You get the **same scenario ID** → Results **overwrite** the previous run.

**Example:**
```bash
# Run 1 (today)
./run_all_experiments.sh --smoke-test --envs native
# Creates: results/native/rsa2048-smoketest-p256-r50/

# Run 2 (tomorrow) - same parameters
./run_all_experiments.sh --smoke-test --envs native
# OVERWRITES: results/native/rsa2048-smoketest-p256-r50/
```

**Protection**: The script checks if results already exist and **skips** re-running:
```bash
# Check if already completed
if [[ -f "$output_dir/stats/summary.json" ]] || [[ -f "$output_dir/merged/merged.jsonl" ]]; then
    log_info "  Skipping (already completed): $run_scenario_id"
    # Adds to index as "cached" status
fi
```

**To preserve old results**: Rename or move the directory before re-running:
```bash
# Before re-running
mv results/native/rsa2048-smoketest-p256-r50 \
   results/native/rsa2048-smoketest-p256-r50-2025-12-06
```

### 2. Smoke Test vs Full Scale Runs

**Separate directories - NO OVERWRITE between them:**

- **Smoke test**: `final-results-smoke/`
- **Full scale**: `final-results/`

**Example:**
```bash
# Run smoke test
./run_all_experiments.sh --smoke-test --envs native
# Creates: final-results-smoke/

# Run full scale (later)
./run_all_experiments.sh --envs native  # (no --smoke-test)
# Creates: final-results/  (separate directory!)
```

✅ **They don't overwrite each other** - smoke and full runs are completely separate.

### 3. Multiple Full Runs (Same Day vs Different Days)

**Behavior**: ⚠️ **WILL OVERWRITE** if same parameters

If you run the full suite twice with the same:
- Matrix file
- Environments
- Replicas
- Smoke-test flag

→ The `final-results/` (or `final-results-smoke/`) directory will be **overwritten**.

**What gets overwritten:**
- `index.json` - Completely rewritten
- `aggregated_stats.json` - Overwritten
- `hypothesis_tests.json` - Overwritten
- `figures/*.png` - Overwritten
- `stats/*.json` - Overwritten
- `report.pdf` - Overwritten

**What's preserved:**
- Individual experiment results in `results/` (if scenario IDs match, they're skipped)

**To preserve multiple runs:**
```bash
# Run 1
./run_all_experiments.sh --envs native
mv final-results final-results-2025-12-06

# Run 2 (next day)
./run_all_experiments.sh --envs native
mv final-results final-results-2025-12-07
```

### 4. Analysis and Visualization Artifacts

**Location**: `final-results/figures/` and `final-results/stats/`

**Behavior**: ⚠️ **ALWAYS OVERWRITTEN** on each run

Every time you run `run_all_experiments.sh`, the analysis phase:
1. **Overwrites** all figures in `figures/`
2. **Overwrites** all statistics in `stats/`
3. **Overwrites** `aggregated_stats.json`
4. **Overwrites** `hypothesis_tests.json`
5. **Overwrites** `index.json`

**Why**: The analysis scripts use `open(..., 'w')` (write mode), not append mode.

**Example:**
```bash
# Day 1: Generate figures
./run_all_experiments.sh --smoke-test
# Creates: final-results-smoke/figures/combined_ecdf.png

# Day 2: Re-run (even with different experiments)
./run_all_experiments.sh --smoke-test
# OVERWRITES: final-results-smoke/figures/combined_ecdf.png
```

## Best Practices

### 1. Preserve Multiple Runs

**Option A: Timestamp directories**
```bash
# After each run
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
mv final-results-smoke "final-results-smoke-${TIMESTAMP}"
```

**Option B: Use version control**
```bash
# Commit results after each run
git add final-results-smoke/
git commit -m "Results from smoke test run $(date)"
```

**Option C: Archive script**
```bash
# Create archive after run
tar -czf "results-$(date +%Y%m%d).tar.gz" final-results-smoke/
```

### 2. Compare Smoke Test vs Full Scale

Since they use different directories, you can compare them:
```bash
# Compare aggregated stats
diff final-results-smoke/aggregated_stats.json \
     final-results/aggregated_stats.json

# Compare figures
diff final-results-smoke/figures/ \
     final-results/figures/
```

### 3. Re-run Individual Experiments

If you need to re-run a specific experiment:
```bash
# Remove just that experiment's results
rm -rf results/native/rsa2048-smoketest-p256-r50/

# Re-run (will only re-run that one)
./run_all_experiments.sh --smoke-test --envs native
```

### 4. Preserve Individual Experiment Results

Before re-running, archive old results:
```bash
# Archive all native results
tar -czf "results-native-$(date +%Y%m%d).tar.gz" results/native/

# Or move to archive directory
mkdir -p archive/2025-12-06/
mv results/native/* archive/2025-12-06/
```

## Summary Table

| Location | Overwrites? | When | How to Preserve |
|----------|-------------|------|-----------------|
| `results/<env>/<scenario-id>/` | ✅ Yes (same scenario ID) | Same parameters | Rename directory before re-run |
| `final-results-smoke/` vs `final-results/` | ❌ No | Different smoke-test flag | Already separate |
| `final-results/` (multiple runs) | ✅ Yes | Same parameters | Rename/timestamp directory |
| `final-results/figures/` | ✅ Yes | Every analysis run | Copy/archive before re-run |
| `final-results/index.json` | ✅ Yes | Every run | Archive before re-run |

## Recommendations

1. **For production runs**: Always archive `final-results/` after each complete run
2. **For development**: Use `--smoke-test` flag to keep smoke and full runs separate
3. **For reproducibility**: Commit results to git or use timestamped directories
4. **For comparison**: Keep multiple runs in separate timestamped directories

## Example Workflow

```bash
# Day 1: Smoke test
./run_all_experiments.sh --smoke-test --envs native,minikube
mv final-results-smoke final-results-smoke-day1

# Day 2: Full scale run
./run_all_experiments.sh --envs native,minikube
mv final-results final-results-day2

# Day 3: Re-run smoke test (different parameters)
./run_all_experiments.sh --smoke-test --envs native
mv final-results-smoke final-results-smoke-day3

# All three runs are preserved!
```

