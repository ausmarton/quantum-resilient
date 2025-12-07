# Quick Reference: Storage and Overwrite Behavior

## TL;DR

| What | Where | Overwrites? |
|------|-------|--------------|
| Individual experiment data | `results/<env>/<scenario-id>/` | ✅ Yes (same scenario ID) |
| Smoke test outputs | `final-results-smoke/` | ✅ Yes (each run) |
| Full scale outputs | `final-results/` | ✅ Yes (each run) |
| Smoke vs Full | Different directories | ❌ No (separate) |

## Key Points

### ✅ Safe (Won't Overwrite)
- **Smoke test** and **full scale** runs use **different directories**
  - Smoke: `final-results-smoke/`
  - Full: `final-results/`
- Running smoke test today and full scale tomorrow → **Both preserved**

### ⚠️ Will Overwrite
- **Same scenario ID** run twice → Individual results overwritten
- **Same final-results directory** run twice → All analysis artifacts overwritten
- **Same parameters** (matrix, envs, smoke-test flag) → Complete overwrite

### 🛡️ Protection Built-In
- Script **skips** already-completed experiments (checks for `stats/summary.json`)
- If results exist, marks as "cached" in index instead of re-running

## Common Scenarios

### Scenario 1: Run smoke test today, full scale tomorrow
```bash
# Today
./run_all_experiments.sh --smoke-test --envs native
# → Creates: final-results-smoke/

# Tomorrow  
./run_all_experiments.sh --envs native  # (no --smoke-test)
# → Creates: final-results/  (separate!)
```
✅ **Both preserved** - different directories

### Scenario 2: Run full suite twice (same day)
```bash
# Morning
./run_all_experiments.sh --envs native,minikube
# → Creates: final-results/

# Afternoon (same parameters)
./run_all_experiments.sh --envs native,minikube
# → OVERWRITES: final-results/
```
⚠️ **Second run overwrites first** - same directory

### Scenario 3: Run same scenario twice
```bash
# Run 1
./run_all_experiments.sh --smoke-test --envs native
# Creates: results/native/rsa2048-smoketest-p256-r50/

# Run 2 (same parameters)
./run_all_experiments.sh --smoke-test --envs native
# Script SKIPS (finds existing results)
# But if you delete and re-run → OVERWRITES
```
✅ **Script protects you** - skips existing results

## How to Preserve Multiple Runs

### Option 1: Timestamp directories (Recommended)
```bash
# After each run
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
mv final-results-smoke "final-results-smoke-${TIMESTAMP}"
```

### Option 2: Archive
```bash
# After each run
tar -czf "results-$(date +%Y%m%d).tar.gz" final-results-smoke/
```

### Option 3: Git commit
```bash
# After each run
git add final-results-smoke/
git commit -m "Results: $(date +%Y-%m-%d)"
```

## Directory Structure

```
results/
├── native/
│   └── <scenario-id>/          ← Individual experiment (overwrites if same ID)
│       ├── raw/
│       ├── merged/
│       ├── stats/
│       └── figures/
├── minikube/
│   └── <scenario-id>/
└── gcp/
    └── <scenario-id>/

final-results-smoke/            ← Smoke test outputs (overwrites on re-run)
├── index.json
├── aggregated_stats.json
├── figures/                     ← All figures (overwritten each run)
└── stats/

final-results/                   ← Full scale outputs (overwrites on re-run)
├── index.json
├── aggregated_stats.json
├── figures/                     ← All figures (overwritten each run)
└── stats/
```

## What Gets Overwritten When?

| Action | What Overwrites |
|--------|----------------|
| Re-run same scenario | `results/<env>/<scenario-id>/` (but script skips if exists) |
| Re-run with same parameters | `final-results/` or `final-results-smoke/` (entire directory) |
| Re-run analysis phase | All files in `figures/` and `stats/` |
| Run smoke then full | Nothing (different directories) |

## Best Practice Workflow

```bash
# 1. Run experiments
./run_all_experiments.sh --smoke-test --envs native,minikube

# 2. Immediately archive
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
mv final-results-smoke "final-results-smoke-${TIMESTAMP}"

# 3. Verify results
./scripts/verify_experiments.sh "final-results-smoke-${TIMESTAMP}"

# 4. (Optional) Commit to git
git add "final-results-smoke-${TIMESTAMP}/"
git commit -m "Smoke test results: ${TIMESTAMP}"
```

