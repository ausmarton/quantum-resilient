# Script Architecture and Purpose

This document explains the purpose of each experiment execution script and when to use them.

## Primary Scripts (Use These)

### `run_all_experiments.sh` - **PRIMARY ENTRY POINT** ⭐

**Purpose**: Complete end-to-end experiment orchestration

**What it does**:
1. Generates scenarios from experiment matrix
2. Runs experiments across all specified environments (native, minikube, gcp)
3. Collects and analyzes data
4. Generates dissertation-ready outputs (figures, stats, hypothesis tests)
5. Creates run tracking directory (`run-YYYYMMDD-HHMMSS/`)

**When to use**:
- ✅ **Always use this for complete experiments** (smoke test or full scale)
- ✅ For end-to-end runs with analysis
- ✅ For smoke tests
- ✅ For full-scale data collection (with `--skip-analysis`)

**Usage**:
```bash
# Smoke test (all environments)
./run_all_experiments.sh \
  --smoke-test \
  --envs native,minikube,gcp \
  --project <project> \
  --bucket <bucket>

# Full scale (data collection only)
./run_all_experiments.sh \
  --envs native,minikube,gcp \
  --project <project> \
  --bucket <bucket> \
  --skip-analysis
```

**Status**: ✅ **ACTIVE - PRIMARY SCRIPT**

---

## Convenience Wrappers

### `run_full_scale_data_collection.sh` - Data Collection Wrapper

**Purpose**: Convenience wrapper for data collection workflows

**What it does**:
- Calls `run_all_experiments.sh` with `--skip-analysis`
- Creates `data-collection-YYYYMMDD-HHMMSS/` directories
- Runs environments separately (one at a time)
- Generates data collection manifest and summary

**When to use**:
- ✅ When you want per-environment data collection tracking
- ✅ When running environments separately (to avoid resource conflicts)
- ⚠️ Otherwise, just use `run_all_experiments.sh --skip-analysis` directly

**Usage**:
```bash
# Run all environments sequentially
./run_full_scale_data_collection.sh --all --project <project> --bucket <bucket>

# Run single environment
./run_full_scale_data_collection.sh --env native
```

**Status**: ✅ **ACTIVE - Convenience wrapper** (can be replaced with `run_all_experiments.sh --skip-analysis`)

---

### `scripts/run_experiment.sh` - Single Experiment Router

**Purpose**: Convenience router for running a single experiment

**What it does**:
- Routes to `run_local.sh`, `run_minikube.sh`, or `deploy_gcp.sh` based on `--env`
- Provides unified interface for single experiments

**When to use**:
- ✅ For running a single experiment manually
- ✅ For testing/debugging individual scenarios
- ⚠️ For batch runs, use `run_all_experiments.sh` instead

**Usage**:
```bash
# Single native experiment
./scripts/run_experiment.sh --env native --scenario scenarios/kyber512.yaml --out results/test

# Single minikube experiment
./scripts/run_experiment.sh --env minikube --scenario scenarios/kyber512.yaml --out results/test --exp-id test
```

**Status**: ✅ **ACTIVE - Convenience router** (optional, for single experiments)

---

## Environment-Specific Execution Scripts

These are used internally by `run_all_experiments.sh` and `scripts/run_experiment.sh`.

### `run_local.sh` - Native Execution

**Purpose**: Run single experiment natively on local machine

**Used by**: `run_all_experiments.sh`, `scripts/run_experiment.sh`

**Status**: ✅ **ACTIVE - Internal use**

---

### `run_minikube.sh` - Minikube Execution

**Purpose**: Run single experiment on Minikube Kubernetes cluster

**Used by**: `run_all_experiments.sh`, `scripts/run_experiment.sh`

**Status**: ✅ **ACTIVE - Internal use**

---

### `deploy_gcp.sh` - GCP Execution

**Purpose**: Run single experiment on GKE (ephemeral or persistent cluster mode)

**Used by**: 
- `run_all_experiments.sh` (ephemeral mode only)
- `scripts/run_experiment.sh`

**Status**: ✅ **ACTIVE - Internal use**

---

## GCP Job Submission Scripts

### `scripts/submit_gcp_job_parallel.sh` - Single Job Submission

**Purpose**: Submit a single GCP experiment as a Kubernetes Job (lightweight, for persistent clusters)

**What it does**:
- Creates ConfigMaps (scenario + GCP config)
- Submits Kubernetes Job using unified `submit_k8s_job()` function
- Sets up GCP service account (Workload Identity)
- Returns job name for tracking

**Used by**: `run_all_experiments.sh` (persistent cluster mode)

**Status**: ✅ **ACTIVE - Internal use** (just refactored with unified functions)

---

### `scripts/submit_parallel_gcp_jobs.sh` - **REMOVED** ✅

**Status**: ✅ **REMOVED** (2025-12-11)

**Reason for removal**: 
- `run_all_experiments.sh` already handles parallel job submission internally
- Uses unified `submit_k8s_job()` function
- Better integrated with experiment tracking and result retrieval
- No longer needed - functionality fully replaced

---

## Summary

### Keep (Active)
1. ✅ `run_all_experiments.sh` - **PRIMARY SCRIPT**
2. ✅ `run_full_scale_data_collection.sh` - Convenience wrapper (optional)
3. ✅ `scripts/run_experiment.sh` - Single experiment router (optional)
4. ✅ `run_local.sh` - Internal
5. ✅ `run_minikube.sh` - Internal
6. ✅ `deploy_gcp.sh` - Internal
7. ✅ `scripts/submit_gcp_job_parallel.sh` - Internal

### Removed (Obsolete)
1. ✅ `scripts/submit_parallel_gcp_jobs.sh` - **REMOVED** (2025-12-11) - Functionality replaced by `run_all_experiments.sh`

---

## Recommended Usage

### For Most Users
**Always use `run_all_experiments.sh`**:
```bash
# Smoke test
./run_all_experiments.sh --smoke-test --envs native,minikube,gcp --project <p> --bucket <b>

# Full scale with analysis
./run_all_experiments.sh --envs native,minikube,gcp --project <p> --bucket <b>

# Full scale data collection only
./run_all_experiments.sh --envs native,minikube,gcp --project <p> --bucket <b> --skip-analysis
```

### For Data Collection Workflows
Use `run_full_scale_data_collection.sh` if you want:
- Per-environment tracking (`data-collection-*/`)
- Running environments separately
- Otherwise, use `run_all_experiments.sh --skip-analysis`

### For Single Experiments
Use `scripts/run_experiment.sh` for:
- Testing individual scenarios
- Debugging
- Manual single experiment runs

---

## Migration Guide

If you're using obsolete scripts:

### ~~Old: `scripts/submit_parallel_gcp_jobs.sh`~~ (REMOVED)
```bash
# OLD (obsolete)
# OLD (REMOVED): ./scripts/submit_parallel_gcp_jobs.sh --scenarios manifest.json --project <p> --bucket <b> --parallel 20
# NEW: Use run_all_experiments.sh instead (handles parallel execution internally)
```

**New**: Use `run_all_experiments.sh`:
```bash
# NEW (recommended)
./run_all_experiments.sh --envs gcp --project <p> --bucket <b> --parallel 20 --skip-analysis
```
