# Unified Benchmark Flow

**Date**: 2025-12-10  
**Status**: Active  
**Purpose**: Single unified flow for both smoke-test and full-scale benchmarks

---

## Overview

The benchmark system uses a **single unified flow** for both smoke-test and full-scale benchmarks. The distinction comes from experiment matrix filtering and parameters, not separate code paths or directory structures.

## Key Principles

1. **Same Code Path**: Both smoke-test and full-scale use the same scripts and code
2. **Same Structure**: Both use `results/<env>/<scenario-id>/` for individual experiments
3. **Same IDs**: Scenario IDs are identical format regardless of mode
4. **Same Pipeline**: Analysis pipeline works identically for both modes
5. **Controlled by Flag**: `--smoke-test` flag controls filtering and parameters only

## Unified Flow

### Execution

```bash
# Full-scale benchmarks
./run_all_experiments.sh \
  --envs native,minikube,gcp \
  --project <gcp-project> \
  --bucket <gcs-bucket>

# Smoke-test benchmarks (same command, add --smoke-test)
./run_all_experiments.sh \
  --smoke-test \
  --envs native,minikube,gcp
```

### What Changes with --smoke-test Flag

**Experiment Matrix Filtering**:
- Algorithms: Subset (rsa2048, kyber512, dilithium2, hybrid_kyber_dilithium)
- Payload sizes: Reduced (256B, 1024B instead of full set)
- Rates: Reduced (100, 500 msg/s instead of full set)
- Duration: Reduced (5s instead of 30s)
- Runs: Reduced (1 instead of 5)
- Excludes: 10K msg/s experiments, 5-minute duration experiments

**Final Results Directory**:
- Both modes: `final-results/` (unified)

**What Stays the Same**:
- Scenario IDs: Same format (`rsa2048_p256_r100_5s_run1_abc123`)
- Results structure: `results/<env>/<scenario-id>/`
- Final results: `final-results/` (unified for both modes)
- Scenario generation: Same `orchestration/generate_scenarios.py`
- Analysis pipeline: Same scripts and logic
- Validation scripts: Same scripts work for both
- Directories: Single set of directories for all experiments

## Directory Structure

```
quantum-resilient/
├── generated-scenarios/          # Same for both smoke-test and full-scale
│   ├── rsa2048/
│   │   ├── p256/
│   │   │   └── r100/
│   │   │       └── run-1/
│   │   │           └── scenario.yaml
│   │   └── ...
│   └── ...
│
├── results/                      # Same structure for both modes
│   ├── native/
│   │   └── <scenario-id>/       # Same format regardless of mode
│   │       ├── raw/
│   │       ├── merged/
│   │       └── stats/
│   ├── minikube/
│   │   └── <scenario-id>/
│   └── gcp/
│       └── <scenario-id>/
│
├── final-results/                # Full-scale aggregated results
│   ├── index.json
│   ├── aggregated_stats.json
│   └── figures/
│
└── final-results-smoke/          # Smoke-test aggregated results
    ├── index.json
    ├── aggregated_stats.json
    └── figures/
```

## Scenario IDs

Scenario IDs use the same format regardless of mode:

```
<algorithm>_p<payload>_r<rate>[_<pattern>][_<duration>][_scaling]_run<N>_<hash>
```

Examples:
- `rsa2048_p256_r100_5s_run1_ca05ed4b` (smoke-test: 5s duration)
- `rsa2048_p256_r100_run1_ca05ed4b` (full-scale: 30s default duration)
- `kyber512_p1024_r500_burst_5s_run1_abc123` (smoke-test burst)
- `kyber512_p1024_r500_burst_run1_abc123` (full-scale burst)

**Note**: The same experiment parameters produce the same scenario ID regardless of `--smoke-test` flag. The distinction comes from which scenarios are generated (matrix filtering) and their parameters (duration, runs).

## Validation

All validation scripts work with both modes:

```bash
# Validate smoke-test results
./scripts/validate_experiment_suite.sh \
  --results-dir results \
  --scenarios-dir generated-scenarios

# Validate full-scale results (same command)
./scripts/validate_experiment_suite.sh \
  --results-dir results \
  --scenarios-dir generated-scenarios

# Other validation scripts (work with both)
./scripts/validate_data_quality.sh --env native
./scripts/validate_data_integrity.sh --env native
```

## Benefits

1. ✅ **No Code Duplication**: Single codebase for both modes
2. ✅ **Consistent Structure**: Same directory layout and naming
3. ✅ **Easy Comparison**: Same scenario IDs allow direct comparison
4. ✅ **Maintainability**: Changes apply to both modes automatically
5. ✅ **Validation**: Same validation scripts work for both
6. ✅ **Analysis**: Same analysis pipeline processes both

## Migration Notes

- **Old**: `scripts/run_smoke_test.sh` → **New**: `run_all_experiments.sh --smoke-test` (removed)
- **Old**: `scripts/validate_smoke_test.sh` → **New**: `scripts/validate_experiment_suite.sh` (renamed)
- **Old**: `smoke-test-scenarios/` → **New**: `generated-scenarios/` (same directory)
- **Old**: `results/smoke-test/<env>/` → **New**: `results/<env>/` (same structure)
- **Old**: Scenario IDs with `-smoketest-` prefix → **New**: Same format as full-scale

**Note**: All deprecated scripts have been removed. Use the unified flow directly.
