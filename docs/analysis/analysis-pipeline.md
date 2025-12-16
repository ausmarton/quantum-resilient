# Analysis Pipeline Documentation

**Date**: 2025-12-14  
**Status**: Active  
**Purpose**: Complete documentation of the data analysis pipeline

---

## Overview

This document describes the complete analysis pipeline for processing experiment data, generating summaries, and producing dissertation-ready artifacts.

**Related Documentation**:
- **[Framework Architecture Diagram](../diagrams/framework-architecture.mmd)** - Visual representation of all framework components including the analysis layer
- **[Diagrams README](../diagrams/README.md)** - Complete description of all architectural diagrams

The analysis pipeline corresponds to the **Analysis Layer** in the framework architecture diagram (Figure 3.1), which includes:
- Hypothesis Testing (t-test, Mann-Whitney U)
- Effect Size Computation (Cohen's d, Confidence Intervals)
- Statistics Computation (Percentiles, Aggregates)
- Visualization Scripts (CDFs, Comparison Charts)
- Jupyter Notebooks (Exploratory Analysis)
- Export Utilities (Dataset Export, Merge)

---

## Pipeline Stages

### Stage 1: Data Collection ✅

**Status**: Complete  
**Output**: Raw JSONL files in `results/env/experiment/run-X/raw/run.jsonl`

**Process**:
- Experiments run across three environments: native, minikube, GCP
- Each experiment has 1-5 runs
- Each run produces a JSONL file with event-level telemetry
- Total: 396 experiments with ECDHE (120 native + 138 minikube + 138 gcp), runs, events

**Validation**:
```bash
./scripts/validate_dissertation_data.sh --results-dir results
```

---

### Stage 2: Summary Generation ✅

**Status**: Complete (331/330 summaries)  
**Script**: `scripts/generate_experiment_summaries.sh`  
**Output**: `results/env/experiment/stats/summary.json`

**Process**:
1. **Merge Runs**: Concatenate all `run-X/raw/run.jsonl` files into `merged/merged.jsonl`
2. **Compute Statistics**: Run `analysis/scripts/compute_statistics.py` on merged data
3. **Generate Summary**: Create `stats/summary.json` with:
   - Latency statistics (p50, p95, p99, mean, std) in nanoseconds and microseconds
   - Throughput statistics (mean, max messages/second)
   - Memory utilization (mean, max RSS)
   - CPU utilization (if available)
   - Worker skew and drift detection

**Optimizations**:
- **Parallelization**: 16 parallel jobs (auto-detected from CPU cores)
- **Chunked Reading**: Files >100MB processed in chunks to manage memory
- **Resume Capability**: Auto-skips already-processed experiments
- **Container Support**: Uses `scripts/lib/run-python-container.sh` with SELinux fix

**Usage**:
```bash
# Generate all summaries
./scripts/generate_experiment_summaries.sh --parallel 16

# Generate for specific environment
./scripts/generate_experiment_summaries.sh --env native --parallel 16

# Resume (skip already-processed)
./scripts/generate_experiment_summaries.sh --resume
```

**Validation**:
```bash
# Validate summary structure
./scripts/validate_summaries.sh

# Validate against raw data
./scripts/lib/run-python-container.sh scripts/validate_summaries_against_raw.py
```

**Known Issues & Fixes**:
- **SELinux Permission Errors**: Fixed by using `:z` flag (shared context) instead of `:Z` in Podman volume mounts
- **Unit Mismatch**: Summaries store latency in both nanoseconds (`latency_ns`) and microseconds (`latency`) for compatibility

---

### Stage 3: Aggregation

**Status**: Pending (waiting for validation)  
**Script**: `analysis/aggregate_results.py`  
**Output**: `final-results/aggregated_stats.json`, `aggregated_stats.csv`

**Process**:
1. Load all summaries from `final-results/index.json`
2. Group by algorithm, environment, payload size, rate
3. Compute aggregated statistics (mean, std, confidence intervals)
4. Calculate effect sizes (Cohen's d) for comparisons
5. Compute environment deltas (native→minikube, native→GCP)

**Usage**:
```bash
./scripts/lib/run-python-container.sh analysis/aggregate_results.py \
  --index final-results/index.json \
  --output final-results/
```

**Outputs**:
- `aggregated_stats.json`: Complete aggregated statistics
- `aggregated_stats.csv`: CSV format for tables
- `stats/effect_sizes.json`: Effect size calculations
- `stats/environment_deltas.json`: Environment comparisons

---

### Stage 4: Visualization

**Status**: Pending  
**Scripts**: Multiple plot scripts in `analysis/`

**Visualizations Generated**:

1. **CDF Plots** (`plot_combined_cdfs.py`):
   - Combined ECDF for all algorithms
   - Per-environment ECDFs (native, minikube, GCP)
   - Environment comparison panels
   - Payload-size panels

2. **Scaling Curves** (`plot_scaling_curves.py`):
   - Latency vs. replica count
   - Throughput vs. replica count
   - Resource utilization vs. scale

3. **Environment Comparisons** (`compare_all_environments.py`):
   - Side-by-side comparisons
   - Delta visualizations

4. **Replica Scaling** (`plot_replica_scaling.py`):
   - Horizontal scaling analysis

**Usage**:
```bash
# Generate all CDF plots
./scripts/lib/run-python-container.sh analysis/plot_combined_cdfs.py \
  --index final-results/index.json \
  --output final-results/figures

# Generate scaling curves
./scripts/lib/run-python-container.sh analysis/plot_scaling_curves.py \
  --index final-results/index.json \
  --output final-results/figures
```

---

### Stage 5: Statistical Testing

**Status**: Pending  
**Script**: `analysis/hypothesis_tests.py`  
**Output**: `final-results/hypothesis_tests.json`, `hypothesis_table.csv`

**Process**:
1. Load aggregated statistics
2. Perform statistical tests:
   - Kolmogorov-Smirnov test (distribution comparison)
   - Mann-Whitney U test (non-parametric comparison)
   - Welch's t-test (parametric comparison with unequal variances)
3. Apply Holm-Bonferroni correction for multiple comparisons
4. Calculate effect sizes (Cohen's d) with confidence intervals

**Usage**:
```bash
./scripts/lib/run-python-container.sh analysis/hypothesis_tests.py \
  --index final-results/index.json \
  --matrix orchestration/experiment_matrix.yaml \
  --output final-results/
```

**Outputs**:
- `hypothesis_tests.json`: Complete test results
- `hypothesis_table.csv`: CSV format
- `hypothesis_interpretation.txt`: Human-readable summary

---

### Stage 6: Table Extraction

**Status**: Pending  
**Script**: `scripts/extract_dissertation_tables.py`  
**Output**: `final-results/tables/*.csv`, `*.tex`

**Process**:
1. Load aggregated statistics and hypothesis tests
2. Generate tables:
   - Performance comparison table
   - Effect size table
   - Environment delta table
3. Export in CSV and LaTeX formats

**Usage**:
```bash
./scripts/lib/run-python-container.sh scripts/extract_dissertation_tables.py \
  --aggregated final-results/aggregated_stats.json \
  --hypothesis final-results/hypothesis_tests.json \
  --output final-results/tables/
```

---

### Stage 7: Interpretation

**Status**: ✅ Complete  
**Script**: `scripts/generate_interpretation_docs.py`  
**Template**: `docs/analysis/interpretation-framework.md`

**Process**:
1. Extract key findings from analysis results
2. Map to dissertation claims (from `docs/dissertation-requirements.md`)
3. Update interpretation documents:
   - Executive summary
   - Algorithm performance analysis
   - Environment comparison
   - Horizontal scaling analysis
   - Statistical significance
   - Size/bandwidth analysis
   - Practical implications

**Idempotency**: Safe to re-run (updates existing markdown files)

**Usage**:
```bash
./scripts/lib/run-python-container.sh scripts/generate_interpretation_docs.py \
  --stats final-results/aggregated_stats.json \
  --hypothesis final-results/hypothesis_tests.json \
  --output docs/analysis
```

---

### Stage 8: Additional Metrics (Optional)

**Status**: ✅ Implemented  
**Scripts**: 
- `scripts/compute_cost_efficiency.py` (FR13)
- `scripts/generate_analysis_report.py` (NFR8)

**Process**:
1. Compute cost efficiency metrics for GCP deployments
2. Generate comprehensive analysis report

**Idempotency**: Checks for existing outputs before generating

---

### Stage 9: Requirements Compliance

**Status**: ✅ Complete  
**Script**: `scripts/verify_requirements_compliance.py`

**Process**:
1. Verify all required outputs exist
2. Check data completeness
3. Validate statistical requirements
4. Verify dissertation claims support

**Idempotency**: Always runs (validation step)

---

## Complete Pipeline Execution

**Script**: `scripts/run_full_analysis_pipeline.sh` (Recommended)

**Usage**:
```bash
# Run full pipeline (idempotent - skips existing outputs)
./scripts/run_full_analysis_pipeline.sh

# Force regeneration of all outputs
./scripts/run_full_analysis_pipeline.sh --force

# Skip specific stages if already done
./scripts/run_full_analysis_pipeline.sh --skip-summaries --skip-visualizations

# Process only specific environment
./scripts/run_full_analysis_pipeline.sh --env native
```

**What it does** (all stages are idempotent):
1. **Stage 1**: Generate experiment summaries (from raw JSONL)
   - Merges runs, computes statistics
   - Skips if `stats/summary.json` exists (unless `--force`)
2. **Stage 2**: Aggregate statistics
   - Groups by algorithm/environment/payload/rate
   - Skips if `aggregated_stats.json` exists
3. **Stage 3**: Hypothesis tests
   - Statistical comparisons with corrections
   - Skips if `hypothesis_tests.json` exists
4. **Stage 4**: Generate visualizations
   - CDF plots, scaling curves, environment comparisons
   - Skips if figures exist
5. **Stage 5**: Generate tables
   - Performance, effect size, environment delta tables
   - Skips if tables exist
6. **Stage 6**: Generate interpretation documents
   - Updates markdown docs with extracted data
   - Safe to re-run (updates existing docs)
7. **Stage 7**: Verify requirements compliance
   - Checks all outputs against requirements
   - Always runs (validation)

**Idempotency**: The pipeline checks for existing outputs and skips regeneration unless `--force` is used. This ensures:
- No unnecessary re-processing
- Safe to re-run after partial failures
- Can resume from any stage

**Legacy Script**: `scripts/run_complete_analysis.sh` (may exist, but use new script)

---

## Troubleshooting

### SELinux Permission Errors

**Problem**: Container cannot read files created by host  
**Solution**: Use `:z` flag (shared SELinux context) in Podman volume mounts

**Fixed in**: `scripts/lib/run-python-container.sh`
```bash
# Changed from:
VOLUME_FLAGS="-v $SCRIPT_DIR:/workspace:rw,Z"

# To:
VOLUME_FLAGS="-v $SCRIPT_DIR:/workspace:rw,z"
```

### Unit Mismatch in Validation

**Problem**: Validation script reports 99.9% difference  
**Cause**: Raw data in nanoseconds, summaries also store microseconds  
**Solution**: Validation script now detects and handles unit conversion

### Missing Dependencies

**Problem**: `ModuleNotFoundError` when running scripts  
**Solution**: Use containerized execution:
```bash
./scripts/lib/run-python-container.sh <script.py> [args...]
```

---

## File Structure

```
results/
├── native/
│   └── <experiment>/
│       ├── run-1/raw/run.jsonl
│       ├── run-2/raw/run.jsonl
│       ├── merged/merged.jsonl
│       └── stats/
│           ├── summary.json
│           ├── latency_hist.png
│           ├── queue_hist.png
│           └── throughput_curve.png
├── minikube/...
└── gcp/...

final-results/
├── index.json
├── aggregated_stats.json
├── aggregated_stats.csv
├── hypothesis_tests.json
├── hypothesis_table.csv
├── figures/
│   ├── combined_ecdf.png
│   ├── combined_ecdf_native.png
│   └── ...
└── tables/
    ├── performance_table.csv
    ├── performance_table.tex
    └── ...
```

---

## Validation & Quality Assurance

### Summary Validation
```bash
# Structure validation
./scripts/validate_summaries.sh

# Accuracy validation (against raw data)
./scripts/lib/run-python-container.sh scripts/validate_summaries_against_raw.py
```

### Data Quality Checks
- Event count consistency
- Latency percentile ordering (p50 < p95 < p99)
- Throughput sanity checks
- Memory/CPU data validity

---

## Performance

**Summary Generation**:
- **Time**: ~3-6 minutes for 396 experiments (with 16 parallel jobs, includes ECDHE)
- **Rate**: ~100+ summaries/minute
- **Optimizations**: Parallelization, chunked reading, resume capability

**Analysis Pipeline**:
- **Aggregation**: ~1-2 minutes
- **Visualizations**: ~2-3 minutes
- **Hypothesis Tests**: ~1-2 minutes
- **Total**: ~5-10 minutes for complete pipeline

---

## Dependencies

**Container Image**: `quantum-resilient-analysis:latest`
- Python 3.11
- pandas, numpy, matplotlib, scipy
- rich (for console output)

**Build**:
```bash
podman build -t quantum-resilient-analysis -f analysis/Dockerfile analysis/
```

---

## Related Documentation

- `docs/dissertation-requirements.md` - Requirements and claims
- `docs/analysis/interpretation-framework.md` - Interpretation template
- `DEVELOPMENT_GUIDELINES.md` - Development standards
- `docs/REQUIREMENTS_SPECIFICATION.md` - Complete requirements

---

**Last Updated**: 2025-12-14
