# Reproducible Analysis Guide

**Date**: 2025-12-14  
**Purpose**: Guide for running reproducible analysis from raw data to dissertation artifacts

---

## Overview

The analysis pipeline is designed to be **fully reproducible** and **idempotent**. You can re-run the complete pipeline from raw data to final artifacts at any time, and it will intelligently skip stages that have already been completed.

---

## Quick Start

### Run Complete Pipeline

```bash
# Run full pipeline (idempotent - skips existing outputs)
./scripts/run_full_analysis_pipeline.sh

# Force regeneration of everything
./scripts/run_full_analysis_pipeline.sh --force

# Skip specific stages
./scripts/run_full_analysis_pipeline.sh --skip-summaries --skip-visualizations
```

---

## Pipeline Stages

The pipeline consists of 9 stages, each with built-in idempotency:

### Stage 1: Summary Generation
**Input**: Raw JSONL files (`results/*/run-*/raw/run.jsonl`)  
**Output**: Experiment summaries (`results/*/stats/summary.json`)  
**Idempotency**: Skips experiments that already have `stats/summary.json` (unless `--force`)

```bash
# Manual execution
./scripts/generate_experiment_summaries.sh --resume
```

### Stage 2: Aggregation
**Input**: Experiment summaries  
**Output**: `final-results/aggregated_stats.json`  
**Idempotency**: Checks if output exists before running

```bash
# Manual execution
./scripts/lib/run-python-container.sh analysis/aggregate_results.py \
  --index final-results/index.json \
  --output final-results/
```

### Stage 3: Hypothesis Tests
**Input**: Aggregated statistics  
**Output**: `final-results/hypothesis_tests.json`  
**Idempotency**: Checks if output exists before running

```bash
# Manual execution
./scripts/lib/run-python-container.sh analysis/hypothesis_tests.py \
  --index final-results/index.json \
  --matrix orchestration/experiment_matrix.yaml \
  --output final-results/
```

### Stage 4: Visualizations
**Input**: Aggregated statistics, experiment summaries  
**Output**: Figures in `final-results/figures/`  
**Idempotency**: Checks for existing figure files

```bash
# Manual execution
./scripts/lib/run-python-container.sh analysis/plot_combined_cdfs.py \
  --index final-results/index.json \
  --output final-results/figures
```

### Stage 5: Tables
**Input**: Aggregated statistics, hypothesis tests  
**Output**: CSV and LaTeX tables in `final-results/tables/`  
**Idempotency**: Checks for existing table files

```bash
# Manual execution
./scripts/lib/run-python-container.sh scripts/extract_dissertation_tables.py \
  --input final-results/aggregated_stats.json \
  --output final-results/tables/
```

### Stage 6: Interpretation Documents
**Input**: Aggregated statistics, hypothesis tests  
**Output**: Updated markdown files in `docs/analysis/`  
**Idempotency**: Safe to re-run (updates existing files)

```bash
# Manual execution
./scripts/lib/run-python-container.sh scripts/generate_interpretation_docs.py \
  --stats final-results/aggregated_stats.json \
  --hypothesis final-results/hypothesis_tests.json \
  --output docs/analysis
```

### Stage 7: Additional Metrics (Optional)
**Input**: Aggregated statistics  
**Output**: 
- `final-results/cost_efficiency.json` (FR13)
- `final-results/analysis_report.md` (NFR8)  
**Idempotency**: Checks for existing outputs

### Stage 8: Requirements Compliance
**Input**: All generated outputs  
**Output**: `final-results/compliance_report.json`  
**Idempotency**: Always runs (validation step)

```bash
# Manual execution
./scripts/lib/run-python-container.sh scripts/verify_requirements_compliance.py \
  --base-dir final-results \
  --output final-results/compliance_report.json
```

---

## Idempotency Guarantees

### Summary Generation
- **Check**: `results/env/experiment/stats/summary.json` exists
- **Skip**: If exists and valid JSON
- **Force**: Use `--force` flag or delete summaries manually

### Aggregation
- **Check**: `final-results/aggregated_stats.json` exists
- **Skip**: If exists and valid JSON
- **Force**: Delete file or use `--force` flag

### Hypothesis Tests
- **Check**: `final-results/hypothesis_tests.json` exists
- **Skip**: If exists and valid JSON
- **Force**: Delete file or use `--force` flag

### Visualizations
- **Check**: Key figure files exist (e.g., `combined_ecdf.png`)
- **Skip**: If all expected figures exist
- **Force**: Delete figures or use `--force` flag

### Tables
- **Check**: Key table files exist (e.g., `performance_table.csv`)
- **Skip**: If all expected tables exist
- **Force**: Delete tables or use `--force` flag

---

## Re-running After Data Collection

If you've collected new data or need to re-run on existing environments:

```bash
# 1. Regenerate index (if new experiments added)
./scripts/lib/run-python-container.sh scripts/lib/regenerate_index.py \
  --results-dir results \
  --output final-results/index.json

# 2. Run full pipeline (will regenerate summaries for new/updated experiments)
./scripts/run_full_analysis_pipeline.sh

# 3. Or force full regeneration
./scripts/run_full_analysis_pipeline.sh --force
```

---

## Partial Re-runs

You can re-run specific stages without re-processing earlier stages:

```bash
# Only regenerate visualizations (assumes summaries and aggregation done)
./scripts/run_full_analysis_pipeline.sh \
  --skip-summaries \
  --skip-aggregation \
  --skip-tables \
  --skip-interpretation

# Only regenerate tables (assumes aggregation done)
./scripts/run_full_analysis_pipeline.sh \
  --skip-summaries \
  --skip-aggregation \
  --skip-visualizations \
  --skip-interpretation
```

---

## Environment-Specific Re-runs

Process only specific environment:

```bash
# Process only native environment
./scripts/run_full_analysis_pipeline.sh --env native

# Process only GCP environment
./scripts/run_full_analysis_pipeline.sh --env gcp
```

---

## Verification

After running the pipeline, verify outputs:

```bash
# Check compliance
cat final-results/compliance_report.json | jq '.compliance_percentage'

# Validate summaries
./scripts/validate_summaries.sh

# Check file counts
echo "Summaries: $(find results -name summary.json | wc -l)"
echo "Figures: $(find final-results/figures -name '*.png' | wc -l)"
echo "Tables: $(find final-results/tables -name '*.csv' | wc -l)"
```

---

## Troubleshooting

### Pipeline Fails at Stage X

1. Check the error message
2. Fix the issue
3. Re-run with `--skip-summaries` (or appropriate skip flags) to resume from the failed stage

### Need to Regenerate Everything

```bash
# Remove all generated outputs
rm -rf final-results/*
find results -name summary.json -delete

# Run full pipeline
./scripts/run_full_analysis_pipeline.sh
```

### Container Issues

If container execution fails:
```bash
# Try without container (requires host dependencies)
QR_USE_CONTAINER=false ./scripts/run_full_analysis_pipeline.sh
```

---

## Best Practices

1. **Always use the pipeline script** for reproducibility
2. **Don't manually delete outputs** unless you need to force regeneration
3. **Use `--resume` or skip flags** when re-running after partial failures
4. **Verify outputs** after pipeline completion
5. **Commit pipeline script** to version control for reproducibility

---

## Related Documentation

- `docs/analysis/analysis-pipeline.md` - Detailed pipeline documentation
- `scripts/run_full_analysis_pipeline.sh` - Pipeline script source
- `DEVELOPMENT_GUIDELINES.md` - Development standards

---

**Last Updated**: 2025-12-14
