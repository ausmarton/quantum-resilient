# Research Artifact Generation

Tools for generating dissertation-ready artifacts from benchmark experiments.

## Overview

This module provides:
- **Provenance tracking**: Capture all metadata about experiment runs
- **Dataset versioning**: Deterministic checksums and semantic versions
- **Table generation**: LaTeX and Markdown tables for publications
- **Figure bundling**: PDF, EPS, and high-DPI PNG exports with captions
- **Report generation**: Jinja2-based LaTeX and Markdown reports
- **Pipeline automation**: End-to-end reproducible workflow

## Directory Structure

```
research/
├── templates/           # Jinja2 templates
│   ├── report.md.j2
│   ├── report.tex.j2
│   ├── table_summary.tex.j2
│   ├── figure_caption.tex.j2
│   └── metadata.json.j2
├── scripts/             # Generation scripts
│   ├── generate_report.py
│   ├── generate_tables.py
│   ├── generate_figures_bundle.py
│   ├── provenance.py
│   ├── version_dataset.py
│   └── pipeline_runner.py
├── output/              # Generated artifacts (gitignored)
│   └── <exp-id>/
│       ├── provenance.json
│       ├── dataset_version.json
│       ├── tables/
│       ├── figures/
│       ├── report.md
│       └── report.tex
└── README.md
```

## Quick Start

### Full Pipeline (Recommended)

Run the complete research documentation pipeline:

```bash
python research/scripts/pipeline_runner.py \
  --exp-id exp_2025_02_01_001 \
  --uri gs://qr-results/exp_2025_02_01_001 \
  --generate-all
```

This executes:
1. Fetch results from storage
2. Merge JSONL files
3. Compute statistics
4. Compute effect sizes
5. Generate provenance metadata
6. Version dataset
7. Generate LaTeX/Markdown tables
8. Bundle figures (PDF, EPS, PNG)
9. Generate final report

### Individual Scripts

#### 1. Provenance Generation

```bash
python research/scripts/provenance.py \
  --exp-id exp_2025_02_01_001 \
  --data-dir analysis/data/exp_2025_02_01_001 \
  --out research/output/exp_2025_02_01_001/
```

#### 2. Dataset Versioning

```bash
python research/scripts/version_dataset.py \
  --exp-id exp_2025_02_01_001 \
  --data-dir analysis/data/exp_2025_02_01_001 \
  --version 1.0.0 \
  --out research/output/exp_2025_02_01_001/
```

#### 3. Table Generation

```bash
python research/scripts/generate_tables.py \
  --exp-id exp_2025_02_01_001 \
  --stats-file analysis/data/exp_2025_02_01_001/stats/summary.json \
  --out research/output/exp_2025_02_01_001/tables/
```

#### 4. Figure Bundle

```bash
python research/scripts/generate_figures_bundle.py \
  --exp-id exp_2025_02_01_001 \
  --figures-dir analysis/figures/exp_2025_02_01_001 \
  --out research/output/exp_2025_02_01_001/figures/
```

#### 5. Report Generation

```bash
# LaTeX report
python research/scripts/generate_report.py \
  --exp-id exp_2025_02_01_001 \
  --format tex \
  --out research/output/exp_2025_02_01_001/

# Markdown report
python research/scripts/generate_report.py \
  --exp-id exp_2025_02_01_001 \
  --format md \
  --out research/output/exp_2025_02_01_001/
```

## Output Artifacts

### provenance.json

Contains complete experiment metadata:

```json
{
  "experiment_id": "exp_2025_02_01_001",
  "git_commit": "abc123...",
  "timestamp": "2025-02-01T12:00:00Z",
  "scenario_yaml": "...",
  "checksums": {
    "merged.jsonl": "sha256:...",
    "merged.parquet": "sha256:..."
  },
  "cluster_config": {...},
  "worker_jitter_stats": {...}
}
```

### dataset_version.json

Semantic versioning for datasets:

```json
{
  "version": "1.0.0",
  "experiment_id": "exp_2025_02_01_001",
  "checksums": {...},
  "changelog": "Initial release"
}
```

### Tables

- `latency_quantiles.tex` / `.md`
- `throughput_summary.tex` / `.md`
- `queue_delay_stats.tex` / `.md`
- `adapter_comparison.tex` / `.md`
- `effect_sizes.tex` / `.md`

### Figures

- High-DPI PNG (300 DPI)
- PDF (vector)
- EPS (vector, for LaTeX)
- `manifest.json` with captions and labels
- `figures_bundle.tar.gz` for dissertation appendix

### Reports

- `report.tex` - Full LaTeX report with embedded tables and figure references
- `report.md` - Markdown version for documentation

## Publishing Workflow

1. **Run pipeline**:
   ```bash
   python research/scripts/pipeline_runner.py --exp-id exp_001 --uri gs://... --generate-all
   ```

2. **Verify provenance**:
   ```bash
   cat research/output/exp_001/provenance.json
   ```

3. **Check dataset version**:
   ```bash
   cat research/output/exp_001/dataset_version.json
   ```

4. **Insert into dissertation**:
   ```latex
   \input{research/output/exp_001/tables/latency_quantiles.tex}
   \includegraphics{research/output/exp_001/figures/latency_cdf.pdf}
   ```

5. **Tag release**:
   ```bash
   git tag -a exp_001_published -m "Published experiment exp_001"
   git push --tags
   ```

## Templates

Templates use Jinja2 syntax. Customize by editing files in `templates/`.

### Variables Available

- `experiment_id`: Experiment identifier
- `provenance`: Full provenance metadata dict
- `stats`: Statistical summary dict
- `tables`: Dict of table names to content
- `figures`: List of figure metadata
- `timestamp`: Generation timestamp

## Requirements

Install additional dependencies:

```bash
pip install jinja2 pylatexenc pyyaml tabulate
```

