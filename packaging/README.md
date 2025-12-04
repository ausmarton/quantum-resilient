# Packaging and Distribution Tools

Tools for creating archival bundles, publication-ready exports, and distributing experiment results.

## Overview

This module provides:
- **Manifest generation**: Machine-readable metadata with checksums
- **Archive creation**: ZIP and TAR.GZ bundles
- **Export structure**: Clean folder layout for publications
- **Release notes**: Human-readable documentation
- **Publishing**: Upload to GCS, S3, or GitHub Releases

## Quick Start

### Using the CLI

```bash
# Create complete bundle (manifest + archives)
python -m packaging bundle exp_001

# Create export folder for publication
python -m packaging export exp_001

# Generate manifest only
python -m packaging manifest exp_001

# Generate release notes
python -m packaging notes exp_001

# Publish to GCS
python -m packaging publish exp_001 \
  --target gcs \
  --uri gs://bucket/path

# Run all packaging steps
python -m packaging all exp_001
```

### Integration with Pipeline

```bash
# Run research pipeline with packaging
python research/scripts/pipeline_runner.py \
  --exp-id exp_001 \
  --generate-all \
  --package

# With publishing
python research/scripts/pipeline_runner.py \
  --exp-id exp_001 \
  --generate-all \
  --package \
  --publish-target gcs \
  --publish-uri gs://bucket/path
```

## Output Structure

### Bundle Archives

```
packaging/output/exp_001/
├── manifest.json
├── release_notes.md
├── exp_001-research-bundle.zip
├── exp_001-research-bundle.tar.gz
└── export/
    ├── data/
    │   ├── merged.parquet
    │   └── summary.json
    ├── figures/
    ├── tables/
    ├── report/
    ├── metadata/
    └── README.md
```

### Manifest Format

```json
{
  "schema_version": "1.0.0",
  "experiment_id": "exp_001",
  "experiment_timestamp_utc": "2025-01-01T00:00:00Z",
  "git_commit": "abc123...",
  "dataset_version": "1.0.0",
  "dataset_checksum": "sha256:...",
  "files": [
    {"path": "...", "sha256": "...", "size_bytes": 0}
  ],
  "figures": [...],
  "tables": [...],
  "stats_summary": {...}
}
```

## CLI Commands

### `bundle`

Create research bundles (ZIP and TAR.GZ archives):

```bash
python -m packaging bundle exp_001 \
  --data-dir analysis/data/exp_001 \
  --research-dir research/output/exp_001 \
  --out packaging/output/exp_001 \
  --formats zip,tar.gz \
  --verify
```

### `export`

Create publication-ready export folder:

```bash
python -m packaging export exp_001 \
  --data-dir analysis/data/exp_001 \
  --research-dir research/output/exp_001 \
  --out packaging/output/exp_001 \
  --lite  # Excludes JSONL files
```

### `manifest`

Generate experiment manifest:

```bash
python -m packaging manifest exp_001 \
  --data-dir analysis/data/exp_001 \
  --research-dir research/output/exp_001 \
  --uri gs://bucket/exp_001
```

### `notes`

Generate release notes:

```bash
python -m packaging notes exp_001 \
  --data-dir analysis/data/exp_001 \
  --research-dir research/output/exp_001 \
  --description "Experiment description"
```

### `publish`

Publish to cloud storage or GitHub:

```bash
# Google Cloud Storage
python -m packaging publish exp_001 \
  --bundle packaging/output/exp_001/exp_001-research-bundle.zip \
  --target gcs \
  --uri gs://bucket/path \
  --public

# AWS S3
python -m packaging publish exp_001 \
  --bundle packaging/output/exp_001/exp_001-research-bundle.zip \
  --target s3 \
  --uri s3://bucket/path

# GitHub Releases
python -m packaging publish exp_001 \
  --bundle packaging/output/exp_001/exp_001-research-bundle.zip \
  --target github \
  --uri owner/repo
```

### `all`

Run complete packaging pipeline:

```bash
python -m packaging all exp_001 \
  --data-dir analysis/data/exp_001 \
  --research-dir research/output/exp_001 \
  --uri gs://bucket/exp_001
```

## Publishing Targets

### Google Cloud Storage (GCS)

Requires GCP authentication:

```bash
gcloud auth application-default login
```

Features:
- Uploads bundle, manifest, and checksum files
- Optional public-read access
- Verification after upload

### AWS S3 / MinIO

Requires AWS credentials or `S3_ENDPOINT_URL` for MinIO:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export S3_ENDPOINT_URL=http://localhost:9000  # For MinIO
```

### GitHub Releases

Requires `GITHUB_TOKEN` environment variable:

```bash
export GITHUB_TOKEN=ghp_...
```

Creates or updates a release tagged `exp-<experiment_id>` with the bundle as an asset.

## Templates

Templates in `templates/` use Jinja2 syntax:

- `manifest.json.j2` - Manifest template
- `release_notes.md.j2` - Release notes template

## Requirements

```
typer>=0.9.0
rich>=13.0.0
jinja2>=3.1.0
google-cloud-storage>=2.0.0  # For GCS
boto3>=1.26.0                # For S3
requests>=2.31.0             # For GitHub
```

