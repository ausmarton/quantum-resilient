# Documentation Index

This directory contains all documentation for the Quantum-Resilient Cryptography Benchmark Framework, organized by category.

## Quick Links

- **[Getting Started](guides/getting-started.md)** - Quick start guide (coming soon)
- **[Running Experiments](guides/running-experiments.md)** - How to run experiments (coming soon)
- **[Data Collection](guides/data-collection.md)** - Full-scale data collection guide
- **[Storage and Output](guides/storage-and-output.md)** - Where results are stored

## Documentation Structure

### 📚 Guides (`guides/`)

User-facing guides for running experiments and using the framework:

- **[Data Collection](guides/data-collection.md)** - Complete guide for full-scale data collection
- **[Storage and Output](guides/storage-and-output.md)** - Directory structure and overwrite behavior
- **[Stop and Resume](guides/stop-and-resume.md)** - Safely interrupting and resuming experiments
- **[Re-running Experiments](guides/re-running-experiments.md)** - How to delete and re-run experiments

### 📖 Reference (`reference/`)

Technical reference documentation:

- **[System Requirements](reference/system-requirements.md)** - System load and variability handling
- **[GCP Deployment](reference/gcp-deployment.md)** - Complete GCP/GKE deployment guide
- **[Scaling Experiments](reference/scaling-experiments.md)** - Horizontal scaling experiments guide
- **[Data Validation](reference/data-validation.md)** - Data quality validation and status

### 🔬 Analysis (`analysis/`)

Research analysis and experimental design documents:

- **[Experimental Design](analysis/experimental-design.md)** - Experimental design analysis
- **[Hardware Consistency](analysis/hardware-consistency.md)** - Cross-environment hardware analysis
- **[Cost Analysis](analysis/cost-analysis.md)** - Cost and time analysis
- **[Enterprise Representativeness](analysis/enterprise-representativeness.md)** - Enterprise deployment analysis

### 🔧 Troubleshooting (`troubleshooting/`)

Historical fixes and troubleshooting guides:

- **[Git Push Fix](troubleshooting/git-push-fix.md)** - Resolving Git push failures with large files
- **[Scaling Fix](troubleshooting/scaling-fix.md)** - Fix for scaling experiment execution

## Main Documentation

The main project README is located at the repository root:
- **[README.md](../README.md)** - Project overview and quick start

## Finding Documentation

### By Task

- **"How do I run experiments?"** → [Data Collection Guide](guides/data-collection.md)
- **"Where are my results?"** → [Storage and Output Guide](guides/storage-and-output.md)
- **"How do I deploy to GCP?"** → [GCP Deployment Guide](reference/gcp-deployment.md)
- **"How do scaling experiments work?"** → [Scaling Experiments Guide](reference/scaling-experiments.md)
- **"What about system load?"** → [System Requirements](reference/system-requirements.md)
- **"How do I validate my data?"** → [Data Validation](reference/data-validation.md)

### By Environment

- **Native (local)**: See [Data Collection Guide](guides/data-collection.md) - Native section
- **Minikube**: See [Data Collection Guide](guides/data-collection.md) - Minikube section
- **GCP**: See [GCP Deployment Guide](reference/gcp-deployment.md)

## Documentation Status

This documentation was reorganized on 2025-12-10 to:
- ✅ Eliminate redundancies
- ✅ Organize by category
- ✅ Update outdated information
- ✅ Improve discoverability

If you find outdated information or have suggestions, please update the relevant document or create an issue.

