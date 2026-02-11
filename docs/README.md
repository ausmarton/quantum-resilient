# Documentation Index

This directory contains all documentation for the Quantum-Resilient Cryptography Benchmark Framework, organized by category.

## Quick Links

- **[Complete System Guide](COMPLETE_SYSTEM_GUIDE.md)** - ⭐ **COMPREHENSIVE END-TO-END GUIDE** - Complete low-level documentation covering code organization, development, execution, data capture, analysis, and reporting
- **[Getting Started](guides/getting-started.md)** - Quick start guide (coming soon)
- **[Running Experiments](guides/running-experiments.md)** - How to run experiments (coming soon)
- **[Data Collection](guides/data-collection.md)** - Full-scale data collection guide
- **[Storage and Output](guides/storage-and-output.md)** - Where results are stored

## Documentation Structure

### 📚 Guides (`guides/`)

User-facing guides for running experiments and using the framework:

- **[Unified Benchmark Flow](guides/unified-benchmark-flow.md)** - **Single unified flow for smoke-test and full-scale benchmarks**
- **[Data Collection](guides/data-collection.md)** - Complete guide for full-scale data collection
- **[Storage and Output](guides/storage-and-output.md)** - Directory structure and overwrite behavior
- **[Stop and Resume](guides/stop-and-resume.md)** - Safely interrupting and resuming experiments
- **[Re-running Experiments](guides/re-running-experiments.md)** - How to delete and re-run experiments
- **[Researcher Guide](guides/researcher-guide.md)** - Comprehensive guide for researchers
- **[Horizontal Scaling Guide](guides/horizontal-scaling-guide.md)** - Horizontal scaling experiments guide
- **[Parallel Execution](guides/parallel-execution.md)** - Running experiments in parallel
- **[Containerization](guides/containerization.md)** - Using containerized analysis pipeline (Podman/Docker)

### 📖 Reference (`reference/`)

Technical reference documentation:

- **[Complete System Guide](COMPLETE_SYSTEM_GUIDE.md)** - ⭐ **COMPREHENSIVE END-TO-END GUIDE** - Complete low-level documentation covering everything from code organization to execution, data capture, analysis, and reporting (1,381 lines)
- **[Requirements Specification](REQUIREMENTS_SPECIFICATION.md)** - **Single source of truth** for dissertation requirements and codebase capabilities
- **[Component Documentation](reference/component-documentation.md)** - **NEW** - Detailed documentation of all major components (rust-core, orchestrator, analysis suite, etc.)
- **[Codebase Inventory](CODEBASE_INVENTORY.md)** - **NEW** - Comprehensive inventory of all files and directories
- **[Algorithm Naming Standard](reference/algorithm-naming-standard.md)** - **NEW** - Standardized algorithm names for dissertation
- **[System Requirements](reference/system-requirements.md)** - System load and variability handling
- **[GCP Deployment](reference/gcp-deployment.md)** - Complete GCP/GKE deployment guide
- **[Scaling Experiments](reference/scaling-experiments.md)** - Horizontal scaling experiments guide
- **[Data Validation](reference/data-validation.md)** - Data quality validation and status
- **[Precision Implementation](reference/precision-implementation.md)** - Sub-microsecond latency measurement implementation
- **[Option 2 Precision](reference/option2-precision.md)** - Alternative floating-point microseconds approach
- **[Test Coverage](reference/test-coverage.md)** - Test coverage, gaps, and strategy

### 🔬 Analysis (`analysis/`)

Research analysis and experimental design documents:

- **[Detailed Methodology](dissertation-methodology-detailed.md)** - **NEW** - Comprehensive methodology documentation for dissertation Chapter 3
- **[Detailed Analysis](dissertation-analysis-detailed.md)** - **NEW** - Comprehensive analysis documentation for dissertation Chapter 4
- **[Experimental Design](analysis/experimental-design.md)** - Experimental design analysis
- **[Hardware Consistency](analysis/hardware-consistency.md)** - Cross-environment hardware analysis
- **[Cost Analysis](analysis/cost-analysis.md)** - Cost and time analysis
- **[Enterprise Representativeness](analysis/enterprise-representativeness.md)** - Enterprise deployment analysis
- **[Telemetry Assessment](analysis/telemetry-assessment.md)** - Comprehensive telemetry assessment for dissertation objectives
- **[Workflow](analysis/workflow.md)** - Analysis workflow guide
- **[Dissertation Guide](analysis/dissertation-guide.md)** - Guide for dissertation analysis
- **[Cluster Sizing Analysis](analysis/cluster-sizing-analysis.md)** - Cluster sizing analysis
- **[Horizontal Scaling Analysis](analysis/horizontal-scaling-analysis.md)** - Horizontal scaling analysis
- **[GCP Optimization](analysis/gcp-optimization.md)** - GCP optimization analysis
- **[ECDHE Reference](analysis/ecdhe-reference.md)** - ECDHE P-256 implementation reference
- **[Comparison Issue Assessment](analysis/comparison-issue-assessment.md)** - Apples-to-apples comparison analysis

### 🔧 Troubleshooting (`troubleshooting/`)

Historical fixes and troubleshooting guides:

- **[Git Push Fix](troubleshooting/git-push-fix.md)** - Resolving Git push failures with large files
- **[Scaling Fix](troubleshooting/scaling-fix.md)** - Fix for scaling experiment execution
- **[GKE Node Pool Troubleshooting](troubleshooting/gke-node-pool.md)** - Troubleshooting GKE node pool creation errors

## Main Documentation

The main project README is located at the repository root:
- **[README.md](../README.md)** - Project overview and quick start

## Additional Documentation

- **[Development Guidelines](../DEVELOPMENT_GUIDELINES.md)** - Development guidelines and practices (also available at root)
- **[Writing Guidelines](writing-guidelines.md)** - Writing guidelines for dissertation
- **[Dissertation Readiness Checklist](dissertation-readiness-checklist.md)** - Checklist for dissertation readiness

## Finding Documentation

### By Task

- **"I want to understand the entire codebase at a low level"** → [Complete System Guide](COMPLETE_SYSTEM_GUIDE.md) ⭐ **COMPREHENSIVE GUIDE**
- **"What are the requirements for the dissertation?"** → [Requirements Specification](REQUIREMENTS_SPECIFICATION.md) ⭐ **START HERE**
- **"How do I run experiments?"** → [Data Collection Guide](guides/data-collection.md)
- **"Where are my results?"** → [Storage and Output Guide](guides/storage-and-output.md)
- **"How do I deploy to GCP?"** → [GCP Deployment Guide](reference/gcp-deployment.md)
- **"How do scaling experiments work?"** → [Scaling Experiments Guide](reference/scaling-experiments.md)
- **"What about system load?"** → [System Requirements](reference/system-requirements.md)
- **"How do I validate my data?"** → [Data Validation](reference/data-validation.md)
- **"How do I use the containerized analysis pipeline?"** → [Containerization Guide](guides/containerization.md)
- **"How do I download results from GCP?"** → [Data Collection Guide](guides/data-collection.md) - Retrieving Results from GCP section
- **"GKE node pool creation failed?"** → [GKE Node Pool Troubleshooting](troubleshooting/gke-node-pool.md)

### By Environment

- **Native (local)**: See [Data Collection Guide](guides/data-collection.md) - Native section
- **Minikube**: See [Data Collection Guide](guides/data-collection.md) - Minikube section
- **GCP**: See [GCP Deployment Guide](reference/gcp-deployment.md)

## Documentation Status

This documentation was consolidated and reorganized on 2025-12-10 to:
- ✅ Eliminate redundancies (consolidated 38+ root-level files)
- ✅ Organize by category (guides, reference, analysis, troubleshooting)
- ✅ Update outdated information (removed completed TODOs, archived outdated docs)
- ✅ Improve discoverability (clear structure, updated index)
- ✅ Centralize action items (all in `TODO.md`)

### Consolidated Documents

**Telemetry Documents** (5 → 1):
- Consolidated into `analysis/telemetry-assessment.md`

**Data Quality Documents** (6 → 1):
- Consolidated into `reference/data-validation.md`

**Implementation Documents** (3 → 1):
- Consolidated into `reference/precision-implementation.md`
- Option 2 preserved as `reference/option2-precision.md`

**Archived Documents**:
- See `archive/documentation-2025-12-10/` for archived files (2025-12-10 consolidation)
- See `archive/documentation-2025-01-27/` for archived files (2025-01-27 cleanup)
- See `ARCHIVE.md` for archive index

### Active Work Tracking

All outstanding work items are tracked in:
- **[TODO.md](../TODO.md)** - Single source of truth for all action items

If you find outdated information or have suggestions, please update the relevant document or create an issue.

## Documentation Audit Status

✅ **Documentation Audit: COMPLETE** (2025-12-15)

All documentation has been audited, consolidated, and verified:
- ✅ Complete codebase inventory created
- ✅ All components documented and verified
- ✅ Workflow and architecture diagrams created
- ✅ Dissertation methodology and analysis documentation ready
- ✅ All references updated and verified

See **[VERIFICATION_REPORT.md](VERIFICATION_REPORT.md)**, **[VALIDATION_REPORT.md](VALIDATION_REPORT.md)**, **[COMPREHENSIVE_VALIDATION.md](COMPREHENSIVE_VALIDATION.md)**, and **[FINAL_AUDIT_SUMMARY.md](FINAL_AUDIT_SUMMARY.md)** for details.

