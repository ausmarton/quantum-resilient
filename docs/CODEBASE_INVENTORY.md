# Codebase Inventory

**Last Updated**: 2025-12-15  
**Purpose**: Comprehensive inventory of all files and directories in the quantum-resilient codebase with descriptions and documentation status.

---

## Directory Structure Overview

```
quantum-resilient/
├── rust-core/              # Core Rust library and benchmark binary
├── orchestrator/          # Distributed experiment orchestrator (Rust)
├── analysis/              # Python analysis and visualization suite
├── orchestration/          # Experiment orchestration and scenario generation
├── scripts/               # Shell and Python utility scripts
├── scenarios/             # Benchmark scenario YAML definitions
├── k8s/                   # Kubernetes manifests
├── helm/                  # Helm charts for deployment
├── iac/                   # Infrastructure as Code (Terraform)
├── research/              # Research artifact generation
├── packaging/             # Experiment packaging and distribution
├── reproducibility/       # Reproducibility test suite
├── docs/                  # Consolidated documentation
├── diagrams/              # Mermaid diagrams and visualizations
└── archive/               # Archived documentation and old files
```

---

## Core Components

### `rust-core/` - Core Benchmark Framework

**Purpose**: Core Rust library providing cryptographic benchmarking functionality.

**Files**:
- `Cargo.toml` - Rust package configuration
- `src/lib.rs` - Library entry point, re-exports all public APIs
- `src/main.rs` - Binary entry point (`pqc-bench`), handles CLI args, orchestrator registration, pipeline execution
- `src/scenario.rs` - Scenario loading from YAML, validation
- `src/workload.rs` - Workload generation (constant, burst, ramp, trace patterns)

**Subdirectories**:
- `src/crypto_adapter/` - Cryptographic algorithm adapters
  - `mod.rs` - Adapter trait definitions and registry
  - `noop_adapter.rs` - NoOp baseline adapter
  - `rsa_adapter.rs` - RSA-2048 classical cryptography
  - `ecdsa_adapter.rs` - ECDSA P-256 classical cryptography
  - `ecdh_adapter.rs` - ECDH P-256 key exchange
  - `kyber_adapter.rs` - Kyber-512 PQC KEM
  - `dilithium_adapter.rs` - Dilithium PQC signature
  - `kem_hybrid.rs` - Hybrid KEM→AEAD encryption helpers
  - `registry.rs` - Adapter factory and registration
- `src/pipeline/` - Async streaming pipeline
  - `mod.rs` - Pipeline types and interfaces
  - `execution.rs` - Execution models (single, fixed_pool, elastic)
  - `workload.rs` - Workload model integration
- `src/telemetry/` - Metrics and logging
  - `mod.rs` - Telemetry module exports
  - `metrics.rs` - Prometheus metrics collection
  - `jsonl_logger.rs` - JSONL event logging
  - `sysinfo_sampler.rs` - System resource sampling (CPU, memory)
  - `tracing_setup.rs` - Structured logging setup
- `src/controlplane/` - Kubernetes control plane endpoints
  - `mod.rs` - Control plane module exports
  - `http.rs` - HTTP handlers (/healthz, /readyz, /workers, /shutdown)

**Tests**:
- `tests/pqc_adapter_smoke.rs` - Smoke tests for all crypto adapters

**Documentation Status**: ⚠️ **NEEDS AUDIT** - Verify README.md matches implementation

---

### `orchestrator/` - Distributed Experiment Orchestrator

**Purpose**: Manages distributed multi-pod benchmark experiments across Kubernetes clusters.

**Files**:
- `Cargo.toml` - Rust package configuration
- `Dockerfile` - Container image for orchestrator
- `src/main.rs` - Orchestrator entry point, HTTP API server setup
- `src/api.rs` - REST API endpoints (experiment CRUD, status, control)
- `src/controller.rs` - Experiment lifecycle management
- `src/coordinator.rs` - Worker coordination and synchronization
- `src/k8s_client.rs` - Kubernetes API integration
- `src/aggregator.rs` - Result aggregation from multiple workers
- `src/storage.rs` - Object storage integration (S3/GCS)
- `src/scheduler.rs` - Experiment scheduling (cron-based)

**Documentation Status**: ⚠️ **NEEDS AUDIT** - Verify documentation matches implementation

---

### `analysis/` - Analysis and Visualization Suite

**Purpose**: Python-based analysis tools for processing benchmark results and generating publication-quality figures.

**Files**:
- `README.md` - Analysis suite documentation
- `requirements.txt` - Python dependencies
- `pyproject.toml` - Python project configuration
- `Dockerfile` - Containerized analysis environment
- `Dockerfile.jupyter` - JupyterLab environment
- `run_full_pipeline.sh` - Complete analysis pipeline script

**Scripts** (`scripts/` subdirectory):
- `aggregate_results.py` - Aggregate statistics across experiments
- `aggregate_runs.py` - Aggregate multiple runs of same experiment
- `build_final_report.py` - Generate dissertation-ready reports
- `check_hardware_compatibility.py` - Hardware consistency validation
- `compare_all_environments.py` - Cross-environment comparison
- `compare_native_vs_minikube.py` - Native vs Minikube comparison
- `hypothesis_tests.py` - Statistical hypothesis testing
- `plot_combined_cdfs.py` - Combined CDF plots
- `plot_effect_size_forest.py` - Effect size visualization
- `plot_payload_scaling_loglog.py` - Payload scaling analysis
- `plot_pqc_vs_classical_distribution.py` - PQC vs classical comparison
- `plot_replica_scaling.py` - Replica scaling analysis
- `plot_scaling_curves.py` - Scaling curve visualization

**Notebooks** (`notebooks/` subdirectory):
- Jupyter notebooks for interactive analysis (9 files)

**Documentation Status**: ⚠️ **NEEDS AUDIT** - Verify README.md and script documentation

---

### `orchestration/` - Experiment Orchestration

**Purpose**: Declarative experiment configuration and scenario generation.

**Files**:
- `experiment_matrix.yaml` - Declarative experiment matrix definition
- `generate_scenarios.py` - Scenario auto-generator from matrix

**Documentation Status**: ⚠️ **NEEDS AUDIT**

---

### `scripts/` - Utility Scripts

**Purpose**: Shell and Python scripts for automation, validation, and data management.

**Categories**:

**Data Collection**:
- `run_experiment.sh` - Run single experiment
- `run_all_experiments.sh` - Master orchestration script
- `run_full_scale_data_collection.sh` - Full-scale data collection
- `check_progress.sh` - Check experiment progress
- `validate_data_collection.sh` - Validate data collection completeness

**GCP Deployment**:
- `deploy_gcp.sh` - GCP deployment automation
- `fetch_and_analyse_from_gcs.sh` - Fetch and analyze GCS results
- `submit_gcp_job_parallel.sh` - Parallel GCP job submission
- `list_gcp_experiments.sh` - List GCP experiments
- `download_all_gcp_results.sh` - Download all GCP results
- `fetch_all_gcp_results.sh` - Fetch all GCP results
- `cleanup_gcp_resources.sh` - Cleanup GCP resources

**Analysis**:
- `run_complete_analysis.sh` - Complete analysis pipeline
- `run_full_analysis_pipeline.sh` - Full analysis pipeline
- `generate_experiment_summaries.sh` - Generate experiment summaries
- `generate_all_summaries.sh` - Generate all summaries
- `generate_missing_summaries.sh` - Generate missing summaries
- `regenerate_index_from_results.sh` - Regenerate experiment index

**Validation**:
- `validate_data_integrity.sh` - Validate data integrity
- `validate_data_quality.sh` - Validate data quality
- `validate_experiment_data.sh` - Validate experiment data
- `validate_experiment_suite.sh` - Validate experiment suite
- `validate_dissertation_data.sh` - Validate dissertation data
- `validate_gcp_downloads.sh` - Validate GCP downloads
- `validate_summaries.sh` - Validate summaries
- `verify_experiments.sh` - Verify experiments
- `verify_chapter3_structure.sh` - Verify Chapter 3 structure
- `verify_chapter4_consistency.py` - Verify Chapter 4 consistency
- `verify_chapter4_data.py` - Verify Chapter 4 data
- `verify_dissertation_readiness.py` - Verify dissertation readiness
- `verify_environment_comparison.py` - Verify environment comparison
- `verify_requirements_compliance.py` - Verify requirements compliance

**Analysis Scripts** (Python):
- `analyze_per_environment_rankings.py` - Environment ranking analysis
- `analyze_queue_delay.py` - Queue delay analysis
- `analyze_resource_utilization.py` - Resource utilization analysis
- `analyze_scaling.py` - Scaling analysis
- `check_data_sufficiency.py` - Data sufficiency check
- `compute_cost_efficiency.py` - Cost efficiency computation
- `extract_dissertation_tables.py` - Extract dissertation tables
- `generate_analysis_report.py` - Generate analysis report
- `generate_interpretation_docs.py` - Generate interpretation docs
- `verify_payload_workload_analysis.py` - Payload/workload analysis verification

**Utility Scripts**:
- `check_system_load.sh` - Check system load before experiments
- `capture_hardware_metadata.sh` - Capture hardware metadata
- `cleanup_experiment_artifacts.sh` - Cleanup experiment artifacts
- `cleanup_failed_data_collections.sh` - Cleanup failed collections
- `cleanup_processed_data.sh` - Cleanup processed data
- `cleanup_results.sh` - Cleanup results
- `complete_incomplete_experiments.sh` - Complete incomplete experiments
- `remove_unusable_data.sh` - Remove unusable data
- `start-jupyter.sh` - Start Jupyter server

**Library Scripts** (`lib/` subdirectory):
- `common.sh` - Common shell functions
- `directories.sh` - Directory management functions
- `analysis.sh` - Analysis helper functions
- `k8s-cluster.sh` - Kubernetes cluster management
- `k8s-configmap.sh` - Kubernetes ConfigMap management
- `k8s-image.sh` - Kubernetes image management
- `k8s-job.sh` - Kubernetes Job management
- `k8s-job-generator.py` - Kubernetes Job generator
- `manifest.sh` - Manifest generation
- `regenerate_index.py` - Index regeneration
- `run-python-container.sh` - Run Python in container
- `scenario-patch.py` - Scenario patching

**Documentation Status**: ⚠️ **NEEDS AUDIT** - Many scripts lack documentation

---

### `scenarios/` - Benchmark Scenarios

**Purpose**: YAML scenario definitions for benchmark experiments.

**Files**:
- `smoke_noop.yaml` - NoOp baseline smoke test
- `rsa_smoke.yaml` - RSA-2048 smoke test
- `ecdsa_smoke.yaml` - ECDSA P-256 smoke test
- `kyber_hybrid_encrypt.yaml` - Kyber hybrid encryption
- `kyber_hybrid_decrypt.yaml` - Kyber hybrid decryption
- `fixed_pool_burst.yaml` - Fixed pool burst workload
- `elastic_ramp.yaml` - Elastic ramp workload
- `hybrid_kyber_dilithium.yaml` - Full PQC hybrid benchmark

**Documentation Status**: ⚠️ **NEEDS AUDIT** - Verify scenario format documentation

---

### `k8s/` - Kubernetes Manifests

**Purpose**: Kubernetes resource definitions for deployment.

**Structure**:
- `base/` - Base Kubernetes manifests (deployments, services, configmaps, etc.)
- `gcp/` - GCP-specific manifests (if any)

**Documentation Status**: ⚠️ **NEEDS AUDIT**

---

### `helm/` - Helm Charts

**Purpose**: Helm charts for Kubernetes deployment.

**Charts**:
- `quantum-resilient/` - Main benchmark framework chart
- `quantum-resilient-orchestrator/` - Orchestrator chart

**Documentation Status**: ⚠️ **NEEDS AUDIT**

---

### `iac/` - Infrastructure as Code

**Purpose**: Terraform configurations for cloud infrastructure.

**Structure**:
- Terraform modules for GCP/GKE deployment
- Variable definitions
- Output definitions

**Documentation Status**: ⚠️ **NEEDS AUDIT**

---

### `research/` - Research Artifact Generation

**Purpose**: Tools for generating dissertation-ready research artifacts.

**Files**:
- `README.md` - Research artifact generation documentation
- `scripts/provenance.py` - Provenance metadata generation
- `scripts/version_dataset.py` - Dataset versioning
- `scripts/generate_tables.py` - LaTeX/Markdown table generation
- `scripts/generate_figures_bundle.py` - Figure bundle generation
- `scripts/generate_report.py` - Report generation
- `scripts/pipeline_runner.py` - Complete pipeline runner
- `templates/` - Jinja2 templates for reports

**Documentation Status**: ⚠️ **NEEDS AUDIT**

---

### `packaging/` - Experiment Packaging

**Purpose**: Tools for packaging and distributing experiment results.

**Files**:
- `README.md` - Packaging documentation
- `cli.py` - Typer CLI interface
- `manifest.py` - Manifest generation
- `archiver.py` - Archive creation (ZIP/TAR.GZ)
- `exporter.py` - Publication-ready exports
- `release_notes.py` - Release notes generation
- `publish.py` - Publishing to GCS/S3/GitHub
- `templates/` - Jinja2 templates

**Documentation Status**: ⚠️ **NEEDS AUDIT**

---

### `reproducibility/` - Reproducibility Test Suite

**Purpose**: Tools for validating experimental reproducibility.

**Files**:
- `README.md` - Reproducibility documentation
- `runner.py` - Multi-run execution
- `variance.py` - Variance analysis
- `confidence.py` - Confidence intervals
- `stability.py` - Distribution stability
- `regression.py` - Regression detection
- `cluster_scaling.py` - Cluster scaling analysis
- `templates/` - Report templates

**Documentation Status**: ⚠️ **NEEDS AUDIT**

---

### `docs/` - Consolidated Documentation

**Purpose**: All project documentation organized by category.

**Structure**:
- `README.md` - Documentation index
- `guides/` - User guides
- `reference/` - Technical reference
- `analysis/` - Research analysis documents
- `troubleshooting/` - Troubleshooting guides
- `REQUIREMENTS_SPECIFICATION.md` - Requirements specification
- `VERIFICATION_CHECKLIST.md` - Verification checklist
- `dissertation-methodology.md` - Dissertation methodology
- `dissertation-requirements.md` - Dissertation requirements

**Documentation Status**: ✅ **ORGANIZED** - Needs content audit

---

### `diagrams/` - Diagrams and Visualizations

**Purpose**: Mermaid diagrams and visualizations for documentation.

**Files**:
- Mermaid diagram files (`.mmd`)
- Markdown files with diagrams
- HTML exports

**Documentation Status**: ⚠️ **NEEDS AUDIT** - Verify diagrams match implementation

---

## Root-Level Files

### Configuration Files
- `Cargo.toml` - Rust workspace definition
- `Cargo.lock` - Rust dependency lock file
- `Makefile` - Build automation
- `Containerfile` - Multi-stage container build
- `Dockerfile.podman` - Podman-specific Dockerfile
- `docker-compose.yml` - Docker Compose configuration
- `podman-compose.yml` - Podman Compose configuration

### Documentation Files
- `README.md` - Main project README (comprehensive)
- `DEVELOPMENT_GUIDELINES.md` - Development guidelines
- `ALGORITHM_NAMING_STANDARD.md` - Algorithm naming standard
- `WRITING_GUIDELINES.md` - Writing guidelines
- `TODO.md` - Outstanding work items
- `ARCHIVE.md` - Archive index
- `DISSERTATION_READINESS_CHECKLIST.md` - Dissertation readiness checklist
- `DISSERTATION_READY.md` - Dissertation ready status
- `DISSERTATION_READY_ECDHE.md` - ECDHE-specific dissertation status
- `REDUNDANCY_ANALYSIS.md` - Redundancy analysis
- `REDUNDANCY_LOG.md` - Redundancy removal log
- `REDUNDANCY_REMOVAL_SUMMARY.md` - Redundancy removal summary

### Scripts
- `run_local.sh` - Local native experiment runner
- `run_minikube.sh` - Minikube Kubernetes experiment runner
- `deploy_gcp.sh` - GCP deployment script
- `fetch_and_analyse_from_gcs.sh` - GCS results fetcher
- `run_all_experiments.sh` - Master orchestration script
- `run_full_scale_data_collection.sh` - Full-scale data collection
- `validate_counting.sh` - Validation script

### Other Files
- `Chapter3-review` - Chapter 3 review notes
- `feedback-draft-1` - Feedback draft
- `FERNANDES_H2807295_F87_dissertation (2).md` - Dissertation draft

---

## Data Directories

### `data-collection-*/` - Data Collection Runs
**Purpose**: Timestamped directories containing raw experiment data from data collection runs.

**Contents**:
- `manifest.json` - Run metadata
- `summary.txt` - Run summary
- `*_run.log` - Execution logs
- `*_validation.log` - Validation logs

### `run-*/` - Experiment Run Directories
**Purpose**: Timestamped directories containing processed experiment results.

### `results/` - Results Directory
**Purpose**: Processed experiment results.

### `final-results/` - Final Results
**Purpose**: Aggregated final results for dissertation.

### `generated-scenarios/` - Generated Scenarios
**Purpose**: Auto-generated scenario files from experiment matrix.

### `generated-scenarios-mini/` - Mini Generated Scenarios
**Purpose**: Mini test scenarios.

### `generated-scenarios-test/` - Test Generated Scenarios
**Purpose**: Test scenarios.

### `figures/` - Generated Figures
**Purpose**: Publication-quality figures.

---

## Documentation Gaps Identified

1. **rust-core/**: Missing detailed component documentation
2. **orchestrator/**: Missing API documentation
3. **scripts/**: Many scripts lack individual documentation
4. **scenarios/**: Missing scenario format specification
5. **k8s/**: Missing deployment guide
6. **helm/**: Missing chart documentation
7. **iac/**: Missing infrastructure documentation
8. **Workflow diagrams**: Missing complete pipeline diagrams
9. **Architecture diagrams**: Missing system architecture diagrams

---

## Next Steps

1. ✅ Create comprehensive inventory (this document)
2. ⏳ Audit rust-core implementation and documentation
3. ⏳ Audit orchestrator implementation and documentation
4. ⏳ Audit analysis scripts and documentation
5. ⏳ Audit orchestration scripts and documentation
6. ⏳ Audit infrastructure (Terraform, Helm, K8s) and documentation
7. ⏳ Consolidate root-level documentation into docs/
8. ⏳ Create workflow diagrams
9. ⏳ Create architecture diagrams
10. ⏳ Update README.md to reflect current state
11. ⏳ Create methodology documentation for Chapter 3
12. ⏳ Create analysis documentation for Chapter 4

---

**Last Updated**: 2025-12-15

