# Low-Level Documentation Index

**Purpose**: This document lists all documentation files that provide detailed, low-level understanding of the codebase. Use this as a guide when you need to understand the system at a deep technical level.

---

## Primary Comprehensive Guide

### ⭐ [Complete System Guide](COMPLETE_SYSTEM_GUIDE.md)

**1,381 lines** of comprehensive, end-to-end documentation covering:

- **Code Organization and Structure**: Complete repository structure, module organization, workspace setup
- **Development Practices**: Code style, testing strategy, development workflow, adding new components
- **Core Components Deep Dive**: Detailed explanation of scenario loading, workload generation, cryptographic adapters, pipeline execution, telemetry collection
- **Running Benchmarks**: Setup and execution for native, Minikube, and GCP environments
- **Data Capture and Telemetry**: Instrumentation points, telemetry collection flow, precision implementation, data storage structure
- **Data Analysis Pipeline**: Complete analysis workflow, key analysis scripts, visualization generation
- **Report and Graph Generation**: Report generation process, graph types, generation process
- **Complete End-to-End Workflow**: Step-by-step from code to report
- **Troubleshooting and Debugging**: Common issues, debugging tools, logging

**Use this when**: You need to understand the entire system from code to execution to analysis.

---

## Component-Level Documentation

### [Component Documentation](reference/component-documentation.md)

**Detailed documentation of all major components**:

- **Rust Core**: Entry points, execution flow, scenario loading, workload generation, cryptographic adapters, pipeline execution, telemetry collection
- **Orchestrator**: REST API, experiment lifecycle, worker coordination, Kubernetes integration, result aggregation, storage
- **Analysis Suite**: Scripts, notebooks, statistical analysis, visualization
- **Orchestration System**: Experiment matrix, scenario generation, execution scripts
- **Scenario Format**: YAML structure, supported adapters, supported operations, execution modes

**Use this when**: You need detailed information about specific components.

---

## Codebase Structure

### [Codebase Inventory](CODEBASE_INVENTORY.md)

**Comprehensive inventory of all files and directories**:

- Complete directory structure
- File descriptions and purposes
- Documentation status for each component
- Component categorization

**Use this when**: You need to find specific files or understand the overall codebase structure.

---

## Development and Code Organization

### [Development Guidelines](../DEVELOPMENT_GUIDELINES.md)

**Development practices and guidelines**:

- TODO item requirements
- Testing requirements
- Requirements compliance
- Change safety practices
- Development workflow

**Use this when**: You're developing new features or making changes to the codebase.

---

## Execution and Data Collection

### [Data Collection Guide](guides/data-collection.md)

**Complete guide for data collection**:

- Native execution
- Minikube execution
- GCP execution
- Data validation
- Complete workflow

**Use this when**: You need to run experiments and collect data.

### [Unified Benchmark Flow](guides/unified-benchmark-flow.md)

**Single unified flow for smoke-test and full-scale benchmarks**:

- Unified flow principles
- Execution modes
- What changes with `--smoke-test` flag
- Directory structure

**Use this when**: You need to understand how benchmarks are executed.

---

## Data Analysis

### [Analysis Workflow](analysis/workflow.md)

**Complete guide to analyzing collected experiment data**:

- Prerequisites
- Step-by-step analysis workflow
- Key analysis scripts
- Visualization generation

**Use this when**: You need to analyze collected data.

### [Analysis Pipeline](analysis/analysis-pipeline.md)

**Detailed analysis pipeline documentation**:

- Pipeline stages
- Script descriptions
- Data flow
- Output formats

**Use this when**: You need detailed information about the analysis pipeline.

---

## Technical Reference

### [Precision Implementation](reference/precision-implementation.md)

**Sub-microsecond latency measurement implementation**:

- Problem statement
- Solution: Nanosecond precision
- Implementation details
- Analysis script updates

**Use this when**: You need to understand how timing measurements work.

### [System Requirements](reference/system-requirements.md)

**System load and variability handling**:

- System requirements
- Load considerations
- Variability handling

**Use this when**: You need to understand system requirements and constraints.

---

## Infrastructure and Deployment

### [GCP Deployment](reference/gcp-deployment.md)

**Complete GCP/GKE deployment guide**:

- Prerequisites
- Terraform setup
- Deployment process
- Troubleshooting

**Use this when**: You need to deploy to GCP.

### [Scaling Experiments](reference/scaling-experiments.md)

**Horizontal scaling experiments guide**:

- Scaling configuration
- Execution
- Analysis

**Use this when**: You need to run scaling experiments.

---

## Requirements and Specifications

### [Requirements Specification](REQUIREMENTS_SPECIFICATION.md)

**Single source of truth for research requirements**:

- Functional requirements
- Non-functional requirements
- Requirements traceability

**Use this when**: You need to understand what the system must do.

---

## Quick Reference by Task

### "I want to understand the entire codebase"
→ **[Complete System Guide](COMPLETE_SYSTEM_GUIDE.md)** ⭐

### "I want to understand a specific component"
→ **[Component Documentation](reference/component-documentation.md)**

### "I want to find a specific file"
→ **[Codebase Inventory](CODEBASE_INVENTORY.md)**

### "I want to develop new features"
→ **[Development Guidelines](../DEVELOPMENT_GUIDELINES.md)**

### "I want to run experiments"
→ **[Data Collection Guide](guides/data-collection.md)**

### "I want to analyze data"
→ **[Analysis Workflow](analysis/workflow.md)**

### "I want to understand how timing works"
→ **[Precision Implementation](reference/precision-implementation.md)**

### "I want to deploy to GCP"
→ **[GCP Deployment](reference/gcp-deployment.md)**

---

## Documentation Hierarchy

For a complete understanding, read in this order:

1. **[Complete System Guide](COMPLETE_SYSTEM_GUIDE.md)** - Start here for comprehensive understanding
2. **[Component Documentation](reference/component-documentation.md)** - Deep dive into specific components
3. **[Codebase Inventory](CODEBASE_INVENTORY.md)** - Understand file structure
4. **[Requirements Specification](REQUIREMENTS_SPECIFICATION.md)** - Understand requirements
5. **[Data Collection Guide](guides/data-collection.md)** - Understand execution
6. **[Analysis Workflow](analysis/workflow.md)** - Understand analysis

---

**Last Updated**: 2025-12-15
