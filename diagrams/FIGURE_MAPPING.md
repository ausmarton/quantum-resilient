# Figure Mapping

This document maps diagram source files (`.mmd`) to generated outputs (PNG/SVG) for reproducibility.

## Figure Mapping

### Figure: High-level Architecture of the Experimental Benchmarking Framework

**Description**: High-level architecture of the benchmarking framework: Configuration, Orchestration, Execution, Instrumentation, Analysis (five layers).

**Source**: `framework-architecture.mmd`

**Outputs**:
- `figures/framework-architecture.png` (2400x1800, high resolution)
- `figures/framework-architecture.svg` (vector format)

**Status**: ✅ Ready

---

### Figure: High-level Framework Architecture

**Description**: End-to-end architecture: input-event stream (configuration and orchestration), cryptographic operation (execution), output evidence (instrumentation and analysis). Includes framework components, execution environments (Native, Minikube, GCP), core components (Pipeline, Workload, Crypto Adapters), orchestrator, and analysis layer.

**Source**: `system-architecture.mmd`

**Outputs**:
- `figures/system-architecture.png` (2400x1800, high resolution)
- `figures/system-architecture.svg` (vector format)

**Status**: ✅ Ready

---

### Figure: Framework Representation of Live Production System Components

**Description**: Production-system components (ingestion, per-event processing, telemetry capture, offline analysis).

**Source**: `live-system-comparison.mmd`

**Outputs**:
- `figures/live-system-comparison.png` (2400x1800, high resolution)
- `figures/live-system-comparison.svg` (vector format)

**Status**: ✅ Ready

---

### Figure: Detailed Research System Implementation

**Description**: Detailed implementation view: workload generation (RNG seed), pipeline execution, cryptographic execution (operation boundary), instrumentation (timing, resource metrics), telemetry outputs (event-level data, run-level summaries).

**Source**: `detailed-implementation.mmd`

**Outputs**:
- `figures/detailed-implementation.png` (2400x1800, high resolution)
- `figures/detailed-implementation.svg` (vector format)

**Status**: ✅ Ready

---

### Figure: Experimental evaluation pipeline

**Description**: Conceptual pipeline: Ingress → Parse/Validate → Crypto stage (KEM/sign) → Downstream processing → Egress/Telemetry. Environments (Bare-metal | Local-K8s | Cloud-K8s) and metrics (latency p50/p95/p99, throughput, CPU/memory).

**Source**: `figure-ea1.mmd`

**Outputs**:
- `figures/figure-ea1.png` (2400x1800)
- `figures/figure-ea1.svg`

**Status**: ✅ Ready

---

## Additional Diagrams

### High-Level Overview (Research Methodology)

**Source**: `high-level-overview.mmd`

**Purpose**: Research methodology flow from literature analysis through experimental framework to outputs (performance metrics and engineering recommendations).

**Status**: Available

---

### Complete Workflow

**Source**: `complete-workflow.mmd`

**Purpose**: Full pipeline from experiment matrix through data collection, processing, analysis, to final reporting.

**Status**: Available

---

## Conversion

All diagrams are converted using `convert_diagrams.sh` which:
1. Converts each `.mmd` file to SVG (vector format)
2. Converts each `.mmd` file to PNG (2400x1800, high resolution)

**Command**: `cd diagrams && bash convert_diagrams.sh`

**Requirements**: `npm install -g @mermaid-js/mermaid-cli`

---

## Verification Checklist

- [x] Figure: `framework-architecture.mmd` exists and is converted
- [x] Figure: `system-architecture.mmd` exists and is converted
- [x] Figure: `live-system-comparison.mmd` exists and is converted
- [x] Figure: `detailed-implementation.mmd` exists and is converted
- [x] Figure: `figure-ea1.mmd` exists and is converted
- [x] All diagrams converted to PNG and SVG formats
- [x] All diagrams use consistent color scheme (see `README.md`)
- [x] All diagrams have proper styling (3px borders, rounded corners)
