# Dissertation Figure Mapping

This document maps dissertation figures (Chapter 3) to Mermaid diagram source files and generated outputs.

**Simplification (supervisor feedback)**: Diagrams were simplified so that every box in each figure is referenced in the dissertation text. Only the five layers (Figure 3.1), production-system components (Figure 3.3), and execution/data-collection elements (Figure 3.4) named in the prose are shown.

## Figure Mapping

### Figure 3.1: High-level Architecture of the Experimental Benchmarking Framework

**Description**: "High-level architecture of the benchmarking framework, showing the flow from configuration and workload generation through execution and measurement to analysis outputs." Five layers only (Configuration, Orchestration, Execution, Instrumentation, Analysis), as referenced in Section 3.1.4.

**Source**: `framework-architecture.mmd`

**Outputs**:
- `figures/framework-architecture.png` (2400x1800, high resolution)
- `figures/framework-architecture.svg` (vector format)

**Status**: ✅ Ready

**Dissertation Reference**: `[image1]`

---

### Figure 3.2: High-level Framework Architecture

**Description**: "End-to-end architecture at a high level." Maps input-event stream (configuration and orchestration), cryptographic operation (execution), and output evidence (instrumentation and analysis), as in Section 3.3.1.

**Source**: `system-architecture.mmd`

**Outputs**:
- `figures/system-architecture.png` (2400x1800, high resolution)
- `figures/system-architecture.svg` (vector format)

**Status**: ✅ Ready

**Dissertation Reference**: `[image-high-level]`

**Note**: This diagram shows all framework components, execution environments (Native, Minikube, GCP), core components (Pipeline, Workload, Crypto Adapters), orchestrator, and analysis layer with data flows.

---

### Figure 3.3: Framework Representation of Live Production System Components

**Description**: "Production-system components (ingestion, per-event processing, telemetry capture, and offline analysis)" as referenced in Section 3.3.1. Simplified to these four components only.

**Source**: `live-system-comparison.mmd`

**Outputs**:
- `figures/live-system-comparison.png` (2400x1800, high resolution)
- `figures/live-system-comparison.svg` (vector format)

**Status**: ✅ Ready

**Dissertation Reference**: `[image-live-system]`

---

### Figure 3.4: Detailed Research System Implementation

**Description**: "Detailed implementation view used for execution and data collection." Only elements referenced in Sections 3.3.3–3.3.5: workload generation (RNG seed), pipeline execution, cryptographic execution (operation boundary), instrumentation (timing, resource metrics), telemetry outputs (event-level data, run-level summaries).

**Source**: `detailed-implementation.mmd`

**Outputs**:
- `figures/detailed-implementation.png` (2400x1800, high resolution)
- `figures/detailed-implementation.svg` (vector format)

**Status**: ✅ Ready (simplified per supervisor feedback)

**Dissertation Reference**: `[image-detailed-research]`

**Key Features** (all referenced in text):
- Workload generation (deterministic RNG seed)
- Pipeline execution (streaming pipeline)
- Cryptographic execution (operation boundary, crypto adapter)
- Instrumentation (timing, resource metrics)
- Telemetry outputs (event-level data, run-level summaries)

---

### Figure EA1: Experimental evaluation pipeline (Extended abstract)

**Description**: Conceptual pipeline for the extended abstract: Ingress → Parse/Validate → Crypto stage (KEM/sign) → Downstream processing → Egress/Telemetry. Environments (Bare-metal | Local-K8s | Cloud-K8s) and metrics (latency p50/p95/p99, throughput, CPU/memory) are in the caption.

**Source**: `figure-ea1.mmd`

**Outputs**:
- `figures/figure-ea1.png` (2400x1800)
- `figures/figure-ea1.svg`

**Dissertation Reference**: `[imageEA1]` (Extended abstract section)

**Status**: ✅ Ready

---

## Additional Diagrams (Not Used in Dissertation)

### High-Level Overview (Research Methodology)

**Source**: `high-level-overview.mmd`

**Purpose**: Shows research methodology flow from literature analysis through experimental framework to outputs (performance metrics and engineering recommendations). This is about the research process, not framework architecture.

**Status**: Available but not used in dissertation figures

---

### Complete Workflow

**Source**: `complete-workflow.mmd`

**Purpose**: Complete workflow diagram showing the full pipeline from experiment matrix through data collection, processing, analysis, to final reporting.

**Status**: Available but not used in dissertation figures

---

## Conversion

All diagrams are converted using `convert_diagrams.sh` which:
1. Converts each `.mmd` file to SVG (vector format)
2. Converts each `.mmd` file to PNG (2400x1800, high resolution for Word documents)

**Command**: `cd diagrams && bash convert_diagrams.sh`

**Requirements**: `npm install -g @mermaid-js/mermaid-cli`

---

## Verification Checklist

- [x] Figure 3.1: `framework-architecture.mmd` exists and is converted
- [x] Figure 3.2: `system-architecture.mmd` exists and is converted
- [x] Figure 3.3: `live-system-comparison.mmd` exists and is converted
- [x] Figure 3.4: `detailed-implementation.mmd` created and converted
- [x] All diagrams converted to PNG and SVG formats
- [x] All diagrams use consistent color scheme (see `README.md`)
- [x] All diagrams have proper styling (3px borders, rounded corners)

---

## Next Steps

1. Verify that dissertation image references match the generated files
2. Update dissertation image references if needed:
   - `[image1]` → `framework-architecture.png`
   - `[image-high-level]` → `system-architecture.png`
   - `[image-live-system]` → `live-system-comparison.png`
   - `[image-detailed-research]` → `detailed-implementation.png`
3. Ensure all figures are properly embedded in the dissertation document

