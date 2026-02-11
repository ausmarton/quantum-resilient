# Figure and Table Placement Audit (Supervisor Feedback)

## Objectives
1. **Diagrams**: Remove boxes not referenced in text; keep figures simple; every box must be discussed.
2. **Placement**: Keep figures/tables close to first reference; avoid scrolling to find them.

## Diagram–Text Alignment (Figures 3.1–3.4)

### Figure 3.1 (framework-architecture.mmd)
**Text references (Section 3.1.4, 3.1.2)**: "five layers", "Configuration layer", "Orchestration layer", "Execution layer", "Instrumentation layer", "Analysis layer", "flow from configuration and workload generation through execution and measurement to analysis outputs".

**Current diagram**: Many nodes (Experiment Matrix, Scenario Generator, RNG Seed, Bare-metal, Kubernetes, Cloud, Rust Orchestrator, Python Orchestration, Data Aggregator, Statistical Analysis, Pipeline, Execution Modes, Workload Generator, Payload Generation, Crypto Adapters, Telemetry Collection, Control Plane, JSONL, CSV, Metadata, Hypothesis Testing, Effect Size, Statistics Computation, Visualization Scripts, Jupyter Notebooks, Export).

**Simplification**: Use five boxes only—Configuration Layer, Orchestration Layer, Execution Layer, Instrumentation Layer, Analysis Layer—with arrows showing flow. Remove sub-boxes that are not named in the prose.

### Figure 3.2 (system-architecture.mmd)
**Text**: "end-to-end architecture at a high level" (no specific boxes named).

**Simplification**: Keep high-level only: configuration → orchestration → execution environments → analysis. Reduce internal nodes to those that appear in Section 3.3.1 (e.g. "configuration and orchestration layers", "execution layer", "instrumentation and analysis layers").

### Figure 3.3 (live-system-comparison.mmd)
**Text**: "production-system components (ingestion, per-event processing, telemetry capture, and offline analysis)".

**Current diagram**: Representative Enterprise AML Pipeline (Transaction Ingestion, Streaming Pipeline, Production Cryptography, ML/AI Models, Alert Generation, Compliant Output, Production Monitoring); Framework (Workload Generator, Streaming Pipeline, Crypto Adapters, Instrumentation subgraph: Timing, Resources, Events; Telemetry Collection); Telemetry Outputs (JSONL, Statistical Summaries, Metadata).

**Simplification**: Show only: ingestion, per-event processing, telemetry capture, offline analysis (and optionally framework mirror). Remove boxes not named in text (e.g. ML/AI Models, Alert Generation, Production Monitoring, Instrumentation sub-boxes unless referenced).

### Figure 3.4 (detailed-implementation.mmd)
**Text**: "detailed implementation view used for execution and data collection".

**Referenced in 3.3.3–3.3.5**: workload generation (deterministic, RNG seed), pipeline execution, cryptographic operation boundary, instrumentation (timing, resource), telemetry outputs (event-level, run-level), analysis (distributional summaries). Not every sub-box (e.g. Health Endpoint, Readiness Endpoint, Queue Delay) is named.

**Simplification**: Keep: Workload Generation, Pipeline Execution, Cryptographic Execution, Instrumentation (or timing/resource only), Telemetry Collection, Telemetry Outputs. Remove or merge Control Plane and fine-grained instrumentation labels if not referenced.

## First Reference vs Current Position

| Item       | First reference (line) | Current position (line) | Action |
|-----------|------------------------|-------------------------|--------|
| Table 2.1 | 298                    | 307                     | OK     |
| Figure 3.1| 385                    | 387–389                 | OK     |
| Table 3.1 | 445                    | 447                     | OK     |
| Figure 3.2| 509                    | 511–513                 | OK     |
| Figure 3.3| 515                    | 517–519                 | OK     |
| Figure 3.4| 521                    | 523–525                 | OK     |
| Table 3.2 | 553                    | 555                     | OK     |
| Table 4.1 | 633                    | 641                     | OK     |
| Table 4.2 | 664                    | 666                     | OK     |
| Figure 4.1| 685                    | 701–703                 | OK (same section) |
| Figure 4.1a| 697                   | 705–707                 | OK     |
| Table 4.7 | 717 (caption only)     | 717                     | OK     |
| Table 4.4 | 734                    | 742                     | OK     |
| **Table 4.4a** | **766**           | **754**                 | **Move table block to after first reference (after Fig 4.2 caption)** |
| Figure 4.2| 766                    | 766–768                 | OK     |
| Figure 4.3| 776                    | 799                     | Move block up: after paragraph ending "...Table 4.5." |
| Figure 4.5a| 776                   | 778–780                 | OK (keep after 4.3 when reordering) |
| Table 4.5 | 776                    | 782                     | OK (keep order: 4.3, 4.5a, 4.5) |
| Figure 4.4| 795                    | 803                     | Move block up: after "Workload Rate Impact" paragraph |
| Figure 4.5| 795                    | 807                     | Move block up: after Figure 4.4 |
| Table 4.6 | 821                    | 821                     | OK     |

## Placement Edits Applied
1. **Table 4.4a**: Moved from before its first reference to after Figure 4.2 caption (so it appears after the paragraph that says "Table 4.4a indicates...").
2. **Figure 4.3**: Moved image10 + caption to immediately after the "Payload Size Impact" paragraph (ending "...presented in Table 4.5."). Order is now: paragraph → Figure 4.3 → Figure 4.5a → Table 4.5.
3. **Figures 4.4 and 4.5**: Already immediately after the "Workload Rate Impact" paragraph; no change.

## Diagram Simplification Applied
- **Figure 3.1** (`framework-architecture.mmd`): Replaced with five-layer flow only (Configuration, Orchestration, Execution, Instrumentation, Analysis), matching Section 3.1.4.
- **Figure 3.2** (`system-architecture.mmd`): Replaced with high-level flow (input-event stream → cryptographic operation → output evidence), matching Section 3.3.1.
- **Figure 3.3** (`live-system-comparison.mmd`): Replaced with four components only (ingestion, per-event processing, telemetry capture, offline analysis), matching Section 3.3.1.
- **Figure 3.4** (`detailed-implementation.mmd`): Replaced with elements referenced in 3.3.3–3.3.5 only (workload generation/RNG seed, pipeline execution, cryptographic execution, instrumentation, telemetry outputs).

**Regenerate figures**: Run `cd diagrams && bash convert_diagrams.sh` to produce updated PNG/SVG from the simplified .mmd files. Re-embed or re-export figures in the dissertation document as needed.
