Documentation
=============

This directory contains comprehensive documentation for the PQC Performance Benchmarking Framework.

## For Researchers Looking to Replicate/Validate

**Start here** (in order):

1. **[Quick Reference Diagram](quick_reference_diagram.md)** ⭐ 
   - Single-page overview with architecture summary
   - Key components, data flow, metrics captured
   - Quick commands and sample output structure
   - **Read this first for a 5-minute overview**

2. **[Implementation Guide](implementation_guide.md)** 📘
   - Step-by-step replication instructions
   - Detailed code examples with explanations
   - Validation checklist
   - Troubleshooting guide
   - **Read this to replicate the framework**

3. **[Framework Diagrams](framework_diagrams.md)** 🎨
   - 7 comprehensive diagrams covering:
     - System Architecture
     - Data Flow Architecture
     - Experimental Workflow (sequence diagram)
     - Component Interaction
     - Deployment Architecture
     - Algorithm Adapter Pattern (class diagram)
     - Metrics Collection Pipeline
   - **Read this for deep architectural understanding**

## Additional Documentation

- **[Architecture](architecture.md)**: High-level system design, components, data flow
- **[Benchmark Methodology](benchmark_methodology.md)**: Metrics definitions, measurement procedures
- **[Reproducibility](reproducibility.md)**: Determinism guarantees, validation procedures
- **[Security Compliance](security_compliance.md)**: Threat models, data handling, ethical considerations

## Documentation Map

```
docs/
├── README.md                          ← You are here
│
├── Quick Start (5-10 minutes)
│   └── quick_reference_diagram.md     ← Single-page overview
│
├── Replication (1-2 hours)
│   └── implementation_guide.md        ← Step-by-step guide
│
├── Deep Dive (2-4 hours)
│   ├── framework_diagrams.md          ← Comprehensive diagrams
│   ├── architecture.md                ← System design
│   └── benchmark_methodology.md       ← Measurement procedures
│
└── Validation & Compliance
    ├── reproducibility.md             ← Validation procedures
    └── security_compliance.md         ← Ethical/security considerations
```

## Diagram Types

The documentation includes multiple diagram types for different purposes:

| Diagram Type | Purpose | Find In |
|--------------|---------|---------|
| **System Architecture** | Shows overall structure (layers, components) | framework_diagrams.md |
| **Data Flow** | Shows how data moves through pipeline | framework_diagrams.md, quick_reference_diagram.md |
| **Sequence** | Shows step-by-step execution timeline | framework_diagrams.md (Experimental Workflow) |
| **Component Interaction** | Shows module dependencies | framework_diagrams.md |
| **Class Diagram** | Shows adapter pattern structure | framework_diagrams.md (Algorithm Adapter Pattern) |
| **Deployment** | Shows infrastructure options | framework_diagrams.md |
| **Flowchart** | Shows decision logic and processes | framework_diagrams.md (Metrics Collection Pipeline) |

## For Different Audiences

### Academic Researchers
**Goal**: Validate and cite the research

Read:
1. Quick Reference Diagram (understand approach)
2. Benchmark Methodology (understand metrics)
3. Framework Diagrams (understand implementation)
4. Reproducibility (validate results)

### Software Engineers
**Goal**: Replicate or extend the framework

Read:
1. Implementation Guide (step-by-step setup)
2. Framework Diagrams (architectural patterns)
3. Architecture (design decisions)

### System Architects
**Goal**: Deploy in production or understand scalability

Read:
1. Quick Reference Diagram (overview)
2. Framework Diagrams → Deployment Architecture
3. Architecture (design considerations)

### Security Auditors
**Goal**: Assess compliance and security

Read:
1. Security Compliance (threat models, data handling)
2. Reproducibility (validation procedures)
3. Benchmark Methodology (measurement integrity)

## Key Features Documented

- ✅ **Complete System Architecture**: All components and their interactions
- ✅ **Reproducibility Instructions**: Step-by-step replication guide
- ✅ **Statistical Methods**: Hypothesis testing, effect sizes, confidence intervals
- ✅ **Deployment Options**: Local, Docker, Kubernetes, GCP GKE
- ✅ **Algorithm Adapters**: Trait-based design pattern with examples
- ✅ **Metrics Pipeline**: From instrumentation to visualization
- ✅ **Validation Checklist**: Verify correct implementation
- ✅ **Extension Points**: How to add new algorithms, metrics, or visualizations

## Original Module Responsibilities (Reference)

### Module responsibilities and interactions

- **Rust `pqc_core`**
  - **Adapters**: Thin wrappers around PQC algorithm implementations (e.g., external crates or FFI), exposing a uniform trait-based interface for keygen/sign/verify or KEM operations.
  - **Metrics**: Collect fine-grained timing, throughput, memory usage, error rates; expose counters/histograms for Prometheus via an optional HTTP endpoint.
  - **Workload engine**: Executes deterministic and stress workloads (single-threaded and concurrent), driven by parameters provided by the Python orchestrator.

- **Python orchestrator**
  - **Experiment execution**: Parses experiment YAML, validates schema, resolves datasets and parameters, and invokes `pqc_core` operations via PyO3 bindings.
  - **Analysis**: Post-processes raw metrics, aggregates statistics, and performs comparisons across algorithms/parameters/hardware.
  - **Reporting**: Writes results to `results/` in CSV/JSON; can render lightweight Markdown/HTML summaries for quick review.

- **Docker/K8s deployment & Terraform**
  - **Docker**: Container image wraps Python orchestrator and Rust core, providing reproducible runs locally and in CI.
  - **K8s/Helm**: Deploys the container to a cluster for scalable workloads; optional Service to expose the Prometheus `/metrics` endpoint.
  - **Terraform (GCP)**: Provisions a minimal GKE cluster and node pool for running orchestrated experiments at scale.

### Textual data-flow diagram

```text
Experiment YAML (configs/*.yaml)
        |
        v
Python Orchestrator
  - parse/validate YAML
  - build experiment plan
  - call Rust via PyO3
        |
        v
Rust pqc_core
  - adapters -> algorithm ops
  - workload engine executes runs
  - metrics collected (timings/mem/errs)
        |                          \
        |                           \ (optional) HTTP /metrics
        v                            \--> Prometheus scrape
Raw Metrics (in-memory / files)
        |
        v
Python Analysis & Reporting
  - aggregate/compare
  - write CSV/JSON to results/
  - generate brief reports (MD/HTML)
```

### Communication and metrics

- **PyO3 bindings**: The orchestrator imports a Python extension module compiled from `pqc_core`. Calls are in-process, avoiding serialization overhead. Data passed includes workload parameters and buffer payloads for crypto operations; results/metrics returned as structured objects.
- **Prometheus endpoint**: `pqc_core` can expose an optional HTTP server (configurable port) for counters/histograms during long runs. In K8s, this can be scraped by Prometheus via Service annotations; locally, it's reachable from the host.

### Local vs cloud parity

- **Same image, same config**: Local `docker compose` and K8s/Helm use the same container image and experiment YAMLs, ensuring identical behavior across environments.
- **Environment-driven overrides**: Minor differences (e.g., concurrency, dataset paths, metrics port) are controlled via environment variables or Helm values, not code changes.
- **Results location**: Both local and cloud runs write to `results/` (bind mount or PVC) to preserve artifacts and facilitate comparison.

---

## Questions?

- Open an issue on GitHub: https://github.com/your-org/quantum-resilient/issues
- Email: research-support@example.com
- See main [README.md](../README.md) for project overview

---

**Last Updated**: November 10, 2024  
**Documentation Version**: 1.0
