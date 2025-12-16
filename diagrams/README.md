# Dissertation Diagrams

This directory contains Mermaid diagram source files for Chapter 3 of the dissertation.

## Diagrams

1. **framework-architecture.mmd** - Block diagram showing the three principal layers of the benchmarking framework and their interactions (Figure 3.1, Section 3.3)

2. **live-system-comparison.mmd** - Comparison diagram showing live production system vs experimental framework with instrumentation points marked (Section 3.3)

3. **high-level-overview.mmd** - High-level overview of the research methodology and framework components (Section 3.1.1)

## Converting to SVG/PNG

### Prerequisites

Install mermaid-cli:
```bash
npm install -g @mermaid-js/mermaid-cli
```

### Conversion

Run the conversion script:
```bash
./convert_diagrams.sh
```

This will generate SVG and PNG files in the `../figures/` directory.

### Manual Conversion

To convert individual diagrams:
```bash
# To SVG
mmdc -i framework-architecture.mmd -o ../figures/framework-architecture.svg -b transparent

# To PNG (high resolution for Word doc)
mmdc -i framework-architecture.mmd -o ../figures/framework-architecture.png -b transparent -w 2400 -H 1800
```

## Color Scheme and Design Principles

The diagrams use a semantic color-coding system designed to improve readability and visual organization while maintaining a modern, professional appearance.

### Design Philosophy

**High-Level Categorization**: Colors are organized into families that represent high-level functional categories (e.g., deployment infrastructure, orchestration, execution, telemetry). This allows readers to quickly identify the role and context of each component.

**Within-Category Distinction**: Within each color family, different shades are used to distinguish individual components. This two-level approach ensures that:
- Components are immediately recognizable as belonging to the same functional category
- Individual components remain visually distinct and easy to identify
- The color scheme remains cohesive and avoids visual confusion

**Visual Clarity**: All nodes use:
- **Thick black borders** (3px) for strong definition and modern appearance
- **Flat colors** (no gradients) for clean, professional aesthetics
- **Rounded corners** (Mermaid default) for softer, contemporary styling
- **Sufficient contrast** to ensure readability without eye strain

### Color Families

The following color families are used consistently across all diagrams:

#### 🔵 **Deployment/Infrastructure (Blue Family)**
- **Purpose**: Infrastructure and deployment environments
- **Shades**: 
  - Darker blue (`#64b5f6`): Bare-metal execution
  - Medium blue (`#90caf9`): Kubernetes orchestration
  - Light blue (`#bbdefb`): Cloud deployment
- **Rationale**: Blue is commonly associated with infrastructure and cloud services, making it an intuitive choice for deployment-related components.

#### 🟣 **Orchestration/Management (Purple Family)**
- **Purpose**: Coordination, management, and configuration
- **Color**: `#ce93d8` (medium purple)
- **Rationale**: Purple distinguishes management functions from execution, providing clear visual separation between coordination and operational components.

#### 🔷 **Configuration (Light Blue)**
- **Purpose**: Configuration files and settings
- **Color**: `#b3e5fc` (light cyan-blue)
- **Rationale**: Light blue maintains connection to the infrastructure theme while clearly distinguishing configuration from deployment.

#### 🟠 **Core Execution (Orange Family)**
- **Purpose**: Core execution components (pipelines, workload generation)
- **Shades**:
  - Darker orange (`#ffb74d`): Primary execution components (e.g., streaming pipeline)
  - Lighter orange (`#ffcc80`): Supporting execution components (e.g., workload generator)
- **Rationale**: Orange provides warm, energetic tones that suggest active processing and execution, distinct from infrastructure and management.

#### 🟡 **Cryptographic Operations (Amber/Yellow)**
- **Purpose**: Cryptographic adapters and operations
- **Color**: `#ffe082` (amber)
- **Rationale**: Amber/yellow stands out as a distinct category for security-critical cryptographic operations, positioned between execution (orange) and instrumentation (yellow).

#### 🟨 **Instrumentation/Telemetry (Yellow Family)**
- **Purpose**: Monitoring, measurement, and telemetry collection
- **Shades**:
  - Darker yellow (`#fff59d`): Primary instrumentation points
  - Medium yellow (`#fff9c4`): Telemetry collection
  - Light yellow (`#fffde7`): Secondary instrumentation points
- **Rationale**: Yellow is associated with observation and measurement, making it ideal for instrumentation components. The light tones ensure visibility without overwhelming the diagram.

#### 🟢 **Output/Data (Green Family)**
- **Purpose**: Data outputs, event logs, and processed results
- **Shades**:
  - Darker green (`#a5d6a7`): Primary outputs (e.g., JSONL event logs)
  - Medium green (`#c8e6c9`): Secondary outputs (e.g., CSV summaries)
  - Light green (`#e8f5e9`): Metadata and tertiary outputs
- **Rationale**: Green suggests completion and output, providing a natural visual endpoint for data flow. The progression from darker to lighter shades can represent data refinement stages.

#### 🌿 **Analysis/Statistics (Darker Green Family)**
- **Purpose**: Statistical analysis, data aggregation, and comparative analysis
- **Shades**:
  - Darker green (`#81c784`): Primary analysis (e.g., statistical analysis)
  - Medium green (`#a5d6a7`): Supporting analysis (e.g., data aggregation)
- **Rationale**: Darker greens distinguish analytical processes from raw outputs, suggesting deeper processing and insight generation.

#### ⚪ **Production System (Gray Family)**
- **Purpose**: Live production system components (for comparison diagrams)
- **Shades**: Multiple gray tones from `#90a4ae` to `#f5f5f5`
- **Rationale**: Gray tones clearly distinguish production systems from experimental framework components, emphasizing the comparison nature of the diagram.

### Consistency Across Diagrams

The color scheme is applied consistently across all diagrams:
- Components serving the same function use the same color family
- Shades within families are assigned based on component hierarchy and importance
- The semantic meaning of colors is preserved, allowing readers to transfer understanding between diagrams

This approach creates a visual language that enhances comprehension and makes the diagrams more accessible to readers at all levels of technical expertise.

## Diagram Descriptions

### Framework Architecture (Figure 3.1)

**Components:**
- **Configuration Layer**: Experiment matrix (declarative YAML), scenario generator (Python script), individual scenario YAML files, and deterministic RNG seed computation from experiment parameters
- **Deployment Layer**: Bare-metal execution, Kubernetes orchestration, and cloud deployment (GCP GKE) environments
- **Orchestration and Metrics Layer**: Rust orchestrator (coordinator, controller), Python orchestration (experiment execution), data aggregator (multi-run combination), and statistical analysis (hypothesis testing, effect sizes)
- **Cryptographic Execution Layer**: Streaming pipeline (async event processing), execution modes (Single, FixedPool, Elastic), workload generator (patterns: Constant, Burst, Ramp), payload generation (deterministic RNG), cryptographic adapters (PQC & classical unified interface), telemetry collection (timing, resources, events), and control plane (health, readiness endpoints)
- **Telemetry Outputs**: Event logs (JSONL format), statistical summaries (CSV/JSON), and environment metadata (deployment context)
- **Analysis Layer**: Hypothesis testing (t-test, Mann-Whitney U), effect size computation (Cohen's d, confidence intervals), statistics computation (percentiles, aggregates), visualization scripts (CDFs, comparison charts), Jupyter notebooks (exploratory analysis), and export utilities (dataset export, merge)

**Data Flow**: 
1. **Configuration**: Experiment Matrix → Scenario Generator → Individual Scenarios + RNG Seed Computation
2. **Execution**: Scenarios → Python Orchestration → Rust Orchestrator → Pipeline → Execution Modes → Workload Generator → Payload Generation → Crypto Adapters → Telemetry Collection → Outputs
3. **Analysis**: Outputs → Data Aggregator → Statistical Analysis → Hypothesis Testing + Effect Sizes + Statistics Computation → Visualization + Notebooks + Export

### Live System Comparison

**Live Production System (Enterprise AML Transaction Monitoring Pipeline):**
- **Transaction Ingestion**: High-volume real-time streams (100-10,000 transactions/s) with variable payload sizes
- **Streaming Pipeline**: Event processing and routing with async processing architecture
- **Production Cryptography**: Digital signatures, encryption for audit trail security and regulatory compliance
- **ML/AI Models**: Anomaly detection, risk scoring, pattern recognition
- **Alert Generation**: Suspicious activity flags and compliance reporting
- **Compliant Output**: Regulatory reporting and audit logs
- **Production Monitoring**: SLA tracking, health checks, operational metrics (dashed line indicates optional/monitoring only)

**Experimental Benchmarking Framework (Representative of Production System):**
- **Workload Generator**: Deterministic patterns (constant rate: 100-2,000 msg/s, burst patterns for enterprise load), payload sizes 256B-16KB, represents transaction volumes
- **Streaming Pipeline**: Async event processing with modular architecture that mirrors production design
- **Crypto Adapters**: PQC & classical unified interface, same cryptographic operations and security primitives as production
- **Telemetry Collection**: Comprehensive instrumentation (framework enhancement for measurement)

**Instrumentation Points (Framework Enhancement):**
- **Timing Measurement**: Nanosecond precision latency capture (exceeds production monitoring)
- **Resource Monitoring**: CPU, memory, I/O system-level metrics with production-grade precision
- **Event Logging**: Operation-level metrics in JSONL format with structured telemetry

**Representativeness Claims:**
- **Workload Representativeness**: Framework implements workload patterns (constant rate, burst patterns) and payload sizes (256B-16KB) representative of real-time streaming applications. Workload rates (100-10,000 msg/s) span operational ranges from moderate to enterprise-scale throughput
- **Pipeline Architecture**: Framework implements modular data streaming pipeline architecture that mirrors production systems, with cryptographic operations integrated at appropriate stages
- **Cryptographic Operations**: Framework uses same cryptographic primitives (signing, encryption) as production systems, ensuring performance measurements reflect operational contexts
- **Measurement Precision**: Framework employs high-resolution telemetry instrumentation (nanosecond precision timing, system-level resource monitoring) that matches or exceeds precision required for production monitoring

**Key Difference**: The framework adds comprehensive telemetry instrumentation at multiple points, enabling detailed performance measurement that would be impractical in live production systems, while maintaining representativeness of operational conditions through deterministic workload generation and production-aligned architecture.

### High-Level Overview

Shows the research methodology flow from literature analysis through experimental framework to outputs (performance metrics and engineering recommendations).
