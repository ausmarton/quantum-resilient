
3. # **Chapter 3 Methodology** {#chapter-3-methodology}

   1. ## Methods and techniques selected {#methods-and-techniques-selected}

This section provides an overview of the research methodology, methods, and techniques employed in this study. The methodology combines systematic literature analysis, experimental framework development, quantitative performance measurement, and statistical analysis to address the research objectives. Detailed descriptions of procedures and implementation are provided in Section 3.3.

   ### **3.1.1 Research Methodology Overview**

To evaluate the performance impact of PQC algorithms on real-time data streaming pipelines, this research adopts an experimental approach comprising four integrated components:

**Systematic Literature Analysis**: A structured review identifies candidate PQC algorithms using inclusion criteria based on NIST standardisation status, implementation maturity, and documented empirical performance. This component addresses Objective 1 by establishing criteria for algorithm selection.

**Prototype-Based Experimental Design**: A high-fidelity framework implements real-time data pipeline primitives with modular cryptographic components, enabling controlled evaluation of PQC and classical algorithms under identical conditions. This component addresses Objective 2 by developing the modular test framework.

**Quantitative Performance Measurement**: Instrumentation captures fine-grained metrics including latency, throughput, and resource utilisation through system-level monitoring interfaces. This component addresses Objective 3 by enabling comprehensive benchmarking across key performance metrics.

**Statistical Hypothesis Testing**: Performance comparisons employ parametric and non-parametric statistical methods with effect size quantification, enabling rigorous comparative analysis. This component addresses Objectives 4 and 5 by providing quantitative comparison and supporting engineering recommendations.

The methodology utilises open-source implementations from the Open Quantum Safe (OQS) project, ensuring reproducibility and alignment with NIST-standardised algorithms. Through this approach, the research contributes detailed performance analysis, benchmark datasets, and engineering design recommendations for quantum-resilient real-time systems.

   ### **3.1.2 Framework Architecture Overview**

The experimental framework consists of five principal layers that work together to enable comprehensive telemetry collection and reproducible experimentation (detailed architecture is described in Section 3.3.3):

**Configuration Layer**: Manages experiment definition through declarative YAML configuration defining the parameter space (algorithms, environments, workload configurations) and ensures deterministic parameter generation.

**Deployment Layer**: Enables reproducible experimentation across three execution contexts: bare-metal native execution (baseline performance), containerised local Kubernetes (containerised experimentation), and cloud-managed Kubernetes (cloud environment evaluation).

**Orchestration and Metrics Layer**: Governs experiment execution and coordinates telemetry collection, combining results from multiple runs and performing statistical analysis including hypothesis testing and effect size computation.

**Cryptographic Execution Layer**: Provides the high-performance runtime environment for executing cryptographic operations, implementing both PQC and classical primitives through a unified interface. This layer includes the streaming pipeline, workload generator, crypto adapters, and telemetry collection components that capture timing, resource utilisation, and event-level metrics.

**Analysis Layer**: Processes telemetry outputs into statistical summaries, hypothesis test results, and visualisations, enabling comprehensive performance analysis and comparison.

The framework architecture is illustrated in Figure 3.0, which shows the high-level flow from configuration through execution to analysis.

![Figure 3.0: High-Level Framework Overview](figures/high-level-overview.png)

*Figure 3.0: High-level overview of the experimental framework showing the research methodology flow. **Literature Analysis** (left): Systematic review identifies candidate PQC algorithms based on NIST standardisation, implementation maturity, and performance characteristics. **Experimental Framework** (centre): Prototype-based design enables controlled evaluation with modular cryptographic components, deterministic workload generation, and comprehensive telemetry collection across multiple deployment environments. **Performance Metrics** (right): Quantitative measurement captures latency, throughput, and resource utilisation through system-level instrumentation. **Engineering Recommendations** (bottom): Statistical analysis and comparative evaluation support evidence-based design recommendations for quantum-resilient real-time systems. Arrows indicate the flow from algorithm selection through framework development and measurement to recommendations.*

   ### **3.1.3 Data Collection Overview**

Empirical performance data is collected through systematic measurements on the prototype framework, utilising open-source implementations from the Open Quantum Safe (OQS) project. The framework integrates PQC and classical cryptographic algorithms into real-time streaming pipeline stages, enabling controlled measurement of key performance metrics.

Data collection employs telemetry instrumentation that captures operation-level metrics including latency (measured with high-resolution timing), throughput (messages per second), and resource utilisation (CPU consumption, memory allocation, I/O statistics). Latency measurements employ nanosecond-precision timing instrumentation, enabling accurate characterisation of sub-microsecond performance differences. Measurements are conducted across multiple deployment environments (bare-metal, containerised local, cloud-managed) and workload configurations (varying payload sizes, message rates, and workload patterns).

Each experimental configuration is executed with multiple independent runs to ensure statistical robustness. Raw event-level telemetry is captured in structured JSONL format, with run-level aggregation producing statistical summaries including percentiles, means, standard deviations, and confidence intervals. This data collection approach enables comprehensive comparative analysis between PQC and classical algorithms under controlled, reproducible conditions.

   ### **3.1.4 Analysis Approach Overview**

The analysis approach employs statistical hypothesis testing to determine the significance of observed performance differences between PQC and classical algorithms. Performance comparisons utilise parametric (independent samples t-test) and non-parametric (Mann-Whitney U) methods with effect size quantification (Cohen's d) and confidence intervals.

**Statistical Unit of Analysis**: Although latency is captured at the event level, statistical hypothesis testing is performed exclusively on run-level aggregates. Individual events are not treated as independent observations for significance testing, thereby avoiding pseudo-replication and ensuring statistical independence. This two-level structure (operation-level measurements → run-level aggregates → cross-run statistics) enables both fine-grained distributional analysis and robust statistical inference, ensuring that distributional claims are based on large operation-level samples while statistical significance is assessed at the run level to account for experimental independence.

Sample sizes are designed to provide adequate statistical power for detecting medium effects. The analysis produces comparative performance characterisation across latency, throughput, and resource utilisation metrics, enabling quantitative assessment of performance–security trade-offs. Findings are synthesised into engineering design recommendations for optimising quantum-resilient real-time data pipelines, addressing the research objectives through evidence-based conclusions.

   ### **3.1.5 Mapping of Research Objectives to Methodology**

Table 3.1 explicitly maps each research objective to the measurement approach, metrics, and statistical methods employed to address it. This mapping demonstrates how the experimental methodology directly addresses each research objective through systematic data collection and analysis.

**Table 3.1**: Mapping of Research Objectives to Methodology

| Research Objective | Measurement Approach | Metrics | Statistical Method |
| :---- | :---- | :---- | :---- |
| Objective 1: Establish criteria for selecting PQC algorithms | Systematic literature analysis and framework integration validation | Algorithm integration feasibility, implementation maturity, performance characteristics | Descriptive analysis, validation metrics |
| Objective 2: Develop modular framework | Framework functionality and modularity assessment | Integration success, interface uniformity, component modularity | Validation metrics, descriptive analysis |
| Objective 3: Benchmark PQC and classical algorithms | Controlled pipeline execution under varying workload conditions | p50, p95, p99 latency (high-resolution timing), throughput (msg/s), CPU time, memory footprint, I/O statistics | Descriptive statistics (percentiles, means, standard deviations, confidence intervals) |
| Objective 4: Compare PQC performance against classical | Comparative analysis under identical experimental conditions | Relative latency differences, throughput differences, resource utilisation differences | t-test, Mann-Whitney U test, effect size (Cohen's d), confidence intervals |
| Objective 5: Provide engineering recommendations | Synthesis of performance findings across algorithms, environments, and workloads | Performance trade-offs, deployment context comparisons, scalability assessments | Comparative analysis, effect size quantification, synthesis of statistical findings |

This explicit mapping ensures that each research objective is addressed through appropriate measurement techniques and statistical methods, providing clear traceability from research questions to methodology.

2. ## Justification {#justification}

This section justifies the experimental methodology adopted in this research, explaining how it addresses each research objective, why the experimental method is appropriate, how the framework represents live production systems, and why alternative research methods are excluded.

   ### **3.2.1 Methodology Alignment with Research Objectives**

The experimental methodology is designed to address each of the five research objectives through systematic data collection and analysis:

**Objective 1: Establish criteria for selecting PQC algorithms relevant to real-time data streaming.** The systematic literature analysis component (Section 3.1.1) addresses this objective by identifying candidate algorithms based on NIST standardisation status, implementation maturity, and documented empirical performance. The experimental framework then validates these selection criteria by demonstrating that selected algorithms can be successfully integrated and evaluated within real-time streaming contexts. Data obtained from experimental runs includes algorithm integration feasibility, implementation maturity assessment, and performance characteristics that validate or refine selection criteria.

**Objective 2: Develop a modular framework for evaluating cryptographic algorithm performance.** The prototype-based experimental design addresses this objective by implementing a modular framework with unified cryptographic interfaces, enabling seamless integration of both PQC and classical algorithms. Data obtained from experimental runs includes framework functionality validation, modularity assessment, and integration success metrics that demonstrate the framework's capability to evaluate diverse algorithms under controlled conditions.

**Objective 3: Benchmark selected PQC algorithms and classical algorithms across key performance metrics.** The quantitative performance measurement component addresses this objective by systematically collecting latency, throughput, and resource utilisation data across multiple algorithms, environments, and workload configurations. Data obtained from experimental runs includes operation-level latency measurements (using high-resolution timing instrumentation), throughput measurements (messages per second), and resource utilisation metrics (CPU consumption, memory allocation, I/O statistics) that enable comprehensive performance characterisation.

**Objective 4: Compare PQC performance against existing classical encryption techniques.** The statistical hypothesis testing component addresses this objective by enabling rigorous comparative analysis between PQC and classical algorithms under identical experimental conditions. Data obtained from experimental runs includes comparative performance metrics, statistical significance test results, and effect size quantification that enable quantitative assessment of performance differences and trade-offs.

**Objective 5: Provide engineering recommendations for designing optimised quantum-resilient real-time data pipelines.** The statistical analysis and comparative evaluation components address this objective by synthesising performance findings into evidence-based recommendations. Data obtained from experimental runs includes performance trade-off analysis, deployment context comparisons, and scalability assessments that inform engineering design decisions.

The experimental methodology enables controlled, repeatable measurements across multiple algorithmic configurations, deployment environments, and workload conditions, providing the comprehensive data required to address all five objectives through a unified experimental framework.

   ### **3.2.2 Framework Representation of Live Production Systems**

The experimental framework is designed to represent the cryptographic and streaming characteristics of live production systems while operating as a closed-system prototype that avoids operational risks and ethical constraints associated with live data. The framework achieves this representation through several design principles:

**Workload Representativeness**: The framework's workload generator creates patterns (constant rate, burst patterns) that span operational ranges (100 to 10,000 messages per second) and payload sizes (256B to 16KB) representative of real-time streaming applications. These workload characteristics approximate production system transaction volumes and message sizes, enabling performance measurements that reflect operational conditions.

**Pipeline Architecture Fidelity**: The framework implements a streaming pipeline architecture that mirrors production system designs, with cryptographic operations integrated at appropriate stages. This architectural similarity ensures that performance measurements reflect operational contexts rather than isolated algorithmic benchmarks, capturing the interactions between algorithmic complexity, system architecture, and runtime behaviour that influence performance in production systems.

**Deployment Context Fidelity**: The framework supports evaluation across multiple deployment contexts (bare-metal, containerised, cloud-managed) that represent common production deployment models. This enables assessment of how deployment characteristics (containerisation overhead, cloud infrastructure) affect performance, providing insights relevant to production deployment decisions.

**Measurement Precision**: The framework employs telemetry instrumentation with high-resolution timing (nanosecond precision) that captures fine-grained performance characteristics. While production systems typically cannot instrument at this level due to performance concerns, the framework's comprehensive instrumentation enables detailed performance analysis that would be impractical in live systems.

**Deterministic Reproducibility**: The framework ensures deterministic workload generation through seeded random number generation, enabling reproducible experiments that support statistical analysis and independent verification. This reproducibility is essential for rigorous experimental evaluation but may differ from production systems where workload patterns are inherently variable.

The framework intentionally excludes domain-specific components such as ML inference, business logic, and network jitter that would be present in full production systems, focusing instead on the cryptographic processing stages that are the primary concern of this research. This abstraction enables controlled evaluation of cryptographic performance while maintaining representativeness of the cryptographic processing context.

**Scope Boundaries**: This research deliberately excludes end-to-end application latency, network jitter, machine learning inference, and business logic execution. These factors are orthogonal to cryptographic cost and would confound attribution of performance differences. The framework focuses exclusively on cryptographic operation performance within the streaming pipeline context, enabling clear attribution of performance characteristics to algorithmic choices rather than system-level factors.

   ### **3.2.3 Justification for Experimental Method**

The experimental research method is appropriate for this study because it enables systematic manipulation and observation of variables influencing performance, providing a rigorous empirical foundation for addressing the research objectives. Performance characteristics of post-quantum cryptographic algorithms can vary significantly depending on the architecture of the real-time system in which they are deployed, making controlled experimental evaluation essential to capture performance data that accurately reflects operational conditions.

Existing empirical work in this domain has frequently been limited to synthetic or laboratory-based environments that fail to capture the complexities and constraints of production-grade streaming systems (Chen, Zhao and Wang, 2022; Zhang and Kumar, 2020). By integrating PQC algorithms into a realistic, modular data streaming pipeline representative of anti-money laundering (AML) systems, this research addresses a critical gap in the current literature and provides results that are directly relevant to real-world deployments.

Conducting an "open-system" experiment on a live production platform was considered impractical due to operational risks and stringent legal and ethical constraints associated with the use of live data, especially under the UK General Data Protection Regulation (GDPR) and Data Protection Act 2018 (Information Commissioner's Office, 2018; Voigt and Von dem Bussche, 2017). Engaging with live systems for experimental purposes poses risks of service disruption and potential violations of data privacy, which this research aims to avoid (Smith, Anderson and Taylor, 2020). Instead, the methodology leverages a "closed-system" prototype that simulates the complexities of a production-grade environment while mitigating these risks.

The methodology leverages open-source, production-quality reference implementations, such as those developed by the Open Quantum Safe (OQS) project. This ensures reproducibility of results and alignment with algorithms that have been standardised by the National Institute of Standards and Technology (NIST), thereby enhancing the applicability of findings to industry adoption (NIST, 2022; Li and Patel, 2023). Through replication and statistical analysis, the experimental framework provides both internal validity (by controlling for confounding factors such as hardware variation or workload imbalance) and external validity (through reproducibility across local and cloud-based environments). Consistent with the principles of empirical software engineering and systems research (Fowler and Kelleher, 2019), this methodology supports the derivation of evidence-based conclusions that extend beyond the implementation itself, offering quantifiable insights into the practical feasibility and performance trade-offs of quantum-resilient data architectures.

   ### **3.2.4 Exclusion of Alternative Methods**

Other research methodologies were carefully evaluated but were ultimately deemed unsuitable for addressing the study's objectives:

**Theoretical or Analytical Approaches**: Purely theoretical or analytical approaches, while valuable for establishing cryptographic soundness, are limited in their ability to quantify the implementation-specific overheads that emerge when cryptographic primitives are deployed in operational systems (Gonzalez and Ramirez, 2018). These approaches cannot account for the interactions between algorithmic complexity, system architecture, and runtime behaviour that directly influence latency, throughput, and resource utilisation in real-time data pipelines. The research objectives require empirical performance data that can only be obtained through experimental measurement.

**Surveys and Expert Interviews**: Qualitative methodologies such as surveys or expert interviews, though effective in capturing practitioner perspectives and adoption trends, do not yield the empirical, measurable data required to evaluate algorithmic performance across controlled experimental conditions (Bryman, 2016). The research objectives require quantitative performance metrics (latency, throughput, resource utilisation) that cannot be obtained through qualitative methods.

**Observations**: Observational methods, while valuable for understanding system behaviour in natural settings, are unsuitable for this research because they cannot provide the controlled conditions necessary for fair algorithmic comparison. Observations of live systems would be confounded by operational variability, workload fluctuations, and the inability to instrument at the required precision without impacting performance.

**Case Studies**: Case study methodologies were also considered but found inadequate, as the insights derived from a single organisational or system context would lack the external validity necessary to inform generalisable engineering recommendations (Yin, 2017). The research objectives require comparative evaluation across multiple algorithms, environments, and workload configurations that cannot be achieved through single-case study approaches.

In contrast, the experimental research method allows for the systematic manipulation and observation of variables influencing performance, providing a rigorous empirical foundation for analysis that directly addresses the research objectives through controlled, repeatable measurements.

3. ## Research procedures {#research-procedures}

This section provides the most detailed description of the research procedures, building upon the overview in Section 3.1 and justification in Section 3.2. The section describes the framework architecture, measurement techniques, data collection procedures, and validation activities in sufficient detail to enable replication by other researchers.

The experimental framework architecture is illustrated in Figure 3.1, which shows the five principal layers and their interactions. This diagram structures the narrative that follows, with each layer described in detail to explain how the framework enables reproducible performance measurement and telemetry collection.

![Figure 3.1: Framework Architecture](figures/framework-architecture.png)

*Figure 3.1: Block diagram showing the five principal layers of the benchmarking framework and their component interactions. **Configuration Layer** (top-left): Experiment matrix (declarative YAML configuration) defines experimental parameters; scenario generator creates individual scenario YAML files; RNG seed computation ensures deterministic workload generation from experiment parameters. **Deployment Layer** (top-right): Bare-metal native execution provides baseline performance measurement; Kubernetes container orchestration enables containerised local experimentation; Cloud deployment (GCP GKE) enables cloud-managed environment evaluation. **Orchestration and Metrics Layer** (middle): Orchestrator manages experiment lifecycle and worker coordination; data aggregator combines results from multiple runs; statistical analysis performs hypothesis testing and effect size computation. **Cryptographic Execution Layer** (bottom): Streaming pipeline performs async event processing; execution modes (Single, FixedPool, Elastic) control concurrency; workload generator creates patterns (Constant, Burst, Ramp) with configurable rates; payload generation uses deterministic RNG; crypto adapters implement PQC and classical algorithms through unified interface; telemetry collection captures timing (nanosecond precision), resources (CPU, memory, I/O), and events; control plane provides health and readiness endpoints. **Telemetry Outputs** (right): Event logs (JSONL format) capture operation-level metrics; statistical summaries (CSV/JSON) provide aggregated run-level statistics; environment metadata records deployment context. **Analysis Layer** (bottom-right): Hypothesis testing (t-test, Mann-Whitney U), effect size computation (Cohen's d, confidence intervals), statistics computation (percentiles, aggregates), visualization scripts (CDFs, comparison charts), Jupyter notebooks (exploratory analysis), and export utilities (dataset export, merge). Arrows indicate data flow: configuration flows from experiment matrix through scenario generation to orchestration; execution flows through pipeline to telemetry collection; telemetry flows to outputs and aggregation; aggregated data flows to statistical analysis and visualization. The telemetry collection component is the primary instrumentation point where all performance measurements are captured.*

   ### **3.3.1 Framework Implementation**

The framework was developed to evaluate performance implications of integrating PQC algorithms within real-time data processing pipelines. Its design emphasises modularity, reproducibility, and portability, enabling controlled comparison between PQC and classical cryptographic methods under uniform experimental conditions. The framework acknowledges that different algorithms employ different computational models (e.g., elliptic curve operations vs. lattice-based polynomial arithmetic), but ensures uniform measurement methodology across all algorithms.

**Methodological Justification**: The architectural design described in this section is not presented for implementation completeness, but to justify how the experimental framework controls for confounding variables, ensures repeatability, and enables fair comparison across cryptographic algorithms. Each layer and component serves a specific methodological purpose: controlling experimental conditions, isolating algorithmic performance from environmental factors, ensuring measurement consistency, and enabling statistical analysis.

As introduced in Section 3.1.2, the framework consists of five principal layers that work together to enable comprehensive telemetry collection and reproducible experimentation. As illustrated in Figure 3.1, each layer serves a distinct functional purpose in the measurement process, as described below.

#### **Configuration Layer**

The configuration layer manages experiment definition through a declarative YAML experiment matrix defining the parameter space (algorithms, environments, workload configurations) and ensures deterministic parameter generation. The scenario generator processes this matrix to create individual scenario YAML files specifying exact experimental conditions. Deterministic random number generator seeds computed from experiment parameters ensure that identical configurations produce identical workload patterns across repeated runs, enabling reproducibility while maintaining realistic workload characteristics.

This architectural design controls for workload variability as a confounding factor, ensuring that observed performance differences arise from algorithmic characteristics rather than workload pattern differences. The deterministic approach enables statistical replication by ensuring identical experimental conditions across runs.

#### **Deployment Layer**

The deployment layer enables reproducible, environment-agnostic experimentation across three execution contexts: (1) bare-metal native execution providing baseline performance without containerisation overhead, (2) Kubernetes container orchestration enabling containerised local experimentation, and (3) cloud deployment (GCP GKE) enabling cloud-managed environment evaluation. The layer ensures uniform telemetry collection methodology across all contexts, enabling direct comparison of deployment impacts on performance.

Environment isolation capabilities ensure experimental runs do not interfere with each other and that external factors do not contaminate measurements, maintaining internal validity and enabling confident attribution of performance differences to algorithmic characteristics rather than environmental variability. Consistent data format generation across all execution environments supports the research objective of assessing performance across different deployment models.

This architectural design isolates deployment context as an experimental variable, enabling assessment of how containerisation and cloud infrastructure affect performance while maintaining measurement consistency. The uniform telemetry methodology ensures that performance differences within each environment are attributable to algorithmic characteristics rather than measurement artifacts.

#### **Orchestration and Metrics Layer**

The orchestration and metrics layer governs experiment execution and coordinates telemetry collection. The orchestrator provides functionality for managing experiment lifecycle and worker coordination in distributed deployments. The orchestration component handles experiment execution, reading scenario configuration files and invoking the execution layer under controlled conditions, coordinating execution and collecting raw event-level telemetry data for each experimental configuration.

A data aggregator combines results from multiple runs, computing run-level statistics (percentiles, means, standard deviations, confidence intervals). Statistical analysis components perform hypothesis testing (t-tests, Mann-Whitney U tests) and effect size quantification (Cohen's d), enabling rigorous assessment of whether observed performance differences are statistically significant and practically meaningful, supporting the comparative analysis objectives of the research.

#### **Cryptographic Execution Layer**

The cryptographic execution layer provides the high-performance runtime environment for executing cryptographic operations. This layer includes the streaming pipeline, which performs asynchronous event processing, enabling efficient handling of high-throughput workloads. The layer implements multiple execution modes (Single, FixedPool, Elastic) that control concurrency and resource allocation, enabling evaluation of performance under different execution models.

The workload generator creates deterministic workload patterns including constant rate, burst patterns, and ramp patterns, with configurable message rates (100 to 10,000 messages per second) and payload sizes (256B to 16KB). The payload generation component uses deterministic random number generation based on seeds computed from experiment parameters, ensuring that identical experimental configurations produce identical workload patterns across repeated runs.

The crypto adapters implement both PQC and classical primitives through a unified interface that exposes standard operations including key generation, encapsulation, decapsulation, signing, and verification. **All cryptographic algorithms are executed through an identical adapter interface and measured using the same instrumentation paths, ensuring that observed differences arise from algorithmic characteristics rather than framework artefacts.** This uniform interface ensures that all algorithms are measured using identical instrumentation, enabling fair comparison despite their different computational models.

The telemetry collection component instruments precise timing measurements using monotonic clock primitives that provide high-resolution timing (nanosecond precision), enabling accurate characterisation of sub-microsecond performance differences. Resource utilization telemetry is captured through system-level monitoring interfaces that provide access to CPU consumption (user and system time), memory allocation (maximum resident set size), and I/O statistics. Event logging captures operation-level metrics including operation type, latency, resource utilization, and workload parameters in structured JSONL format.

The control plane component provides health and readiness endpoints that support distributed experiment coordination and monitoring, enabling the orchestrator to manage experiment lifecycle and ensure all components are ready before execution begins.

This architectural design ensures uniform measurement methodology across all algorithms, controlling for instrumentation bias and enabling fair comparative analysis. The unified interface and identical instrumentation paths eliminate framework artefacts as confounding factors, ensuring that performance differences reflect algorithmic characteristics.

#### **Analysis Layer**

The analysis layer processes telemetry outputs into statistical summaries, hypothesis test results, and visualisations. This layer includes hypothesis testing components that perform parametric (t-test) and non-parametric (Mann-Whitney U) statistical tests to determine significance of performance differences. Effect size computation calculates Cohen's d and confidence intervals to quantify the magnitude of observed differences. Statistics computation components compute percentiles, means, standard deviations, and other aggregate metrics from event-level telemetry.

The layer includes visualization scripts that generate cumulative distribution functions (CDFs), comparison charts, and statistical summaries, providing intuitive representation of complex performance relationships. Jupyter notebooks enable exploratory analysis of telemetry data, supporting ad-hoc investigation and verification of findings. Export utilities provide functionality for dataset export and merging, enabling independent analysis and verification of results by other researchers.

#### **Framework Integration and Reproducibility**

The five layers are integrated through clearly defined interfaces and a standardised output schema that together support reproducible execution and transparent data handling. Data flows from configuration through execution to telemetry outputs, then through aggregation to statistical analysis and visualization. All artefacts generated during experimentation, including configuration files, raw performance metrics, environment metadata, and analytical reports, are maintained under version control to enable independent verification and traceability.

The framework's modular architecture, standardised interfaces, and comprehensive documentation enable replication by other researchers. The layered design allows researchers to understand and potentially modify individual components while maintaining the overall measurement methodology.

#### **Framework Comparison with Live Production Systems**

Figure 3.2 illustrates how the experimental framework compares to live production systems and shows where telemetry instrumentation is placed to enable comprehensive performance measurement.

![Figure 3.2: Live System vs Framework Comparison](figures/live-system-comparison.png)

*Figure 3.2: Comparison diagram showing representative enterprise AML pipeline (left) versus experimental benchmarking framework (right) with instrumentation points and representativeness mappings. **Representative Enterprise AML Pipeline**: Transaction ingestion (100-10,000 tx/s) flows through streaming pipeline (event processing) to production cryptography (signing, encryption) for audit trail security and regulatory compliance, then through ML/AI models (anomaly detection) to alert generation (compliance reporting) and compliant output. Production monitoring (dashed line) provides operational monitoring but cannot capture fine-grained telemetry without impacting performance. **Experimental Benchmarking Framework**: Workload generator creates representative workload patterns (constant, burst patterns; 100-10,000 msg/s; 256B-16KB payloads) that represent transaction volumes; streaming pipeline (async processing) mirrors production architecture; crypto adapters (PQC & classical, sign, encrypt) enable same cryptographic operations as production. **Instrumentation Points** (highlighted subgraph): (1) Timing Measurement captures operation latency with nanosecond precision at the cryptographic operation boundary; (2) Resource Monitoring captures CPU consumption, memory allocation, and I/O statistics via system-level interfaces; (3) Event Logging captures operation-level metrics in JSONL format including operation type, latency, resource utilization, and workload parameters. **Telemetry Outputs**: Event logs (JSONL), statistical summaries (CSV/JSON), and environment metadata. **Representativeness Mappings** (dashed arrows): Transaction ingestion maps to workload generator (representative workload); production pipeline maps to framework pipeline (mirrors architecture); production crypto maps to framework crypto (same operations). The framework's comprehensive instrumentation enables detailed performance measurement that would be impractical in live production systems due to performance impact and operational constraints, while maintaining representativeness of operational conditions.*

The key difference between the live system and the framework is the comprehensive telemetry instrumentation integrated throughout the framework. In live production systems, monitoring is typically limited to aggregate metrics to avoid performance impact, and fine-grained telemetry collection is impractical due to operational constraints. The framework, operating as a closed-system prototype, can instrument every operation without performance concerns, enabling the detailed telemetry collection required for comparative performance analysis. The framework approximates production system characteristics through workload patterns that span operational ranges (100-10,000 messages per second), payload sizes (256B-16KB) representative of real-time streaming applications, pipeline architecture that captures cryptographic processing stages, and cryptographic operations that use the same primitives (signing, encryption) as production systems. However, the framework intentionally excludes domain-specific components such as ML inference, business logic, and network jitter that would be present in full production systems.

   ### **3.3.2 Data Collection Procedures**

Data collection follows a structured process that ensures comprehensive telemetry capture and statistical robustness. For each experimental configuration defined in the experiment matrix, the framework executes multiple independent runs to enable statistical analysis. Each run produces event-level telemetry data capturing individual cryptographic operation metrics.

The data collection process begins with scenario configuration, where the scenario generator creates individual scenario YAML files specifying exact experimental conditions including algorithm selection, deployment environment, workload parameters (payload size, message rate, workload pattern), and execution duration. Deterministic random number generator seeds computed from experiment parameters ensure that identical configurations produce identical workload patterns across repeated runs.

During execution, the telemetry collection component captures operation-level metrics for each cryptographic operation, including operation type, latency (using the high-resolution timing instrumentation described above), resource utilization (CPU consumption, memory allocation, I/O statistics), and workload parameters. These metrics are logged in structured JSONL format, with each event representing a single cryptographic operation invocation.

After execution, run-level aggregation processes event-level telemetry into statistical summaries including percentiles (p50, p95, p99), means, standard deviations, and confidence intervals. Multiple runs per configuration enable cross-run statistical analysis, computing mean values, standard deviations, and 95% confidence intervals across runs. This two-level structure (operation-level measurements → run-level aggregates → cross-run statistics) enables both fine-grained distributional analysis and robust statistical inference.

The resulting dataset includes raw event logs (JSONL format), aggregated statistical summaries (CSV/JSON format), and environment metadata recording deployment context, hardware characteristics, and configuration parameters. This comprehensive data collection approach enables the comparative analysis required to address the research objectives.

   ### **3.3.3 Framework Validation**

A series of validation activities were conducted to confirm that the framework approximates the cryptographic and streaming characteristics of production systems and produces reliable, reproducible measurements. These validation activities addressed three critical aspects: (1) measurement accuracy and precision, (2) framework approximation of production system characteristics, and (3) experimental reproducibility.

#### **Measurement Accuracy and Precision Validation**

Validation of measurement accuracy focused on confirming that telemetry instrumentation captures performance characteristics with sufficient precision and without introducing significant measurement overhead. Timing precision was validated by comparing framework measurements against known system clock characteristics, confirming that nanosecond-resolution timing provides adequate precision for sub-microsecond performance characterisation. Resource utilization measurement accuracy was validated by comparing framework measurements against system-level monitoring tools, confirming that CPU time, memory footprint, and I/O statistics are accurately captured.

Measurement overhead was assessed by comparing performance measurements with and without telemetry instrumentation enabled. This assessment confirmed that instrumentation overhead is negligible (<1% of measured latencies) and consistent across algorithms, ensuring that relative performance comparisons are not affected by measurement artifacts.

**Measurement Noise and Noise Floor**: Operating system scheduling, context switching, CPU frequency scaling, and container/cloud virtualization introduce variability that can exceed nanosecond-scale measurement precision. Modern OS scheduling noise and container/cloud jitter can produce microsecond-scale variations that may exceed the sub-microsecond deltas being measured. To mitigate this, the framework employs several strategies: (1) multi-run aggregation across independent experimental replications enables statistical separation of algorithmic signal from system-level noise, (2) distributional analysis (CDFs, percentiles) provides robust characterization that is less sensitive to individual measurement outliers than mean-based statistics, (3) relative performance comparisons (PQC vs classical within the same environment) are prioritized over absolute latency measurements, as relative deltas are more stable than absolute values in the presence of OS noise, and (4) statistical hypothesis testing with non-parametric methods (Mann-Whitney U) provides robustness to distributional assumptions when noise may affect distribution shape. This approach acknowledges that absolute latencies include measurement noise, while relative performance differences remain valid for comparative analysis.

#### **Framework Representativeness Validation**

Validation of framework representativeness addressed whether experimental measurements reflect performance characteristics that would be observed in live production systems. This validation employed several approaches:

**Workload Pattern Validation**: The framework's workload patterns (constant rate, burst patterns) were compared against documented characteristics of real-time streaming applications. This comparison confirmed that experimental workloads span operational ranges (100 to 10,000 messages per second) and payload sizes (256B to 16KB) that approximate production system characteristics.

**Pipeline Architecture Validation**: The framework's pipeline architecture was validated against production system designs, confirming that cryptographic operations are integrated at appropriate stages and that pipeline interactions mirror production contexts. This validation ensures that performance measurements reflect operational contexts rather than isolated algorithmic benchmarks.

**Deployment Context Validation**: Performance measurements across different deployment contexts (bare-metal, containerised, cloud-managed) were compared to assess whether framework behavior reflects expected production deployment characteristics. This validation confirmed that containerisation and cloud deployment overheads observed in the framework align with documented production system behavior, supporting the conclusion that framework measurements approximate production performance characteristics within the evaluated cryptographic processing context.

**Measurement Consistency Validation**: Consistency of measurements across repeated runs was validated to confirm that observed performance differences reflect algorithmic characteristics rather than measurement variability. Statistical analysis of repeated runs demonstrated that variance is within expected stochastic fluctuations, confirming that the framework produces stable, reproducible measurements suitable for comparative analysis.

#### **Experimental Reproducibility Validation**

Validation of experimental reproducibility addressed whether identical experimental configurations produce consistent results, enabling independent verification of findings. Reproducibility was validated through:

**Deterministic Workload Validation**: The framework's deterministic workload generation was validated by executing identical configurations multiple times and confirming that workload patterns are identical across runs. This validation ensures that observed performance differences reflect algorithmic characteristics rather than workload variability.

**Cross-Environment Reproducibility**: Measurements were validated across different execution environments to confirm that relative performance characteristics (PQC vs classical comparisons) are consistent regardless of deployment context. This validation supports the conclusion that framework findings are generalisable across deployment contexts.

**Statistical Robustness Validation**: The framework's statistical analysis capabilities were validated by confirming that hypothesis test results and effect size calculations are consistent with expected statistical behavior. This validation ensures that statistical conclusions drawn from experimental data are methodologically sound.

These validation activities confirmed that the framework produces accurate, representative, and reproducible measurements suitable for addressing the research objectives. The validation results support the conclusion that experimental findings can be confidently applied to inform real-world deployment decisions.

#### **Implemented vs Planned Capabilities**

This subsection clarifies which framework capabilities were implemented and actively used in the experimental campaign described in Chapter 4, distinguishing these from capabilities that were designed but not exercised in the final data collection.

**Implemented and Used in This Study (✔)**: The following capabilities were implemented and actively used in the experimental campaign: (1) **Multi-environment execution** across bare-metal, containerised local (Minikube), and cloud-managed (GCP GKE) deployments, (2) **Comprehensive telemetry collection** including nanosecond-precision timing, CPU/memory/I/O resource monitoring, and event-level JSONL logging, (3) **Multiple workload patterns** including constant rate and burst patterns across payload sizes (256B-16KB) and rates (100-10,000 msg/s), (4) **Statistical analysis pipeline** including run-level aggregation, hypothesis testing (t-tests, Mann-Whitney U), and effect size computation (Cohen's d), (5) **Deterministic workload generation** using seeded random number generation for reproducibility, and (6) **Six algorithm evaluation** covering three classical (RSA-2048, ECDSA P-256, ECDHE P-256) and three post-quantum (Kyber-512, Dilithium-2, Hybrid) algorithms.

**Implemented but Not Used in Final Experiments (△)**: Some framework capabilities were implemented but not exercised in the final experimental campaign: (1) **Extended duration tests** (300s) were implemented but only baseline configurations (30s) were used for primary analysis, (2) **Ramp workload patterns** were implemented but only constant and burst patterns were used in final experiments, and (3) **Elastic execution mode** was implemented but primary experiments used FixedPool mode.

**Designed but Deferred for Future Work (✗)**: The following capabilities were designed as part of the framework architecture but were not implemented or were deferred: (1) **OpenTelemetry (OTel) integration** for standardized observability, (2) **Prometheus metrics export** for real-time monitoring dashboards, and (3) **Advanced workload patterns** such as variable-rate simulation and complex burst profiles. These capabilities were designed to support future extensibility but were not required for the current research objectives.

This distinction clarifies that the framework's design supports extensibility beyond the current study's requirements, while the experimental campaign focused on capabilities necessary to address the five research objectives. The implemented capabilities were sufficient to conduct the comprehensive evaluation described in Chapter 4.

   ### **3.3.4 Threats to Validity**

This subsection explicitly addresses threats to validity and the controls employed to mitigate them, ensuring that experimental findings are methodologically sound and free from bias.

**Internal Validity**: Internal validity concerns whether observed performance differences are attributable to the experimental variables (algorithmic choice) rather than confounding factors. The framework controls for internal validity threats through: (1) **Identical hardware and environment** within each experimental run, ensuring that all algorithms are evaluated under identical conditions, (2) **Identical workload patterns** through deterministic workload generation, ensuring that workload characteristics do not vary between algorithm comparisons, (3) **Identical instrumentation paths** through the unified adapter interface, ensuring that measurement methodology does not favour any algorithm class, and (4) **Environment isolation** ensuring that experimental runs do not interfere with each other. These controls ensure that observed performance differences arise from algorithmic characteristics rather than experimental artefacts.

**Construct Validity**: Construct validity concerns whether the measurements accurately capture the intended performance characteristics. The framework addresses construct validity through: (1) **Precise latency definition** as the time required to complete a single cryptographic operation, measured at the operation boundary using monotonic clock primitives, (2) **Consistent measurement methodology** across all algorithms through the unified interface, ensuring that latency is measured identically for all algorithms, (3) **Resource utilization measurement** through system-level interfaces that provide accurate CPU, memory, and I/O statistics, and (4) **Throughput measurement** as aggregate message processing rate under parallel execution, clearly distinguished from per-operation latency. These measures ensure that the metrics accurately represent the performance characteristics of interest.

**Conclusion Validity**: Conclusion validity concerns whether statistical conclusions are valid and not subject to Type I or Type II errors. The framework addresses conclusion validity through: (1) **Multiple independent runs** per experimental configuration, providing adequate sample sizes for statistical inference, (2) **Run-level statistical analysis** treating runs as independent observations rather than individual events, avoiding pseudo-replication, (3) **Non-parametric statistical tests** (Mann-Whitney U) providing robustness to distributional assumptions, (4) **Effect size quantification** (Cohen's d) ensuring that statistically significant differences are also practically meaningful, and (5) **Multiple comparison correction** (Holm-Bonferroni) controlling for Type I error inflation. These measures ensure that statistical conclusions are methodologically sound.

**External Validity**: External validity concerns whether experimental findings generalise to real-world production systems. The framework addresses external validity through: (1) **Representative workload patterns** spanning operational ranges (100-10,000 msg/s, 256B-16KB payloads) that approximate production system characteristics, (2) **Multiple deployment contexts** (bare-metal, containerised, cloud-managed) representing common production deployment models, (3) **Production-quality algorithms** using NIST-standardised implementations that are representative of real-world deployments, and (4) **Pipeline architecture fidelity** mirroring production system designs. However, external validity is limited by the exclusion of domain-specific components (ML inference, business logic, network jitter) and the focus on cryptographic processing stages. Findings are generalisable to cryptographic performance within streaming pipeline contexts but may not extend to end-to-end application performance.

These validity controls ensure that experimental findings are methodologically sound, free from bias, and suitable for addressing the research objectives. The framework's design prioritises internal and construct validity to enable confident attribution of performance differences to algorithmic characteristics, while acknowledging limitations to external validity arising from the controlled experimental context.

4. ## Ethical considerations {#ethical-considerations}

This research is conducted in compliance with the Data Protection Act (2018) and the UK General Data Protection Regulation (UK GDPR), which govern the lawful, fair, and secure handling of personal data. The study does not involve human participants, personal data, or live production systems. All workloads are synthetic, and all cryptographic implementations are publicly available, standardised algorithms. No novel cryptographic primitives are proposed, and no activities pose ethical or legal risk.

**No Human Participants or Personal Data**: The research involves no human participants, no collection of personal data, and no handling of personally identifiable information (PII) or sensitive operational data. All experimental work is conducted using synthetic datasets generated specifically for testing purposes, with no connection to production infrastructure or handling of real-time data at any stage of the project.

**Synthetic Workloads Only**: All experimental workloads are synthetically generated using deterministic random number generation. No real-world transaction data, financial records, or operational data are used. The synthetic workloads are designed to represent operational characteristics (message rates, payload sizes) but contain no actual data content.

**No Cryptographic Misuse**: The research employs only publicly available, NIST-standardised cryptographic algorithms from the Open Quantum Safe (OQS) project. No novel cryptographic primitives are proposed, no cryptographic vulnerabilities are exploited, and no cryptographic misuse occurs. All implementations are used for their intended purpose of secure communication and data protection.

**Compliance with Institutional Guidelines**: The project design incorporates secure data management practices including strict access controls, logging, and compliance with institutional data handling protocols in the experimental framework. The research follows guidance from the Information Commissioner's Office (ICO) and reflects best practices for the ethical evaluation of emerging cybersecurity technologies, including quantum-resilient encryption, in non-operational settings.

**No Export-Controlled Material**: All cryptographic implementations used in this research are publicly available, open-source software. No export-controlled cryptographic material is employed, and all algorithms are standardised and publicly documented.

**Data Management**: All experimental datasets are hosted in isolated, access-controlled environments entirely separate from live systems. Raw telemetry data, statistical summaries, and analytical reports contain no personal or sensitive information and are maintained under version control for reproducibility purposes only.

This research poses no ethical concerns and complies with all relevant data protection regulations and institutional research ethics guidelines. The closed-system experimental approach ensures that no operational risks or privacy violations occur.

5. ## Summary of Chapter 3 {#summary-of-chapter-3}
