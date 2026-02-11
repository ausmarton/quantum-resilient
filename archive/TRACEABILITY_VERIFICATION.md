# Traceability Verification

**Date**: 2025-12-15  
**Purpose**: Verify traceability for objectives, aim, methodologies, and data analysis artefacts after Table 3.1 removal

---

## 1. Research Aim Traceability

### Aim Statement
**Location**: Section 2.3 (Line 313)
"Aim: To assess the performance impacts of quantum-resilient algorithms on real-time data streaming pipelines, focusing on latency, throughput, and resource utilisation."

### Aim → Objectives Connection
**Location**: Section 2.3 (Line 333)
"These objectives combine to support the aim by systematically quantifying the performance overhead of PQC, identifying bottlenecks, and providing practical guidance for mitigating these challenges."

### Aim → Methodology Connection
**Location**: Section 3.1.1 (Line 347)
"To evaluate the performance impact of PQC algorithms on real-time data streaming pipelines, this research adopted an experimental approach..."

### Aim → Results Connection
**Location**: Section 4.4.1 (Line 735)
"The empirical evaluation provides assessment of performance impacts across the three key dimensions specified in the research aim: latency, throughput, and resource utilisation."

**Location**: Section 4.4.5 (Line 765)
"The research aim is achieved through empirical evaluation (Section 4.1), statistical analysis (Section 4.2), multi-dimensional assessment of latency, throughput, and resource utilisation impacts..."

**Status**: ✅ **TRACEABLE** - Aim is connected to objectives, methodology, and results

---

## 2. Objectives Traceability

### Objective 1: Establish criteria for selecting PQC algorithms

**Objective → Methodology**:
- **Section 3.1.1**: "Systematic Literature Analysis: A structured review... identified candidate PQC algorithms... establishing criteria for algorithm selection."
- **Section 3.2.1**: "Systematic literature analysis (Section 3.1.1) identified candidate algorithms based on NIST standardisation status, implementation maturity, and documented empirical performance. The experimental framework validated these selection criteria..."

**Objective → Results**:
- **Section 4.3.1**: "The experimental evaluation validates the algorithm selection criteria. The selected algorithms, Kyber-512 and Dilithium-2, demonstrate performance characteristics that align with real-time system requirements (Section 4.2.1)."

**Status**: ✅ **TRACEABLE**

---

### Objective 2: Develop modular framework

**Objective → Methodology**:
- **Section 3.1.1**: "Prototype-Based Experimental Design: A high-fidelity framework implemented real-time data pipeline primitives with modular cryptographic components..."
- **Section 3.2.1**: "Framework development implemented a modular architecture with unified interfaces enabling seamless integration of both PQC and classical algorithms. Framework validation runs confirmed successful integration..."

**Objective → Results**:
- **Section 4.3.2**: "The experimental framework (Section 3.3.3) successfully enabled performance evaluation across six algorithms, three environments, and multiple workload configurations. The framework integrated all algorithms through a unified interface..."

**Status**: ✅ **TRACEABLE**

---

### Objective 3: Benchmark PQC and classical algorithms

**Objective → Methodology**:
- **Section 3.1.1**: "Quantitative Performance Measurement: Instrumentation captured fine-grained metrics including latency, throughput, and resource utilisation..."
- **Section 3.2.1**: "Controlled experiments executed across multiple workload configurations captured operation-level performance data (latency, throughput, resource utilisation) for each algorithm."

**Objective → Results**:
- **Section 4.3.3**: "Benchmarking was conducted across all specified performance metrics (Section 4.2), providing quantitative characterisation of algorithm performance under real-time streaming conditions."

**Status**: ✅ **TRACEABLE**

---

### Objective 4: Compare PQC performance against classical

**Objective → Methodology**:
- **Section 3.1.1**: "Statistical Hypothesis Testing: Performance comparisons employed parametric and non-parametric statistical methods with effect size quantification..."
- **Section 3.2.1**: "Pairwise comparisons between PQC and classical algorithms were conducted under identical experimental conditions within the same environment. Statistical analysis of run-level data employed hypothesis testing and effect size quantification."

**Objective → Results**:
- **Section 4.3.4**: "Statistical comparative analysis (Section 4.2.2) confirms the performance advantages identified in Section 4.2.1. All primary pairwise comparisons relevant to the research objectives were statistically significant..."

**Status**: ✅ **TRACEABLE**

---

### Objective 5: Provide engineering recommendations

**Objective → Methodology**:
- **Section 3.1.1**: "Statistical Hypothesis Testing: ... enabling rigorous comparative analysis and supporting engineering recommendations."
- **Section 3.2.1**: "Performance findings were synthesised across algorithms, environments, and workloads. Comprehensive experimental runs identified performance trade-offs, deployment context comparisons, and scalability assessments..."

**Objective → Results**:
- **Section 4.3.5**: "The empirical findings enable formulation of evidence-based engineering recommendations synthesising performance analysis, statistical findings, and deployment considerations (Sections 4.1 and 4.2)."

**Status**: ✅ **TRACEABLE**

---

## 3. Methodology Traceability

### Methodology Components

**Systematic Literature Analysis**:
- **Section 3.1.1**: Overview description
- **Section 3.2.1**: How it addresses Objective 1
- **Section 3.3.1**: Detailed implementation description

**Prototype-Based Experimental Design**:
- **Section 3.1.1**: Overview description
- **Section 3.1.2**: Framework architecture overview
- **Section 3.2.1**: How it addresses Objective 2
- **Section 3.2.2**: How framework represents live systems
- **Section 3.3.3**: Detailed framework implementation

**Quantitative Performance Measurement**:
- **Section 3.1.1**: Overview description
- **Section 3.1.3**: Data collection overview
- **Section 3.2.1**: How it addresses Objective 3
- **Section 3.3.2**: Detailed data collection procedures

**Statistical Hypothesis Testing**:
- **Section 3.1.1**: Overview description
- **Section 3.1.4**: Analysis approach overview
- **Section 3.2.1**: How it addresses Objectives 4 and 5
- **Section 3.3.2**: Detailed statistical methods

**Status**: ✅ **TRACEABLE** - All methodology components are described at overview level (3.1), justified (3.2), and detailed (3.3)

---

## 4. Data Analysis Artefacts Traceability

### Data Collection Artefacts

**What was collected**:
- **Section 3.1.3**: "Operation-level data was collected... Run-level aggregation computed... Cross-run statistics combined..."
- **Section 3.3.2**: Detailed data collection procedures, experimental matrix, environments

**Where results are presented**:
- **Section 4.1**: Summary of data collected (396 experiments, 1,836 runs, 134,621,400 operations)
- **Section 4.2**: Data analysis results

**Status**: ✅ **TRACEABLE**

---

### Statistical Methods Artefacts

**What methods were used**:
- **Section 3.1.4**: "Descriptive statistics... Inferential statistics... Visualisation..."
- **Section 3.3.2**: Detailed statistical methods (t-test, Mann-Whitney U, Cohen's d, Holm-Bonferroni)

**Where results are presented**:
- **Section 4.2.2**: Statistical hypothesis testing results
- **Section 4.3.4**: "All primary pairwise comparisons... were statistically significant (p < 0.05 after Holm-Bonferroni correction), with 59 comparisons showing large effect sizes (|d| ≥ 0.8)..."

**Status**: ✅ **TRACEABLE**

---

### Analysis Pipeline Artefacts

**What pipeline was used**:
- **Section 3.1.4**: Overview of analysis approach
- **Section 3.3.2**: "Data Processing Pipeline: Raw event-level data was processed through a multi-stage analysis pipeline: (1) run aggregation... (2) cross-run statistics... (3) statistical testing... (4) visualisation generation..."

**Where results are presented**:
- **Section 4.1**: Data completeness and coverage
- **Section 4.2**: Analysis results using these methods

**Status**: ✅ **TRACEABLE**

---

## 5. Missing Traceability (If Any)

### Check: Are all objectives explicitly connected to methodology?
✅ **YES** - Section 3.2.1 explicitly describes how each objective is addressed

### Check: Are all objectives explicitly connected to results?
✅ **YES** - Section 4.3 has subsections for each objective

### Check: Is aim connected to objectives?
✅ **YES** - Section 2.3 states connection

### Check: Is aim connected to methodology?
✅ **YES** - Section 3.1.1 references the aim

### Check: Is aim connected to results?
✅ **YES** - Section 4.4 discusses interpretation in relation to research aim

### Check: Are methodology components traceable to detailed procedures?
✅ **YES** - Section 3.1 (overview) → Section 3.3 (detailed)

### Check: Are data analysis artefacts traceable from methodology to results?
✅ **YES** - Section 3.1.3/3.1.4 (overview) → Section 3.3.2 (detailed) → Section 4.1/4.2 (results)

---

## Summary

### ✅ All Traceability Maintained

**Objectives → Methodology**: ✅ Section 3.2.1 explicitly describes how each objective is addressed

**Objectives → Results**: ✅ Section 4.3 has explicit subsections for each objective

**Aim → Objectives**: ✅ Section 2.3 states connection

**Aim → Methodology**: ✅ Section 3.1.1 references aim

**Aim → Results**: ✅ Section 4.4 discusses aim achievement

**Methodology Components**: ✅ Traced through 3.1 (overview) → 3.2 (justification) → 3.3 (detailed)

**Data Analysis Artefacts**: ✅ Traced through 3.1.3/3.1.4 (overview) → 3.3.2 (detailed) → 4.1/4.2 (results)

---

## Conclusion

**Table 3.1 was redundant** - all traceability is maintained through:
1. Section 3.1.1: Methodology components (implicitly address objectives)
2. Section 3.2.1: Explicit description of how each objective is addressed
3. Section 4.3: Explicit interpretation in relation to each objective
4. Section 4.4: Explicit interpretation in relation to research aim

**No traceability lost** by removing the table. The prose descriptions provide better, more complete traceability than the table would have.

