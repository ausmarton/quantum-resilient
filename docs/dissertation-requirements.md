# Dissertation Requirements and Claims

**Source**: Extracted from FERNANDES_H2807295_F87_dissertation (1).docx  
**Date**: 2025-12-14  
**Purpose**: Comprehensive reference for all dissertation requirements and claims that must be supported by data analysis

---

## Executive Summary

This document extracts and consolidates all requirements, claims, and objectives from the dissertation document to ensure complete alignment between data analysis outputs and dissertation needs.

**Key Principle**: Every claim made in the dissertation must be supported by:
1. Appropriate data collection
2. Statistical analysis with proper rigor
3. Visualizations/figures where needed
4. Interpretation and discussion

---

## Research Questions

Based on the dissertation abstract and content:

1. **PQC Performance Impact**: What is the operational impact of NIST-standardized PQC algorithms on real-time data streaming pipelines?
2. **Latency Overhead**: How much latency overhead do PQC algorithms incur compared to classical algorithms?
3. **Performance Characteristics**: How do PQC algorithms (ML-KEM/Kyber, ML-DSA/Dilithium) compare to classical counterparts (RSA-2048, ECDSA-P256, ECDHE-P256)?
4. **Resource Utilization**: What are the CPU and memory utilization differences between PQC and classical algorithms?
5. **Environment Impact**: How does performance vary across native, containerized (Minikube), and cloud (GCP) environments?
6. **Scaling Behavior**: How do algorithms scale horizontally in production deployments?
7. **Size Inflation**: What is the impact of key and signature size inflation (2-45x increases) on bandwidth-constrained environments?

---

## Key Claims from Dissertation

### Performance Claims

1. **PQC Key Generation Overhead**
   - **Claim**: "PQC key generation incurs measurable but operationally manageable latency overhead (1 - 3 microseconds) relative to classical algorithms"
   - **Support Required**:
     - Latency measurements for key generation operations
     - Comparison: PQC vs Classical (RSA, ECDSA)
     - Statistical significance testing
     - Effect size calculation (Cohen's d)

2. **Dilithium vs ECDSA Performance**
   - **Claim**: "Dilithium signature generation demonstrated comparable performance to ECDSA at equivalent security levels"
   - **Support Required**:
     - Signature generation latency comparison
     - Statistical tests (t-test, Mann-Whitney U)
     - Effect size analysis
     - Distribution analysis (CDFs)

3. **User-Perceived Impact**
   - **Claim**: "Negligible absolute impact on user-perceived performance"
   - **Support Required**:
     - Total latency analysis (including queue delay)
     - Sub-second processing verification
     - Real-world context interpretation

### Statistical Claims

1. **Effect Sizes**
   - **Claim**: "Large effect sizes (Cohen's d > 1.2, p < 0.001)"
   - **Support Required**:
     - Cohen's d calculation for all comparisons
     - P-value reporting with multiple comparison correction
     - Confidence intervals (95% CI)

2. **Statistical Significance**
   - **Claim**: Statistical analysis using "independent samples t-tests and Mann-Whitney U tests"
   - **Support Required**:
     - Welch's t-test results (for unequal variances)
     - Mann-Whitney U test results (non-parametric)
     - Multiple comparison correction (Holm-Bonferroni)
     - All p-values < 0.05 threshold

### Experimental Design Claims

1. **Scale of Experiments**
   - **Claim**: "810 cryptographic operations across 8 algorithms with comprehensive instrumentation"
   - **Support Required**:
     - Complete experiment matrix coverage
     - All algorithms tested (5 PQC + 3 Classical = 8 total)
     - All payload sizes, rates, and patterns tested
     - Comprehensive telemetry (latency, throughput, CPU, memory)

2. **Deterministic Conditions**
   - **Claim**: "Controlled experimental evaluation... under deterministic conditions"
   - **Support Required**:
     - Experiment isolation verification
     - Hardware consistency checks
     - Reproducibility documentation

### Size and Bandwidth Claims

1. **Key/Signature Size Inflation**
   - **Claim**: "Key and signature size inflation (2 - 45x increases) represents the primary deployment challenge"
   - **Support Required**:
     - Ciphertext size measurements
     - Signature size measurements
     - Comparison tables (PQC vs Classical)
     - Bandwidth impact analysis

---

## Required Analysis Outputs

### 1. Performance Comparison Tables

**Algorithm Performance (Native Baseline)**
- Mean latency (p50, p95, p99) for each algorithm
- Throughput (ops/sec) for each algorithm
- Statistical significance indicators
- Effect sizes (Cohen's d)

**Environment Comparison**
- Native vs Minikube overhead (%)
- Native vs GCP overhead (%)
- Minikube vs GCP comparison
- Statistical significance for each comparison

**Payload Size Impact**
- Latency vs payload size (256B, 1KB, 4KB, 16KB)
- Throughput vs payload size
- Scaling factors (% per KB)

### 2. Statistical Test Results

**Required Tests for Each Comparison**:
- Kolmogorov-Smirnov test (distribution shape)
- Mann-Whitney U test (distribution location)
- Welch's t-test (mean difference)
- Cohen's d effect size with 95% CI
- Holm-Bonferroni corrected p-values

**Comparisons Required**:
- PQC vs Classical (each PQC algo vs each classical)
- Environment comparisons (native vs minikube vs GCP)
- Payload size impact (within each algorithm)
- Workload pattern impact (constant vs burst)

### 3. Visualizations

**Required Figures**:
1. **Latency Distribution (CDF)**
   - Combined CDF for all algorithms
   - Separate CDFs by algorithm
   - Environment comparison panels
   - Payload size panels

2. **Performance Comparison**
   - Bar charts: Mean latency by algorithm
   - Box plots: Latency distribution by algorithm
   - Violin plots: Distribution shapes

3. **Environment Overhead**
   - Overhead percentage charts
   - Native baseline normalization
   - Statistical significance indicators

4. **Scaling Analysis** (if applicable)
   - Throughput vs replicas
   - Latency vs replicas
   - Scaling efficiency curves

5. **Queue Delay Analysis**
   - Queue delay vs total latency
   - Queue delay distribution
   - Queue delay by workload rate

6. **Resource Utilization**
   - CPU utilization by algorithm
   - Memory utilization by algorithm
   - Resource efficiency (ops/CPU-second)

### 4. Size and Bandwidth Analysis

**Required Outputs**:
- Ciphertext size comparison table
- Signature size comparison table
- Size inflation factors (PQC/Classical ratio)
- Bandwidth impact estimates

---

## Dissertation Structure Mapping

### Chapter 5: Results and Analysis

**5.1 Algorithmic Performance (Native Baseline)**
- **Requirement**: Complete performance comparison for all algorithms
- **Data Needed**: All native experiments with 5 runs each
- **Outputs**: 
  - Performance tables
  - CDF plots
  - Statistical test results

**5.2 Deployment Context Analysis**

**5.2.1 Containerization Overhead (Minikube)**
- **Requirement**: Quantify containerization overhead
- **Data Needed**: Native vs Minikube comparison
- **Outputs**: Overhead percentage, statistical significance

**5.2.2 Production Scaling (GCP)**
- **Requirement**: Horizontal scaling analysis
- **Data Needed**: Scaling experiments (replicas 1, 2, 4, 8)
- **Outputs**: Scaling curves, efficiency metrics

**5.3 Cross-Environment Insights**
- **Requirement**: Cross-environment comparison and recommendations
- **Data Needed**: All three environments
- **Outputs**: Environment comparison plots, deltas

### Chapter 6: Discussion

**6.1 Algorithm Selection Guidelines**
- **Requirement**: Provide algorithm recommendations
- **Data Needed**: Complete performance analysis
- **Outputs**: Performance comparison, use case recommendations

**6.2 Deployment Strategy Guidelines**
- **Requirement**: Provide deployment recommendations
- **Data Needed**: Scaling analysis, environment comparison
- **Outputs**: Scaling recommendations, environment selection guide

**6.3 Limitations and Future Work**
- **Requirement**: Document limitations accurately
- **Data Needed**: All experimental data
- **Outputs**: Limitations documentation, future work suggestions

---

## Success Criteria

### Data Completeness
- ✅ All experiments have raw data (396 total with ECDHE: 120 native + 138 minikube + 138 gcp)
- ⚠️ All experiments need summary.json files (in progress)
- ⚠️ All experiments need aggregated statistics

### Statistical Rigor
- ✅ 5 runs per baseline configuration
- ✅ 3 runs per scaling configuration
- ✅ Statistical tests implemented
- ✅ Effect size calculation implemented
- ✅ Multiple comparison correction implemented

### Visualization Quality
- ✅ CDF plots generated
- ⚠️ All required comparison plots needed
- ⚠️ Publication-quality formatting needed

### Claim Support
- ⚠️ All performance claims need statistical backing
- ⚠️ All effect size claims need calculation
- ⚠️ All environment claims need comparison data

---

## Verification Checklist

### Performance Claims
- [ ] PQC key generation overhead quantified (1-3μs claim)
- [ ] Dilithium vs ECDSA comparison complete
- [ ] User-perceived impact assessed
- [ ] All algorithms compared (8 total)

### Statistical Claims
- [ ] Cohen's d > 1.2 verified for key comparisons
- [ ] P < 0.001 verified for key comparisons
- [ ] All tests use appropriate methods (t-test, Mann-Whitney U)
- [ ] Multiple comparison correction applied

### Experimental Design Claims
- [ ] 810 operations verified (or current count documented)
- [ ] 8 algorithms tested
- [ ] Comprehensive instrumentation verified
- [ ] Deterministic conditions verified

### Size and Bandwidth Claims
- [ ] Key size inflation measured (2-45x range)
- [ ] Signature size inflation measured
- [ ] Bandwidth impact analyzed

---

## Next Steps

1. **Complete Summary Generation**: Generate all experiment summaries (396 total with ECDHE: 120 native + 138 minikube + 138 gcp)
2. **Run Complete Analysis**: Generate aggregated stats, visualizations, hypothesis tests
3. **Verify Claim Support**: Check each claim against generated data
4. **Create Interpretation Document**: Document findings supporting each claim
5. **Generate Final Figures**: Create publication-quality visualizations
6. **Requirements Compliance Check**: Verify all requirements met

---

**Last Updated**: 2025-12-14  
**Status**: Living document - update as analysis progresses
