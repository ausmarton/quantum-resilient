# Chapter 3 vs Chapter 4: Comprehensive Analysis

**Date**: 2025-12-15  
**Purpose**: Extract and synthesize requirements from university guidance documents to clarify what belongs in Chapter 3 (Methodology) vs Chapter 4 (Data Analysis and Presentation)

---

## Executive Summary

**Chapter 3 (Methodology)** focuses on **HOW** you will conduct your research: the methods, techniques, framework design, data collection procedures, and validation activities. It describes the research design **before** data collection.

**Chapter 4 (Data Analysis and Presentation)** focuses on **WHAT** you found: processing, analyzing, presenting, and interpreting the actual data collected. It describes what happened **during and after** data collection.

**Key Distinction**: Chapter 3 = "How will I do it?" | Chapter 4 = "What did I find?"

---

## PART 1: CHAPTER 3 REQUIREMENTS (Methodology and Techniques)

### 1.1 Core Purpose of Chapter 3

According to `methodology-and-techniques.pdf`, Chapter 3 should:

1. **Describe the research methodology** - the combination of methods used and how they will be implemented
2. **Justify method selection** - explain why chosen methods are appropriate
3. **Detail research procedures** - provide sufficient detail for replication
4. **Address validity** - demonstrate technical validity and practicability
5. **Map objectives to methods** - show how methodology addresses research objectives

### 1.2 Required Sections for Chapter 3

#### **Section 3.1: Methods and Techniques Selected**

**What to Include:**
- Overview of research methodology (combination of methods)
- Framework architecture overview (high-level description)
- Data collection overview (what will be collected, not actual results)
- Analysis approach overview (what statistical methods will be used, not actual results)
- Mapping of research objectives to methodology (Table showing: Objective → Measurement Approach → Metrics → Statistical Method)

**What NOT to Include:**
- Actual experimental results
- Actual statistical test results
- Actual performance measurements
- Data interpretation or conclusions

**Key Phrases:**
- "The methodology combines..."
- "The framework enables..."
- "Data collection will employ..."
- "The analysis approach will utilize..."

---

#### **Section 3.2: Justification**

**What to Include:**
- How methodology addresses each research objective
- Why the experimental method is appropriate
- How framework represents live production systems
- Why alternative methods are excluded
- Justification for framework design choices

**What NOT to Include:**
- Results from validation activities (those go in 3.3.3)
- Actual performance comparisons
- Evidence from data collection

**Key Questions to Answer:**
1. Why is this method appropriate for addressing the research objectives?
2. How does the framework represent real-world systems?
3. Why were other methods (surveys, case studies, etc.) excluded?
4. What are the limitations of the chosen approach?

---

#### **Section 3.3: Research Procedures**

**What to Include:**

**3.3.1 Framework Implementation**
- Detailed description of framework architecture (5 layers)
- **Methodological justification** for each component (how it controls for confounding variables, ensures repeatability, enables fair comparison)
- Component descriptions with **methodological purpose** (not just implementation details)
- Framework comparison with live production systems
- **Key**: Frame technical details as methodological necessities, not engineering specifications

**3.3.2 Data Collection Procedures**
- **Process description**: How data will be collected (scenario configuration, execution process, aggregation)
- **Data structure**: What format data will be in (JSONL, CSV, etc.)
- **Measurement methodology**: How measurements will be taken (timing precision, resource monitoring)
- **Statistical structure**: Two-level structure (operation-level → run-level → cross-run statistics)
- **What NOT to Include**: Actual collected data, actual results, actual measurements

**3.3.3 Framework Validation**
- **Validation activities** conducted to confirm framework validity
- **Measurement accuracy and precision validation** (how you verified instrumentation)
- **Framework representativeness validation** (how you confirmed framework approximates production)
- **Experimental reproducibility validation** (how you confirmed reproducibility)
- **Implemented vs Planned Capabilities** (what was actually used vs designed)

**3.3.4 Threats to Validity**
- **Internal validity**: Controls for confounding factors
- **Construct validity**: How measurements capture intended characteristics
- **Conclusion validity**: Statistical validity controls
- **External validity**: Generalisability and limitations

**Key Distinction:**
- **Chapter 3**: "We validated that the framework produces accurate measurements"
- **Chapter 4**: "The measurements show that Algorithm X has latency Y"

---

#### **Section 3.4: Ethical Considerations**

**What to Include:**
- Compliance with data protection regulations
- No human participants or personal data
- Synthetic workloads only
- No cryptographic misuse
- Data management practices

---

### 1.3 Key Terminology for Chapter 3

**Use These Terms:**
- "The framework **enables** measurement of..."
- "The methodology **controls for** confounding variables..."
- "Data collection **will employ** telemetry instrumentation..."
- "The analysis approach **will utilize** statistical hypothesis testing..."
- "The framework **is designed to** represent production characteristics..."

**Avoid These Terms (Save for Chapter 4):**
- "The results show..."
- "Analysis revealed..."
- "Measurements indicate..."
- "Statistical tests found..."
- "The data demonstrates..."

---

## PART 2: CHAPTER 4 REQUIREMENTS (Data Analysis and Presentation)

### 2.1 Core Purpose of Chapter 4

According to `data-analysis-and-presentation.pdf`, Chapter 4 should:

1. **Process raw data** into useful information
2. **Present data** in appropriate formats (tables, graphs, figures)
3. **Analyze data** using appropriate techniques
4. **Interpret data** to reach conclusions
5. **Explain findings** in context of research objectives

### 2.2 Required Sections for Chapter 4

#### **Section 4.1: Data Processing and Preparation**

**What to Include:**
- How raw data was processed (aggregation, filtering, transformation)
- Data quality checks (outlier detection, missing data handling)
- Data structure and organization
- Sample sizes and data completeness
- **What NOT to Include**: Detailed methodology of how data was collected (that's Chapter 3)

**Key Distinction:**
- **Chapter 3**: "Data collection employs telemetry instrumentation that captures..."
- **Chapter 4**: "The collected data was processed by aggregating event-level measurements into run-level statistics..."

---

#### **Section 4.2: Data Presentation**

**What to Include:**

**4.2.1 Descriptive Statistics**
- Summary statistics (means, medians, percentiles, standard deviations)
- Performance tables (algorithm comparisons, environment comparisons)
- **Key**: Present the data clearly, but don't interpret yet

**4.2.2 Graphical Representations**
- **Line charts**: Trends over time, scaling behavior
- **Bar charts**: Discrete comparisons (algorithm performance, environment overhead)
- **CDFs (Cumulative Distribution Functions)**: Latency distributions
- **Scatter diagrams**: Correlations between variables
- **Box plots**: Distribution comparisons
- **Histograms**: Distribution shapes

**Selection Criteria for Visualizations:**
- Choose representations that best convey key points
- Use initial plots to understand data "shape"
- Select final representations that support conclusions
- **Avoid**: Misleading scales, inappropriate chart types, cherry-picking data

**What NOT to Include:**
- Detailed explanation of how visualizations were generated (implementation detail)
- Framework architecture descriptions (Chapter 3)

---

#### **Section 4.3: Statistical Analysis**

**What to Include:**

**4.3.1 Hypothesis Testing**
- **Actual test results**: t-test results, Mann-Whitney U results, p-values
- **Significance levels**: Which differences are statistically significant (p < 0.05, p < 0.01)
- **Multiple comparison correction**: Holm-Bonferroni adjustments
- **Interpretation**: What the statistical tests tell us about performance differences

**4.3.2 Effect Size Quantification**
- **Cohen's d values**: Magnitude of performance differences
- **Confidence intervals**: Uncertainty in effect size estimates
- **Practical significance**: Whether differences are practically meaningful
- **Interpretation**: What effect sizes mean for real-world deployment

**4.3.3 Comparative Analysis**
- **Algorithm comparisons**: PQC vs classical performance
- **Environment comparisons**: Native vs containerized vs cloud
- **Workload comparisons**: Different payload sizes, message rates, patterns
- **Cross-cutting insights**: Patterns across multiple dimensions

**Key Distinction:**
- **Chapter 3**: "Statistical analysis will employ t-tests and Mann-Whitney U tests to determine significance..."
- **Chapter 4**: "Mann-Whitney U tests revealed statistically significant differences (p < 0.01) between Kyber-512 and ECDSA P-256..."

---

#### **Section 4.4: Data Interpretation**

**What to Include:**

**4.4.1 Performance Findings**
- **What the data shows**: Actual performance characteristics
- **Patterns and trends**: Observable relationships in the data
- **Anomalies**: Unexpected findings that require explanation
- **Context**: How findings relate to research objectives

**4.4.2 Correlation and Causality**
- **Correlations observed**: Relationships between variables
- **Causal inferences**: What can be inferred about causes
- **Limitations**: What correlations don't tell us
- **Key Principle**: Correlation does not imply causation

**4.4.3 Context and Limitations**
- **Sample limitations**: How sample size affects conclusions
- **Generalizability**: Extent to which findings apply beyond experimental context
- **Confidence in conclusions**: How certain we can be about findings
- **Boundaries**: What contexts findings apply to

**What NOT to Include:**
- Detailed methodology descriptions (Chapter 3)
- Framework implementation details (Chapter 3)
- Engineering recommendations (Chapter 5)

---

#### **Section 4.5: Addressing Research Objectives**

**What to Include:**
- **For each objective**: What the data reveals about that objective
- **Evidence**: Specific findings that address each objective
- **Gaps**: Any objectives not fully addressed by the data
- **Connections**: How findings across objectives relate to each other

**Key Structure:**
- Objective 1: [What data shows about Objective 1]
- Objective 2: [What data shows about Objective 2]
- etc.

---

### 2.3 Key Terminology for Chapter 4

**Use These Terms:**
- "The results show..."
- "Analysis revealed..."
- "Measurements indicate..."
- "Statistical tests found..."
- "The data demonstrates..."
- "Findings suggest..."
- "Comparison reveals..."

**Avoid These Terms (Save for Chapter 3):**
- "The framework enables..."
- "The methodology controls for..."
- "Data collection will employ..."
- "The analysis approach will utilize..."

---

## PART 3: KEY DISTINCTIONS AND DECISION FRAMEWORK

### 3.1 Temporal Distinction

| Aspect | Chapter 3 | Chapter 4 |
|--------|-----------|-----------|
| **Timeframe** | Before/during research design | During/after data collection |
| **Tense** | Future/present (planning) | Past/present (completed/ongoing) |
| **Focus** | "How will we do it?" | "What did we find?" |

---

### 3.2 Content Distinction

| Topic | Chapter 3 | Chapter 4 |
|-------|-----------|-----------|
| **Framework Architecture** | Detailed description with methodological justification | Brief reference if needed for context |
| **Data Collection** | How data will be collected, what will be collected | What data was collected, how it was processed |
| **Statistical Methods** | Which methods will be used, why they're appropriate | Actual test results, p-values, effect sizes |
| **Validation** | Validation activities conducted, how validity was ensured | Validation results if they inform data interpretation |
| **Performance Measurements** | How measurements will be taken, what precision | Actual measured values, performance characteristics |
| **Visualizations** | What types of visualizations will be used | Actual figures, charts, tables with data |
| **Findings** | None - no data yet | All findings, patterns, trends, conclusions |

---

### 3.3 Decision Framework: Where Does This Content Belong?

**Ask These Questions:**

1. **Is this about HOW the research was designed/conducted?**
   - YES → Chapter 3
   - NO → Continue to question 2

2. **Is this about WHAT was found/discovered?**
   - YES → Chapter 4
   - NO → Continue to question 3

3. **Is this about framework design/architecture?**
   - YES → Chapter 3 (if methodological justification) or Appendix (if implementation detail)
   - NO → Continue to question 4

4. **Is this about actual data/results/measurements?**
   - YES → Chapter 4
   - NO → May belong elsewhere (Introduction, Literature Review, etc.)

---

### 3.4 Common Mistakes to Avoid

#### **Mistake 1: Implementation Details in Chapter 4**
- **Wrong**: "Latency measurements employ Rust's `Instant::now()` function..."
- **Right (Chapter 3)**: "Latency measurements employ high-resolution timing instrumentation with nanosecond precision..."
- **Right (Chapter 4)**: "Latency measurements were captured with nanosecond precision, enabling characterization of sub-microsecond differences..."

#### **Mistake 2: Results in Chapter 3**
- **Wrong (Chapter 3)**: "The framework measured Kyber-512 latency at 2.3μs..."
- **Right (Chapter 3)**: "The framework enables measurement of cryptographic operation latency with nanosecond precision..."
- **Right (Chapter 4)**: "Kyber-512 exhibited a median latency of 2.3μs (p50) with a 95th percentile of 3.1μs..."

#### **Mistake 3: Methodology in Chapter 4**
- **Wrong (Chapter 4)**: "We designed the framework with five layers to enable comprehensive telemetry collection..."
- **Right (Chapter 3)**: "The framework consists of five principal layers that enable comprehensive telemetry collection..."
- **Right (Chapter 4)**: "The comprehensive telemetry collection enabled detailed performance characterization across all algorithms..."

#### **Mistake 4: Statistical Methods vs Results**
- **Wrong (Chapter 4)**: "We used t-tests to compare algorithms..." (this is methodology)
- **Right (Chapter 3)**: "Statistical analysis employs independent samples t-tests to compare algorithm performance..."
- **Right (Chapter 4)**: "Independent samples t-tests revealed statistically significant differences (t=4.32, p<0.01) between Kyber-512 and ECDSA P-256..."

---

## PART 4: SPECIFIC GUIDANCE FOR YOUR DISSERTATION

### 4.1 Chapter 3 Structure (Based on Your Current Draft)

**Section 3.1: Methods and Techniques Selected** ✅
- Overview of methodology
- Framework architecture overview
- Data collection overview
- Analysis approach overview
- **ADD**: Explicit mapping table (Objective → Measurement → Metrics → Statistical Method)

**Section 3.2: Justification** ✅
- Methodology alignment with objectives
- Framework representation of production systems
- Justification for experimental method
- Exclusion of alternative methods

**Section 3.3: Research Procedures** ✅
- Framework implementation (with methodological framing)
- Data collection procedures
- Framework validation
- Threats to validity

**Section 3.4: Ethical Considerations** ✅

---

### 4.2 Chapter 4 Structure (Recommended)

**Section 4.1: Data Collection and Processing**
- Overview of collected data (sample sizes, completeness)
- Data processing pipeline (aggregation, quality checks)
- Data structure and organization

**Section 4.2: Performance Characterization**
- Algorithmic performance (native baseline)
- Performance distributions (CDFs, percentiles)
- Performance tables (comparative statistics)

**Section 4.3: Statistical Analysis**
- Hypothesis testing results (actual p-values, test statistics)
- Effect size quantification (Cohen's d, confidence intervals)
- Statistical significance interpretation

**Section 4.4: Comparative Analysis**
- PQC vs Classical comparisons
- Environment comparisons (native vs containerized vs cloud)
- Workload impact analysis (payload size, message rate, patterns)

**Section 4.5: Interpretation and Findings**
- Key findings for each research objective
- Patterns and trends observed
- Anomalies and unexpected results
- Context and limitations

---

### 4.3 Content Migration Checklist

**Move FROM Chapter 4 TO Chapter 3:**
- [ ] Framework architecture descriptions (if not already there)
- [ ] Data collection methodology details
- [ ] Statistical method descriptions (which methods, why chosen)
- [ ] Measurement methodology (how measurements are taken)
- [ ] Validation activities (how validation was conducted)

**Keep IN Chapter 4:**
- [ ] Actual performance measurements
- [ ] Statistical test results (p-values, test statistics)
- [ ] Effect sizes (Cohen's d values)
- [ ] Visualizations (CDFs, comparison charts, tables)
- [ ] Data interpretation and findings
- [ ] Patterns and trends in the data

**Remove FROM Chapter 4:**
- [ ] Implementation details (function names, file formats, directory structures)
- [ ] Framework design rationale (unless needed for data interpretation context)
- [ ] Methodology descriptions (how data was collected - that's Chapter 3)

---

## PART 5: VALIDATION CHECKLIST

### Chapter 3 Validation

- [ ] Does it describe HOW research will be conducted?
- [ ] Does it justify method selection?
- [ ] Does it provide sufficient detail for replication?
- [ ] Does it address validity threats?
- [ ] Does it map objectives to methodology?
- [ ] Does it avoid presenting actual results?
- [ ] Does it use future/present tense (planning)?
- [ ] Does it frame technical details as methodological necessities?

### Chapter 4 Validation

- [ ] Does it present WHAT was found?
- [ ] Does it include actual data/results/measurements?
- [ ] Does it present statistical test results?
- [ ] Does it interpret findings in context?
- [ ] Does it address research objectives with evidence?
- [ ] Does it avoid detailed methodology descriptions?
- [ ] Does it use past/present tense (completed/ongoing)?
- [ ] Does it focus on data interpretation rather than methodology?

---

## PART 6: SUMMARY TABLES

### Table 1: Content Placement Guide

| Content Type | Chapter 3 | Chapter 4 | Notes |
|--------------|-----------|-----------|-------|
| Framework architecture | ✅ Detailed with methodological justification | ❌ Brief reference only if needed | Chapter 3 explains WHY design choices were made |
| Data collection process | ✅ How data will be collected | ✅ What data was collected, how processed | Different aspects |
| Statistical methods | ✅ Which methods, why chosen | ✅ Actual test results | Methodology vs results |
| Performance measurements | ✅ How measurements taken | ✅ Actual measured values | Methodology vs data |
| Validation activities | ✅ How validation conducted | ❌ Results only if inform interpretation | Chapter 3 focuses on process |
| Visualizations | ❌ What types will be used | ✅ Actual figures with data | Planning vs presentation |
| Findings/conclusions | ❌ None | ✅ All findings | No data in Chapter 3 |

### Table 2: Language Patterns

| Chapter 3 Patterns | Chapter 4 Patterns |
|-------------------|-------------------|
| "The framework enables..." | "The results show..." |
| "Data collection will employ..." | "The collected data was..." |
| "Statistical analysis will utilize..." | "Statistical tests revealed..." |
| "The methodology controls for..." | "Analysis indicates..." |
| "The framework is designed to..." | "Measurements demonstrate..." |
| "Validation activities confirmed..." | "Findings suggest..." |

---

## CONCLUSION

**Chapter 3 (Methodology)** = The research design and procedures. It answers: "How will I conduct this research?" It describes the framework, methods, techniques, and validation **before** data collection.

**Chapter 4 (Data Analysis and Presentation)** = The research findings. It answers: "What did I discover?" It presents, analyzes, and interprets the actual data collected **during** the research.

**Key Principle**: If it's about **HOW** you did it → Chapter 3. If it's about **WHAT** you found → Chapter 4.

**Validation Question**: Could another researcher replicate your research using Chapter 3 alone? If yes, Chapter 3 is complete. Could they understand your findings using Chapter 4? If yes, Chapter 4 is complete.

