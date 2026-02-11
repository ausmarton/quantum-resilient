# Table 3.1 Removal Rationale

**Date**: 2025-12-15  
**Decision**: Removed Table 3.1 "Mapping of Research Objectives to Methodology"

---

## Issues Identified

### 1. Redundancy
The table duplicated information already well-covered in prose:
- **Section 3.1.1**: Describes the four methodology components and implicitly shows which objectives they address
- **Section 3.2.1**: Explicitly describes how each of the five objectives is addressed through the methodology

### 2. "Statistical Method" Column Problems
The "Statistical Method" column was problematic for several objectives:
- **Objective 1** (Algorithm selection): "Descriptive analysis, validation metrics" - not really statistical methods
- **Objective 2** (Framework development): "Validation metrics, descriptive analysis" - not really statistical methods  
- **Objective 3** (Benchmarking): Descriptive statistics - OK
- **Objective 4** (Comparison): Inferential statistics - OK
- **Objective 5** (Engineering recommendations): "Comparative analysis, effect size quantification" - not really a statistical method per se

### 3. Column Redundancy
- "Methodology Component" duplicates Section 3.1.1
- "Measurement Approach" and "Metrics" duplicate Section 3.2.1
- "Statistical Method" is only appropriate for Objectives 3 and 4

---

## Information Already Covered

### Section 3.1.1 (Research Methodology Overview)
Describes the four methodology components:
- Systematic Literature Analysis (addresses Objective 1)
- Prototype-Based Experimental Design (addresses Objective 2)
- Quantitative Performance Measurement (addresses Objective 3)
- Statistical Hypothesis Testing (addresses Objectives 4 and 5)

### Section 3.2.1 (Methodology Alignment with Research Objectives)
Explicitly describes how each objective is addressed:
- Objective 1: Systematic literature analysis + framework validation
- Objective 2: Framework development with modular architecture
- Objective 3: Controlled experiments capturing performance data
- Objective 4: Pairwise comparisons with statistical analysis
- Objective 5: Synthesis of findings across algorithms, environments, workloads

---

## Decision

**Removed Table 3.1** because:
1. Information is already comprehensively covered in Sections 3.1.1 and 3.2.1
2. Table structure was problematic (Statistical Method column inappropriate for some objectives)
3. Redundancy violates principle of avoiding unnecessary duplication
4. Prose descriptions in Section 3.2.1 provide better traceability than a table

---

## Alternative Considered

Could have simplified the table to only show Objectives 3 and 4 (which actually use statistical methods), but:
- This would be incomplete (missing Objectives 1, 2, 5)
- Still redundant with Section 3.2.1
- Not worth the space

---

## Traceability Maintained

Objective traceability is maintained through:
- Section 3.1.1: Methodology components described
- Section 3.2.1: Explicit description of how each objective is addressed
- Section 4.3: Interpretation in relation to objectives (results connected to objectives)

No table needed - prose provides better, more complete traceability.

