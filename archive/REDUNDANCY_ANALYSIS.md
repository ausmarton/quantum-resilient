# Redundancy Analysis Report

**Date**: 2025-12-15  
**Purpose**: Identify redundancies, duplicated ideas, and repeated terminology/acronym explanations

---

## CATEGORY 1: REPEATED TERMINOLOGY AND ACRONYM EXPANSIONS

### 1.1 Post-Quantum Cryptography (PQC)
**Found in:**
- Abstract: "post-quantum cryptography (PQC)"
- Section 1.1: "post-quantum cryptography (PQC)" (line 218, 220)
- Section 1.2: "PQC protocols" (line 232)
- Section 2.1: "PQC algorithms" (line 256)
- Section 2.2: "post-quantum cryptography (PQC)" (line 268)
- Glossary: "Post-Quantum Cryptography (PQC)" (line 172)

**Issue**: Acronym expanded 6+ times. After first expansion, should use "PQC" only.

**Recommendation**: Expand once in Abstract or Section 1.1, then use "PQC" throughout.

---

### 1.2 Anti-Money Laundering (AML)
**Found in:**
- Abstract: "anti-money laundering (AML)"
- Section 1.1: "anti-money laundering (AML)" (line 214)
- Section 1.2: "anti-money laundering (AML)" (line 234)
- Section 2.1: "Anti-Money Laundering (AML)" (line 260)
- Section 2.2: "anti-money laundering (AML)" (line 268, 284, 288, 309)
- Glossary: "Anti-Money Laundering (AML)" (line 100)

**Issue**: Acronym expanded 8+ times.

**Recommendation**: Expand once in Abstract or Section 1.1, then use "AML" only.

---

### 1.3 Deployment Environments
**Found in:**
- Section 3.1.2: "Bare-metal, Local-K8s, Cloud-K8s" (line 371)
- Section 3.1.3: "Bare-metal, Local-K8s, Cloud-K8s" (line 379)
- Section 3.3.2: Full definitions with explanations (line 463)
- Section 4.1.1: "Bare-metal, Local-K8s, Cloud-K8s" (line 557, 565)
- Table 3.2: Full definitions (line 465-473)
- Table 4.0: Full definitions (line 562-570)

**Issue**: Environments defined/explained 6+ times with full terms.

**Recommendation**: Define once in Section 3.3.2 (Table 3.2), then use labels (Bare-metal, Local-K8s, Cloud-K8s) throughout.

---

### 1.4 Algorithm Names
**Found in:**
- Abstract: "ML-KEM/Kyber and ML-DSA/Dilithium" vs "RSA-2048, ECDSA-P256, ECDHE-P256"
- Section 1.1: "ML-KEM (Kyber) and ML-DSA (Dilithium)" (line 228)
- Section 2.2.2: "ML-KEM (Kyber)" and "ML-DSA (Dilithium)" (line 278)
- Section 2.2.3: "Kyber-512" and "Dilithium-2" (line 311)
- Section 4.1.1: "Kyber-512, Dilithium-2" vs "RSA-2048, ECDSA P-256, ECDHE P-256" (line 563)

**Issue**: Inconsistent naming (ML-KEM vs Kyber, ML-DSA vs Dilithium, ECDSA-P256 vs ECDSA P-256).

**Recommendation**: Standardize on one form (e.g., "Kyber-512" and "Dilithium-2" for PQC, "RSA-2048" and "ECDSA P-256" for classical) after first mention.

---

## CATEGORY 2: DUPLICATED CONCEPT EXPLANATIONS

### 2.1 Tail Latency Definition
**Found in:**
- Section 2.2.3: Full definition and explanation (line 317) - **CORRECT LOCATION**
- Section 4.2.1: "tail latency (p95, p99) is critical" (line 596, 617) - Uses term without redefining

**Status**: ✅ **GOOD** - Defined once in Section 2.2.3, referenced in Chapter 4.

---

### 2.2 Two-Level Data Structure
**Found in:**
- Section 3.1.3: "Operation-level data... Run-level aggregation... Cross-run statistics" (line 379)
- Section 3.1.4: "two-level structure: operation-level measurements were aggregated into run-level statistics" (line 383)
- Section 3.3.2: Detailed explanation (line 449, 475)
- Section 4.1.1: "two-level structure (operation-level measurements → run-level aggregates → cross-run statistics)" (line 559)

**Issue**: Explained 4 times with similar wording.

**Recommendation**: Define once in Section 3.3.2, then reference briefly in Section 4.1.1.

---

### 2.3 Performance Metrics Framework
**Found in:**
- Section 2.2.3: "Performance Metrics Selection" with latency, throughput, resource utilisation (line 315-319)
- Section 3.3.2: "Performance Metrics" with detailed explanation (line 451)
- Section 4.2: "Performance Metrics Framework" with similar explanation (line 590)

**Issue**: Performance metrics explained 3 times with overlapping content.

**Recommendation**: Define once in Section 2.2.3, reference in Section 3.3.2, use in Section 4.2 without redefinition.

---

### 2.4 Statistical Methods
**Found in:**
- Section 3.1.4: "parametric (independent samples t-test) and non-parametric (Mann-Whitney U) hypothesis tests with effect size quantification (Cohen's d) and multiple comparison correction (Holm-Bonferroni)" (line 383)
- Section 3.3.1: Detailed explanation of each method (line 443)
- Section 3.3.2: "Statistical testing applied hypothesis tests (Welch's t-test, Mann-Whitney U) with effect size quantification (Cohen's d)" (line 475)
- Section 4.2.2: "Hypothesis tests (Welch's t-test, Mann-Whitney U) with effect size quantification (Cohen's d) and multiple comparison correction (Holm-Bonferroni)" (line 637)

**Issue**: Statistical methods explained 4 times with similar wording.

**Recommendation**: Define once in Section 3.3.1, reference briefly elsewhere.

---

### 2.5 Experimental Matrix Parameters
**Found in:**
- Section 3.3.2: Full explanation of payload sizes, workload rates, patterns, durations (line 461)
- Section 4.1.1: "payload sizes (256B, 1KB, 4KB, 16KB), workload rates (100, 500, 2,000, 10,000 msg/s), workload patterns (constant, burst), and durations (30s primary, 300s extended)" (line 567)

**Status**: ✅ **GOOD** - Defined in Section 3.3.2, listed in Section 4.1.1 (appropriate).

---

### 2.6 Data Processing Pipeline
**Found in:**
- Section 3.1.4: Overview (line 383)
- Section 3.3.2: Detailed explanation (line 475)
- Section 4.1.1: Summary (line 586)

**Issue**: Explained 3 times with similar content.

**Recommendation**: Define once in Section 3.3.2, reference briefly in Section 4.1.1.

---

### 2.7 Framework Architecture Layers
**Found in:**
- Section 3.1.2: Five layers listed (line 371)
- Section 3.3.3: Detailed explanation of three functional layers (line 491-497)

**Issue**: Different levels of detail, but some overlap. Section 3.1.2 mentions 5 layers, Section 3.3.3 describes 3 functional layers.

**Recommendation**: Clarify relationship between "five principal layers" and "three functional layers" or consolidate.

---

### 2.8 Environment Comparison Methodology
**Found in:**
- Section 3.3.6: "Inferential comparisons" vs "Descriptive comparisons" with detailed explanation (line 537-539)
- Section 4.2.3: Similar explanation repeated (line 649)

**Issue**: Methodology explained twice with similar wording.

**Recommendation**: Define once in Section 3.3.6, reference in Section 4.2.3.

---

## CATEGORY 3: REDUNDANT PHRASES AND VERBOSE EXPLANATIONS

### 3.1 "as defined in Section 3.3.2"
**Found in:**
- Line 557: "as defined in Section 3.3.2"
- Line 561: "defined in Section 3.3.2"
- Line 565: "as defined in Section 3.3.2"
- Line 567: "as defined in Section 3.3.2"

**Issue**: Repeated 4 times in Section 4.1.1.

**Recommendation**: Use once at start of section, then just reference "Section 3.3.2" or omit.

---

### 3.2 "as described in Section X"
**Found in:**
- Line 353: "Detailed descriptions... are provided in Section 3.3"
- Line 371: "Detailed descriptions... are provided in Section 3.3.3"
- Line 379: "Section 3.3.2 describes the data collection methodology in detail"
- Line 383: "Section 3.3.2 details the analysis approach"
- Line 613: "Algorithm equivalence and comparison methodology are described in Section 2.2.3"
- Line 637: "as described in Section 3.3.1 and Section 3.3.2"

**Issue**: Overused phrase pattern.

**Recommendation**: Vary language or reduce frequency.

---

### 3.3 Repetitive Performance Characterizations
**Found in:**
- Multiple places: "comparable or superior performance"
- Multiple places: "lower latency than classical alternatives"
- Multiple places: "statistically significant and practically meaningful"

**Issue**: Same phrases repeated multiple times.

**Recommendation**: Vary wording or consolidate statements.

---

### 3.4 Verbose Explanations of Same Concept
**Example - Latency Metrics:**
- Section 2.2.3: "Latency metrics (median p50, 95th percentile p95, 99th percentile p99) are selected because..."
- Section 3.3.2: "Per-operation efficiency (latency): The time required to complete a single cryptographic operation, measured as median (p50), 95th percentile (p95), and 99th percentile (p99) latencies..."
- Section 4.2: "Per-operation efficiency (latency): The time required to complete a single cryptographic operation, measured as median (p50), 95th percentile (p95), and 99th percentile (p99) latencies..."

**Issue**: Same explanation repeated 3 times with minor variations.

**Recommendation**: Define once, reference elsewhere.

---

## CATEGORY 4: DUPLICATED CONTENT ACROSS SECTIONS

### 4.1 Framework Overview vs Detailed Description
**Found in:**
- Section 3.1.2: High-level overview of 5 layers (line 371)
- Section 3.3.3: Detailed description of 3 functional layers (line 491-497)

**Issue**: Different levels of detail, but some conceptual overlap.

**Recommendation**: Ensure Section 3.1.2 is truly overview, Section 3.3.3 is detailed.

---

### 4.2 Algorithm Selection Criteria
**Found in:**
- Section 2.2.3: Detailed criteria and rationale (line 286-319)
- Section 4.3.1: Summary of validation (line 736-740)

**Status**: ✅ **GOOD** - Different purposes (selection vs validation).

---

### 4.3 Environment Specifications
**Found in:**
- Table 3.2: Full specifications (line 465-473)
- Table 4.0: Similar content (line 562-570)

**Issue**: Table 4.0 duplicates Table 3.2 content.

**Recommendation**: Remove Table 4.0, reference Table 3.2 in Chapter 4.

---

## CATEGORY 5: GLOSSARY REDUNDANCY

### 5.1 Terms Defined in Glossary and Text
**Found in:**
- Glossary defines: PQC, AML, Kyber, Dilithium, Latency, Throughput, etc.
- Text also defines/explains these terms

**Issue**: Some terms explained in both glossary and text.

**Recommendation**: Glossary should be reference only; definitions in text should be sufficient.

---

## SUMMARY OF RECOMMENDATIONS

### High Priority (Significant Redundancy)
1. **Remove Table 4.0** - Duplicates Table 3.2
2. **Consolidate Performance Metrics explanations** - Currently in 3 places
3. **Consolidate Statistical Methods explanations** - Currently in 4 places
4. **Reduce "as defined in Section 3.3.2" repetitions** - 4 times in Section 4.1.1
5. **Standardize algorithm naming** - Inconsistent forms used

### Medium Priority (Moderate Redundancy)
6. **Consolidate Two-Level Data Structure explanation** - 4 times
7. **Consolidate Data Processing Pipeline explanation** - 3 times
8. **Reduce acronym expansions** - PQC (6+), AML (8+)
9. **Vary repetitive phrases** - "comparable or superior", "statistically significant and practically meaningful"

### Low Priority (Minor Redundancy)
10. **Clarify Framework Layers** - 5 layers vs 3 functional layers
11. **Reduce "as described in Section X" frequency**
12. **Review Glossary vs Text definitions**

---

## ESTIMATED WORD COUNT SAVINGS

- Table 4.0 removal: ~50 words
- Consolidate Performance Metrics: ~150 words
- Consolidate Statistical Methods: ~200 words
- Reduce Section 4.1.1 repetitions: ~50 words
- Reduce acronym expansions: ~100 words
- Consolidate other explanations: ~200 words

**Total Estimated Savings: ~750 words**

