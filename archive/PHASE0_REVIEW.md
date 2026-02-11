# Phase 0: Pre-Implementation Review

**Date**: 2025-12-15  
**Status**: In Progress

---

## Task 0.1: Review Current Section 3.1 Content

### Current Content Analysis

**Location**: Section 3.1 "Methods and techniques selected"  
**Subsection**: 3.1.1 "Evidence and data"

**What's Currently There**:
- Brief mention of experimental approach
- Reference to high-level diagram (image1)
- Mention of prototype framework
- Reference to OQS project
- Brief mention of metrics: latency, throughput, resource utilisation
- Future tense: "will adopt", "will be collected", "will contribute"

### Requirements Check

| Requirement | Source | Current Status | Action Needed |
|------------|--------|---------------|---------------|
| Overview of everything | Chapter3-review | ⚠️ Partial | Needs expansion |
| Methodology overview | methodology-and-techniques.txt | ⚠️ Partial | Needs expansion |
| Framework architecture overview | methodology-and-techniques.txt | ❌ Missing | **ADD** |
| Data collection overview | methodology-and-techniques.txt | ⚠️ Partial | Needs expansion |
| Analysis approach overview | methodology-and-techniques.txt | ❌ Missing | **ADD** |
| Objective-to-methodology mapping table | methodology-and-techniques.txt | ❌ Missing | **ADD** |
| Figure description (3.1.1) | Chapter3-review | ❌ Missing | **ADD** - describe each block |

### Findings

**Missing Elements**:
1. Framework architecture overview (high-level description)
2. Data collection overview (what will be collected, not results)
3. Analysis approach overview (what statistical methods will be used)
4. Objective-to-methodology mapping table
5. Description of figure in 3.1.1 (if figure exists, describe each block)

**Issues**:
- Uses future tense (should be past tense - work is complete)
- Too brief - needs more detail for "overview of everything"
- No explicit mapping to research objectives

---

## Task 0.2: Review Current Section 3.2 Content

### Current Content Analysis

**Location**: Section 3.2 "Justification"

**What's Currently There**:
- Justification for experimental methodology
- Explanation of why controlled experimental evaluation is essential
- Discussion of why live production platform was impractical
- Reference to closed-system prototype
- Discussion of OQS project implementations
- **Good**: Explains why other methodologies were excluded (theoretical, surveys, case studies)
- Mentions internal and external validity

### Requirements Check

| Requirement | Source | Current Status | Action Needed |
|------------|--------|---------------|---------------|
| What/why with respect to objectives | Chapter3-review | ⚠️ Partial | Needs explicit mapping to each objective |
| Why exclude methods (survey/observations) | Chapter3-review | ✅ Covered | Already explains exclusion |
| How framework represents live/real-world systems | Chapter3-review | ⚠️ Partial | Needs explicit explanation |
| Justify method selection | methodology-and-techniques.txt | ✅ Covered | Already justified |
| Justify experimental method | methodology-and-techniques.txt | ✅ Covered | Already justified |

### Findings

**Missing Elements**:
1. Explicit mapping to each research objective (what data from runs addresses each objective)
2. Explicit explanation of how framework represents live/real-world systems
3. More detail on framework design choices justification

**Good Elements**:
- Already explains why surveys/observations excluded
- Already justifies experimental method
- Already discusses alternative methods

---

## Task 0.3: Identify Code Snippets

### Code/Implementation Details Found

**Location**: Section 3.3.3 "Framework Implementation"

**Items Found**:
1. **Line 389**: "implemented in Rust" - mentions language
2. **Line 389**: "CryptoAdapter trait interface" - mentions specific implementation detail
3. **Line 389**: "Instant::now() monotonic clock" - mentions specific function
4. **Line 389**: "getrusage, /proc filesystem" - mentions specific system calls
5. **Line 391**: "implemented in Python" - mentions language
6. **Line 391**: "PyO3 bindings" - mentions specific library
7. **Line 391**: "YAML configuration files" - mentions file format
8. **Line 391**: "JSONL, CSV/JSON" - mentions file formats
9. **Line 403**: "Rust's monotonic clock primitives" - mentions specific implementation

### Assessment

**Should Move to Appendices**:
- Specific function names (Instant::now(), getrusage)
- Specific library names (PyO3)
- Specific file formats (JSONL, CSV/JSON) - these might be OK if they're part of methodology
- Implementation language mentions (Rust, Python) - these might be OK if they're part of methodology justification

**Keep in Main Text** (with functional framing):
- "CryptoAdapter trait interface" - but frame as "unified interface" (functionality)
- "monotonic clock" - but frame as "high-precision timing" (functionality)
- "system-level monitoring" - keep (methodological purpose)

**Recommendation**: 
- Reframe implementation details as functionality
- Move specific function/library names to appendices if they're not critical
- Focus on "what it does" not "how it's implemented"

---

## Task 0.4: Missing Concepts (3.e. and 5.f. Items)

### Already Documented in FINAL_VERIFICATION.md

**3.e. Items (New Concepts)** - 8 identified:
1. Performance Metrics Framework
2. Inferential Comparison Scope
3. Throughput vs Latency Efficiency
4. Scaling Factor concepts
5. Resource Utilisation Analysis
6. Deployment Implications
7. Migration Strategy concepts

**5.f. Items (New Terminology)** - 20+ identified:
- Statistical: p50/p95/p99, CDF, violin/box plots, Holm-Bonferroni, Cohen's d, effect size, inferential/descriptive, hardware confounding
- Performance: scaling factor, sub-linear scaling, throughput saturation, capacity limits
- System: containerisation, virtualisation, on-premise, horizontal scaling
- Deployment: bandwidth-constrained, certificate infrastructure, capacity planning, phased migration, hybrid approaches, transitional deployments
- Measurement: resource utilisation, CPU utilisation, memory consumption, deployment capacity

**Status**: ✅ Already systematically scanned and documented

---

## Summary of Phase 0 Findings

### Section 3.1 Needs:
1. ✅ Expand to provide "overview of everything"
2. ✅ Add framework architecture overview
3. ✅ Add data collection overview
4. ✅ Add analysis approach overview
5. ✅ Add objective-to-methodology mapping table
6. ✅ Describe figure in 3.1.1 (if exists)
7. ✅ Fix future tense → past tense

### Section 3.2 Needs:
1. ✅ Add explicit mapping to each research objective
2. ✅ Add explicit explanation of how framework represents live/real-world systems
3. ✅ Enhance framework design choices justification

### Code Snippets:
1. ✅ Reframe implementation details as functionality
2. ✅ Move specific function/library names to appendices (if not critical)
3. ✅ Focus on "what it does" not "how it's implemented"

### Missing Concepts:
1. ✅ Already documented in FINAL_VERIFICATION.md
2. ✅ Tasks 3.8-3.15 added to action plan

---

## Next Steps

1. ✅ Phase 0 complete - findings documented
2. ⏭️ Proceed to Phase 1: Chapter 3 Enhancements
   - Task 1.0: Enhance Section 3.1
   - Task 1.0a: Enhance Section 3.2
   - Task 1.1-1.6: Other Chapter 3 enhancements

---

**Status**: ✅ Phase 0 Review Complete - Ready for Phase 1

