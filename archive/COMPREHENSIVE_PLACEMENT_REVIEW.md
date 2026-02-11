# Comprehensive Placement Review

**Date**: 2025-12-15  
**Purpose**: Review all content placement against supervisor feedback and university guidance

---

## ISSUE 1: Tail Latency Content Placement

### Current Location
- **Section 1.1** (line 230): "Tail Latency and Real-Time Performance Metrics" paragraph

### Supervisor Feedback
- "These percentiles are critical for real-time systems, where tail latency determines user-perceived performance and system reliability." - New information - needs to be discussed earlier **either section 2.1 or section 2.2 or section 3.3**

### Analysis
**Section 1.1 Purpose**: "Background to the problem / issue" - Introduces the problem context, not metric definitions

**Section 2.1 Purpose**: "The practical problem" - Defines the practical problem being addressed

**Section 2.2 Purpose**: "Existing relevant knowledge" (ERK) - Reviews existing knowledge, including performance characteristics

**Section 3.3 Purpose**: "Research procedures" - Describes methodology and measurement techniques

### Recommendation
**Move tail latency content from Section 1.1 to Section 2.2.3** (Performance Metrics Selection), where it logically belongs as part of existing knowledge about real-time system performance metrics. We already mention tail latency there (line 317), so we should consolidate and expand that discussion rather than having it in Section 1.1.

---

## COMPREHENSIVE REVIEW CHECKLIST

### Chapter 1: Introduction
- [ ] Section 1.1: Should only contain background/problem context, not metric definitions
- [ ] Tail latency content: Should be moved to Section 2.2.3

### Chapter 2: Research Definition
- [ ] Section 2.1: Practical problem definition
- [ ] Section 2.2: ERK (Existing Relevant Knowledge)
  - [ ] Section 2.2.3: Performance Metrics Selection - Should include tail latency definition here
- [ ] All concepts used in Chapter 4 should be defined here or in Chapter 3

### Chapter 3: Methodology
- [ ] Section 3.1: Overview of methods
- [ ] Section 3.2: Justification for methods
- [ ] Section 3.3: Detailed research procedures
  - [ ] All experimental details (environments, matrix, pipeline) should be here
  - [ ] No new concepts should appear in Chapter 4

### Chapter 4: Data Analysis
- [ ] Section 4.1: Summary of data collected (results, not procedures)
- [ ] Section 4.2: Data analysis (results, not procedures)
- [ ] All concepts should be referenced from earlier chapters, not introduced here

---

## GUIDANCE FROM methodology-and-techniques.txt

**Chapter 3 Structure**:
- 3.1: Methods and techniques selected (overview)
- 3.2: Justification (what/why with respect to objectives)
- 3.3: Research procedures (all the details)

**Key Principle**: "Chapter 3 needs to describe the exact methods you have used in sufficient detail that there are no new concepts introduced in chapters 4 and 5."

---

## GUIDANCE FROM data-analysis-and-presentation.txt

**Chapter 4 Structure**:
- 4.1: Summary of data collected
- 4.2: Data analysis
- 4.3: Interpretation in relation to objectives
- 4.4: Interpretation in relation to research aim

**Key Principle**: "Chapter 4 discusses results not procedures since procedures should be discussed in section 3.3."

---

## ACTION ITEMS

1. **Move tail latency content from Section 1.1 to Section 2.2.3**
   - Remove paragraph from Section 1.1 (line 230)
   - Expand discussion in Section 2.2.3 where tail latency is already mentioned
   - Ensure comprehensive definition before Chapter 4

2. **Verify all content is in correct chapters**
   - Chapter 1: Background/introduction only
   - Chapter 2: Problem definition and ERK (including metric definitions)
   - Chapter 3: Methodology and procedures (all experimental details)
   - Chapter 4: Results and analysis only (no new concepts)

3. **Cross-reference check**
   - All concepts in Chapter 4 should be defined in Chapter 2 or 3
   - All procedures in Chapter 4 should reference Chapter 3
   - All metrics should be defined before use

