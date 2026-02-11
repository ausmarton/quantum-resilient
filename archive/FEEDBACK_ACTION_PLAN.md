# Feedback Action Plan: Draft 1 → Draft 2

**Date Created**: 2025-12-15  
**Purpose**: Comprehensive plan to address supervisor feedback and university guidance for dissertation revision

---

## Executive Summary

This plan organizes feedback from:
1. **Supervisor feedback** (`feedback-draft-1`) - Main focus on Chapters 3 & 4
2. **Earlier Chapter 3 feedback** (`Chapter3-review`) - Structure and detail level
3. **University guidance** (`methodology-and-techniques.txt`) - Chapter 3 requirements
4. **University guidance** (`data-analysis-and-presentation.txt`) - Chapter 4 requirements

**Key Principle**: Chapter 3 describes **HOW** (methods, procedures, framework). Chapter 4 describes **WHAT** (results, analysis, findings). No new concepts should appear in Chapter 4.

---

## PART 1: FEEDBACK ORGANIZATION

### 1.1 Overall Structure Issues

| Issue | Location | Priority | Dependency |
|-------|----------|----------|------------|
| Dissertation should be integrated whole | All chapters | HIGH | None |
| Section 2.2 ERK important for Chapter 3+ | 2.2 | MEDIUM | None |
| Chapter 3 must describe exact methods in detail | 3.3 | HIGH | None |
| No new concepts in Chapter 4 | 4.x | HIGH | Move content to 3.3 |
| Explain method choices, don't just state | 3.3 | HIGH | None |
| Discuss new ideas earlier, reference in Chapter 4 | 3.3 → 4.x | HIGH | Move content |
| Many graphs not discussed in text | 4.x | MEDIUM | Add discussion |
| Discuss algorithm characteristics/metrics | 2.2 or 3.3 | MEDIUM | None |
| Reduce software implementation detail | 3.3 | MEDIUM | Move to appendix |

### 1.2 Chapter 3 Specific Issues

#### Section 3.3.1 Methods and techniques
- **Issue**: Too high-level, needs more explanation
- **Issue**: "trait-based adapter pattern" and "deterministic workload generation" are obscure - needs explanation
- **Action**: Expand with methodological justification

#### Section 3.3.2 Data collection and analysis
- **Issue**: Uses future tense - should be past tense (work is complete)
- **Action**: Change all future tense to past tense

#### Section 3.3.3 Framework Implementation
- **Issue**: Too dense, needs more explanation
- **Issue**: Emphasis on software development (languages, tools) - should emphasize functionality
- **Issue**: Should start with diagram and structure narrative around it
- **Action**: Restructure with diagram first, focus on functionality

#### Section 3.3.4 Pilot activities
- **Issue**: Should describe tests proving framework represents live system
- **Issue**: Ignore debugging details
- **Action**: Focus on validation activities

### 1.3 Chapter 4 Specific Issues

#### Section 4.1.1 Experimental Scope and Scale
**Content to MOVE to Section 3.3**:
- Environments: Bare-metal, Local-K8s, Cloud-K8s (first mention should be in 3.3)
- Experimental matrix dimensions (payload sizes, workload rates, patterns, durations, replication)
- Defined parameter space
- Data processing pipeline (run aggregation, cross-run statistics, statistical testing, visualization)

**Content to KEEP in Section 4.1.1**:
- Summary of data collected (actual numbers: 396 experiments, 1,836 runs, 134,621,400 operations)
- Reference back to Section 3.3 for methodology
- Table 4.0 (environment specifications) - but should be introduced in 3.3 first

#### Section 4.1.3 Methodology Recap
- **Issue**: Should be in Section 3.3, not Chapter 4
- **Action**: Move to 3.3, reference from 4.1

#### Section 4.2 Algorithm Performance Comparison
- **Issue**: "classical implementations" - which are discussed where?
- **Issue**: Why focus on Bare-metal as baseline? (needs justification earlier)
- **Issue**: "tail latency determines user-perceived performance" - new information, needs earlier discussion
- **Issue**: "tail latency scenarios" - where defined?
- **Action**: Add definitions/justifications to Section 2.1, 2.2, or 3.3

#### Section 4.2.1 Algorithm Equivalence
- **Issue**: All new information, should be discussed earlier
- **Action**: Move to Section 2.2 or 3.3

#### Section 4.2.2 Statistical Hypothesis Testing
- **Issue**: New idea - is this discussed in Chapter 3?
- **Action**: Add to Section 3.3.2 (data collection and analysis)

#### Section 4.2.3 Environment Comparison
- **Issue**: Figure 4.2 not referenced in text
- **Action**: Add explicit reference to figure in text

#### Section 4.3.4 Objective 4: Comparative Analysis
- **Issue**: "59 comparisons showing large effect sizes" - which figure does this refer to?
- **Action**: Add figure reference

### 1.4 New Concepts to Introduce Earlier

These concepts appear in Chapter 4 but should be introduced in Chapter 2 or 3:

1. **Tail Amplification and Jitter Assessment** → Section 2.1 or 2.2
2. **Throughput Analysis** → Section 3.3.2 (as a metric to be measured)
3. **Latency Distribution Characteristics** → Section 2.1 or 3.3.2
4. **Environment Overhead Analysis** → Section 3.3 (as part of framework design)
5. **Normalised Throughput Analysis** → Section 3.3.2 (as an analysis technique)

---

## PART 2: DEPENDENCY RESOLUTION

### 2.1 Content Movement Dependencies

```
Priority 1 (Foundation - Must be done first):
├── Define environments in Section 3.3
├── Define experimental matrix in Section 3.3
├── Define data processing pipeline in Section 3.3.2
├── Define statistical methods in Section 3.3.2
└── Define metrics (tail latency, throughput, etc.) in Section 2.1/2.2 or 3.3.2

Priority 2 (Structure - Depends on Priority 1):
├── Move methodology recap from 4.1.3 to 3.3
├── Add references from Chapter 4 back to Chapter 3
└── Update Section 4.1.1 to reference 3.3 instead of introducing concepts

Priority 3 (Enhancement - Depends on Priority 2):
├── Add figure references in Chapter 4 text
├── Add discussion of all graphs
├── Add justifications for method choices
└── Expand explanations in Section 3.3.1
```

### 2.2 Section Numbering Tracking

**Original Section Numbers** (for reference):
- 3.3.1 Methods and techniques
- 3.3.2 Data collection and analysis
- 3.3.3 Framework Implementation
- 3.3.4 Pilot activities
- 4.1.1 Experimental Scope and Scale
- 4.1.3 Methodology Recap
- 4.2.1 Algorithm Performance Comparison
- 4.2.2 Statistical Hypothesis Testing
- 4.2.3 Environment Comparison

**Note**: When moving content, we'll track original locations using comments like `[MOVED FROM 4.1.1]` temporarily, then remove after verification.

---

## PART 3: DETAILED ACTION ITEMS

### Writing Style Guidelines

**IMPORTANT**: All changes must follow academic writing best practices:
- **Avoid meta-language**: Don't describe what you're doing, just do it
- **Be direct**: Let content speak for itself
- **Remove self-referential statements**: "This section provides...", "This ensures that...", "This component addresses..."
- **Remove explanatory commentary**: If content is clear, don't explain that it's clear
- **Let tables/figures speak for themselves**: Don't explain what they show if it's obvious
- **Maintain objective traceability**: Show connections between objectives, methodology, and results through factual statements and section references, not meta-commentary

See `WRITING_GUIDELINES.md` for detailed examples and patterns to avoid.
See `OBJECTIVE_TRACEABILITY.md` for how to maintain objective connections without meta-language.

### Redundancy Checking

**IMPORTANT**: Before and after each change:
1. Check for duplication with existing content
2. Review for redundant statements
3. Consolidate or remove redundant text
4. Ensure appropriate level of detail (3.1 overview → 3.2 justification → 3.3 detailed)

See `REDUNDANCY_LOG.md` for tracking redundancy issues.

### Scope Limitations

**CRITICAL**: Only work on Chapters 3 and 4 based on feedback:
- **DO**: Fix issues in Chapters 3 and 4
- **DON'T**: Fix issues in Chapters 1, 2, or 5
- **DO**: Log issues found in other chapters for future work
- **DON'T**: Make changes outside scope even if issues are obvious

See `ISSUES_OUTSIDE_SCOPE.md` for logging issues found in other chapters.

### Phase 0: Pre-Implementation Review and Verification

#### Task 0.1: Review Current Section 3.1 Content
**Location**: Section 3.1  
**Action**: Verify current content against requirements:
- Does it provide overview of everything? (Chapter3-review)
- Does it include: methodology overview, framework architecture overview, data collection overview, analysis approach overview? (methodology-and-techniques.txt)
- Does it have a figure that needs block-by-block description? (Chapter3-review)
- Does it need objective-to-methodology mapping table? (methodology-and-techniques.txt)

**Checkpoint**: Document what's missing in Section 3.1

#### Task 0.2: Review Current Section 3.2 Content
**Location**: Section 3.2  
**Action**: Verify current content against requirements:
- Does it talk about what/why with respect to objectives? (Chapter3-review)
- Does it explain why exclude methods like survey/observations? (Chapter3-review)
- Does it explain how framework represents live/real-world systems? (Chapter3-review)
- Does it justify method selection and experimental method? (methodology-and-techniques.txt)

**Checkpoint**: Document what's missing in Section 3.2

#### Task 0.3: Identify Code Snippets
**Location**: Throughout Chapter 3  
**Action**: Find any code snippets, implementation details that should move to appendices
**Note**: Per Chapter3-review: "Code shouldn't be in main text - can go into appendices. Don't waste too much time on appendices though."

**Checkpoint**: List code snippets to move

#### Task 0.4: Identify Missing Concepts (3.e. and 5.f. Items)
**Location**: Systematic scan of Chapter 4  
**Action**: 
- Scan Chapter 4 for new concepts that should be in Chapter 3 (3.e. items)
- Scan Chapter 4 for new terminology/jargon that should be defined earlier (5.f. items)
- Document findings in FINAL_VERIFICATION.md
**Note**: Per user clarification:
- **3.e.**: Other new concepts in Chapter 4 that should be in Chapter 3 (extending list 3.a-3.d)
- **5.f.**: Other terminology/jargon in Chapter 4 that should be defined earlier (extending list 5.a-5.e)

**Checkpoint**: All 3.e. and 5.f. items identified and documented

### Phase 1: Chapter 3 Enhancements (Foundation)

#### Task 1.0: Enhance Section 3.1 (if needed after review)
**Location**: Section 3.1  
**Dependencies**: Task 0.1  
**Required Changes** (based on review):
1. Ensure overview of everything is covered
2. Add framework architecture overview (high-level)
3. Add data collection overview (what will be collected)
4. Add analysis approach overview (what statistical methods will be used)
5. Add objective-to-methodology mapping table (Objective → Measurement → Metrics → Statistical Method)
6. If figure exists in 3.1.1, describe each block and what it does

**Checkpoint**: Section 3.1 covers all required overview content

#### Task 1.0a: Enhance Section 3.2 (if needed after review)
**Location**: Section 3.2  
**Dependencies**: Task 0.2  
**Required Changes** (based on review):
1. Ensure it talks about what/why with respect to objectives
2. Explicitly explain why exclude methods like survey/observations (may already be there, verify)
3. Add explicit explanation of how framework represents live/real-world systems
4. Ensure justification for experimental method is clear
5. Ensure justification for framework design choices

**Checkpoint**: Section 3.2 fully justifies methodology and framework design

#### Task 1.1: Expand Section 3.3.1 Methods and techniques
**Location**: Section 3.3.1  
**Current State**: Too high-level, obscure terms  
**Required Changes**:
1. Expand "trait-based adapter pattern" explanation - explain how it ensures uniform measurement
2. Expand "deterministic workload generation (seeded ChaCha20 RNG)" - explain why this ensures reproducibility
3. Add methodological justification for each component
4. Explain WHY these choices were made (not just WHAT)

**Checkpoint**: Section 3.3.1 should be understandable to another T802 student

#### Task 1.2: Fix Section 3.3.2 Tense Issues
**Location**: Section 3.3.2  
**Current State**: Uses future tense  
**Required Changes**:
1. Change all future tense to past tense
2. Example: "will be collected" → "was collected"
3. Example: "will enable" → "enabled"

**Checkpoint**: No future tense verbs in Section 3.3.2

#### Task 1.3: Restructure Section 3.3.3 Framework Implementation
**Location**: Section 3.3.3  
**Current State**: Too dense, emphasis on software tools  
**Required Changes**:
1. **Create THREE diagrams** (per user clarification):
   - **Diagram 1**: High-level implementation diagram (explains implementation at high level)
   - **Diagram 2**: Live system diagram (shows live system and which parts represent live system components, shows where instrumentation is placed)
   - **Diagram 3**: Detailed research system diagram (describes what was built as part of research)
2. Start section with high-level diagram
3. Structure narrative around diagrams
4. Change emphasis from languages/tools to functionality
5. Explain methodological purpose of each component
6. Add framework comparison with live production systems (using live system diagram)
7. Focus on telemetry - "how we make the measurements" (per Chapter3-review: "Focus on the bits that we're interested in - how we make the measurements")
8. Remove content that doesn't affect telemetry (per Chapter3-review: "There's no need to talk about stuff that doesn't affect telemetry")

**Checkpoint**: Section should be structured around three diagrams, focus on functionality and telemetry

#### Task 1.4: Enhance Section 3.3.4 Pilot activities
**Location**: Section 3.3.4  
**Current State**: May include debugging details  
**Required Changes**:
1. Focus on validation activities proving framework represents live system
2. Remove debugging details
3. Describe tests performed to validate framework

**Checkpoint**: Section describes validation, not debugging

#### Task 1.5: Add Validity Considerations
**Location**: Section 3.3 (new subsection 3.3.5 or integrate into existing)  
**Source**: methodology-and-techniques.txt  
**Required Changes**:
1. Add discussion of construct validity (relationship of data to research problem)
2. Add discussion of internal validity (cause and effect, controlling confounding factors)
3. Add discussion of external validity (generalisability)
4. Add discussion of data validity (accuracy, sufficiency, relevance)
5. Frame these as methodological considerations, not just checkboxes

**Checkpoint**: Validity considerations addressed in Section 3.3

#### Task 1.6: Move Code to Appendices
**Location**: Throughout Chapter 3  
**Dependencies**: Task 0.3  
**Required Changes**:
1. Identify code snippets in Chapter 3
2. Move to appropriate appendix
3. Replace in main text with high-level description
4. Keep only "key bits of source code if they're really critical" (per Chapter3-review)

**Checkpoint**: No code in main text, only in appendices if critical

### Phase 2: Move Content from Chapter 4 to Chapter 3

#### Task 2.1: Define Environments in Section 3.3
**Location**: Section 3.3 (new subsection or integrate into 3.3.3)  
**Source**: Section 4.1.1  
**Content to Move**:
- First mention of: Bare-metal (non-containerised), Containerised local Kubernetes (Local-K8s), Cloud-managed Kubernetes (Cloud-K8s)
- Table 4.0 (or create Table 3.X in Chapter 3)
- Explanation of why these environments were chosen

**Checkpoint**: Environments defined in Chapter 3, referenced in Chapter 4

#### Task 2.2: Define Experimental Matrix in Section 3.3
**Location**: Section 3.3.2  
**Source**: Section 4.1.1  
**Content to Move**:
- Payload sizes (256B, 1KB, 4KB, 16KB)
- Workload rates (100, 500, 2,000, 10,000 msg/s)
- Workload patterns (constant, burst)
- Durations (30s primary, 300s extended)
- Replication strategy (5 runs for baseline, 3 for scaling/extended)
- Rationale for these choices

**Checkpoint**: Experimental matrix fully defined in Section 3.3.2

#### Task 2.3: Define Data Processing Pipeline in Section 3.3.2
**Location**: Section 3.3.2  
**Source**: Section 4.1.1 (implied)  
**Content to Add**:
- Run aggregation (computing run-level statistics: percentiles, means, standard deviations)
- Cross-run statistics (combining multiple runs: mean values, standard deviations, 95% confidence intervals)
- Statistical testing (hypothesis tests: Welch's t-test, Mann-Whitney U with effect size: Cohen's d)
- Visualization generation (CDFs, comparison charts, statistical summaries)
- Justification for these statistical methods

**Checkpoint**: Data processing pipeline fully described in Section 3.3.2

#### Task 2.4: Move Methodology Recap from 4.1.3 to 3.3
**Location**: Section 3.3 (appropriate subsection)  
**Source**: Section 4.1.3  
**Content to Move**:
- Entire "Methodology Recap and Environment Control" section
- Integrate into appropriate 3.3 subsection

**Checkpoint**: Methodology recap removed from Chapter 4, integrated into Chapter 3

### Phase 3: Add Missing Definitions and Justifications

#### Task 3.1: Define Tail Latency and Real-Time Metrics
**Location**: Section 2.1 or 2.2  
**Content to Add**:
- Definition of tail latency (p95, p99)
- Why tail latency is critical for real-time systems
- How tail latency determines user-perceived performance
- Why this is important for AML systems

**Checkpoint**: Tail latency defined before Chapter 4

#### Task 3.2: Define Throughput Analysis
**Location**: Section 3.3.2  
**Content to Add**:
- Throughput as a performance metric
- How throughput will be measured
- Why throughput is important for real-time systems

**Checkpoint**: Throughput defined in Chapter 3

#### Task 3.3: Define Latency Distribution Characteristics
**Location**: Section 2.1 or 3.3.2  
**Content to Add**:
- What latency distribution characteristics mean
- Why distribution shape matters
- How CDFs will be used to analyze distributions

**Checkpoint**: Latency distribution concepts defined before Chapter 4

#### Task 3.4: Define Environment Overhead Analysis
**Location**: Section 3.3  
**Content to Add**:
- Why environment overhead is being measured
- How containerisation overhead will be assessed
- How cloud deployment overhead will be assessed
- Why Bare-metal is used as baseline

**Checkpoint**: Environment overhead analysis rationale in Chapter 3

#### Task 3.5: Define Statistical Hypothesis Testing
**Location**: Section 3.3.2  
**Content to Add**:
- Which statistical tests will be used (Welch's t-test, Mann-Whitney U)
- Why these tests are appropriate
- How effect size (Cohen's d) will be computed
- Why statistical testing is necessary

**Checkpoint**: Statistical methods defined in Section 3.3.2

#### Task 3.6: Define Algorithm Equivalence Concept
**Location**: Section 2.2 or 3.3  
**Content to Add**:
- What "algorithm equivalence" means in this context
- How equivalence will be assessed
- Why equivalence is important

**Checkpoint**: Algorithm equivalence concept defined before Chapter 4

#### Task 3.7: Clarify "Classical Implementations"
**Location**: Section 2.2  
**Content to Add**:
- Which classical algorithms are being evaluated
- Why these specific implementations were chosen
- Where they are discussed in detail

**Checkpoint**: Classical implementations clearly identified

#### Task 3.8: Define Performance Metrics Framework
**Location**: Section 3.3.2  
**Source**: Chapter 4, Section 4.2 (3.e. item)  
**Content to Add**:
- Distinction between "Per-operation efficiency (latency)" and "System capacity (throughput under concurrency)"
- Why these are independent dimensions
- How they will be measured

**Checkpoint**: Performance metrics framework defined in Section 3.3.2

#### Task 3.9: Define Statistical Comparison Terminology
**Location**: Section 3.3.2  
**Source**: Chapter 4, Section 4.2.3 (3.e. item)  
**Content to Add**:
- Inferential vs descriptive comparisons
- Within-environment baselines
- Hardware confounding
- Why cross-environment comparisons are descriptive only

**Checkpoint**: Statistical comparison terminology defined in Section 3.3.2

#### Task 3.10: Define Scaling Concepts
**Location**: Section 3.3.2  
**Source**: Chapter 4, Section 4.2.4 (3.e. item)  
**Content to Add**:
- Scaling factor
- Sub-linear scaling
- Throughput saturation
- Capacity limits

**Checkpoint**: Scaling concepts defined in Section 3.3.2

#### Task 3.11: Define Resource Utilisation Metrics
**Location**: Section 3.3.2  
**Source**: Chapter 4, Section 4.2.5 (3.e. item)  
**Content to Add**:
- Resource utilisation
- CPU utilisation
- Memory consumption
- Deployment capacity

**Checkpoint**: Resource utilisation metrics defined in Section 3.3.2

#### Task 3.12: Define Deployment and Migration Terminology
**Location**: Section 2.1/2.2 or 3.3  
**Source**: Chapter 4, various sections (5.f. items)  
**Content to Add**:
- Bandwidth-constrained environments
- Certificate infrastructure
- Capacity planning
- Phased migration
- Hybrid approaches
- Transitional deployments
- Horizontal scaling
- On-premise (if not already defined)
- Containerisation/virtualisation (if not already defined)

**Checkpoint**: Deployment and migration terminology defined

#### Task 3.13: Define Visualization Methods
**Location**: Section 3.3.2  
**Source**: Chapter 4, Figure 4.1a (5.f. items)  
**Content to Add**:
- CDF (Cumulative Distribution Function)
- Violin plots
- Box plots
- Why these visualization methods were chosen

**Checkpoint**: Visualization methods defined and justified in Section 3.3.2

#### Task 3.14: Define Statistical Correction Methods
**Location**: Section 3.3.2  
**Source**: Chapter 4, Section 4.3.4 (5.f. items)  
**Content to Add**:
- Holm-Bonferroni correction
- Why multiple comparison correction is needed
- How it will be applied

**Checkpoint**: Statistical correction methods defined in Section 3.3.2

#### Task 3.15: Define Percentile Terminology
**Location**: Section 3.3.2  
**Source**: Chapter 4, throughout (5.f. items)  
**Content to Add**:
- p50, p95, p99 (median, 95th percentile, 99th percentile)
- Why these percentiles are used
- How they relate to tail latency

**Checkpoint**: Percentile terminology defined in Section 3.3.2

### Phase 4: Chapter 4 Improvements

#### Task 4.1: Update Section 4.1.1 to Reference Chapter 3
**Location**: Section 4.1.1  
**Required Changes**:
1. Remove first mention of environments (reference Section 3.3 instead)
2. Remove experimental matrix definition (reference Section 3.3.2 instead)
3. Remove data processing pipeline description (reference Section 3.3.2 instead)
4. Keep actual data summary (396 experiments, 1,836 runs, etc.)
5. Add explicit references: "As described in Section 3.3..."

**Checkpoint**: Section 4.1.1 references Chapter 3, doesn't introduce new concepts

#### Task 4.2: Add Figure References in Text
**Location**: Throughout Chapter 4  
**Required Changes**:
1. Find all figures mentioned in feedback
2. Ensure each figure is explicitly referenced in text before or after it appears
3. Add discussion of what each figure shows
4. Connect figures to research objectives

**Specific Figures to Address**:
- Figure 4.1: Add discussion of why steeper curves are important
- Figure 4.1a: Explain why violin/box plots were chosen
- Figure 4.2: Add explicit reference (currently missing)
- Figure 4.3: Ensure referenced and discussed

**Checkpoint**: All figures referenced and discussed in text

#### Task 4.3: Add Discussion of All Graphs
**Location**: Throughout Chapter 4  
**Required Changes**:
1. For each graph/figure, add:
   - Why this visualization was chosen
   - What it shows
   - How it relates to research objectives
   - What conclusions can be drawn

**Checkpoint**: Every graph has accompanying discussion

#### Task 4.4: Add Justification for Bare-metal Baseline
**Location**: Section 4.2 (early in section)  
**Required Changes**:
1. Explain why Bare-metal is used as baseline
2. Reference back to Section 3.3 where this was established
3. Explain that containerisation/cloud overheads are examined separately

**Checkpoint**: Bare-metal baseline choice justified

#### Task 4.5: Add Figure Reference for Statistical Results
**Location**: Section 4.3.4  
**Required Changes**:
1. Identify which figure shows "59 comparisons showing large effect sizes"
2. Add explicit reference to that figure
3. Ensure figure exists and is properly labeled

**Checkpoint**: Statistical results linked to appropriate figure

#### Task 4.7: Map Claims to Objectives
**Location**: Throughout Chapter 4  
**Source**: feedback-draft-1 item 69  
**Required Changes**:
1. Ensure all performance claims map to research objectives
2. **Specific**: "Which objective is this?" for consistency claim in 4.2.3 Environment-Algorithm Interaction Analysis
3. Add explicit objective mapping where claims are made
4. Example: "This consistency [claim] addresses Objective 4 by demonstrating..."

**Checkpoint**: All claims in Chapter 4 explicitly linked to research objectives

#### Task 4.6: Remove Methodology Recap from 4.1.3
**Location**: Section 4.1.3  
**Required Changes**:
1. Remove "Methodology Recap and Environment Control" subsection
2. Replace with reference to Section 3.3
3. Keep only results-related content

**Checkpoint**: Methodology recap removed from Chapter 4

### Phase 5: Enhance Explanations and Justifications

#### Task 5.1: Add Method Choice Justifications
**Location**: Section 3.3  
**Required Changes**:
1. For each method/technique, add:
   - Why this method was chosen
   - What alternatives were considered
   - Why alternatives were rejected
   - How this method addresses research objectives

**Checkpoint**: All method choices justified

#### Task 5.2: Add Statistical Method Justifications
**Location**: Section 3.3.2  
**Required Changes**:
1. Explain why Welch's t-test (not standard t-test)
2. Explain why Mann-Whitney U (non-parametric alternative)
3. Explain why Cohen's d (effect size)
4. Explain why 95% confidence intervals
5. Explain why Holm-Bonferroni correction

**Checkpoint**: All statistical methods justified

#### Task 5.3: Discuss Algorithm Characteristics/Metrics
**Location**: Section 2.2 or 3.3  
**Required Changes**:
1. Similar to selection criteria discussion
2. Explain what metrics will be measured
3. Why these metrics matter
4. How metrics relate to real-time system requirements

**Checkpoint**: Algorithm characteristics/metrics discussed

---

## PART 4: IMPLEMENTATION CHECKLIST

### Pre-Implementation
- [ ] Review plan with user
- [ ] Identify any unclear feedback items
- [ ] Create section number mapping table
- [ ] Backup current dissertation file
- [ ] **NEW**: Complete Phase 0 review tasks (0.1, 0.2, 0.3, 0.4)
- [ ] **NEW**: Clarify conflicts with supervisor if needed

### Phase 0: Pre-Implementation Review
- [ ] Task 0.1: Review Current Section 3.1
- [ ] Task 0.2: Review Current Section 3.2
- [ ] Task 0.3: Identify Code Snippets
- [ ] Task 0.4: Identify Missing Concepts
- [ ] **Checkpoint 0**: Document findings, update plan if needed

### Phase 1: Chapter 3 Enhancements
- [ ] Task 1.0: Enhance Section 3.1 (if needed)
- [ ] Task 1.0a: Enhance Section 3.2 (if needed)
- [ ] Task 1.1: Expand Section 3.3.1
- [ ] Task 1.2: Fix Section 3.3.2 tense
- [ ] Task 1.3: Restructure Section 3.3.3 (with live system diagram)
- [ ] Task 1.4: Enhance Section 3.3.4
- [ ] Task 1.5: Add Validity Considerations
- [ ] Task 1.6: Move Code to Appendices
- [ ] **Checkpoint 1**: Review Chapter 3 with user

### Phase 2: Move Content from Chapter 4 to Chapter 3
- [ ] Task 2.1: Define environments in 3.3
- [ ] Task 2.2: Define experimental matrix in 3.3.2
- [ ] Task 2.3: Define data processing pipeline in 3.3.2
- [ ] Task 2.4: Move methodology recap to 3.3
- [ ] **Checkpoint 2**: Verify all content moved correctly

### Phase 3: Add Missing Definitions
- [ ] Task 3.1: Define tail latency
- [ ] Task 3.2: Define throughput analysis
- [ ] Task 3.3: Define latency distribution characteristics
- [ ] Task 3.4: Define environment overhead analysis
- [ ] Task 3.5: Define statistical hypothesis testing
- [ ] Task 3.6: Define algorithm equivalence
- [ ] Task 3.7: Clarify classical implementations
- [ ] Task 3.8: Define Performance Metrics Framework
- [ ] Task 3.9: Define Statistical Comparison Terminology
- [ ] Task 3.10: Define Scaling Concepts
- [ ] Task 3.11: Define Resource Utilisation Metrics
- [ ] Task 3.12: Define Deployment and Migration Terminology
- [ ] Task 3.13: Define Visualization Methods
- [ ] Task 3.14: Define Statistical Correction Methods
- [ ] Task 3.15: Define Percentile Terminology
- [ ] **Checkpoint 3**: Verify all definitions in place

### Phase 4: Chapter 4 Improvements
- [ ] Task 4.1: Update Section 4.1.1
- [ ] Task 4.2: Add figure references
- [ ] Task 4.3: Add graph discussions
- [ ] Task 4.4: Add Bare-metal justification
- [ ] Task 4.5: Add statistical results figure reference
- [ ] Task 4.6: Remove methodology recap
- [ ] Task 4.7: Map Claims to Objectives
- [ ] **Checkpoint 4**: Review Chapter 4 with user

### Phase 5: Enhance Explanations
- [ ] Task 5.1: Add method choice justifications
- [ ] Task 5.2: Add statistical method justifications
- [ ] Task 5.3: Discuss algorithm characteristics
- [ ] **Checkpoint 5**: Final review

### Post-Implementation
- [ ] Verify all section numbers still correct
- [ ] Verify all cross-references work
- [ ] Check for any remaining future tense
- [ ] Verify no new concepts in Chapter 4
- [ ] Final proofread

---

## PART 5: CLARIFICATION NEEDED

### Items Requiring Supervisor Clarification

1. **Section 2.2 ERK**: ✅ **CLARIFIED** - ERK = "Existing Relevant Knowledge" (Section 2.2)
   - **Action**: Ensure Section 2.2 supports Chapter 3+ (verify content supports methodology)

2. **Figure 4.2 Reference**: Feedback says "When you produce a graph like this you must refer to it in the text… in figure 4.2"
   - **Question**: Should the reference be "as shown in Figure 4.2" or "see Figure 4.2"?
   - **Action**: Add reference using standard academic format, supervisor can confirm if needed

3. **Statistical Results Figure**: Feedback asks "Which figure does this refer to?" for the 59 comparisons
   - **Question**: Does such a figure exist, or should we create one?
   - **Action**: Check if figure exists during implementation, create if needed

4. **"Classical implementations"**: Feedback asks "Which are discussed where?"
   - **Question**: Should we add a dedicated section, or is Section 2.2 sufficient?
   - **Action**: Add discussion in Section 2.2 (Task 3.7), verify during implementation

5. **Block Diagram Requirements**: ✅ **CLARIFIED** - Need 3 diagrams:
   - High-level implementation diagram (explains implementation at high level)
   - Live system diagram (shows live system and which parts represent live system components)
   - Detailed research system diagram (describes what was built as part of research)
   - **Action**: Create all 3 diagrams as specified

6. **Section 3.1/3.2 Sufficiency**: ✅ **CLARIFIED** - Phase 0 will identify what's needed; overall review needed at end
   - **Action**: Complete Phase 0 review tasks, then decide if expansion needed

7. **Ellipsis Items (3.e. and 5.f.)**: ✅ **CLARIFIED**
   - **3.e.**: Other new concepts in Chapter 4 that should be in Chapter 3 (extending list 3.a-3.d)
   - **5.f.**: Other terminology/jargon in Chapter 4 that should be defined earlier (extending list 5.a-5.e)
   - **Action**: Systematic scan completed, tasks added (3.8-3.15)

---

## PART 6: TRACKING ORIGINAL LOCATIONS

### Section Number Mapping

When moving content, we'll use this table to track:

| Original Location | New Location | Content Moved | Status |
|------------------|--------------|---------------|--------|
| 4.1.1 | 3.3 | Environments definition | Pending |
| 4.1.1 | 3.3.2 | Experimental matrix | Pending |
| 4.1.1 | 3.3.2 | Data processing pipeline | Pending |
| 4.1.3 | 3.3 | Methodology recap | Pending |
| 4.2.1 | 2.2 or 3.3 | Algorithm equivalence | Pending |
| 4.2.1 | 2.1 or 2.2 | Tail latency definition | Pending |
| 4.2.2 | 3.3.2 | Statistical hypothesis testing | Pending |

**Note**: After implementation, verify all content is in correct location and remove tracking comments.

---

## PART 7: SUCCESS CRITERIA

### Chapter 3 Success Criteria
- [ ] All methods and techniques explained in sufficient detail
- [ ] No future tense (work is complete)
- [ ] Framework described with functionality focus, not tools
- [ ] All environments, experimental matrix, and data processing defined
- [ ] All statistical methods defined and justified
- [ ] All metrics (tail latency, throughput, etc.) defined
- [ ] Section structured around diagram
- [ ] Understandable to another T802 student

### Chapter 4 Success Criteria
- [ ] No new concepts introduced
- [ ] All concepts reference back to Chapter 3
- [ ] All figures referenced and discussed in text
- [ ] All graphs have accompanying discussion
- [ ] Methodology recap removed
- [ ] Results and analysis only (no procedures)
- [ ] Clear connection to research objectives

### Overall Success Criteria
- [ ] Dissertation is integrated whole
- [ ] Each chapter provides information for subsequent chapters
- [ ] Can refer back when needed
- [ ] Method choices explained, not just stated
- [ ] Right level of detail in each chapter
- [ ] Software implementation detail in appendix (if needed)

---

## PART 8: RISK MITIGATION

### Risks and Mitigation Strategies

1. **Risk**: Losing track of original section numbers
   - **Mitigation**: Use tracking table, add temporary comments

2. **Risk**: Breaking cross-references
   - **Mitigation**: Update all cross-references after moves, verify with grep

3. **Risk**: Missing some feedback items
   - **Mitigation**: Use checklist, review after each phase

4. **Risk**: Unclear feedback items
   - **Mitigation**: Document in "Clarification Needed" section, ask supervisor

5. **Risk**: Over-editing and losing good content
   - **Mitigation**: Backup file, review changes after each phase

---

## NEXT STEPS

1. **Review this plan** with user
2. **Clarify unclear items** with supervisor if needed
3. **Begin Phase 1** (Chapter 3 enhancements)
4. **Checkpoint after each phase**
5. **Final review** before submission

---

**Document Status**: Draft - Awaiting user review and approval before implementation

