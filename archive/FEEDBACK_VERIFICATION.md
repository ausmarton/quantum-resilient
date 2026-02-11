# Feedback Verification: Complete Cross-Reference Check

**Date**: 2025-12-15  
**Purpose**: Verify that all feedback items from all 4 source documents are covered in our action plan

**Source Documents**:
1. `feedback-draft-1` - Main supervisor feedback
2. `Chapter3-review` - Earlier Chapter 3 feedback  
3. `methodology-and-techniques.txt` - University guidance for Chapter 3
4. `data-analysis-and-presentation.txt` - University guidance for Chapter 4

---

## PART 1: VERIFICATION FROM feedback-draft-1

### Overall Structure Issues

| Item | Status | Location in Plan | Notes |
|------|--------|------------------|-------|
| 1. Dissertation should be integrated whole | ✅ Covered | Part 1.1, Overall Structure | General principle noted |
| 2. Section 2.2 ERK important (now clarified as "Existing Relevant Knowledge") | ✅ Covered | Part 1.1, Task 3.1-3.7 | Need to ensure Section 2.2 supports Chapter 3+ |
| 3. Chapter 3 must describe exact methods in detail | ✅ Covered | Phase 1, Phase 2 | Multiple tasks address this |
| 4. No new concepts in Chapter 4 | ✅ Covered | Phase 2, Phase 3 | Content movement tasks |
| 5. Explain method choices, don't just state | ✅ Covered | Task 5.1, Task 5.2 | Justification tasks |
| 6. Discuss new ideas earlier | ✅ Covered | Phase 3 | Missing definitions tasks |
| 7. Many graphs not discussed | ✅ Covered | Task 4.2, Task 4.3 | Figure reference tasks |
| 8. Discuss algorithm characteristics/metrics | ✅ Covered | Task 5.3 | Algorithm characteristics task |
| 9. Reduce software implementation detail | ✅ Covered | Part 1.1 | Move to appendix noted |

### Specific Feedback Items from feedback-draft-1

| Item | Status | Location in Plan | Notes |
|------|--------|------------------|-------|
| 3.a. Environments (Bare-metal, Local-K8s, Cloud-K8s) | ✅ Covered | Task 2.1 | Move to 3.3 |
| 3.b. Experimental parameters (payload sizes, rates, patterns, durations, replication) | ✅ Covered | Task 2.2 | Move to 3.3.2 |
| 3.c. Defined parameter space | ✅ Covered | Task 2.2 | Part of experimental matrix |
| 3.d. Data processing pipeline (4 stages) | ✅ Covered | Task 2.3 | Move to 3.3.2 |
| 3.e. (ellipsis - other items) | ⚠️ Needs check | - | May need to identify what else |
| 4. Explain why these statistics? | ✅ Covered | Task 5.2 | Statistical method justifications |
| 5.a. Tail Amplification and Jitter Assessment | ✅ Covered | Task 3.1 | Define in 2.1/2.2 |
| 5.b. Throughput Analysis | ✅ Covered | Task 3.2 | Define in 3.3.2 |
| 5.c. Latency Distribution Characteristics | ✅ Covered | Task 3.3 | Define in 2.1/3.3.2 |
| 5.d. Environment Overhead Analysis | ✅ Covered | Task 3.4 | Define in 3.3 |
| 5.e. Normalised Throughput Analysis | ✅ Covered | Task 3.2 | Part of throughput analysis |
| 5.f. (ellipsis - other items) | ⚠️ Needs check | - | May need to identify what else |
| 3.3.1: Too high-level, needs explanation | ✅ Covered | Task 1.1 | Expand Section 3.3.1 |
| 3.3.1: "trait-based adapter pattern" obscure | ✅ Covered | Task 1.1 | Expand explanation |
| 3.3.1: "deterministic workload generation" obscure | ✅ Covered | Task 1.1 | Expand explanation |
| 3.3.2: No future tenses | ✅ Covered | Task 1.2 | Fix tense issues |
| 4.1.1: Some content should be in 3.3 | ✅ Covered | Task 2.1, 2.2, 2.3 | Move content |
| 4.1.1: Reference back to 3.3 | ✅ Covered | Task 4.1 | Update 4.1.1 |
| 4.1.1: Environments first mention | ✅ Covered | Task 2.1 | Move to 3.3 |
| 4.1.1: Table 4.0 "Minikube" needs definition | ✅ Covered | Task 2.1 | Define in 3.3 |
| 4.1.1: Experimental matrix dimensions | ✅ Covered | Task 2.2 | Move to 3.3 |
| 4.1.3: Methodology Recap should be in 3.3 | ✅ Covered | Task 2.4 | Move to 3.3 |
| 4.2: "classical implementations" - where discussed? | ✅ Covered | Task 3.7 | Clarify in 2.2 |
| 4.2: Why Bare-metal as baseline? | ✅ Covered | Task 3.4, Task 4.4 | Justify in 3.3 and 4.2 |
| 4.2: "tail latency determines..." - new info | ✅ Covered | Task 3.1 | Define in 2.1/2.2 |
| 4.2: "tail latency scenarios" - where defined? | ✅ Covered | Task 3.1 | Define in 2.1/2.2 |
| 4.2.1: Algorithm Equivalence - all new info | ✅ Covered | Task 3.6 | Define before Chapter 4 |
| 4.2.1: Figure 4.1 - is steeper curve important? Why? | ✅ Covered | Task 4.2, Task 4.3 | Add discussion |
| 4.2.1: Figure 4.1a - why violin/box plots? | ✅ Covered | Task 4.2, Task 4.3 | Explain choice |
| 4.2.2: Statistical Hypothesis Testing - new idea | ✅ Covered | Task 3.5 | Define in 3.3.2 |
| 4.2.3: Which objective is consistency claim? | ⚠️ Needs check | - | May need to add objective mapping |
| 4.2.3: Figure 4.2 not referenced | ✅ Covered | Task 4.2 | Add reference |
| 4.3.4: "59 comparisons" - which figure? | ✅ Covered | Task 4.5 | Add figure reference |

---

## PART 2: VERIFICATION FROM Chapter3-review

| Item | Status | Location in Plan | Notes |
|------|--------|------------------|-------|
| Level of detail: 3.1 → 3.2 → 3.3 (increasing) | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| 3.1 should cover overview of everything | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| 3.2 talks about what/why with respect to objectives | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| 3.2: Why exclude methods like survey/observations | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| 3.2: How framework represents live/real-world systems | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| 3.3: Need block diagram of framework | ✅ Covered | Task 1.3 | Restructure 3.3.3 with diagram |
| 3.3: Need block diagram of live system | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| 3.3: Show where instrumentation is placed | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| 3.3: Draw live system, draw how we compare | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| 3.3: Fine balance - not too detailed | ✅ Covered | Task 1.3 | Restructure with explanation |
| Focus on telemetry - how we make measurements | ⚠️ **PARTIALLY** | Task 1.3 | Need to emphasize telemetry focus |
| Code should go to appendices | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| Figure in 3.1.1 - describe each block | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| 3.3.3: Too dense, more explanation | ✅ Covered | Task 1.3 | Restructure 3.3.3 |
| 3.3.3: Emphasis on functionality, not tools | ✅ Covered | Task 1.3 | Change emphasis |
| 3.3.3: Start section with diagram | ✅ Covered | Task 1.3 | Restructure with diagram |
| 3.3.4: Describe tests proving framework represents live system | ✅ Covered | Task 1.4 | Focus on validation |

---

## PART 3: VERIFICATION FROM methodology-and-techniques.txt

Key requirements from university guidance:

| Requirement | Status | Location in Plan | Notes |
|-------------|--------|------------------|-------|
| Section 3.1: Overview of methodology | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| Section 3.1: Framework architecture overview | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| Section 3.1: Data collection overview | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| Section 3.1: Analysis approach overview | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| Section 3.1: Mapping objectives to methodology (table) | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| Section 3.2: Justify method selection | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| Section 3.2: Explain why experimental method appropriate | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| Section 3.2: How framework represents production systems | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| Section 3.2: Why alternative methods excluded | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| Section 3.3: Framework implementation (methodological framing) | ✅ Covered | Task 1.3 | Restructure 3.3.3 |
| Section 3.3: Data collection procedures | ✅ Covered | Task 2.2, 2.3 | Move to 3.3.2 |
| Section 3.3: Framework validation | ✅ Covered | Task 1.4 | Enhance 3.3.4 |
| Construct validity questions | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| Internal validity considerations | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| External validity considerations | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |
| Data validity considerations | ⚠️ **MISSING** | - | **NEW TASK NEEDED** |

---

## PART 4: VERIFICATION FROM data-analysis-and-presentation.txt

Key requirements from university guidance:

| Requirement | Status | Location in Plan | Notes |
|-------------|--------|------------------|-------|
| Chapter 4: Process raw data | ✅ Covered | Task 4.1 | Reference to 3.3.2 |
| Chapter 4: Present data (tabular/pictorial) | ✅ Covered | Task 4.2, 4.3 | Figure references |
| Chapter 4: Analyze using appropriate techniques | ✅ Covered | Task 4.1 | Reference to 3.3.2 |
| Chapter 4: Reach conclusions | ✅ Covered | - | Keep in Chapter 4 |
| Chapter 4: No new concepts | ✅ Covered | Phase 2, Phase 3 | Content movement |
| Chapter 4: Interpret data carefully | ✅ Covered | Task 4.3 | Graph discussions |
| Chapter 4: Context for conclusions | ✅ Covered | - | Keep in Chapter 4 |
| Statistical tests: appropriate selection | ✅ Covered | Task 5.2 | Justify methods |
| Statistical tests: sufficient data | ✅ Covered | - | Keep in Chapter 4 |
| Graphical representation: appropriate selection | ✅ Covered | Task 4.2, 4.3 | Explain choices |
| All graphs discussed in text | ✅ Covered | Task 4.2, 4.3 | Add discussions |

---

## PART 5: MISSING ITEMS IDENTIFIED

### Critical Missing Items (High Priority)

1. **Section 3.1 Structure and Content** (from Chapter3-review)
   - 3.1 should cover overview of everything
   - Need to verify current 3.1 content
   - May need to restructure

2. **Section 3.2 Structure and Content** (from Chapter3-review)
   - 3.2 should talk about what/why with respect to objectives
   - Why exclude methods like survey/observations
   - How framework represents live/real-world systems
   - Need to verify current 3.2 content

3. **Block Diagram of Live System** (from Chapter3-review)
   - Need block diagram of live system
   - Show where instrumentation is placed
   - Draw live system, draw how we compare
   - This is DIFFERENT from framework diagram

4. **Figure in 3.1.1** (from Chapter3-review)
   - Describe each block, what it does
   - Need to check if figure exists and if it's described

5. **Code to Appendices** (from Chapter3-review)
   - Code shouldn't be in main text
   - Move to appendices (but don't waste time on appendices)

6. **Section 3.1 Requirements** (from methodology-and-techniques.txt)
   - Overview of methodology
   - Framework architecture overview
   - Data collection overview
   - Analysis approach overview
   - Mapping objectives to methodology (table)

7. **Section 3.2 Requirements** (from methodology-and-techniques.txt)
   - Justify method selection
   - Explain why experimental method appropriate
   - How framework represents production systems
   - Why alternative methods excluded

8. **Validity Considerations** (from methodology-and-techniques.txt)
   - Construct validity
   - Internal validity
   - External validity
   - Data validity

9. **Objective Mapping** (from feedback-draft-1)
   - "Which objective is this?" for consistency claim in 4.2.3
   - Need to ensure all claims map to objectives

10. **Ellipsis Items** (from feedback-draft-1)
    - 3.e. (other items)
    - 5.f. (other items)
    - Need to identify what else might be missing

---

## PART 6: CONFLICTS AND CLARIFICATIONS NEEDED

### Potential Conflicts

1. **Section 3.1 Detail Level**
   - Chapter3-review: "3.1 should cover overview of everything"
   - methodology-and-techniques.txt: "Overview of methodology, framework architecture overview, data collection overview, analysis approach overview"
   - **Question**: Is current 3.1 sufficient, or does it need expansion?

2. **Section 3.2 Content**
   - Chapter3-review: "3.2 talks about what/why with respect to objectives; why exclude methods"
   - methodology-and-techniques.txt: "Justify method selection, explain why experimental method appropriate"
   - **Question**: Does current 3.2 cover this, or does it need restructuring?

3. **Block Diagrams**
   - Chapter3-review: "Make up some simple block diagram and a block diagram of a live system"
   - **Question**: Do we need TWO diagrams (framework + live system), or one comparison diagram?

4. **Telemetry Focus**
   - Chapter3-review: "Focus on the bits that we're interested in - how we make the measurements"
   - **Question**: Should we restructure 3.3.3 to focus more on measurement methodology?

5. **Code in Appendices**
   - Chapter3-review: "Code shouldn't be in main text - can go into appendices"
   - **Question**: Is there code in the current dissertation that needs to be moved?

---

## PART 7: UPDATED ACTION PLAN ADDITIONS

### New Tasks to Add

#### Task 0.1: Review Current Section 3.1
- **Location**: Section 3.1
- **Action**: Read current 3.1 content, verify it covers overview
- **Check**: Does it match Chapter3-review and methodology-and-techniques.txt requirements?

#### Task 0.2: Review Current Section 3.2
- **Location**: Section 3.2
- **Action**: Read current 3.2 content, verify it covers justification
- **Check**: Does it explain what/why with respect to objectives? Does it exclude alternative methods?

#### Task 0.3: Enhance Section 3.1 (if needed)
- **Location**: Section 3.1
- **Action**: Add/expand:
  - Overview of methodology
  - Framework architecture overview
  - Data collection overview
  - Analysis approach overview
  - Mapping objectives to methodology (table)

#### Task 0.4: Enhance Section 3.2 (if needed)
- **Location**: Section 3.2
- **Action**: Add/expand:
  - What/why with respect to objectives
  - Why exclude methods like survey/observations
  - How framework represents live/real-world systems
  - Justification for experimental method

#### Task 0.5: Create Block Diagram of Live System
- **Location**: Section 3.3
- **Action**: Create diagram showing:
  - Live system architecture
  - Where instrumentation is placed
  - How framework compares to live system
- **Note**: This is DIFFERENT from framework diagram

#### Task 0.6: Describe Figure in 3.1.1
- **Location**: Section 3.1.1
- **Action**: If figure exists, describe each block and what it does
- **Check**: Does figure exist? Is it described?

#### Task 0.7: Move Code to Appendices
- **Location**: Throughout Chapter 3
- **Action**: Identify any code snippets, move to appendices
- **Note**: Don't waste time on appendices, but key bits if critical

#### Task 0.8: Add Validity Considerations
- **Location**: Section 3.3 (new subsection or integrate)
- **Action**: Add discussion of:
  - Construct validity
  - Internal validity
  - External validity
  - Data validity

#### Task 0.9: Map Claims to Objectives
- **Location**: Throughout Chapter 4
- **Action**: Ensure all performance claims map to research objectives
- **Specific**: "Which objective is this?" for consistency claim in 4.2.3

#### Task 0.10: Emphasize Telemetry Focus
- **Location**: Section 3.3.3
- **Action**: Restructure to focus on "how we make the measurements"
- **Note**: "There's no need to talk about stuff that doesn't affect telemetry"

---

## PART 8: VERIFICATION CHECKLIST

Before proceeding with implementation:

- [ ] Review current Section 3.1 content
- [ ] Review current Section 3.2 content
- [ ] Check if figure exists in 3.1.1
- [ ] Identify any code snippets in Chapter 3
- [ ] Identify what "3.e." and "5.f." might refer to
- [ ] Clarify with supervisor:
  - [ ] Is current 3.1 sufficient?
  - [ ] Is current 3.2 sufficient?
  - [ ] Do we need TWO diagrams (framework + live system)?
  - [ ] What other items might be in "3.e." and "5.f."?
- [ ] Update action plan with new tasks
- [ ] Verify all conflicts resolved

---

## PART 9: SUMMARY

### Items Covered: ✅
- Most feedback-draft-1 items
- Most data-analysis-and-presentation.txt items
- Framework diagram and restructuring
- Content movement from Chapter 4 to Chapter 3
- Figure references and discussions

### Items Missing: ⚠️
- Section 3.1 structure and content verification
- Section 3.2 structure and content verification
- Block diagram of live system (separate from framework)
- Figure in 3.1.1 description
- Code to appendices
- Validity considerations
- Objective mapping for all claims
- Telemetry focus emphasis

### Next Steps:
1. Review current Sections 3.1 and 3.2
2. Identify missing items
3. Add new tasks to action plan
4. Clarify conflicts with supervisor
5. Proceed with updated plan

---

**Status**: Verification complete - **MISSING ITEMS IDENTIFIED** - Plan needs update before implementation

