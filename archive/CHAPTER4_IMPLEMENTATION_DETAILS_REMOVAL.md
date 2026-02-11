# Chapter 4: Implementation Details Removal Summary

**Date**: 2025-12-15  
**Status**: ✅ **COMPLETED**

---

## Objective

Remove implementation/low-level details from Chapter 4 to ensure it focuses on analysis and interpretation rather than technical implementation specifics. The chapter should be accessible to another student at the same level, with implementation details moved to Chapter 3 (Methodology) or an appendix if needed.

---

## Changes Made

### 1. Section 4.1.2: Data Collection Methodology

**Before**:
- Mentioned specific function calls: `Instant::now()`, `getrusage()`, `/proc` filesystem
- Mentioned specific file format: JSONL
- Mentioned specific implementation detail: ChaCha20 random number generation

**After**:
- "high-resolution timing instrumentation with nanosecond precision"
- "system-level resource monitoring interfaces"
- "deterministic workload generation"
- Removed specific function names, file formats, and implementation details

**Rationale**: These are methodology details that belong in Chapter 3. Chapter 4 should focus on what was measured, not how it was measured.

---

### 2. Section 4.1.4: Data Processing and Aggregation

**Before**:
- Mentioned specific directory: `final-results/` directory

**After**:
- Removed directory reference entirely

**Rationale**: Directory structure is an implementation detail. The analysis pipeline description is sufficient without specifying where files are stored.

---

### 3. Section 4.2.3: Measurement Validity and Noise Considerations

**Before**:
- "Latency measurements employ Rust's monotonic clock primitives (`Instant::now()`)"
- "Timing operations (`Instant::now()` calls) and telemetry serialisation"
- "deterministic workload generation using seeded random number generation"

**After**:
- "Latency measurements employ high-resolution timing instrumentation with nanosecond precision"
- "Timing operations and telemetry capture"
- "deterministic workload generation"

**Rationale**: Specific function calls and implementation details are methodology concerns. Chapter 4 should focus on measurement validity concepts, not implementation specifics.

---

### 4. Section 4.3.2: Objective 2 Interpretation

**Before**:
- "The framework collected event-level telemetry (Section 4.1.1), with output schema enabling automated statistical analysis and visualisation generation."

**After**:
- "Event-level telemetry was collected (Section 4.1.1), enabling automated statistical analysis and visualisation generation."

**Rationale**: "Output schema" is an implementation detail. The fact that telemetry was collected is sufficient.

---

## What Was Kept (Appropriately)

The following references to methodology are appropriate and were kept:
- References to Section 3.3.1, Section 3.3.3 (methodology sections)
- High-level descriptions of measurement approach (without specific function calls)
- Statistical methods (Welch's t-test, Mann-Whitney U, Cohen's d) - these are analysis methods, not implementation details
- Performance metrics (latency, throughput, resource utilisation) - these are analysis concepts
- Experimental design elements (environments, algorithms, workload parameters) - these are experimental scope, not implementation

---

## Verification

✅ **Removed**: Specific function calls (`Instant::now()`, `getrusage()`)  
✅ **Removed**: Specific file formats (JSONL)  
✅ **Removed**: Specific directory paths (`final-results/`)  
✅ **Removed**: Specific implementation details (ChaCha20, seeded RNG, output schema)  
✅ **Kept**: High-level methodology references (Section 3.3.1, etc.)  
✅ **Kept**: Statistical methods (appropriate for analysis chapter)  
✅ **Kept**: Performance metrics and experimental scope  

---

## Result

Chapter 4 now focuses on:
- **What** was measured (performance metrics, algorithms, environments)
- **What** was found (statistical results, performance comparisons)
- **What** it means (interpretation, implications, conclusions)

Implementation details (how measurements were made, what tools were used, where files are stored) are appropriately in Chapter 3 (Methodology) or can be moved to an appendix if needed.

The chapter is now more accessible to readers at the same academic level, focusing on analysis and interpretation rather than technical implementation specifics.
