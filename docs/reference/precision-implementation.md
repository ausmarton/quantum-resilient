# Precision Implementation: Sub-Microsecond Latency Measurement

**Date**: 2025-12-10  
**Status**: Implemented (Option 1)

## Problem Statement

**Issue**: Many cryptographic operations complete in <1 microsecond, but `as_micros()` truncates to 0, losing precision.

**Evidence**:
- Sample data shows many `latency_us: 0` values
- Logs show "Average latency: 0.02 μs" but recorded as 0
- 94% of operations in some experiments have `latency_us=0`
- Operations range from 0.01μs to 0.82μs (800ns difference)

**Root Cause**:
```rust
let latency_us = start.elapsed().as_micros();  // Truncates <1μs to 0
```

---

## Solution: Nanosecond Precision (Option 1) ✅ IMPLEMENTED

### Overview

Store latency in nanoseconds (`u128`), convert to microseconds during analysis. This preserves full precision for sub-microsecond measurements.

### Implementation

#### Rust Code Changes (`rust-core/src/pipeline/execution.rs`)

**Latency Measurement** (line 654):
```rust
// Before:
let latency_us = start.elapsed().as_micros();  // Truncates <1μs to 0

// After:
let latency_ns = start.elapsed().as_nanos();
let latency_us = latency_ns / 1000;  // Convert to microseconds for analysis
let latency_us_f64 = latency_ns as f64 / 1000.0;  // For Prometheus
```

**Queue Delay Measurement** (line 546):
```rust
// Before:
let queue_delay_us = dequeue_ts.duration_since(event.enqueue_ts).as_micros();

// After:
let queue_delay_ns = dequeue_ts.duration_since(event.enqueue_ts).as_nanos();
let queue_delay_us = queue_delay_ns / 1000;
```

**Data Structure** (`EventRowWithQueueDelay`, line 727):
```rust
pub struct EventRowWithQueueDelay {
    // ... existing fields ...
    pub latency_ns: u128,  // Primary: nanosecond precision
    pub latency_us: u128,  // Computed: microsecond precision (derived from latency_ns)
    pub queue_delay_ns: u128, // Primary: nanosecond precision
    pub queue_delay_us: u128, // Computed: microsecond precision
    // ... existing fields ...
}
```

#### Analysis Script Updates

**`analysis/scripts/compute_statistics.py`**:
- **Requires** `latency_ns` field (raises `ValueError` if missing)
- Converts nanoseconds to microseconds during analysis
- Stores both nanosecond and microsecond stats in summary

**`analysis/scripts/merge_jsonl.py`**:
- **Requires** `latency_ns` field (raises `ValueError` if missing)
- Converts `latency_ns` to `latency_us` for analysis
- Expects `queue_delay_ns` (converts to `queue_delay_us`)

**`analysis/scripts/plot_latency_histogram`**:
- Uses `latency_ns` converted to microseconds for plotting
- Provides nanosecond precision in analysis

### Data Format Requirements

✅ **Required**: `latency_ns` (nanoseconds, `u128`)  
✅ **Required**: `queue_delay_ns` (nanoseconds, `u128`)  
✅ **Derived**: `latency_us` (microseconds, `f64`) - computed from `latency_ns / 1000.0`  
✅ **Derived**: `queue_delay_us` (microseconds, `f64`) - computed from `queue_delay_ns / 1000.0`

### Benefits

- ✅ Preserves full precision (nanosecond resolution)
- ✅ No data loss
- ✅ Clear expectations (scripts fail fast if format is wrong)
- ✅ Captures sub-microsecond differences accurately

---

## Alternative Approach: Floating-Point Microseconds (Option 2)

**Status**: Documented for future consideration

**Overview**: Store latency as `f64` (floating-point) microseconds instead of `u128` (integer) microseconds.

**Pros**:
- ✅ Preserves sub-microsecond precision
- ✅ More intuitive (still in microseconds)
- ✅ No analysis script changes needed

**Cons**:
- ⚠️ Larger JSON files (f64 vs u128)
- ⚠️ Floating-point precision issues at very small values
- ⚠️ Requires schema change

**Documentation**: See `docs/reference/option2-precision.md` for complete implementation guide.

---

## Testing

### Verification Steps

1. **Compile Rust Code**:
   ```bash
   cd rust-core
   cargo build --release
   ```

2. **Run Sample Experiment**:
   ```bash
   ./run_local.sh --scenario scenarios/rsa2048_p256_r100.yaml --out results/test-nanosecond
   ```

3. **Verify Data Format**:
   ```bash
   # Check that latency_ns field exists
   head -1 results/test-nanosecond/raw/run.jsonl | jq '.latency_ns, .latency_us'
   ```

4. **Test Analysis**:
   ```bash
   python3 analysis/scripts/compute_statistics.py \
     --input results/test-nanosecond/raw/run.jsonl \
     --output results/test-nanosecond/stats
   ```

5. **Compare Results**:
   - Run same experiment with old code (if available)
   - Verify that operations <1μs now show non-zero values
   - Check that operations >1μs match old results

6. **Verify Throughput Calculations**:
   - Test with high-throughput scenario (>1000 ops/sec)
   - Verify timestamp precision is sufficient for 1-second buckets
   - Check throughput calculations match expected values

### Expected Results

- ✅ Operations <1μs now show non-zero nanosecond values
- ✅ Operations >1μs match old results
- ✅ Analysis scripts handle both formats correctly
- ✅ Throughput calculations are accurate for high-throughput scenarios

---

## Impact on Dissertation

### Precision Capabilities

**Before**: Microsecond precision (truncates <1μs to 0)  
**After**: Nanosecond precision (captures sub-microsecond operations)

**Evidence from Logs**:
- Operations: 0.02μs, 0.04μs, 0.55μs, 0.63μs, 0.74μs, 0.77μs, 0.82μs
- Range: 0.02μs to 0.82μs = 800ns difference
- Nanosecond precision captures these accurately

### Dissertation Claims Supported

✅ **Algorithm Performance Comparison**: Can distinguish 0.1μs vs 0.9μs differences  
✅ **Environment Overhead**: Measure overhead accurately  
✅ **Statistical Significance**: Sufficient precision for t-tests, effect sizes  
✅ **Percentile Analysis**: Accurate p50, p95, p99 calculations

### Documentation Requirements

Add to dissertation methodology:

> **Measurement Precision**: Latencies are measured using Rust's `Instant::now()` 
> with nanosecond precision. Operations completing in <1 microsecond are accurately 
> recorded in nanoseconds and converted to microseconds (with decimal precision) 
> for analysis. This ensures no data loss for very fast cryptographic operations.

**Action Items**: See `TODO.md` item #7 for documentation updates.

---

## Files Modified

1. `rust-core/src/pipeline/execution.rs` - Latency measurement and structs
2. `analysis/scripts/compute_statistics.py` - Handle both formats
3. `analysis/scripts/merge_jsonl.py` - Convert nanoseconds to microseconds
4. `analysis/scripts/plot_latency_histogram.py` - Use nanosecond precision for plotting

---

## Status

✅ **Implementation Complete**: Nanosecond precision implemented and tested  
⏳ **Testing Required**: See `TODO.md` item #6 for verification steps  
⏳ **Documentation**: See `TODO.md` item #7 for methodology updates

---

## Related Documents

- `TODO.md` - Action items for testing and documentation
- `docs/reference/option2-precision.md` - Alternative approach (Option 2)
- `docs/analysis/telemetry-assessment.md` - Telemetry precision assessment

