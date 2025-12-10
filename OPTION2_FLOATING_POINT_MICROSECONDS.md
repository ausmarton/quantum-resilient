# Option 2: Floating-Point Microseconds - Detailed Implementation Guide

## Overview

**Option 2** stores latency as `f64` (floating-point) microseconds instead of `u128` (integer) microseconds. This preserves sub-microsecond precision while maintaining the microsecond unit.

---

## Implementation Details

### Changes Required

#### 1. Rust Code Changes

**File**: `rust-core/src/pipeline/execution.rs`

**Current Code** (line 654):
```rust
let latency_us = start.elapsed().as_micros();  // Returns u128, truncates <1μs
```

**New Code**:
```rust
let latency_us = start.elapsed().as_secs_f64() * 1_000_000.0;  // Returns f64, preserves precision
```

**Struct Changes** (line 727):
```rust
// Current:
pub latency_us: u128,

// New:
pub latency_us: f64,  // Floating-point microseconds
```

**ProcessedEvent Struct** (line 35):
```rust
// Current:
pub latency_us: u128,

// New:
pub latency_us: f64,
```

#### 2. Queue Delay Changes

**File**: `rust-core/src/pipeline/execution.rs` (line 546)

**Current Code**:
```rust
let queue_delay_us = dequeue_ts.duration_since(event.enqueue_ts).as_micros();
```

**New Code**:
```rust
let queue_delay_us = dequeue_ts.duration_since(event.enqueue_ts).as_secs_f64() * 1_000_000.0;
```

**Struct Changes**:
```rust
// Current:
pub queue_delay_us: u128,

// New:
pub queue_delay_us: f64,
```

#### 3. Internal State Tracking

**File**: `rust-core/src/pipeline/execution.rs` (line 254, 525)

**Current Code**:
```rust
state.total_latency_us.fetch_add(result.latency_us as u64, Ordering::Relaxed);
```

**New Code**:
```rust
// Need to change AtomicU64 to AtomicU128 or use f64 accumulation
// Option A: Use AtomicU128 for integer nanoseconds, convert at end
let latency_ns = (result.latency_us * 1000.0) as u128;
state.total_latency_us.fetch_add(latency_ns / 1000, Ordering::Relaxed);

// Option B: Change to f64 accumulation (requires different atomic type)
// This is more complex and may have precision issues
```

**Recommendation**: Keep internal state as integer microseconds (`u64`) by rounding:
```rust
state.total_latency_us.fetch_add(result.latency_us.round() as u64, Ordering::Relaxed);
```

#### 4. Analysis Script Changes

**File**: `analysis/scripts/compute_statistics.py`

**Current Code**:
```python
if "latency_us" in df.columns:
    summary["latency"] = compute_basic_stats(df["latency_us"])
```

**New Code**:
```python
if "latency_us" in df.columns:
    # Handle both integer (old) and float (new) formats
    latency_series = df["latency_us"]
    if latency_series.dtype == 'float64':
        # New format: already floating-point
        summary["latency"] = compute_basic_stats(latency_series)
    else:
        # Old format: convert to float for consistency
        summary["latency"] = compute_basic_stats(latency_series.astype('float64'))
```

---

## Pros and Cons

### Pros ✅

1. **Preserves Sub-Microsecond Precision**
   - Operations completing in 0.1μs, 0.5μs, 0.9μs are accurately recorded
   - No data loss for fast operations

2. **Intuitive Unit**
   - Still in microseconds (familiar unit)
   - No need to convert during analysis

3. **Minimal Analysis Script Changes**
   - Most scripts already handle float64
   - Just need to ensure type consistency

4. **Backward Compatible**
   - Can convert old integer data to float during analysis
   - Existing data remains valid

### Cons ⚠️

1. **Larger JSON Files**
   - `f64` (8 bytes) vs `u128` (16 bytes) - actually smaller!
   - But JSON representation is larger (e.g., `0.5` vs `0`)
   - Estimated: ~10-15% larger JSON files

2. **Floating-Point Precision Issues**
   - Very small values (<0.001μs) may lose precision
   - Accumulation errors over many operations
   - **Mitigation**: Use `f64` (double precision) - sufficient for microsecond precision

3. **Type Conversion Complexity**
   - Internal state tracking uses integers
   - Need careful conversion between float and int
   - **Mitigation**: Round to nearest microsecond for state tracking

4. **Prometheus Metrics**
   - Prometheus histograms expect float64 anyway
   - No change needed here

---

## Migration Path

### Phase 1: Code Changes
1. Update Rust structs to use `f64` for `latency_us` and `queue_delay_us`
2. Change measurement to use `as_secs_f64() * 1_000_000.0`
3. Update internal state tracking (round to nearest microsecond)

### Phase 2: Analysis Updates
1. Update analysis scripts to handle both integer and float formats
2. Add type checking and conversion logic
3. Test with both old and new data formats

### Phase 3: Testing
1. Run sample experiment with new format
2. Verify sub-microsecond values are preserved
3. Compare results with old format (should match for >1μs operations)

### Phase 4: Documentation
1. Update dissertation methodology
2. Document precision capabilities
3. Note backward compatibility approach

---

## Code Example: Complete Implementation

### Rust Changes

```rust
// rust-core/src/pipeline/execution.rs

// Line 35: ProcessedEvent struct
pub struct ProcessedEvent {
    pub event_id: u64,
    pub latency_us: f64,  // Changed from u128
    pub queue_delay_us: f64,  // Changed from u128
    pub success: bool,
    pub output_size: Option<usize>,
    pub worker_id: usize,
}

// Line 546: Queue delay measurement
let queue_delay_us = dequeue_ts.duration_since(event.enqueue_ts).as_secs_f64() * 1_000_000.0;

// Line 654: Latency measurement
let latency_us = start.elapsed().as_secs_f64() * 1_000_000.0;

// Line 255, 525: Internal state tracking (round to nearest microsecond)
state.total_latency_us.fetch_add(latency_us.round() as u64, Ordering::Relaxed);

// Line 727: EventRowWithQueueDelay struct
pub struct EventRowWithQueueDelay {
    // ... other fields ...
    pub latency_us: f64,  // Changed from u128
    pub queue_delay_us: f64,  // Changed from u128
    // ... other fields ...
}
```

### Python Analysis Changes

```python
# analysis/scripts/compute_statistics.py

# Handle both integer (old) and float (new) formats
if "latency_us" in df.columns:
    latency_series = df["latency_us"]
    
    # Ensure float64 for consistent analysis
    if latency_series.dtype != 'float64':
        latency_series = latency_series.astype('float64')
    
    summary["latency"] = compute_basic_stats(latency_series)
```

---

## Comparison: Option 1 vs Option 2

| Aspect | Option 1 (Nanoseconds) | Option 2 (Float Microseconds) |
|--------|------------------------|-------------------------------|
| **Precision** | Nanosecond (1ns) | Sub-microsecond (0.001μs = 1ns) |
| **Storage** | `u128` (16 bytes) | `f64` (8 bytes) |
| **JSON Size** | Larger (big integers) | Medium (decimals) |
| **Unit** | Nanoseconds | Microseconds |
| **Analysis Changes** | Convert ns → μs | Minimal (already float) |
| **Backward Compat** | Good (can convert) | Good (can convert) |
| **Precision Loss** | None | Minimal (<0.001μs) |

---

## Recommendation

**For Dissertation**: Option 1 (nanoseconds) is recommended because:
- ✅ No precision loss at any scale
- ✅ More explicit about measurement precision
- ✅ Better for very fast operations (<0.1μs)

**For Future**: Option 2 (float microseconds) could be considered if:
- JSON file size becomes a concern
- You want to maintain microsecond unit throughout
- Analysis scripts prefer float64

---

## Testing Checklist

- [ ] Update Rust code to use `f64` for latency
- [ ] Update internal state tracking (round to nearest μs)
- [ ] Update analysis scripts to handle float format
- [ ] Run sample experiment (RSA-2048, small payload)
- [ ] Verify sub-microsecond values are preserved
- [ ] Compare with old format (should match for >1μs)
- [ ] Check JSON file size increase
- [ ] Verify Prometheus metrics still work
- [ ] Test backward compatibility (old data analysis)

---

## Estimated Effort

- **Code Changes**: 2-3 hours
- **Testing**: 1-2 hours
- **Analysis Updates**: 1 hour
- **Total**: 4-6 hours

---

**Status**: Documented for future consideration. Option 1 (nanoseconds) is currently being implemented.


