# Sub-Microsecond Latency Measurement Solution

## Problem

**Current Issue**: Many operations complete in <1 microsecond, but `as_micros()` truncates to 0, losing precision.

**Evidence**:
- Sample data shows many `latency_us: 0` values
- Logs show "Average latency: 0.02 μs" but recorded as 0
- 94% of operations in some experiments have `latency_us=0`

**Root Cause**:
```rust
let latency_us = start.elapsed().as_micros();  // Truncates <1μs to 0
```

## Solutions

### Option 1: Store Nanoseconds (Recommended) ✅

**Change**: Store latency in nanoseconds, convert to microseconds during analysis.

**Pros**:
- ✅ Preserves full precision (nanosecond resolution)
- ✅ No data loss
- ✅ Backward compatible (can convert existing data)
- ✅ Minimal code changes

**Cons**:
- ⚠️ Requires updating analysis scripts to handle nanoseconds
- ⚠️ Slightly larger JSON files (but negligible)

**Implementation**:
1. Change `latency_us: u128` → `latency_ns: u128` in Rust structs
2. Use `as_nanos()` instead of `as_micros()`
3. Update analysis scripts to convert `latency_ns / 1000.0` for microseconds
4. Keep `latency_us` as computed field for backward compatibility

### Option 2: Floating-Point Microseconds

**Change**: Store latency as `f64` microseconds instead of `u128`.

**Pros**:
- ✅ Preserves sub-microsecond precision
- ✅ No analysis script changes needed
- ✅ More intuitive (still in microseconds)

**Cons**:
- ⚠️ Larger JSON files (f64 vs u128)
- ⚠️ Floating-point precision issues at very small values
- ⚠️ Requires schema change

**Implementation**:
1. Change `latency_us: u128` → `latency_us: f64`
2. Use `start.elapsed().as_secs_f64() * 1_000_000.0`

### Option 3: Dual Storage (latency_us + latency_ns)

**Change**: Store both integer microseconds AND nanoseconds.

**Pros**:
- ✅ Backward compatible (latency_us still exists)
- ✅ Full precision available (latency_ns)
- ✅ No breaking changes

**Cons**:
- ⚠️ Redundant data
- ⚠️ Larger JSON files

## Recommended Approach: Option 1 (Nanoseconds)

### Implementation Steps

#### Step 1: Update Rust Code

**File**: `rust-core/src/pipeline/execution.rs`

```rust
// Change line 654 from:
let latency_us = start.elapsed().as_micros();

// To:
let latency_ns = start.elapsed().as_nanos();
let latency_us = latency_ns as f64 / 1000.0;  // For backward compatibility
```

**File**: `rust-core/src/pipeline/execution.rs` (struct definition)

```rust
// Change line 727 from:
pub latency_us: u128,

// To:
pub latency_ns: u128,  // Primary: nanosecond precision
pub latency_us: f64,   // Computed: microsecond precision (for compatibility)
```

#### Step 2: Update Analysis Scripts

**File**: `analysis/scripts/compute_statistics.py`

```python
# Handle both latency_us (old) and latency_ns (new)
if 'latency_ns' in df.columns:
    df['latency_us'] = df['latency_ns'] / 1000.0
elif 'latency_us' in df.columns:
    # Old data: already in microseconds
    pass
```

#### Step 3: Backward Compatibility

**For existing data**:
- Analysis scripts check for `latency_ns` first, fall back to `latency_us`
- Existing data continues to work
- New data has nanosecond precision

### Alternative: Quick Fix (No Code Changes)

**For Dissertation**: Document that `p50=0.0` means "median latency <1μs"

**Pros**:
- ✅ No code changes needed
- ✅ No re-running experiments
- ✅ Scientifically valid interpretation

**Cons**:
- ⚠️ Cannot distinguish between 0.1μs and 0.9μs
- ⚠️ Less precise for very fast operations

**Documentation**:
```markdown
**Measurement Precision**: Latencies are measured in microseconds (μs). 
Operations completing in <1μs are recorded as 0μs. For these operations, 
we report that "median latency <1μs" rather than exact values. This is 
scientifically valid as it indicates operations are extremely fast.
```

## Impact Assessment

### On Dissertation

**Current State** (with zero values):
- ✅ **Valid**: `p50=0.0` means "median <1μs"
- ✅ **Valid**: Can still compare algorithms (p95, p99 are non-zero)
- ✅ **Valid**: Statistical tests work (zero is a valid measurement)

**With Nanosecond Precision**:
- ✅ **Better**: Can distinguish 0.1μs vs 0.9μs
- ✅ **Better**: More precise percentiles
- ✅ **Better**: Better for very fast operations (RSA, ECDSA)

### On Existing Data

**Option 1 (Nanoseconds)**:
- ✅ Existing data still works (analysis scripts handle both)
- ✅ New experiments have better precision
- ✅ Can re-analyze existing data if needed

**Option 2 (Floating Point)**:
- ⚠️ Requires re-running experiments for full precision
- ⚠️ Existing data remains valid but less precise

**Option 3 (Documentation)**:
- ✅ No changes needed
- ✅ Existing data fully valid
- ⚠️ Less precise but acceptable for dissertation

## Recommendation

**✅ IMPLEMENTED: Option 1 (Nanoseconds)**
- ✅ **Code Updated**: Rust code now measures nanoseconds
- ✅ **Analysis Updated**: Scripts handle both formats
- ✅ **Backward Compatible**: Existing data still works
- ✅ **Better Precision**: Sub-microsecond measurements preserved

**For Future Consideration**:
- 📋 **Option 2**: Floating-point microseconds (see `OPTION2_FLOATING_POINT_MICROSECONDS.md`)
- 📋 **Option 3**: Document zero values (no longer needed with Option 1)

## Implementation Priority

1. **Immediate** (Dissertation): Document zero values as "<1μs"
2. **Short-term** (Post-dissertation): Implement nanosecond precision
3. **Long-term**: Consider floating-point microseconds if needed

---

## Quick Implementation (Option 1)

If you want to implement nanosecond precision now:

### 1. Update Rust Code

```rust
// rust-core/src/pipeline/execution.rs line 654
let latency_ns = start.elapsed().as_nanos();
let latency_us_f64 = latency_ns as f64 / 1000.0;

// Update struct (line 727)
pub latency_ns: u128,
pub latency_us: f64,  // Computed for compatibility
```

### 2. Update Analysis Scripts

Add to `analysis/scripts/compute_statistics.py`:

```python
# Handle nanosecond precision
if 'latency_ns' in df.columns:
    df['latency_us'] = df['latency_ns'] / 1000.0
```

### 3. Test

Run a single experiment and verify:
- `latency_ns` field exists
- Values are non-zero for fast operations
- Analysis scripts handle both formats

---

**Conclusion**: For dissertation, documenting zero values is sufficient. For future precision, implement nanosecond storage.

