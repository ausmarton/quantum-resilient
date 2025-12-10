# Outstanding Work Items

This document tracks all outstanding work items identified across the codebase, with sufficient detail to investigate and implement as separate tasks. Action items from other documentation files have been consolidated here to avoid duplication.

**Last Updated**: 2025-12-10

---

## Critical Priority

### 1. Investigate CPU Sampling Issue

**Status**: 🔴 **CRITICAL - INVESTIGATION REQUIRED**

**Issue**: 
- All CPU values (`cpu_user_seconds`) are 0.0 in sample data
- This prevents CPU utilization analysis
- May affect resource utilization claims in dissertation

**Possible Causes**:
1. Operations too fast (<1ms) for CPU sampling to accumulate measurable time
2. CPU sampling implementation bug
3. Cumulative metric not incrementing correctly
4. `sysinfo` crate limitation for very short-lived processes

**Investigation Steps**:
1. **Check CPU sampling implementation** (`rust-core/src/telemetry/sysinfo_sampler.rs`)
   - Verify `process.cpu_usage()` is being called correctly
   - Check if CPU time accumulates over multiple samples
   - Verify the conversion from percentage to seconds

2. **Test with longer operations**:
   - Run experiment with slower algorithm (e.g., Dilithium)
   - Check if CPU values are non-zero for operations >1 second
   - Verify CPU accumulation over experiment duration

3. **Check system-level CPU metrics**:
   - Verify if `sysinfo` can detect CPU usage for very fast operations
   - Consider alternative: Use `/proc/self/stat` directly for more precision
   - Check if cumulative CPU time is available from system

4. **Verify data collection**:
   - Check if CPU sampling happens before/after crypto operation
   - Verify timing of CPU sample relative to operation

**Expected Outcome**:
- Determine if CPU sampling works for longer operations
- Identify root cause of zero values
- Decide if fix is possible or if limitation should be documented

**Effort**: 1-2 hours (investigation)

**Dependencies**: None

**Impact**: 
- **HIGH**: Affects resource utilization claims
- **MEDIUM**: May need to document limitation for fast operations

**Related Files**:
- `rust-core/src/telemetry/sysinfo_sampler.rs`
- `rust-core/src/pipeline/execution.rs` (line 667)
- Sample data: `results/native/rsa2048_p256_r100_run1_c0098396/raw/run.jsonl`

---

### 2. Add Memory Utilization Analysis

**Status**: 🟡 **HIGH PRIORITY - IMPLEMENTATION REQUIRED**

**Issue**: 
- Memory data (`memory_rss_bytes`) is captured but not analyzed
- Cannot make memory utilization claims without analysis

**Current State**:
- ✅ Memory data is valid (9-10MB native, 6-7MB minikube)
- ✅ Data varies appropriately
- ❌ Analysis not implemented in `compute_statistics.py`

**Implementation Required**:

**File**: `analysis/scripts/compute_statistics.py`

**Add to `compute_statistics()` function**:
```python
# Memory utilization stats
if "memory_rss_bytes" in df.columns:
    summary["memory"] = {
        "mean_rss_bytes": float(df["memory_rss_bytes"].mean()),
        "max_rss_bytes": int(df["memory_rss_bytes"].max()),
        "min_rss_bytes": int(df["memory_rss_bytes"].min()),
        "std_rss_bytes": float(df["memory_rss_bytes"].std()),
        "p50_rss_bytes": float(df["memory_rss_bytes"].quantile(0.50)),
        "p95_rss_bytes": float(df["memory_rss_bytes"].quantile(0.95)),
        "p99_rss_bytes": float(df["memory_rss_bytes"].quantile(0.99)),
    }
```

**Per-Algorithm Memory Stats**:
```python
# In per-algorithm loop
if "memory_rss_bytes" in algo_df.columns:
    summary["per_algorithm"][algo]["memory"] = {
        "mean_rss_bytes": float(algo_df["memory_rss_bytes"].mean()),
        "max_rss_bytes": int(algo_df["memory_rss_bytes"].max()),
    }
```

**Per-Environment Comparison**:
- Add memory comparison to `compare_all_environments.py`
- Add memory plots to visualization scripts

**Testing**:
1. Run `compute_statistics.py` on existing data
2. Verify memory stats appear in `summary.json`
3. Check memory values are reasonable (6-10MB range)
4. Verify per-algorithm memory stats

**Expected Outcome**:
- Memory utilization metrics available in `summary.json`
- Can make memory-related claims in dissertation
- Memory comparison across environments possible

**Effort**: 1-2 hours (implementation + testing)

**Dependencies**: None (data already available)

**Impact**: 
- **HIGH**: Enables memory utilization claims
- **MEDIUM**: Supports resource efficiency analysis

**Related Files**:
- `analysis/scripts/compute_statistics.py`
- `analysis/compare_all_environments.py`
- `analysis/scripts/plot_latency.py` (add memory plots)

---

### 3. Add CPU Utilization Analysis (If Data Valid)

**Status**: 🟡 **HIGH PRIORITY - CONDITIONAL ON INVESTIGATION**

**Issue**: 
- CPU analysis not implemented
- Depends on CPU sampling investigation outcome

**Prerequisites**:
- ✅ Complete CPU sampling investigation (#1)
- ✅ Verify CPU data is valid for longer operations
- ✅ Confirm CPU delta calculation approach

**Implementation Required**:

**File**: `analysis/scripts/compute_statistics.py`

**Add CPU delta calculation**:
```python
# CPU utilization stats (if CPU data is valid)
if "cpu_user_seconds" in df.columns and "timestamp" in df.columns:
    # Calculate CPU delta between events
    df["cpu_delta"] = df["cpu_user_seconds"].diff()
    df["time_delta"] = df["timestamp"].diff().dt.total_seconds()
    
    # Filter out invalid deltas (first row, negative values)
    valid_mask = (df["cpu_delta"] > 0) & (df["time_delta"] > 0)
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) > 0:
        df_valid["cpu_utilization"] = df_valid["cpu_delta"] / df_valid["time_delta"]
        
        summary["cpu"] = {
            "mean_utilization": float(df_valid["cpu_utilization"].mean()),
            "max_utilization": float(df_valid["cpu_utilization"].max()),
            "cpu_per_operation": float(df["cpu_user_seconds"].iloc[-1] / len(df)),
            "total_cpu_seconds": float(df["cpu_user_seconds"].iloc[-1]),
        }
```

**Handling Edge Cases**:
- First event has no delta (skip)
- Negative deltas (system clock adjustment, skip)
- Zero time deltas (concurrent events, skip)
- All zeros (document limitation)

**Testing**:
1. Test with data that has non-zero CPU values
2. Verify CPU delta calculation is correct
3. Check CPU utilization percentages are reasonable (0-100%)
4. Handle edge cases gracefully

**Expected Outcome**:
- CPU utilization metrics available (if data valid)
- CPU efficiency analysis possible
- Resource efficiency claims supported

**Effort**: 1-2 hours (implementation + testing)

**Dependencies**: 
- CPU sampling investigation (#1) must be completed first
- Requires CPU data to be valid

**Impact**: 
- **HIGH**: Enables CPU utilization claims (if data valid)
- **MEDIUM**: Supports resource efficiency analysis

**Related Files**:
- `analysis/scripts/compute_statistics.py`
- `analysis/compare_all_environments.py`

---

## Medium Priority

### 4. Generate Missing Summary Files

**Status**: 🟡 **MEDIUM PRIORITY - DATA COMPLETENESS**

**Issue**: 
- 14 experiments have raw data but are missing `summary.json` files
- Affects: GCP experiments (dilithium2 and hybrid scaling experiments)
- Prevents complete analysis and aggregation

**Affected Experiments**:
- `dilithium2_p1024_r10000_run*` (4 experiments)
- `dilithium2_p1024_r100_run1` (1 experiment)
- `hybrid_kyber_dilithium_p1024_r500_scaling_*` (9 scaling experiments)

**Implementation**:

**Script**: `scripts/generate_missing_summaries.sh` (already exists)

**Steps**:
```bash
# Generate missing summary files
./scripts/generate_missing_summaries.sh

# Then re-run aggregation
python3 analysis/aggregate_results.py \
  --index final-results/index.json \
  --output final-results
```

**Verification**:
```bash
# Check that all experiments have summaries
find results -name "summary.json" | wc -l
# Should match number of experiments with raw data
```

**Expected Outcome**:
- All 14 missing summary files generated
- Complete data for aggregation and analysis
- All figures include complete data

**Effort**: 1-2 hours (generation + verification)

**Dependencies**: None (raw data exists)

**Impact**: 
- **MEDIUM**: Enables complete analysis
- **MEDIUM**: Required for dissertation completeness

**Related Files**:
- `scripts/generate_missing_summaries.sh`
- `analysis/aggregate_results.py`
- `final-results/index.json`

---

### 5. Enhance Data Validation

**Status**: 🟡 **MEDIUM PRIORITY - DATA QUALITY**

**Issue**: 
- Current validation scripts don't check for summary.json existence
- No statistical validity checks
- No cross-environment consistency checks
- Missing validations discovered late (14 missing summaries)

**Current Validation** (`validate_experiment_data.sh`):
- ✅ File existence checks
- ✅ File size validation (non-zero)
- ✅ JSONL format validation
- ✅ Required fields in first record

**Missing Validations**:
- ❌ Summary file existence check
- ❌ Statistical validity checks (percentiles reasonable)
- ❌ Data completeness (expected vs actual event counts)
- ❌ Cross-environment consistency

**Implementation Required**:

**File**: `scripts/validate_data_integrity.sh` (or create new validation script)

**Add Checks**:

1. **Summary File Existence**:
   ```bash
   # Check that summary.json exists after data collection
   if [[ ! -f "$OUTPUT_DIR/stats/summary.json" ]]; then
       echo "WARNING: Missing summary.json for $EXP_ID"
   fi
   ```

2. **Statistical Validity**:
   ```bash
   # Check that percentiles are reasonable (including zero as valid)
   p50=$(jq -r '.latency.p50 // empty' "$OUTPUT_DIR/stats/summary.json")
   if [[ -z "$p50" ]]; then
       echo "ERROR: Missing p50 in summary.json"
   fi
   # Note: p50=0.0 is valid (operations <1μs)
   ```

3. **Data Completeness**:
   ```bash
   # Compare expected events vs actual events
   expected_events=$(jq -r '.total_events_planned' "$SCENARIO_FILE")
   actual_events=$(jq -r '.total_events' "$OUTPUT_DIR/stats/summary.json")
   if [[ "$actual_events" -lt "$expected_events" ]]; then
       echo "WARNING: Expected $expected_events events, got $actual_events"
   fi
   ```

4. **Cross-Environment Consistency**:
   ```bash
   # Check that all environments have data for same experiments
   # Compare index.json across native/minikube/gcp
   ```

**Testing**:
1. Run validation on existing data
2. Verify it catches missing summaries
3. Verify it accepts zero values as valid
4. Test cross-environment consistency check

**Expected Outcome**:
- Enhanced validation catches issues early
- Prevents missing summary files
- Improves data quality assurance

**Effort**: 2-3 hours (implementation + testing)

**Dependencies**: None

**Impact**: 
- **MEDIUM**: Improves data quality assurance
- **LOW**: Prevents future issues

**Related Files**:
- `scripts/validate_data_integrity.sh`
- `scripts/validate_experiment_data.sh`
- `final-results/index.json`

---

### 6. Test Nanosecond Precision Implementation

**Status**: 🟡 **MEDIUM PRIORITY - VERIFICATION REQUIRED**

**Issue**: 
- Nanosecond precision implemented but not tested
- Need to verify data integrity and analysis compatibility
- Need to verify throughput calculations are accurate for high-throughput scenarios

**Testing Steps**:
1. **Compile Rust code**:
   ```bash
   cd rust-core
   cargo build --release
   ```

2. **Run sample experiment**:
   ```bash
   ./run_local.sh --scenario scenarios/rsa2048_p256_r100.yaml --out results/test-nanosecond
   ```

3. **Verify data format**:
   ```bash
   # Check that latency_ns field exists
   head -1 results/test-nanosecond/raw/run.jsonl | jq '.latency_ns, .latency_us'
   ```

4. **Test analysis compatibility**:
   ```bash
   python3 analysis/scripts/compute_statistics.py \
     --input results/test-nanosecond/raw/run.jsonl \
     --output results/test-nanosecond/stats
   ```

5. **Compare with old format**:
   - Run same experiment with old code (if available)
   - Verify operations <1μs now show non-zero values
   - Check operations >1μs match old results

6. **Verify throughput calculations**:
   - Test with high-throughput scenario (>1000 ops/sec)
   - Verify timestamp precision is sufficient for 1-second buckets
   - Check throughput calculations match expected values
   - Verify throughput scaling analysis works correctly

**Expected Outcome**:
- Verify nanosecond precision works correctly
- Confirm analysis scripts handle both formats
- Validate backward compatibility
- Confirm throughput calculations are accurate for high-throughput scenarios

**Effort**: 1 hour (testing)

**Dependencies**: None (implementation complete)

**Impact**: 
- **MEDIUM**: Validates implementation correctness
- **LOW**: Confirms backward compatibility

**Related Files**:
- `rust-core/src/pipeline/execution.rs`
- `analysis/scripts/compute_statistics.py`
- `analysis/scripts/merge_jsonl.py`

---

### 7. Update Dissertation Methodology Documentation

**Status**: 🟡 **MEDIUM PRIORITY - DOCUMENTATION**

**Issue**: 
- Methodology section needs precision documentation
- Resource metric limitations need documentation

**Documentation Required**:

**1. Measurement Precision Section**:
```markdown
## Measurement Precision

Latencies are measured using Rust's `Instant::now()` with nanosecond precision 
(1ns resolution). Operations completing in <1 microsecond are accurately 
recorded in nanoseconds and converted to microseconds (with decimal precision) 
for analysis. This ensures no data loss for very fast cryptographic operations 
and enables precise statistical analysis.

Queue delays are measured with nanosecond precision but stored in microseconds 
(sufficient since queue delays are typically >1μs). System metrics (CPU, memory) 
are sampled per event using the `sysinfo` crate for resource utilization analysis.
```

**2. Resource Metrics Limitations**:
```markdown
## Resource Metrics

**Memory**: Instantaneous RSS (Resident Set Size) is captured per event and 
analyzed directly. Memory metrics enable comparison across environments and 
analysis of memory efficiency.

**CPU**: Cumulative CPU time is captured per event. CPU utilization requires 
delta calculation between events. Note: For very fast operations (<1ms), CPU 
sampling may not accumulate measurable time, limiting CPU utilization analysis 
for sub-millisecond operations.
```

**3. Timestamp Precision**:
```markdown
## Timestamp Precision

Event timestamps are captured in two formats:
- `timestamp_utc_iso`: UTC timestamp in ISO 8601 format (millisecond precision)
- `timestamp_monotonic_ns`: Monotonic timestamp in nanoseconds

Throughput is calculated using 1-second buckets, for which millisecond precision 
is sufficient. Monotonic timestamps provide nanosecond precision for more 
detailed time-series analysis if needed.
```

**Effort**: 1 hour (documentation)

**Dependencies**: 
- CPU investigation (#1) - to document limitations accurately
- Memory analysis (#2) - to document capabilities

**Impact**: 
- **MEDIUM**: Ensures accurate methodology documentation
- **LOW**: Improves dissertation clarity

**Related Files**:
- Dissertation methodology section (external document)

---

## Low Priority (Optional Improvements)

### 8. Add Queue Delay Nanosecond Precision (Consistency)

**Status**: 🟢 **LOW PRIORITY - OPTIONAL CONSISTENCY IMPROVEMENT**

**Issue**: 
- Queue delay measured in nanoseconds but only stored in microseconds
- Inconsistent with latency fields (which store both `latency_ns` and `latency_us`)

**Current State**:
- ✅ Queue delays are typically ≥1μs (microsecond precision sufficient)
- ✅ All queue delays verified ≥1μs in sample data
- ⚠️ Inconsistent with latency field structure

**Implementation**:

**File**: `rust-core/src/pipeline/execution.rs`

**Update struct** (line 736):
```rust
pub struct EventRowWithQueueDelay {
    // ... existing fields ...
    pub queue_delay_ns: u128,  // Primary: nanosecond precision
    pub queue_delay_us: u128,  // Computed: microsecond precision
    // ... existing fields ...
}
```

**Update measurement** (line 546):
```rust
let queue_delay_ns = dequeue_ts.duration_since(event.enqueue_ts).as_nanos();
let queue_delay_us = queue_delay_ns / 1000;

let row = EventRowWithQueueDelay {
    // ... existing fields ...
    queue_delay_ns,  // Add this
    queue_delay_us,  // Keep this
    // ... existing fields ...
};
```

**Update analysis scripts**:
```python
# analysis/scripts/merge_jsonl.py
if "queue_delay_ns" in df.columns:
    df["queue_delay_us"] = df["queue_delay_ns"] / 1000.0
elif "queue_delay_us" in df.columns:
    # Old data: already in microseconds
    pass
```

**Testing**:
1. Verify backward compatibility with old data
2. Check new data includes both fields
3. Verify analysis scripts handle both formats

**Expected Outcome**:
- Consistent field structure with latency
- No functional change (queue delays ≥1μs)
- Improved code consistency

**Effort**: 1-2 hours (implementation + testing)

**Dependencies**: None

**Impact**: 
- **LOW**: Cosmetic consistency improvement
- **NONE**: No functional impact (queue delays ≥1μs)

**Related Files**:
- `rust-core/src/pipeline/execution.rs`
- `analysis/scripts/merge_jsonl.py`
- `analysis/scripts/compute_statistics.py`

---

### 9. Refine Prometheus Histogram Buckets

**Status**: 🟢 **LOW PRIORITY - OPTIONAL MONITORING IMPROVEMENT**

**Issue**: 
- Smallest Prometheus bucket is 0.5μs
- Operations <0.5μs all go into first bucket
- Less granular for very fast operations

**Current Buckets**:
```rust
vec![0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0, 100000.0]
```

**Proposed Buckets**:
```rust
vec![0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0, 100000.0]
```

**Implementation**:

**File**: `rust-core/src/telemetry/metrics.rs`

**Update latency histogram buckets** (line 54):
```rust
.buckets(vec![
    0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 
    1000.0, 5000.0, 10000.0, 50000.0, 100000.0
])
```

**Update queue delay histogram buckets** (line 95):
```rust
.buckets(vec![
    0.1, 0.2, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 
    5000.0, 10000.0, 50000.0, 100000.0, 500000.0, 1000000.0
])
```

**Testing**:
1. Verify Prometheus metrics endpoint still works
2. Check histogram distribution is reasonable
3. Verify no performance impact

**Expected Outcome**:
- Better granularity for very fast operations in Prometheus
- No impact on JSONL data (has full precision)
- Improved monitoring capabilities

**Effort**: 30 minutes (implementation + testing)

**Dependencies**: None

**Impact**: 
- **LOW**: Prometheus is supplementary (primary data is JSONL)
- **LOW**: Monitoring improvement only

**Related Files**:
- `rust-core/src/telemetry/metrics.rs`

---

## Completed Items

### ✅ Nanosecond Precision Implementation
- **Status**: Completed
- **Date**: 2025-12-10
- **Files Modified**: 
  - `rust-core/src/pipeline/execution.rs`
  - `analysis/scripts/compute_statistics.py`
  - `analysis/scripts/merge_jsonl.py`

### ✅ Telemetry Review
- **Status**: Completed
- **Date**: 2025-12-10
- **Outcome**: Identified gaps and created this document

### ✅ Option 2 Documentation
- **Status**: Completed
- **Date**: 2025-12-10
- **File**: `OPTION2_FLOATING_POINT_MICROSECONDS.md`

### ✅ Aggregation Logic Fix
- **Status**: Completed
- **Date**: 2025-12-10
- **File**: `analysis/aggregate_results.py`
- **Fix**: Removed filter `if r.p50 > 0` to accept zero values as valid
- **Result**: All 86 entries now have valid data (zero values included)

### ✅ Zero Value Handling
- **Status**: Completed
- **Date**: 2025-12-10
- **File**: `analysis/aggregate_results.py`
- **Fix**: Updated to check `if r.total_events > 0` instead of `if r.p50 > 0`
- **Result**: Zero latency values correctly recognized as valid measurements

---

## Priority Summary

**Critical** (Must complete before dissertation):
1. 🔴 Investigate CPU Sampling Issue
2. 🟡 Add Memory Utilization Analysis

**High** (Should complete if CPU data valid):
3. 🟡 Add CPU Utilization Analysis

**Medium** (Recommended):
4. 🟡 Generate Missing Summary Files
5. 🟡 Enhance Data Validation
6. 🟡 Test Nanosecond Precision Implementation
7. 🟡 Update Dissertation Methodology Documentation

**Low** (Optional):
8. 🟢 Add Queue Delay Nanosecond Precision
9. 🟢 Refine Prometheus Histogram Buckets

---

## Investigation Checklist

When investigating each item:

- [ ] Read relevant source code files
- [ ] Check existing data for patterns
- [ ] Test with sample experiments
- [ ] Document findings
- [ ] Implement fix (if applicable)
- [ ] Test fix with existing data
- [ ] Update this document with status

---

## Notes

- **CPU Sampling**: May be a fundamental limitation for very fast operations. If CPU values remain zero even for longer operations, consider documenting this as a limitation rather than trying to fix.
- **Memory Analysis**: Straightforward implementation since data is valid. Can be completed quickly.
- **Nanosecond Precision**: Implementation complete, just needs verification testing.
- **Documentation**: Should be updated after CPU investigation to accurately reflect limitations.

---

## Quick Reference: Action Items Summary

### Immediate Actions (Before Dissertation)

1. **Investigate CPU Sampling** (1-2 hours)
   - Check if CPU values are non-zero for longer operations
   - Verify sampling implementation
   - Document limitation if unfixable

2. **Add Memory Analysis** (1-2 hours)
   - Implement memory utilization calculation
   - Add to `compute_statistics.py`
   - Test with existing data

3. **Add CPU Analysis** (1-2 hours, if CPU data valid)
   - Implement CPU utilization calculation
   - Add delta calculation
   - Test with existing data

### Documentation Updates

1. **Methodology Section** (1 hour)
   - Document measurement precision
   - Document resource metric limitations
   - Document timestamp precision

### Data Completeness

1. **Generate Missing Summaries** (1-2 hours) - See item #4
   - Run `generate_missing_summaries.sh`
   - Re-run aggregation
   - Verify all summaries exist

2. **Enhance Validation** (2-3 hours) - See item #5
   - Add summary.json existence check
   - Add statistical validity checks
   - Add cross-environment consistency

### Verification

1. **Test Nanosecond Precision** (1 hour) - See item #6
   - Run sample experiment
   - Verify data format
   - Test analysis compatibility

---

**Last Review**: 2025-12-10
**Next Review**: After CPU investigation completion

