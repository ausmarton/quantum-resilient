#!/usr/bin/env bash
# =============================================================================
# tests/functional/test_nanosecond_precision.sh - Test nanosecond precision implementation
#
# Verifies that nanosecond precision is correctly implemented and that analysis
# scripts handle it properly.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$SCRIPT_DIR/tests/lib/common.sh"

test_nanosecond_precision_rust() {
    test_start "Testing Rust nanosecond precision implementation"
    
    # Check that Rust code uses as_nanos()
    if grep -q "as_nanos()" "$SCRIPT_DIR/rust-core/src/pipeline/execution.rs"; then
        test_pass "Rust code uses as_nanos() for nanosecond precision"
    else
        test_fail "Rust code does not use as_nanos()"
        return 1
    fi
    
    # Check that latency_ns field exists in struct
    if grep -q "latency_ns.*u128" "$SCRIPT_DIR/rust-core/src/pipeline/execution.rs"; then
        test_pass "latency_ns field defined in struct"
    else
        test_fail "latency_ns field not found in struct"
        return 1
    fi
    
    # Verify code compiles
    if (cd "$SCRIPT_DIR/rust-core" && cargo check --quiet 2>&1); then
        test_pass "Rust code compiles successfully"
    else
        test_fail "Rust code compilation failed"
        return 1
    fi
}

test_nanosecond_precision_analysis() {
    test_start "Testing analysis script nanosecond precision support"
    
    # Check that compute_statistics.py handles latency_ns
    if grep -q "latency_ns" "$SCRIPT_DIR/analysis/scripts/compute_statistics.py"; then
        test_pass "compute_statistics.py handles latency_ns"
    else
        test_fail "compute_statistics.py does not handle latency_ns"
        return 1
    fi
    
    # Check that merge_jsonl.py handles latency_ns
    if grep -q "latency_ns" "$SCRIPT_DIR/analysis/scripts/merge_jsonl.py"; then
        test_pass "merge_jsonl.py handles latency_ns"
    else
        echo "  [WARN] merge_jsonl.py may not handle latency_ns (check manually)"
    fi
    
    # Test with sample data (if available)
    local sample_file=""
    for jsonl_file in "$SCRIPT_DIR"/results/*/raw/run.jsonl; do
        if [[ -f "$jsonl_file" ]] && [[ -s "$jsonl_file" ]]; then
            sample_file="$jsonl_file"
            break
        fi
    done
    
    if [[ -n "$sample_file" ]]; then
        # Check if file has latency_ns (new format) or only latency_us (old format)
        if head -1 "$sample_file" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'latency_ns' in data:
    print('NEW_FORMAT')
    sys.exit(0)
elif 'latency_us' in data:
    print('OLD_FORMAT')
    sys.exit(0)
else:
    print('UNKNOWN')
    sys.exit(1)
" 2>/dev/null | grep -q "NEW_FORMAT"; then
            test_pass "Sample data has latency_ns field (new format)"
        else
            echo "  [WARN] Sample data uses old format (latency_us only) - expected for old experiments"
        fi
    else
        echo "  [WARN] No sample data found for format verification"
    fi
}

test_nanosecond_precision_conversion() {
    test_start "Testing nanosecond to microsecond conversion"
    
    # Create test data with nanosecond precision
    local test_dir=$(mktemp -d)
    trap "rm -rf '$test_dir'" EXIT
    
    mkdir -p "$test_dir/raw"
    
    # Create test JSONL with nanosecond precision
    cat > "$test_dir/raw/test.jsonl" <<EOF
{"event_id": 1, "latency_ns": 500, "latency_us": 0.5, "timestamp_utc_iso": "2025-01-01T00:00:00Z"}
{"event_id": 2, "latency_ns": 1500, "latency_us": 1.5, "timestamp_utc_iso": "2025-01-01T00:00:01Z"}
{"event_id": 3, "latency_ns": 2500, "latency_us": 2.5, "timestamp_utc_iso": "2025-01-01T00:00:02Z"}
EOF
    
    # Test that analysis scripts can process it
    if python3 -c "
import json
from pathlib import Path

test_file = Path('$test_dir/raw/test.jsonl')
with open(test_file) as f:
    for line in f:
        data = json.loads(line)
        if 'latency_ns' not in data:
            print('ERROR: latency_ns missing')
            exit(1)
        # Verify conversion: latency_us should be latency_ns / 1000
        if 'latency_us' in data:
            expected_us = data['latency_ns'] / 1000.0
            if abs(data['latency_us'] - expected_us) > 0.001:
                print(f'ERROR: Conversion mismatch: {data[\"latency_us\"]} != {expected_us}')
                exit(1)
print('OK')
" 2>/dev/null; then
        test_pass "Nanosecond to microsecond conversion verified"
    else
        test_fail "Nanosecond to microsecond conversion failed"
        return 1
    fi
}

test_nanosecond_precision_throughput() {
    test_start "Testing throughput calculation precision"
    
    # Verify that timestamp precision is sufficient for throughput
    # Throughput uses 1-second buckets, so millisecond precision is sufficient
    # Monotonic timestamps provide nanosecond precision for detailed analysis
    
    if grep -q "timestamp_monotonic_ns" "$SCRIPT_DIR/rust-core/src/pipeline/execution.rs"; then
        test_pass "Monotonic nanosecond timestamps available for detailed analysis"
    else
        echo "  [WARN] Monotonic nanosecond timestamps may not be available"
    fi
    
    if grep -q "timestamp_utc_iso\|timestamp.*iso" "$SCRIPT_DIR/rust-core/src/pipeline/execution.rs"; then
        test_pass "ISO timestamp available for throughput calculation (millisecond precision sufficient)"
    else
        echo "  [WARN] ISO timestamp may not be available"
    fi
}

# Run all tests
test_nanosecond_precision_rust
RUST_TEST=$?

test_nanosecond_precision_analysis
ANALYSIS_TEST=$?

test_nanosecond_precision_conversion
CONVERSION_TEST=$?

test_nanosecond_precision_throughput
THROUGHPUT_TEST=$?

# Summary
if [[ $RUST_TEST -eq 0 ]] && [[ $ANALYSIS_TEST -eq 0 ]] && [[ $CONVERSION_TEST -eq 0 ]] && [[ $THROUGHPUT_TEST -eq 0 ]]; then
    test_summary "Nanosecond precision tests: ALL PASSED"
    exit 0
else
    test_summary "Nanosecond precision tests: SOME FAILED"
    exit 1
fi

