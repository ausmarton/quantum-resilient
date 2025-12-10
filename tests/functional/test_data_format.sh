#!/usr/bin/env bash
# =============================================================================
# tests/functional/test_data_format.sh - Functional tests for data format expectations
#
# Tests that analysis scripts expect the correct data format (nanosecond precision).
# Since we're re-running all experiments, we only need to support the new format.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$SCRIPT_DIR/tests/lib/common.sh"

test_merge_jsonl_expects_latency_ns() {
    test_start "Testing merge_jsonl.py expects latency_ns"
    
    # Check if pandas is available (required for analysis scripts)
    if ! python3 -c "import pandas" 2>/dev/null; then
        echo "  [SKIP] pandas not available - skipping test (install with: pip install pandas)"
        test_pass "Test skipped (pandas not available)"
        return 0  # Skip test, don't fail suite
    fi
    
    # Create a temporary test JSONL file with nanosecond precision
    local test_dir=$(mktemp -d)
    trap "rm -rf '$test_dir'" EXIT
    
    mkdir -p "$test_dir/raw"
    
    # Create test JSONL with latency_ns (correct format)
    cat > "$test_dir/raw/test.jsonl" <<EOF
{"event_id": 1, "latency_ns": 1000, "queue_delay_ns": 100, "timestamp_utc_iso": "2025-01-01T00:00:00Z"}
{"event_id": 2, "latency_ns": 2000, "queue_delay_ns": 200, "timestamp_utc_iso": "2025-01-01T00:00:01Z"}
EOF
    
    # Test that merge_jsonl.py processes it correctly
    if python3 "$SCRIPT_DIR/analysis/scripts/merge_jsonl.py" \
        --input "$test_dir/raw" \
        --output "$test_dir/merged" 2>&1; then
        test_pass "merge_jsonl.py processes latency_ns correctly"
        
        # Verify output has latency_us derived from latency_ns
        if [[ -f "$test_dir/merged/merged.jsonl" ]]; then
            # Check that latency_us exists and is correct (1000ns = 1.0us)
            if python3 -c "
import json
with open('$test_dir/merged/merged.jsonl') as f:
    for line in f:
        data = json.loads(line)
        if 'latency_ns' not in data:
            print('ERROR: latency_ns missing')
            exit(1)
        if 'latency_us' not in data:
            print('ERROR: latency_us missing')
            exit(1)
        if abs(data['latency_us'] - data['latency_ns'] / 1000.0) > 0.001:
            print(f'ERROR: latency_us incorrect: {data[\"latency_us\"]} != {data[\"latency_ns\"] / 1000.0}')
            exit(1)
print('OK')
" 2>&1; then
                test_pass "Output has correct latency_us derived from latency_ns"
            else
                test_fail "Output missing or incorrect latency_us"
                return 1
            fi
        else
            test_fail "merged.jsonl not created"
            return 1
        fi
    else
        test_fail "merge_jsonl.py failed to process latency_ns"
        return 1
    fi
}

test_merge_jsonl_rejects_missing_latency_ns() {
    test_start "Testing merge_jsonl.py rejects data without latency_ns"
    
    # Check if pandas is available (required for analysis scripts)
    if ! python3 -c "import pandas" 2>/dev/null; then
        echo "  [SKIP] pandas not available - skipping test (install with: pip install pandas)"
        test_pass "Test skipped (pandas not available)"
        return 0  # Skip test, don't fail suite
    fi
    
    # Create a temporary test JSONL file without latency_ns (old format)
    local test_dir=$(mktemp -d)
    trap "rm -rf '$test_dir'" EXIT
    
    mkdir -p "$test_dir/raw"
    
    # Create test JSONL with only latency_us (old format - should be rejected)
    cat > "$test_dir/raw/test.jsonl" <<EOF
{"event_id": 1, "latency_us": 1.0, "timestamp_utc_iso": "2025-01-01T00:00:00Z"}
EOF
    
    # Test that merge_jsonl.py rejects it
    if python3 "$SCRIPT_DIR/analysis/scripts/merge_jsonl.py" \
        --input "$test_dir/raw" \
        --output "$test_dir/merged" 2>&1 | grep -q "latency_ns column not found\|Missing required column"; then
        test_pass "merge_jsonl.py correctly rejects data without latency_ns"
    else
        test_fail "merge_jsonl.py should reject data without latency_ns"
        return 1
    fi
}

test_compute_statistics_expects_latency_ns() {
    test_start "Testing compute_statistics.py expects latency_ns"
    
    # Check if pandas is available (required for analysis scripts)
    if ! python3 -c "import pandas" 2>/dev/null; then
        echo "  [SKIP] pandas not available - skipping test (install with: pip install pandas)"
        test_pass "Test skipped (pandas not available)"
        return 0  # Skip test, don't fail suite
    fi
    
    # Create a temporary test JSONL file with nanosecond precision
    local test_dir=$(mktemp -d)
    trap "rm -rf '$test_dir'" EXIT
    
    mkdir -p "$test_dir/merged"
    
    # Create test JSONL with latency_ns (correct format)
    cat > "$test_dir/merged/merged.jsonl" <<EOF
{"event_id": 1, "latency_ns": 1000, "queue_delay_ns": 100, "timestamp_utc_iso": "2025-01-01T00:00:00Z", "algorithm": "rsa2048"}
{"event_id": 2, "latency_ns": 2000, "queue_delay_ns": 200, "timestamp_utc_iso": "2025-01-01T00:00:01Z", "algorithm": "rsa2048"}
EOF
    
    mkdir -p "$test_dir/stats"
    
    # Test that compute_statistics.py processes it correctly
    if python3 "$SCRIPT_DIR/analysis/scripts/compute_statistics.py" \
        --input "$test_dir/merged/merged.jsonl" \
        --output "$test_dir/stats" \
        --experiment-id "test" 2>&1; then
        test_pass "compute_statistics.py processes latency_ns correctly"
        
        # Verify summary.json has both latency and latency_ns
        if [[ -f "$test_dir/stats/summary.json" ]]; then
            if python3 -c "
import json
with open('$test_dir/stats/summary.json') as f:
    data = json.load(f)
    if 'latency' not in data:
        print('ERROR: latency missing')
        exit(1)
    if 'latency_ns' not in data:
        print('ERROR: latency_ns missing')
        exit(1)
    print('OK')
" 2>&1; then
                test_pass "Summary includes both latency and latency_ns"
            else
                test_fail "Summary missing latency or latency_ns"
                return 1
            fi
        else
            test_fail "summary.json not created"
            return 1
        fi
    else
        test_fail "compute_statistics.py failed to process latency_ns"
        return 1
    fi
}

test_compute_statistics_rejects_missing_latency_ns() {
    test_start "Testing compute_statistics.py rejects data without latency_ns"
    
    # Check if pandas is available (required for analysis scripts)
    if ! python3 -c "import pandas" 2>/dev/null; then
        echo "  [SKIP] pandas not available - skipping test (install with: pip install pandas)"
        test_pass "Test skipped (pandas not available)"
        return 0  # Skip test, don't fail suite
    fi
    
    # Create a temporary test JSONL file without latency_ns (old format)
    local test_dir=$(mktemp -d)
    trap "rm -rf '$test_dir'" EXIT
    
    mkdir -p "$test_dir/merged"
    
    # Create test JSONL with only latency_us (old format - should be rejected)
    cat > "$test_dir/merged/merged.jsonl" <<EOF
{"event_id": 1, "latency_us": 1.0, "timestamp_utc_iso": "2025-01-01T00:00:00Z"}
EOF
    
    mkdir -p "$test_dir/stats"
    
    # Test that compute_statistics.py rejects it
    if python3 "$SCRIPT_DIR/analysis/scripts/compute_statistics.py" \
        --input "$test_dir/merged/merged.jsonl" \
        --output "$test_dir/stats" \
        --experiment-id "test" 2>&1 | grep -q "Missing required column.*latency_ns"; then
        test_pass "compute_statistics.py correctly rejects data without latency_ns"
    else
        test_fail "compute_statistics.py should reject data without latency_ns"
        return 1
    fi
}

# Run all tests
test_merge_jsonl_expects_latency_ns
test_merge_jsonl_rejects_missing_latency_ns
test_compute_statistics_expects_latency_ns
test_compute_statistics_rejects_missing_latency_ns
test_summary

