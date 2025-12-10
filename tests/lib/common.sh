#!/usr/bin/env bash
# =============================================================================
# tests/lib/common.sh - Common test utilities
#
# Provides common functions for test scripts:
# - Assertion functions
# - Test setup/teardown helpers
# - Output comparison utilities
# =============================================================================

set -euo pipefail

# Colors for test output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TEST_COUNT=0
TEST_PASSED=0
TEST_FAILED=0

test_start() {
    TEST_COUNT=$((TEST_COUNT + 1))
    echo -e "${BLUE}[TEST $TEST_COUNT]${NC} $1"
}

test_pass() {
    TEST_PASSED=$((TEST_PASSED + 1))
    echo -e "${GREEN}  ✓ PASS${NC}: $1"
}

test_fail() {
    TEST_FAILED=$((TEST_FAILED + 1))
    echo -e "${RED}  ✗ FAIL${NC}: $1"
    return 1
}

assert_file_exists() {
    local file="$1"
    if [[ -f "$file" ]]; then
        test_pass "File exists: $file"
        return 0
    else
        test_fail "File not found: $file"
        return 1
    fi
}

assert_file_not_empty() {
    local file="$1"
    if [[ -f "$file" ]] && [[ -s "$file" ]]; then
        test_pass "File is not empty: $file"
        return 0
    else
        test_fail "File is empty or missing: $file"
        return 1
    fi
}

assert_directory_exists() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        test_pass "Directory exists: $dir"
        return 0
    else
        test_fail "Directory not found: $dir"
        return 1
    fi
}

assert_equals() {
    local expected="$1"
    local actual="$2"
    local message="${3:-Values should be equal}"
    
    if [[ "$expected" == "$actual" ]]; then
        test_pass "$message"
        return 0
    else
        test_fail "$message (expected: '$expected', actual: '$actual')"
        return 1
    fi
}

assert_not_equals() {
    local val1="$1"
    local val2="$2"
    local message="${3:-Values should not be equal}"
    
    if [[ "$val1" != "$val2" ]]; then
        test_pass "$message"
        return 0
    else
        test_fail "$message (both values: '$val1')"
        return 1
    fi
}

assert_contains() {
    local haystack="$1"
    local needle="$2"
    local message="${3:-String should contain substring}"
    
    if [[ "$haystack" == *"$needle"* ]]; then
        test_pass "$message"
        return 0
    else
        test_fail "$message (haystack: '$haystack', needle: '$needle')"
        return 1
    fi
}

assert_exit_code() {
    local expected_code="$1"
    shift
    local command="$*"
    
    set +e
    eval "$command" >/dev/null 2>&1
    local actual_code=$?
    set -e
    
    if [[ $actual_code -eq $expected_code ]]; then
        test_pass "Command exit code matches: $command"
        return 0
    else
        test_fail "Command exit code mismatch (expected: $expected_code, actual: $actual_code): $command"
        return 1
    fi
}

assert_json_has_field() {
    local json_file="$1"
    local field="$2"
    
    if command -v jq &>/dev/null; then
        if jq -e ".$field" "$json_file" >/dev/null 2>&1; then
            test_pass "JSON has field: $field"
            return 0
        else
            test_fail "JSON missing field: $field"
            return 1
        fi
    else
        test_fail "jq not available for JSON validation"
        return 1
    fi
}

compare_file_hashes() {
    local file1="$1"
    local file2="$2"
    
    if [[ ! -f "$file1" ]] || [[ ! -f "$file2" ]]; then
        test_fail "Cannot compare: one or both files missing"
        return 1
    fi
    
    local hash1=$(sha256sum "$file1" | cut -d' ' -f1)
    local hash2=$(sha256sum "$file2" | cut -d' ' -f1)
    
    if [[ "$hash1" == "$hash2" ]]; then
        test_pass "File hashes match: $file1 == $file2"
        return 0
    else
        test_fail "File hashes differ: $file1 != $file2"
        return 1
    fi
}

test_summary() {
    echo ""
    echo "=============================================================================="
    echo "Test Summary"
    echo "=============================================================================="
    echo "Total: $TEST_COUNT"
    echo "Passed: $TEST_PASSED"
    echo "Failed: $TEST_FAILED"
    
    if [[ $TEST_FAILED -eq 0 ]]; then
        echo -e "${GREEN}All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}$TEST_FAILED test(s) failed${NC}"
        return 1
    fi
}

