#!/usr/bin/env bash
# =============================================================================
# tests/unit/test_directories.sh - Unit tests for directory creation
#
# Tests that directory creation functions work correctly.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$SCRIPT_DIR/tests/lib/common.sh"

test_directory_creation() {
    test_start "Testing directory creation"
    
    # Source the common library
    source "$SCRIPT_DIR/scripts/lib/common.sh"
    source "$SCRIPT_DIR/scripts/lib/directories.sh"
    
    # Create temporary test directory
    local test_dir=$(mktemp -d)
    trap "rm -rf '$test_dir'" EXIT
    
    # Test create_output_directories
    create_output_directories "$test_dir"
    
    # Verify all directories were created
    assert_directory_exists "$test_dir/raw"
    assert_directory_exists "$test_dir/merged"
    assert_directory_exists "$test_dir/stats"
    assert_directory_exists "$test_dir/figures"
    
    # Test verify_output_directories
    verify_output_directories "$test_dir"
    
    # Test with missing directory
    rm -rf "$test_dir/stats"
    if verify_output_directories "$test_dir" 2>/dev/null; then
        test_fail "verify_output_directories should fail when directory missing"
    else
        test_pass "verify_output_directories correctly detects missing directory"
    fi
}

test_directory_creation
test_summary

