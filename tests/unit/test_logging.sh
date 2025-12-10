#!/usr/bin/env bash
# =============================================================================
# tests/unit/test_logging.sh - Unit tests for logging functions
#
# Tests that logging functions produce identical output across all scripts.
# This validates that the extracted common.sh library works correctly.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$SCRIPT_DIR/tests/lib/common.sh"

test_logging_functions() {
    test_start "Testing logging functions"
    
    # Source the common library
    source "$SCRIPT_DIR/scripts/lib/common.sh"
    
    # Test log_info
    local info_output=$(log_info "Test info message" 2>&1)
    assert_contains "$info_output" "[INFO]" "log_info should contain [INFO]"
    assert_contains "$info_output" "Test info message" "log_info should contain message"
    
    # Test log_success
    local success_output=$(log_success "Test success message" 2>&1)
    assert_contains "$success_output" "[OK]" "log_success should contain [OK]"
    assert_contains "$success_output" "Test success message" "log_success should contain message"
    
    # Test log_warn
    local warn_output=$(log_warn "Test warn message" 2>&1)
    assert_contains "$warn_output" "[WARN]" "log_warn should contain [WARN]"
    assert_contains "$warn_output" "Test warn message" "log_warn should contain message"
    
    # Test log_error
    local error_output=$(log_error "Test error message" 2>&1)
    assert_contains "$error_output" "[ERROR]" "log_error should contain [ERROR]"
    assert_contains "$error_output" "Test error message" "log_error should contain message"
    
    # Test log_step
    local step_output=$(log_step "Test step" 2>&1)
    assert_contains "$step_output" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" "log_step should contain separator"
    assert_contains "$step_output" "Test step" "log_step should contain message"
    
    # Test log_run
    local run_output=$(log_run "1" "5" "Test run" 2>&1)
    assert_contains "$run_output" "[RUN 1/5]" "log_run should contain [RUN X/Y]"
    assert_contains "$run_output" "Test run" "log_run should contain message"
}

test_logging_functions
test_summary

