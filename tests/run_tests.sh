#!/usr/bin/env bash
# =============================================================================
# run_tests.sh - Unified test runner for refactoring validation
#
# Runs unit, integration, functional, regression, and smoke tests to ensure
# refactoring doesn't break critical functionality.
#
# Usage:
#   ./tests/run_tests.sh [unit|integration|functional|regression|smoke|all]
#
# Exit codes:
#   0 - All tests passed
#   1 - One or more tests failed
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$SCRIPT_DIR/tests"
FAILED_TESTS=0
TOTAL_TESTS=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

run_test_suite() {
    local suite_name="$1"
    local suite_dir="$TEST_DIR/$suite_name"
    
    if [[ ! -d "$suite_dir" ]]; then
        log_warn "Test suite directory not found: $suite_dir"
        return 0
    fi
    
    log_info "Running $suite_name tests..."
    
    local suite_tests=0
    local suite_failed=0
    
    # Find all test scripts
    while IFS= read -r -d '' test_file; do
        suite_tests=$((suite_tests + 1))
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        
        local test_name=$(basename "$test_file")
        log_info "  Running: $test_name"
        
        if bash "$test_file" 2>&1; then
            log_success "    ✓ $test_name"
        else
            log_error "    ✗ $test_name"
            suite_failed=$((suite_failed + 1))
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
    done < <(find "$suite_dir" -name "test_*.sh" -type f -print0 2>/dev/null | sort -z)
    
    if [[ $suite_tests -eq 0 ]]; then
        log_warn "  No tests found in $suite_name suite"
    else
        log_info "  $suite_name: $((suite_tests - suite_failed))/$suite_tests passed"
    fi
    
    return $suite_failed
}

run_unit_tests() {
    run_test_suite "unit"
}

run_integration_tests() {
    run_test_suite "integration"
}

run_functional_tests() {
    run_test_suite "functional"
}

run_regression_tests() {
    run_test_suite "regression"
}

run_smoke_tests() {
    run_test_suite "smoke"
}

usage() {
    cat <<EOF
Usage: $0 [SUITE]

Run test suites for refactoring validation.

SUITE:
    unit          Run unit tests only
    integration   Run integration tests only
    functional    Run functional tests only
    regression    Run regression tests only
    smoke         Run smoke tests only
    all           Run all test suites (default)

EXAMPLES:
    $0                    # Run all tests
    $0 unit              # Run unit tests only
    $0 integration        # Run integration tests only
EOF
    exit 1
}

main() {
    local suite="${1:-all}"
    
    echo "=============================================================================="
    echo "Test Runner - Refactoring Validation"
    echo "=============================================================================="
    echo ""
    
    case "$suite" in
        unit)
            run_unit_tests
            ;;
        integration)
            run_integration_tests
            ;;
        functional)
            run_functional_tests
            ;;
        regression)
            run_regression_tests
            ;;
        smoke)
            run_smoke_tests
            ;;
        all)
            run_unit_tests || true
            run_integration_tests || true
            run_functional_tests || true
            run_regression_tests || true
            run_smoke_tests || true
            ;;
        *)
            usage
            ;;
    esac
    
    echo ""
    echo "=============================================================================="
    echo "Test Summary"
    echo "=============================================================================="
    echo "Total tests: $TOTAL_TESTS"
    echo "Passed: $((TOTAL_TESTS - FAILED_TESTS))"
    echo "Failed: $FAILED_TESTS"
    
    if [[ $FAILED_TESTS -eq 0 ]]; then
        log_success "All tests passed!"
        exit 0
    else
        log_error "$FAILED_TESTS test(s) failed"
        exit 1
    fi
}

main "$@"

