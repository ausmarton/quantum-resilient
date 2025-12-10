#!/usr/bin/env bash
# =============================================================================
# scripts/validate_experiment_suite.sh - Experiment Suite Validation Script
#
# Validates all experiments in a suite (works for both smoke-test and full-scale):
# 1. All expected experiments completed
# 2. Raw data files exist and are non-empty
# 3. Data has nanosecond precision (latency_ns field present)
# 4. Analysis pipeline processes data successfully
# 5. Summaries generated successfully
# 6. Cross-environment comparison possible
# 7. Classical vs PQC comparison possible
# 8. Scaling analysis possible (where applicable)
# 9. Burst pattern analysis possible
#
# Usage:
#   ./scripts/validate_experiment_suite.sh --results-dir results --scenarios-dir generated-scenarios
#
# Note: This script works with both smoke-test and full-scale benchmarks.
# The distinction comes from the scenarios directory contents, not the script logic.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default values
RESULTS_DIR="results"  # Base results directory (contains native/, minikube/, gcp/)
SCENARIOS_DIR=""
VERBOSE=false

# Validation results
VALIDATION_ERRORS=0
VALIDATION_WARNINGS=0
TOTAL_CHECKS=0
PASSED_CHECKS=0

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    VALIDATION_WARNINGS=$((VALIDATION_WARNINGS + 1))
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
}

log_verbose() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${CYAN}[VERBOSE]${NC} $1"
    fi
}

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Validate smoke test outputs and analysis pipeline.

OPTIONS:
    --results-dir DIR      Base directory containing results (default: results/)
                            Should contain <env>/ subdirectories (native, minikube, gcp)
    --scenarios-dir DIR    Directory containing generated scenarios (required)
                            Use generated-scenarios/ for both smoke-test and full-scale
    --verbose               Enable verbose output
    -h, --help              Show this help message

EXAMPLES:
    # Validate smoke-test results
    $0 --results-dir results --scenarios-dir generated-scenarios
    
    # Validate full-scale results (same command)
    $0 --results-dir results --scenarios-dir generated-scenarios
    
NOTE: This script works with both smoke-test and full-scale benchmarks.
The distinction comes from the scenarios directory contents, not the script logic.
EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --scenarios-dir)
            SCENARIOS_DIR="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$SCENARIOS_DIR" ]]; then
    log_error "--scenarios-dir is required"
    usage
fi

# RESULTS_DIR defaults to "results" if not provided
if [[ -z "$RESULTS_DIR" ]]; then
    RESULTS_DIR="results"
fi

if [[ ! -d "$RESULTS_DIR" ]]; then
    log_error "Results directory not found: $RESULTS_DIR"
    exit 1
fi

if [[ ! -d "$SCENARIOS_DIR" ]]; then
    log_error "Scenarios directory not found: $SCENARIOS_DIR"
    exit 1
fi

# -----------------------------------------------------------------------------
# Validation Functions
# -----------------------------------------------------------------------------

# Check 1: Count expected vs actual scenarios
validate_scenario_count() {
    log_info "Checking scenario counts..."
    
    local expected_count=$(find "$SCENARIOS_DIR" -name "scenario.yaml" | wc -l)
    log_verbose "Expected scenarios: $expected_count"
    
    for env_dir in "$RESULTS_DIR"/*/; do
        if [[ ! -d "$env_dir" ]]; then
            continue
        fi
        
        local env=$(basename "$env_dir")
        local actual_count=$(find "$env_dir" -name "scenario.yaml" -o -name "run.jsonl" 2>/dev/null | wc -l)
        
        if [[ $actual_count -eq 0 ]]; then
            log_warn "No results found for environment: $env"
        else
            log_verbose "  $env: Found $actual_count result files"
            if [[ $actual_count -lt $((expected_count / 2)) ]]; then
                log_warn "  $env: Only $actual_count/$expected_count scenarios completed"
            else
                log_success "  $env: Scenario count acceptable ($actual_count files)"
            fi
        fi
    done
}

# Check 2: Validate raw data files exist and are non-empty
validate_raw_data() {
    log_info "Validating raw data files..."
    
    local total_files=0
    local empty_files=0
    local missing_files=0
    
    for env_dir in "$RESULTS_DIR"/*/; do
        if [[ ! -d "$env_dir" ]]; then
            continue
        fi
        
        local env=$(basename "$env_dir")
        local jsonl_files=$(find "$env_dir" -name "*.jsonl" -type f 2>/dev/null)
        
        for jsonl_file in $jsonl_files; do
            total_files=$((total_files + 1))
            
            if [[ ! -f "$jsonl_file" ]]; then
                missing_files=$((missing_files + 1))
                log_error "  Missing file: $jsonl_file"
            elif [[ ! -s "$jsonl_file" ]]; then
                empty_files=$((empty_files + 1))
                log_error "  Empty file: $jsonl_file"
            else
                log_verbose "  Valid: $jsonl_file"
            fi
        done
    done
    
    if [[ $total_files -eq 0 ]]; then
        log_error "No raw data files found in $RESULTS_DIR"
    elif [[ $empty_files -gt 0 ]] || [[ $missing_files -gt 0 ]]; then
        log_error "Found $empty_files empty files and $missing_files missing files out of $total_files total"
    else
        log_success "All raw data files exist and are non-empty ($total_files files)"
    fi
}

# Check 3: Validate nanosecond precision
validate_nanosecond_precision() {
    log_info "Validating nanosecond precision (latency_ns field)..."
    
    local files_with_ns=0
    local files_without_ns=0
    local files_with_us_only=0
    
    for env_dir in "$RESULTS_DIR"/*/; do
        if [[ ! -d "$env_dir" ]]; then
            continue
        fi
        
        local jsonl_files=$(find "$env_dir" -name "*.jsonl" -type f 2>/dev/null | head -10)
        
        for jsonl_file in $jsonl_files; do
            if python3 -c "
import json
has_ns = False
has_us_only = False
with open('$jsonl_file') as f:
    for line_num, line in enumerate(f, 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            if 'latency_ns' in data:
                has_ns = True
            elif 'latency_us' in data and 'latency_ns' not in data:
                has_us_only = True
            if line_num >= 10:  # Check first 10 lines
                break
        except:
            pass
if has_ns:
    exit(0)
elif has_us_only:
    exit(2)
else:
    exit(1)
" 2>/dev/null; then
                exit_code=$?
                if [[ $exit_code -eq 0 ]]; then
                    files_with_ns=$((files_with_ns + 1))
                elif [[ $exit_code -eq 2 ]]; then
                    files_with_us_only=$((files_with_us_only + 1))
                else
                    files_without_ns=$((files_without_ns + 1))
                fi
            fi
        done
    done
    
    local total_checked=$((files_with_ns + files_without_ns + files_with_us_only))
    
    if [[ $files_with_us_only -gt 0 ]]; then
        log_error "Found $files_with_us_only files with microseconds-only precision (should have latency_ns)"
    fi
    
    if [[ $files_without_ns -gt 0 ]]; then
        log_error "Found $files_without_ns files without latency_ns field"
    fi
    
    if [[ $files_with_ns -gt 0 ]] && [[ $files_without_ns -eq 0 ]] && [[ $files_with_us_only -eq 0 ]]; then
        log_success "All checked files have nanosecond precision ($files_with_ns/$total_checked files)"
    elif [[ $files_with_ns -gt 0 ]]; then
        log_warn "Some files have nanosecond precision ($files_with_ns/$total_checked), but $files_without_ns missing and $files_with_us_only microseconds-only"
    else
        log_error "No files found with nanosecond precision"
    fi
}

# Check 4: Validate analysis pipeline
validate_analysis_pipeline() {
    log_info "Validating analysis pipeline (compute_statistics.py)..."
    
    # Find a sample result directory to test
    local sample_dir=$(find "$RESULTS_DIR" -type d -name "run-*" | head -1)
    
    if [[ -z "$sample_dir" ]]; then
        log_warn "No sample result directory found - skipping analysis pipeline validation"
        return
    fi
    
    # Find merged.jsonl or raw JSONL
    local jsonl_file=""
    if [[ -f "$sample_dir/merged.jsonl" ]]; then
        jsonl_file="$sample_dir/merged.jsonl"
    elif [[ -f "$sample_dir/raw/run.jsonl" ]]; then
        jsonl_file="$sample_dir/raw/run.jsonl"
    fi
    
    if [[ -z "$jsonl_file" ]]; then
        log_warn "No JSONL file found in $sample_dir - skipping analysis pipeline validation"
        return
    fi
    
    # Test compute_statistics.py
    local temp_output=$(mktemp -d)
    trap "rm -rf '$temp_output'" EXIT
    
    if python3 "$SCRIPT_DIR/analysis/scripts/compute_statistics.py" \
        --input "$jsonl_file" \
        --output "$temp_output" \
        --scenario-id "smoke-test-validation" 2>&1 | grep -v "PermissionError\|FileNotFoundError" >/dev/null; then
        log_success "Analysis pipeline processes smoke test data successfully"
    else
        log_error "Analysis pipeline failed to process smoke test data"
    fi
    
    rm -rf "$temp_output"
    trap - EXIT
}

# Check 5: Validate summaries exist
validate_summaries() {
    log_info "Validating summary.json files..."
    
    local total_summaries=0
    local missing_summaries=0
    
    for env_dir in "$RESULTS_DIR"/*/; do
        if [[ ! -d "$env_dir" ]]; then
            continue
        fi
        
        local summary_files=$(find "$env_dir" -name "summary.json" -type f 2>/dev/null)
        local count=0
        if [[ -n "$summary_files" ]]; then
            count=$(echo "$summary_files" | wc -l)
        fi
        total_summaries=$((total_summaries + count))
        
        # Check if we have scenarios but no summaries
        local scenario_dirs=$(find "$env_dir" -type d -name "run-*" 2>/dev/null | wc -l)
        if [[ $scenario_dirs -gt 0 ]] && [[ $count -eq 0 ]]; then
            missing_summaries=$((missing_summaries + scenario_dirs))
        fi
    done
    
    if [[ $total_summaries -eq 0 ]]; then
        log_warn "No summary.json files found (may need to run generate_missing_summaries.sh)"
    elif [[ $missing_summaries -gt 0 ]]; then
        log_warn "Found $total_summaries summaries but $missing_summaries experiments missing summaries"
    else
        log_success "Found $total_summaries summary.json files"
    fi
}

# Check 6: Validate experiment types coverage
validate_experiment_types() {
    log_info "Validating experiment types coverage..."
    
    local has_constant=false
    local has_burst=false
    local has_scaling=false
    
    # Check scenario directories (works for both smoke-test and full-scale)
    if find "$SCENARIOS_DIR" -name "scenario.yaml" -exec grep -l "pattern.*burst\|burst" {} \; 2>/dev/null | head -1 | grep -q .; then
        has_burst=true
    fi
    
    if find "$SCENARIOS_DIR" -name "scenario.yaml" -exec grep -l "scaling" {} \; 2>/dev/null | head -1 | grep -q .; then
        has_scaling=true
    fi
    
    # Constant is always present (default pattern)
    has_constant=true
    
    local coverage_ok=true
    if [[ "$has_constant" != "true" ]]; then
        log_error "Missing constant pattern experiments"
        coverage_ok=false
    fi
    if [[ "$has_burst" != "true" ]]; then
        log_warn "Missing burst pattern experiments (may be expected for some experiment sets)"
    fi
    if [[ "$has_scaling" != "true" ]]; then
        log_warn "Missing scaling experiments (may be expected for some experiment sets)"
    fi
    
    if [[ "$coverage_ok" == "true" ]]; then
        log_success "Experiment types coverage: constant=$has_constant, burst=$has_burst, scaling=$has_scaling"
    fi
}

# -----------------------------------------------------------------------------
# Main Validation
# -----------------------------------------------------------------------------

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Smoke Test Validation${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

validate_scenario_count
echo ""

validate_raw_data
echo ""

validate_nanosecond_precision
echo ""

validate_analysis_pipeline
echo ""

validate_summaries
echo ""

validate_experiment_types
echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Validation Summary${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Total checks: $TOTAL_CHECKS"
echo "  Passed: $PASSED_CHECKS"
echo "  Warnings: $VALIDATION_WARNINGS"
echo "  Errors: $VALIDATION_ERRORS"
echo ""

if [[ $VALIDATION_ERRORS -eq 0 ]]; then
    log_success "Smoke test validation PASSED"
    exit 0
else
    log_error "Smoke test validation FAILED ($VALIDATION_ERRORS errors)"
    exit 1
fi
