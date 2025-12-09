#!/usr/bin/env bash
# =============================================================================
# validate_experiment_data.sh - Validate data files for a specific experiment
#
# Performs comprehensive validation of experiment data across environments:
# - File existence checks
# - File size validation
# - JSONL format validation
# - Data completeness checks
# - Metadata validation
#
# Usage:
#   ./scripts/validate_experiment_data.sh --exp-id <experiment-id> [--env <env>]
#
# Example:
#   ./scripts/validate_experiment_data.sh --exp-id rsa2048_p256_r100_run1_c0098396
#   ./scripts/validate_experiment_data.sh --exp-id rsa2048_p256_r100_run1_c0098396 --env native
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

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Validate experiment data files across environments.

OPTIONS:
    --exp-id ID          Experiment ID to validate (required)
    --env ENV            Validate only specific environment (native, minikube, gcp)
    --detailed           Show detailed validation (sample data, statistics)
    -h, --help           Show this help message

EXAMPLE:
    $0 --exp-id rsa2048_p256_r100_run1_c0098396
    $0 --exp-id rsa2048_p256_r100_run1_c0098396 --env native --detailed
EOF
    exit 1
}

EXP_ID=""
ENV_FILTER=""
DETAILED=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp-id)
            EXP_ID="$2"
            shift 2
            ;;
        --env)
            ENV_FILTER="$2"
            shift 2
            ;;
        --detailed)
            DETAILED=true
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

if [[ -z "$EXP_ID" ]]; then
    log_error "Missing required argument: --exp-id"
    usage
fi

echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Experiment Data Validation${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
log_info "Experiment ID: $EXP_ID"
[[ -n "$ENV_FILTER" ]] && log_info "Environment filter: $ENV_FILTER"
echo ""

# Function to validate a single environment
validate_environment() {
    local env=$1
    local exp_dir="$SCRIPT_DIR/results/$env/$EXP_ID"
    local has_data=false
    local issues=0
    
    echo -e "${CYAN}--- Environment: ${env^^} ---${NC}"
    
    if [[ ! -d "$exp_dir" ]]; then
        log_error "Experiment directory not found: $exp_dir"
        return 1
    fi
    
    log_success "Directory exists: $exp_dir"
    
    # Check raw data
    local raw_file="$exp_dir/raw/run.jsonl"
    if [[ -f "$raw_file" ]]; then
        has_data=true
        local file_size=$(stat -f%z "$raw_file" 2>/dev/null || stat -c%s "$raw_file" 2>/dev/null || echo 0)
        local line_count=$(wc -l < "$raw_file" 2>/dev/null || echo 0)
        
        log_success "Raw data file exists"
        log_info "  File size: $file_size bytes"
        log_info "  Line count: $line_count"
        
        if [[ $file_size -eq 0 ]]; then
            log_error "  Raw data file is empty (0 bytes)"
            issues=$((issues + 1))
        elif [[ $line_count -eq 0 ]]; then
            log_error "  Raw data file has no lines"
            issues=$((issues + 1))
        else
            # Validate JSONL format
            local first_line=$(head -1 "$raw_file" 2>/dev/null || echo "")
            local last_line=$(tail -1 "$raw_file" 2>/dev/null || echo "")
            
            if [[ -n "$first_line" ]]; then
                if [[ "$first_line" =~ ^error: ]]; then
                    log_error "  First line contains error: ${first_line:0:80}..."
                    issues=$((issues + 1))
                elif ! echo "$first_line" | python3 -m json.tool >/dev/null 2>&1; then
                    log_error "  First line is not valid JSON: ${first_line:0:80}..."
                    issues=$((issues + 1))
                else
                    log_success "  First line is valid JSON"
                    
                    if [[ "$DETAILED" == "true" ]]; then
                        local first_json=$(echo "$first_line" | python3 -m json.tool 2>/dev/null || echo "")
                        log_info "  First record sample:"
                        echo "$first_json" | head -10 | sed 's/^/    /'
                    fi
                fi
            fi
            
            if [[ $line_count -gt 1 ]] && [[ -n "$last_line" ]]; then
                if ! echo "$last_line" | python3 -m json.tool >/dev/null 2>&1; then
                    log_error "  Last line is not valid JSON: ${last_line:0:80}..."
                    issues=$((issues + 1))
                else
                    log_success "  Last line is valid JSON"
                fi
            fi
            
            # Check for required fields in first record
            if [[ "$DETAILED" == "true" ]] && [[ -n "$first_line" ]]; then
                local has_latency=$(echo "$first_line" | python3 -c "import sys, json; d=json.load(sys.stdin); print('latency_us' in d)" 2>/dev/null || echo "false")
                local has_algorithm=$(echo "$first_line" | python3 -c "import sys, json; d=json.load(sys.stdin); print('algorithm' in d)" 2>/dev/null || echo "false")
                local has_timestamp=$(echo "$first_line" | python3 -c "import sys, json; d=json.load(sys.stdin); print('timestamp' in d or 'timestamp_utc_iso' in d)" 2>/dev/null || echo "false")
                
                log_info "  Required fields check:"
                [[ "$has_latency" == "True" ]] && log_success "    ✓ latency_us" || log_error "    ✗ latency_us missing"
                [[ "$has_algorithm" == "True" ]] && log_success "    ✓ algorithm" || log_error "    ✗ algorithm missing"
                [[ "$has_timestamp" == "True" ]] && log_success "    ✓ timestamp" || log_error "    ✗ timestamp missing"
            fi
        fi
    else
        log_error "Raw data file not found: $raw_file"
        issues=$((issues + 1))
    fi
    
    # Check merged data
    local merged_file="$exp_dir/merged/merged.jsonl"
    if [[ -f "$merged_file" ]]; then
        local merged_size=$(stat -f%z "$merged_file" 2>/dev/null || stat -c%s "$merged_file" 2>/dev/null || echo 0)
        local merged_lines=$(wc -l < "$merged_file" 2>/dev/null || echo 0)
        log_success "Merged data file exists ($merged_size bytes, $merged_lines lines)"
    else
        log_warn "Merged data file not found (may be generated during analysis)"
    fi
    
    # Check stats
    local stats_file="$exp_dir/stats/summary.json"
    if [[ -f "$stats_file" ]]; then
        if python3 -m json.tool "$stats_file" >/dev/null 2>&1; then
            log_success "Stats file exists and is valid JSON"
            if [[ "$DETAILED" == "true" ]]; then
                log_info "  Stats content:"
                python3 -m json.tool "$stats_file" 2>/dev/null | head -20 | sed 's/^/    /'
            fi
        else
            log_error "Stats file exists but is not valid JSON"
            issues=$((issues + 1))
        fi
    else
        log_warn "Stats file not found (may be generated during analysis)"
    fi
    
    # Check metadata files
    local manifest_file="$exp_dir/manifest.json"
    if [[ -f "$manifest_file" ]]; then
        if python3 -m json.tool "$manifest_file" >/dev/null 2>&1; then
            log_success "Manifest file exists and is valid JSON"
        else
            log_error "Manifest file exists but is not valid JSON"
            issues=$((issues + 1))
        fi
    else
        log_warn "Manifest file not found"
    fi
    
    # Check environment-specific metadata
    if [[ "$env" == "gcp" ]]; then
        local cloud_metadata="$exp_dir/cloud_metadata.json"
        if [[ -f "$cloud_metadata" ]]; then
            log_success "Cloud metadata file exists"
        else
            log_warn "Cloud metadata file not found"
        fi
    fi
    
    # Summary
    echo ""
    if [[ $issues -eq 0 ]] && [[ "$has_data" == "true" ]]; then
        log_success "✓ $env: Validation PASSED"
        return 0
    elif [[ "$has_data" == "true" ]]; then
        log_warn "⚠ $env: Validation completed with $issues issue(s)"
        return 1
    else
        log_error "✗ $env: No data found"
        return 1
    fi
}

# Validate environments
ENVS=("native" "minikube" "gcp")
if [[ -n "$ENV_FILTER" ]]; then
    ENVS=("$ENV_FILTER")
fi

TOTAL_ISSUES=0
VALIDATED_ENVS=0
PASSED_ENVS=0

for env in "${ENVS[@]}"; do
    if validate_environment "$env"; then
        PASSED_ENVS=$((PASSED_ENVS + 1))
    else
        TOTAL_ISSUES=$((TOTAL_ISSUES + 1))
    fi
    VALIDATED_ENVS=$((VALIDATED_ENVS + 1))
    echo ""
done

# Final summary
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
log_info "Validation Summary:"
log_info "  Experiment ID: $EXP_ID"
log_info "  Environments validated: $VALIDATED_ENVS"
log_info "  Passed: $PASSED_ENVS"
log_info "  Issues found: $TOTAL_ISSUES"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"

if [[ $TOTAL_ISSUES -eq 0 ]]; then
    log_success "All validations passed! Data is usable."
    exit 0
else
    log_warn "Some issues found. Review the output above."
    exit 1
fi

