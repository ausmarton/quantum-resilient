#!/usr/bin/env bash
# =============================================================================
# validate_data_integrity.sh - Validate data integrity during/after collection
#
# Checks that collected data files are valid and non-empty.
# Can be run during collection to catch issues early.
#
# Usage:
#   ./scripts/validate_data_integrity.sh [OPTIONS]
#
# Options:
#   --env ENV              Check specific environment (native, minikube, gcp)
#   --results-dir DIR      Results directory (default: results/)
#   --fail-on-empty        Exit with error if empty files found
#   --min-size BYTES       Minimum file size in bytes (default: 1)
#   --min-lines LINES      Minimum JSONL lines (default: 1)
#   --recent MINUTES       Only check files modified in last N minutes
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
ENV=""
FAIL_ON_EMPTY=false
MIN_SIZE=1
MIN_LINES=1
RECENT_MINUTES=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
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

Validate data integrity of collected experiment results.

OPTIONS:
    --env ENV              Check specific environment (native, minikube, gcp)
    --results-dir DIR      Results directory (default: results/)
    --fail-on-empty        Exit with error if empty files found
    --min-size BYTES       Minimum file size in bytes (default: 1)
    --min-lines LINES      Minimum JSONL lines (default: 1)
    --recent MINUTES       Only check files modified in last N minutes
    -h, --help             Show this help message

EXAMPLES:
    # Check all environments
    ./scripts/validate_data_integrity.sh

    # Check only minikube, fail on empty files
    ./scripts/validate_data_integrity.sh --env minikube --fail-on-empty

    # Check files modified in last 10 minutes
    ./scripts/validate_data_integrity.sh --recent 10
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --fail-on-empty)
            FAIL_ON_EMPTY=true
            shift
            ;;
        --min-size)
            MIN_SIZE="$2"
            shift 2
            ;;
        --min-lines)
            MIN_LINES="$2"
            shift 2
            ;;
        --recent)
            RECENT_MINUTES="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Determine which environments to check
if [[ -n "$ENV" ]]; then
    ENVS=("$ENV")
else
    ENVS=("native" "minikube" "gcp")
fi

echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Data Integrity Validation${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

TOTAL_CHECKED=0
TOTAL_VALID=0
TOTAL_EMPTY=0
TOTAL_INVALID=0
TOTAL_MISSING_SUMMARY=0
TOTAL_INVALID_STATS=0
EMPTY_FILES=()
MISSING_SUMMARIES=()
INVALID_STATS=()

for env in "${ENVS[@]}"; do
    ENV_RESULTS_DIR="$RESULTS_DIR/$env"
    
    if [[ ! -d "$ENV_RESULTS_DIR" ]]; then
        log_warn "${env^^}: Results directory not found: $ENV_RESULTS_DIR"
        continue
    fi
    
    echo -e "${MAGENTA}${env^^}:${NC}"
    
    # Find all raw JSONL files
    FIND_CMD="find \"$ENV_RESULTS_DIR\" -name \"run.jsonl\" -type f -path \"*/raw/*\""
    if [[ -n "$RECENT_MINUTES" ]]; then
        FIND_CMD="$FIND_CMD -mmin -$RECENT_MINUTES"
    fi
    
    while IFS= read -r jsonl_file; do
        TOTAL_CHECKED=$((TOTAL_CHECKED + 1))
        
        # Get file size
        FILE_SIZE=$(stat -f%z "$jsonl_file" 2>/dev/null || stat -c%s "$jsonl_file" 2>/dev/null || echo 0)
        
        if [[ $FILE_SIZE -lt $MIN_SIZE ]]; then
            TOTAL_EMPTY=$((TOTAL_EMPTY + 1))
            SCENARIO_ID=$(basename "$(dirname "$(dirname "$jsonl_file")")")
            EMPTY_FILES+=("$SCENARIO_ID:$jsonl_file")
            log_error "  ✗ $SCENARIO_ID: Empty file ($FILE_SIZE bytes)"
            continue
        fi
        
        # Check line count
        LINE_COUNT=$(wc -l < "$jsonl_file" 2>/dev/null || echo 0)
        if [[ $LINE_COUNT -lt $MIN_LINES ]]; then
            TOTAL_INVALID=$((TOTAL_INVALID + 1))
            SCENARIO_ID=$(basename "$(dirname "$(dirname "$jsonl_file")")")
            log_error "  ✗ $SCENARIO_ID: No valid lines ($LINE_COUNT lines, $FILE_SIZE bytes)"
            continue
        fi
        
        # Validate JSONL format (check first line is valid JSON)
        FIRST_LINE=$(head -1 "$jsonl_file" 2>/dev/null || echo "")
        if [[ -n "$FIRST_LINE" ]]; then
            # Check if it's an error message (kubectl errors start with "error:")
            if [[ "$FIRST_LINE" =~ ^error: ]]; then
                TOTAL_INVALID=$((TOTAL_INVALID + 1))
                SCENARIO_ID=$(basename "$(dirname "$(dirname "$jsonl_file")")")
                log_error "  ✗ $SCENARIO_ID: Contains error message (not JSONL data)"
                log_error "    First line: ${FIRST_LINE:0:80}..."
                continue
            fi
            
            # Check if it's valid JSON
            if ! echo "$FIRST_LINE" | python3 -m json.tool >/dev/null 2>&1; then
                TOTAL_INVALID=$((TOTAL_INVALID + 1))
                SCENARIO_ID=$(basename "$(dirname "$(dirname "$jsonl_file")")")
                log_error "  ✗ $SCENARIO_ID: Invalid JSONL format"
                log_error "    First line: ${FIRST_LINE:0:80}..."
                continue
            fi
        fi
        
        # Check for summary.json file
        EXP_DIR=$(dirname "$(dirname "$jsonl_file")")
        SUMMARY_PATHS=(
            "$EXP_DIR/stats/summary.json"
            "$EXP_DIR/merged/stats/summary.json"
            "$EXP_DIR/summary.json"
        )
        
        SUMMARY_FOUND=false
        SUMMARY_FILE=""
        for summary_path in "${SUMMARY_PATHS[@]}"; do
            if [[ -f "$summary_path" ]]; then
                SUMMARY_FOUND=true
                SUMMARY_FILE="$summary_path"
                break
            fi
        done
        
        if [[ "$SUMMARY_FOUND" == "false" ]]; then
            TOTAL_MISSING_SUMMARY=$((TOTAL_MISSING_SUMMARY + 1))
            SCENARIO_ID=$(basename "$EXP_DIR")
            MISSING_SUMMARIES+=("$SCENARIO_ID:$EXP_DIR")
            log_warn "  ⚠ $SCENARIO_ID: Missing summary.json"
        else
            # Validate statistical validity of summary.json
            if ! python3 <<PYTHON_SCRIPT 2>/dev/null; then
import json
import sys

try:
    with open('$SUMMARY_FILE') as f:
        summary = json.load(f)
    
    # Check that latency stats exist (either latency_us or latency_ns)
    has_latency = False
    if 'latency' in summary:
        if 'p50' in summary['latency']:
            has_latency = True
    elif 'latency_ns' in summary:
        if 'p50' in summary['latency_ns']:
            has_latency = True
    
    if not has_latency:
        print("Missing latency p50")
        sys.exit(1)
    
    # Check that total_events exists
    if 'total_events' not in summary:
        print("Missing total_events")
        sys.exit(1)
    
    # Check that total_events is reasonable (> 0)
    if summary.get('total_events', 0) <= 0:
        print("total_events is zero or negative")
        sys.exit(1)
    
    sys.exit(0)
except Exception as e:
    print(f"Error validating summary: {e}")
    sys.exit(1)
PYTHON_SCRIPT
                TOTAL_INVALID_STATS=$((TOTAL_INVALID_STATS + 1))
                SCENARIO_ID=$(basename "$EXP_DIR")
                INVALID_STATS+=("$SCENARIO_ID:$SUMMARY_FILE")
                log_warn "  ⚠ $SCENARIO_ID: Invalid statistics in summary.json"
            fi
        fi
        
        TOTAL_VALID=$((TOTAL_VALID + 1))
        
    done < <(eval "$FIND_CMD")
    
    echo ""
done

# Cross-environment consistency check
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}Cross-Environment Consistency Check:${NC}"
echo ""

# Find all unique scenario IDs across environments
declare -A SCENARIO_ENVS
for env in "${ENVS[@]}"; do
    ENV_RESULTS_DIR="$RESULTS_DIR/$env"
    if [[ ! -d "$ENV_RESULTS_DIR" ]]; then
        continue
    fi
    
    while IFS= read -r jsonl_file; do
        SCENARIO_ID=$(basename "$(dirname "$(dirname "$jsonl_file")")")
        if [[ -n "$SCENARIO_ID" ]]; then
            SCENARIO_ENVS["$SCENARIO_ID"]="${SCENARIO_ENVS[$SCENARIO_ID]:-}$env "
        fi
    done < <(find "$ENV_RESULTS_DIR" -name "run.jsonl" -type f -path "*/raw/*" 2>/dev/null || true)
done

INCONSISTENT_COUNT=0
if [[ ${#SCENARIO_ENVS[@]} -gt 0 ]]; then
    EXPECTED_ENV_COUNT=${#ENVS[@]}
    for scenario_id in "${!SCENARIO_ENVS[@]}"; do
        env_list="${SCENARIO_ENVS[$scenario_id]}"
        env_count=$(echo "$env_list" | wc -w)
        if [[ $env_count -lt $EXPECTED_ENV_COUNT ]]; then
            INCONSISTENT_COUNT=$((INCONSISTENT_COUNT + 1))
            if [[ $INCONSISTENT_COUNT -le 5 ]]; then
                log_warn "  ⚠ $scenario_id: Present in $env_count/$EXPECTED_ENV_COUNT environments"
            fi
        fi
    done
    
    if [[ $INCONSISTENT_COUNT -gt 5 ]]; then
        log_warn "  ... and $((INCONSISTENT_COUNT - 5)) more inconsistent scenarios"
    fi
fi

echo ""

# Summary
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}Validation Summary:${NC}"
echo ""
echo "  Total checked: $TOTAL_CHECKED files"
echo "  Valid: $TOTAL_VALID"
echo "  Empty: $TOTAL_EMPTY"
echo "  Invalid: $TOTAL_INVALID"
echo "  Missing summary.json: $TOTAL_MISSING_SUMMARY"
echo "  Invalid statistics: $TOTAL_INVALID_STATS"
if [[ $INCONSISTENT_COUNT -gt 0 ]]; then
    echo "  Cross-environment inconsistencies: $INCONSISTENT_COUNT"
fi

if [[ $TOTAL_EMPTY -gt 0 ]]; then
    echo ""
    echo -e "${YELLOW}Empty files found:${NC}"
    for entry in "${EMPTY_FILES[@]}"; do
        IFS=':' read -r scenario_id file_path <<< "$entry"
        echo "  - $scenario_id: $file_path"
    done
fi

if [[ $TOTAL_MISSING_SUMMARY -gt 0 ]]; then
    echo ""
    echo -e "${YELLOW}Missing summary.json files:${NC}"
    for entry in "${MISSING_SUMMARIES[@]}"; do
        IFS=':' read -r scenario_id exp_dir <<< "$entry"
        echo "  - $scenario_id: $exp_dir"
    done
fi

if [[ $TOTAL_INVALID_STATS -gt 0 ]]; then
    echo ""
    echo -e "${YELLOW}Invalid statistics in summary.json:${NC}"
    for entry in "${INVALID_STATS[@]}"; do
        IFS=':' read -r scenario_id summary_file <<< "$entry"
        echo "  - $scenario_id: $summary_file"
    done
fi

echo ""

# Exit status
TOTAL_ISSUES=$((TOTAL_EMPTY + TOTAL_INVALID + TOTAL_MISSING_SUMMARY + TOTAL_INVALID_STATS))
if [[ $TOTAL_ISSUES -gt 0 ]]; then
    if [[ "$FAIL_ON_EMPTY" == "true" ]]; then
        log_error "Validation failed: Found $TOTAL_ISSUES issues"
        exit 1
    else
        log_warn "Validation found $TOTAL_ISSUES issues (use --fail-on-empty to exit with error)"
        exit 0
    fi
else
    log_success "All files validated successfully!"
    exit 0
fi

