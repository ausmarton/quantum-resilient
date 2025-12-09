#!/usr/bin/env bash
# =============================================================================
# validate_gcp_downloads.sh - Validate integrity of downloaded GCP experiments
#
# Performs comprehensive validation of downloaded GCP experiment data:
# - File size checks
# - JSONL format validation
# - Line count validation
# - Sample data validation
# - Metadata file checks
#
# Usage:
#   ./scripts/validate_gcp_downloads.sh [OPTIONS]
#
# Options:
#   --results-dir DIR    Results directory (default: results/gcp)
#   --exp-id ID          Validate only specific experiment ID
#   --strict             Exit with error if any validation fails
#   --detailed           Show detailed validation for each file
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Default values
RESULTS_DIR="results/gcp"
EXP_ID_FILTER=""
STRICT=false
DETAILED=false

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

Validate integrity of downloaded GCP experiment data.

OPTIONS:
    --results-dir DIR    Results directory (default: results/gcp)
    --exp-id ID          Validate only specific experiment ID
    --strict             Exit with error if any validation fails
    --detailed           Show detailed validation for each file
    -h, --help           Show this help message

EXAMPLES:
    # Validate all downloaded experiments
    ./scripts/validate_gcp_downloads.sh

    # Validate specific experiment
    ./scripts/validate_gcp_downloads.sh --exp-id rsa2048_p256_r100_run1_c0098396

    # Strict mode (fail on any error)
    ./scripts/validate_gcp_downloads.sh --strict
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
        --exp-id)
            EXP_ID_FILTER="$2"
            shift 2
            ;;
        --strict)
            STRICT=true
            shift
            ;;
        --detailed)
            DETAILED=true
            shift
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

if [[ ! -d "$RESULTS_DIR" ]]; then
    log_error "Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Find all experiment directories
if [[ -n "$EXP_ID_FILTER" ]]; then
    EXP_DIRS=("$RESULTS_DIR/$EXP_ID_FILTER")
else
    EXP_DIRS=("$RESULTS_DIR"/*)
fi

TOTAL_EXPS=0
VALID_EXPS=0
INVALID_EXPS=0
WARNINGS=0

echo -e "${CYAN}=== GCP Download Validation ===${NC}"
echo "Results directory: $RESULTS_DIR"
echo ""

for exp_dir in "${EXP_DIRS[@]}"; do
    [[ ! -d "$exp_dir" ]] && continue
    
    EXP_ID=$(basename "$exp_dir")
    TOTAL_EXPS=$((TOTAL_EXPS + 1))
    
    if [[ "$DETAILED" == "true" ]]; then
        echo -e "${BLUE}Validating: $EXP_ID${NC}"
    fi
    
    EXP_VALID=true
    EXP_WARNINGS=0
    
    # Find data files
    RAW_FILE=""
    MERGED_FILE=""
    
    if [[ -f "$exp_dir/raw/run.jsonl" ]]; then
        RAW_FILE="$exp_dir/raw/run.jsonl"
    fi
    
    if [[ -f "$exp_dir/merged/merged.jsonl" ]]; then
        MERGED_FILE="$exp_dir/merged/merged.jsonl"
    fi
    
    # Prefer merged, fallback to raw
    DATA_FILE="$MERGED_FILE"
    [[ -z "$DATA_FILE" ]] && DATA_FILE="$RAW_FILE"
    
    if [[ -z "$DATA_FILE" ]] || [[ ! -f "$DATA_FILE" ]]; then
        log_error "  ✗ No data file found (raw/run.jsonl or merged/merged.jsonl)"
        EXP_VALID=false
        INVALID_EXPS=$((INVALID_EXPS + 1))
        continue
    fi
    
    # Validate file size
    FILE_SIZE=$(stat -f%z "$DATA_FILE" 2>/dev/null || stat -c%s "$DATA_FILE" 2>/dev/null || echo 0)
    if [[ $FILE_SIZE -eq 0 ]]; then
        log_error "  ✗ File is 0 bytes"
        EXP_VALID=false
    elif [[ "$DETAILED" == "true" ]]; then
        log_info "  File size: $FILE_SIZE bytes"
    fi
    
    # Validate line count
    if [[ "$EXP_VALID" == "true" ]]; then
        LINE_COUNT=$(wc -l < "$DATA_FILE" 2>/dev/null || echo 0)
        if [[ $LINE_COUNT -eq 0 ]]; then
            log_error "  ✗ File has no lines"
            EXP_VALID=false
        elif [[ "$DETAILED" == "true" ]]; then
            log_info "  Line count: $LINE_COUNT"
        fi
    fi
    
    # Validate JSONL format
    if [[ "$EXP_VALID" == "true" ]]; then
        # Check first line
        FIRST_LINE=$(head -1 "$DATA_FILE" 2>/dev/null || echo "")
        if [[ -z "$FIRST_LINE" ]]; then
            log_error "  ✗ File is empty (no first line)"
            EXP_VALID=false
        elif ! echo "$FIRST_LINE" | python3 -m json.tool >/dev/null 2>&1; then
            log_error "  ✗ First line is not valid JSON"
            if [[ "$DETAILED" == "true" ]]; then
                log_error "    First line: ${FIRST_LINE:0:100}..."
            fi
            EXP_VALID=false
        elif [[ "$DETAILED" == "true" ]]; then
            log_success "  ✓ First line is valid JSON"
        fi
        
        # Check last line (if multiple lines)
        if [[ "$EXP_VALID" == "true" ]] && [[ $LINE_COUNT -gt 1 ]]; then
            LAST_LINE=$(tail -1 "$DATA_FILE" 2>/dev/null || echo "")
            if [[ -n "$LAST_LINE" ]]; then
                if ! echo "$LAST_LINE" | python3 -m json.tool >/dev/null 2>&1; then
                    log_warn "  ⚠ Last line is not valid JSON (may indicate incomplete file)"
                    EXP_WARNINGS=$((EXP_WARNINGS + 1))
                elif [[ "$DETAILED" == "true" ]]; then
                    log_success "  ✓ Last line is valid JSON"
                fi
            fi
        fi
        
        # Sample validation (check random lines)
        if [[ "$EXP_VALID" == "true" ]] && [[ $LINE_COUNT -gt 10 ]]; then
            SAMPLE_COUNT=5
            INVALID_SAMPLES=0
            for i in $(seq 1 $SAMPLE_COUNT); do
                # Get random line number
                RANDOM_LINE=$((RANDOM % LINE_COUNT + 1))
                SAMPLE_LINE=$(sed -n "${RANDOM_LINE}p" "$DATA_FILE" 2>/dev/null || echo "")
                if [[ -n "$SAMPLE_LINE" ]]; then
                    if ! echo "$SAMPLE_LINE" | python3 -m json.tool >/dev/null 2>&1; then
                        INVALID_SAMPLES=$((INVALID_SAMPLES + 1))
                    fi
                fi
            done
            
            if [[ $INVALID_SAMPLES -gt 0 ]]; then
                log_warn "  ⚠ Found $INVALID_SAMPLES invalid sample(s) out of $SAMPLE_COUNT"
                EXP_WARNINGS=$((EXP_WARNINGS + 1))
            elif [[ "$DETAILED" == "true" ]]; then
                log_success "  ✓ Sample validation passed ($SAMPLE_COUNT random lines)"
            fi
        fi
    fi
    
    # Check metadata files
    if [[ "$EXP_VALID" == "true" ]]; then
        MISSING_METADATA=0
        
        if [[ ! -f "$exp_dir/cloud_metadata.json" ]]; then
            log_warn "  ⚠ Missing: cloud_metadata.json"
            MISSING_METADATA=$((MISSING_METADATA + 1))
        fi
        
        if [[ ! -f "$exp_dir/provenance.json" ]]; then
            log_warn "  ⚠ Missing: provenance.json"
            MISSING_METADATA=$((MISSING_METADATA + 1))
        fi
        
        if [[ $MISSING_METADATA -gt 0 ]]; then
            EXP_WARNINGS=$((EXP_WARNINGS + MISSING_METADATA))
        elif [[ "$DETAILED" == "true" ]]; then
            log_success "  ✓ Metadata files present"
        fi
    fi
    
    # Summary for this experiment
    if [[ "$EXP_VALID" == "true" ]]; then
        if [[ $EXP_WARNINGS -eq 0 ]]; then
            if [[ "$DETAILED" == "true" ]]; then
                log_success "  ✓ Validation passed: $EXP_ID"
            fi
            VALID_EXPS=$((VALID_EXPS + 1))
        else
            log_warn "  ⚠ Validation passed with $EXP_WARNINGS warning(s): $EXP_ID"
            VALID_EXPS=$((VALID_EXPS + 1))
            WARNINGS=$((WARNINGS + EXP_WARNINGS))
        fi
    else
        INVALID_EXPS=$((INVALID_EXPS + 1))
        if [[ "$STRICT" == "true" ]]; then
            log_error "Strict mode: Exiting due to validation failure"
            exit 1
        fi
    fi
    
    if [[ "$DETAILED" == "true" ]]; then
        echo ""
    fi
done

# Summary
echo ""
echo -e "${CYAN}=== Validation Summary ===${NC}"
echo "Total experiments: $TOTAL_EXPS"
log_success "Valid: $VALID_EXPS"
if [[ $INVALID_EXPS -gt 0 ]]; then
    log_error "Invalid: $INVALID_EXPS"
fi
if [[ $WARNINGS -gt 0 ]]; then
    log_warn "Warnings: $WARNINGS"
fi
echo ""

if [[ $INVALID_EXPS -gt 0 ]]; then
    if [[ "$STRICT" == "true" ]]; then
        exit 1
    else
        exit 0
    fi
fi

exit 0

