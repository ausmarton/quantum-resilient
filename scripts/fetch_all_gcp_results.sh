#!/usr/bin/env bash
# =============================================================================
# fetch_all_gcp_results.sh - Fetch all completed GCP experiments from GCS
#
# Lists all experiments in GCS bucket and downloads those that are missing
# locally or need to be updated.
#
# Usage:
#   ./scripts/fetch_all_gcp_results.sh \
#     --bucket pqc-bench-results \
#     --out results/gcp
#
# Options:
#   --bucket NAME       GCS bucket name (required)
#   --out DIR           Output directory (default: results/gcp)
#   --force             Re-download even if local data exists
#   --exp-id ID         Fetch only specific experiment ID
#   --skip-analysis     Skip analysis pipeline (download only)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Default values
BUCKET=""
OUT_DIR="results/gcp"
FORCE=false
EXP_ID_FILTER=""
SKIP_ANALYSIS=false

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

Fetch all completed GCP experiments from GCS bucket.

OPTIONS:
    --bucket NAME       GCS bucket name (required)
    --out DIR           Output directory (default: results/gcp)
    --force             Re-download even if local data exists
    --exp-id ID         Fetch only specific experiment ID
    --skip-analysis     Skip analysis pipeline (download only)
    -h, --help          Show this help message

EXAMPLE:
    $0 --bucket pqc-bench-results
    $0 --bucket pqc-bench-results --exp-id rsa2048_p256_r100_run1_c0098396
EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --bucket)
            BUCKET="$2"
            shift 2
            ;;
        --out)
            OUT_DIR="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --exp-id)
            EXP_ID_FILTER="$2"
            shift 2
            ;;
        --skip-analysis)
            SKIP_ANALYSIS=true
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
if [[ -z "$BUCKET" ]]; then
    log_error "Missing required argument: --bucket"
    usage
fi

# Check prerequisites
if ! command -v gsutil &> /dev/null; then
    log_error "gsutil not found. Please install Google Cloud SDK."
    exit 1
fi

# Check authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1 | grep -q "@"; then
    log_warn "gcloud may not be authenticated. Run: gcloud auth login"
fi

# Create output directory
mkdir -p "$OUT_DIR"

log_info "Bucket: $BUCKET"
log_info "Output: $OUT_DIR"
log_info "Force: $FORCE"

# List all experiments in GCS
log_info "Listing experiments in GCS bucket..."
GCS_EXPERIMENTS_DIR="gs://${BUCKET}/experiments"

if ! gsutil ls "$GCS_EXPERIMENTS_DIR/" &>/dev/null; then
    log_error "GCS experiments directory not found: $GCS_EXPERIMENTS_DIR"
    log_info "Available paths in bucket:"
    gsutil ls "gs://${BUCKET}/" 2>/dev/null || echo "  (none)"
    exit 1
fi

# Get list of experiment IDs from GCS
log_info "Fetching experiment list from GCS..."
EXPERIMENTS=$(gsutil ls "$GCS_EXPERIMENTS_DIR/" 2>/dev/null | \
    sed "s|${GCS_EXPERIMENTS_DIR}/||" | \
    sed 's|/$||' | \
    grep -v '^$' | \
    sort)

if [[ -z "$EXPERIMENTS" ]]; then
    log_warn "No experiments found in GCS bucket"
    exit 0
fi

# Filter by exp-id if specified
if [[ -n "$EXP_ID_FILTER" ]]; then
    EXPERIMENTS=$(echo "$EXPERIMENTS" | grep -E "^${EXP_ID_FILTER}$" || true)
    if [[ -z "$EXPERIMENTS" ]]; then
        log_warn "Experiment ID '$EXP_ID_FILTER' not found in GCS"
        exit 0
    fi
fi

EXPERIMENT_COUNT=$(echo "$EXPERIMENTS" | wc -l)
log_info "Found $EXPERIMENT_COUNT experiment(s) in GCS"

# Process each experiment
SUCCESS_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

while IFS= read -r EXP_ID; do
    [[ -z "$EXP_ID" ]] && continue
    
    log_info ""
    log_info "Processing: $EXP_ID"
    
    EXP_OUT_DIR="$OUT_DIR/$EXP_ID"
    
    # Check if already exists locally
    if [[ "$FORCE" != "true" ]] && [[ -f "$EXP_OUT_DIR/raw/run.jsonl" ]] || [[ -f "$EXP_OUT_DIR/merged/merged.jsonl" ]]; then
        log_warn "  Already exists locally, skipping (use --force to re-download)"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi
    
    # Check if experiment has data in GCS
    GCS_EXP_PATH="$GCS_EXPERIMENTS_DIR/$EXP_ID"
    HAS_DATA=false
    
    if gsutil ls "$GCS_EXP_PATH/merged.jsonl" &>/dev/null; then
        HAS_DATA=true
    elif gsutil ls "$GCS_EXP_PATH/raw/run.jsonl" &>/dev/null; then
        HAS_DATA=true
    fi
    
    if [[ "$HAS_DATA" != "true" ]]; then
        log_warn "  No data found in GCS, skipping"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi
    
    # Fetch using existing script (which includes validation)
    log_info "  Fetching from GCS..."
    if "$SCRIPT_DIR/fetch_and_analyse_from_gcs.sh" \
        --exp-id "$EXP_ID" \
        --bucket "$BUCKET" \
        --out "$EXP_OUT_DIR" \
        $([ "$SKIP_ANALYSIS" == "true" ] && echo "--skip-analysis" || echo "") 2>&1; then
        # Additional integrity validation after download
        log_info "  Validating downloaded data..."
        VALIDATION_FAILED=false
        
        # Check if we have data files
        if [[ -f "$EXP_OUT_DIR/raw/run.jsonl" ]]; then
            RAW_FILE="$EXP_OUT_DIR/raw/run.jsonl"
        elif [[ -f "$EXP_OUT_DIR/merged/merged.jsonl" ]]; then
            RAW_FILE="$EXP_OUT_DIR/merged/merged.jsonl"
        else
            log_error "  ✗ No data files found after download"
            VALIDATION_FAILED=true
        fi
        
        if [[ "$VALIDATION_FAILED" != "true" ]] && [[ -f "$RAW_FILE" ]]; then
            # Validate file size
            FILE_SIZE=$(stat -f%z "$RAW_FILE" 2>/dev/null || stat -c%s "$RAW_FILE" 2>/dev/null || echo 0)
            if [[ $FILE_SIZE -eq 0 ]]; then
                log_error "  ✗ Validation failed: File is 0 bytes"
                VALIDATION_FAILED=true
            fi
            
            # Validate line count
            if [[ "$VALIDATION_FAILED" != "true" ]]; then
                LINE_COUNT=$(wc -l < "$RAW_FILE" 2>/dev/null || echo 0)
                if [[ $LINE_COUNT -eq 0 ]]; then
                    log_error "  ✗ Validation failed: File has no lines"
                    VALIDATION_FAILED=true
                fi
            fi
            
            # Validate JSONL format (check first and last lines)
            if [[ "$VALIDATION_FAILED" != "true" ]]; then
                FIRST_LINE=$(head -1 "$RAW_FILE" 2>/dev/null || echo "")
                if [[ -n "$FIRST_LINE" ]]; then
                    if ! echo "$FIRST_LINE" | python3 -m json.tool >/dev/null 2>&1; then
                        log_error "  ✗ Validation failed: First line is not valid JSON"
                        VALIDATION_FAILED=true
                    fi
                fi
                
                # Check last line (if file has multiple lines)
                if [[ "$VALIDATION_FAILED" != "true" ]] && [[ $LINE_COUNT -gt 1 ]]; then
                    LAST_LINE=$(tail -1 "$RAW_FILE" 2>/dev/null || echo "")
                    if [[ -n "$LAST_LINE" ]]; then
                        if ! echo "$LAST_LINE" | python3 -m json.tool >/dev/null 2>&1; then
                            log_warn "  ⚠ Last line is not valid JSON (may be incomplete)"
                        fi
                    fi
                fi
            fi
        fi
        
        if [[ "$VALIDATION_FAILED" == "true" ]]; then
            log_error "  ✗ Validation failed: $EXP_ID"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        else
            log_success "  ✓ Fetched and validated: $EXP_ID"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        fi
    else
        log_error "  ✗ Failed: $EXP_ID"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
done <<< "$EXPERIMENTS"

# Summary
echo ""
log_info "=== Summary ==="
log_success "Successfully fetched: $SUCCESS_COUNT"
log_warn "Skipped: $SKIP_COUNT"
if [[ $FAIL_COUNT -gt 0 ]]; then
    log_error "Failed: $FAIL_COUNT"
fi
echo ""

if [[ $SUCCESS_COUNT -gt 0 ]]; then
    log_success "Results available in: $OUT_DIR"
fi

exit 0

