#!/usr/bin/env bash
# =============================================================================
# list_gcp_experiments.sh - List all experiments in GCS bucket
#
# Quick helper to see what experiments are available in GCS and which ones
# are already downloaded locally.
#
# Usage:
#   ./scripts/list_gcp_experiments.sh --bucket pqc-bench-results
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Default values
BUCKET=""
LOCAL_DIR="results/gcp"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

List all experiments in GCS bucket and show which are downloaded locally.

OPTIONS:
    --bucket NAME       GCS bucket name (required)
    --local DIR         Local results directory (default: results/gcp)
    -h, --help          Show this help message

EXAMPLE:
    $0 --bucket pqc-bench-results
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
        --local)
            LOCAL_DIR="$2"
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

if [[ -z "$BUCKET" ]]; then
    echo "Error: --bucket is required"
    usage
fi

# Check prerequisites
if ! command -v gsutil &> /dev/null; then
    echo "Error: gsutil not found"
    exit 1
fi

echo -e "${CYAN}=== GCS Experiments ===${NC}"
echo "Bucket: $BUCKET"
echo ""

# List experiments in GCS
GCS_EXPERIMENTS_DIR="gs://${BUCKET}/experiments"

if ! gsutil ls "$GCS_EXPERIMENTS_DIR/" &>/dev/null; then
    echo "No experiments directory found in bucket"
    exit 0
fi

GCS_EXPERIMENTS=$(gsutil ls "$GCS_EXPERIMENTS_DIR/" 2>/dev/null | \
    sed "s|${GCS_EXPERIMENTS_DIR}/||" | \
    sed 's|/$||' | \
    grep -v '^$' | \
    sort)

if [[ -z "$GCS_EXPERIMENTS" ]]; then
    echo "No experiments found in GCS"
    exit 0
fi

GCS_COUNT=$(echo "$GCS_EXPERIMENTS" | wc -l)
echo -e "${BLUE}Found $GCS_COUNT experiment(s) in GCS:${NC}"
echo ""

# Check local status
LOCAL_COUNT=0
MISSING_COUNT=0

while IFS= read -r EXP_ID; do
    [[ -z "$EXP_ID" ]] && continue
    
    LOCAL_EXP_DIR="$LOCAL_DIR/$EXP_ID"
    HAS_DATA=false
    
    if [[ -f "$LOCAL_EXP_DIR/raw/run.jsonl" ]] || [[ -f "$LOCAL_EXP_DIR/merged/merged.jsonl" ]]; then
        HAS_DATA=true
        LOCAL_COUNT=$((LOCAL_COUNT + 1))
        echo -e "  ${GREEN}✓${NC} $EXP_ID (downloaded)"
    else
        MISSING_COUNT=$((MISSING_COUNT + 1))
        echo -e "  ${YELLOW}○${NC} $EXP_ID (missing)"
    fi
done <<< "$GCS_EXPERIMENTS"

echo ""
echo -e "${CYAN}=== Summary ===${NC}"
echo "Total in GCS: $GCS_COUNT"
echo "Downloaded locally: $LOCAL_COUNT"
echo "Missing locally: $MISSING_COUNT"
echo ""

if [[ $MISSING_COUNT -gt 0 ]]; then
    echo -e "${YELLOW}To fetch missing experiments:${NC}"
    echo "  ./scripts/fetch_all_gcp_results.sh --bucket $BUCKET"
    echo ""
fi

