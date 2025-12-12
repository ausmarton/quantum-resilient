#!/usr/bin/env bash
# =============================================================================
# fetch_and_analyse_from_gcs.sh - Download and analyze GCP experiment results
#
# Downloads results from GCS and runs the local analysis pipeline to generate
# statistics and publication-quality plots.
#
# Usage:
#   ./fetch_and_analyse_from_gcs.sh \
#     --exp-id exp3 \
#     --bucket pqc-bench-results \
#     --out results/exp3
#
# Requirements:
#   - gsutil (gcloud CLI)
#   - Python 3.10+ with analysis dependencies
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
EXP_ID=""
BUCKET=""
OUT_DIR=""
SKIP_DOWNLOAD=false
SKIP_ANALYSIS=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
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

log_step() {
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Download and analyze GCP experiment results.

OPTIONS:
    --exp-id ID         Experiment identifier (required)
    --bucket NAME       GCS bucket name (required)
    --out DIR           Output directory (required)
    --skip-download     Skip GCS download (use existing local data)
    --skip-analysis     Skip analysis pipeline
    -h, --help          Show this help message

EXAMPLE:
    $0 --exp-id exp3 --bucket pqc-results --out results/exp3
EOF
    exit 1
}

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp-id)
            EXP_ID="$2"
            shift 2
            ;;
        --bucket)
            BUCKET="$2"
            shift 2
            ;;
        --out)
            OUT_DIR="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
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
if [[ -z "$EXP_ID" ]]; then
    log_error "Missing required argument: --exp-id"
    usage
fi

if [[ -z "$BUCKET" ]]; then
    log_error "Missing required argument: --bucket"
    usage
fi

if [[ -z "$OUT_DIR" ]]; then
    log_error "Missing required argument: --out"
    usage
fi

# Make output path absolute
OUT_DIR="$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)"

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
START_TIME=$(date +%s)
START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           PQC Benchmark - GCS Fetch & Analyze                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

log_info "Experiment ID: $EXP_ID"
log_info "Bucket: $BUCKET"
log_info "Output: $OUT_DIR"
log_info "Started: $START_ISO"

# =============================================================================
# Step 1: Verify prerequisites
# =============================================================================
log_step "Step 1/4: Verifying prerequisites"

# Check gsutil
if ! command -v gsutil &> /dev/null; then
    log_error "gsutil not found. Please install Google Cloud SDK."
    exit 1
fi
log_success "gsutil available"

# Check gcloud authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1 | grep -q "@"; then
    log_warn "gcloud may not be authenticated. Run: gcloud auth login"
fi

# Determine Python command (use container wrapper if available, else system python3)
CONTAINER_WRAPPER="$SCRIPT_DIR/scripts/lib/run-python-container.sh"
USE_CONTAINER="${QR_USE_CONTAINER:-true}"

if [[ "$USE_CONTAINER" == "true" ]] && [[ -f "$CONTAINER_WRAPPER" ]]; then
    PYTHON_CMD="$CONTAINER_WRAPPER"
    log_info "Using containerized Python environment"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    log_info "Using host Python: $(python3 --version)"
else
    log_error "python3 not found. Please install Python 3.10+ or use containerized environment"
    exit 1
fi

# =============================================================================
# Step 2: Create output directories
# =============================================================================
log_step "Step 2/4: Creating output directories"

mkdir -p "$OUT_DIR/raw"
mkdir -p "$OUT_DIR/merged"
mkdir -p "$OUT_DIR/stats"
mkdir -p "$OUT_DIR/figures"

log_success "Created: $OUT_DIR/{raw,merged,stats,figures}"

# =============================================================================
# Step 3: Download from GCS
# =============================================================================
log_step "Step 3/4: Downloading results from GCS"

GCS_PATH="gs://${BUCKET}/experiments/${EXP_ID}"

if [[ "$SKIP_DOWNLOAD" == "true" ]]; then
    log_warn "Skipping download (--skip-download)"
else
    log_info "Checking GCS path: $GCS_PATH"
    
    # Verify bucket exists and has data
    if ! gsutil ls "$GCS_PATH/" &> /dev/null; then
        log_error "GCS path not found: $GCS_PATH"
        log_info "Available experiments in bucket:"
        gsutil ls "gs://${BUCKET}/experiments/" 2>/dev/null || echo "  (none)"
        exit 1
    fi
    
    # Download merged JSONL
    log_info "Downloading merged.jsonl..."
    if gsutil cp "$GCS_PATH/merged.jsonl" "$OUT_DIR/merged/merged.jsonl" 2>/dev/null; then
        log_success "Downloaded merged.jsonl"
    else
        log_warn "merged.jsonl not found, trying raw data..."
        # Download raw data instead (use direct file path to avoid wildcard issues)
        gsutil -m cp "$GCS_PATH/raw/run.jsonl" "$OUT_DIR/raw/run.jsonl" 2>/dev/null || \
        gsutil -m rsync -r "$GCS_PATH/raw" "$OUT_DIR/raw" 2>/dev/null || true
    fi
    
    # Download manifest
    log_info "Downloading manifest.json..."
    if gsutil cp "$GCS_PATH/manifest.json" "$OUT_DIR/manifest.json" 2>/dev/null; then
        log_success "Downloaded manifest.json"
    else
        log_warn "manifest.json not found"
    fi
    
    # Download provenance
    log_info "Downloading provenance.json..."
    if gsutil cp "$GCS_PATH/provenance.json" "$OUT_DIR/provenance.json" 2>/dev/null; then
        log_success "Downloaded provenance.json"
    else
        log_warn "provenance.json not found"
    fi
    
    # Download cloud metadata
    log_info "Downloading cloud_metadata.json..."
    if gsutil cp "$GCS_PATH/cloud_metadata.json" "$OUT_DIR/cloud_metadata.json" 2>/dev/null; then
        log_success "Downloaded cloud_metadata.json"
    else
        log_warn "cloud_metadata.json not found"
    fi
    
    # Download summary if exists
    log_info "Downloading summary.json (if exists)..."
    gsutil cp "$GCS_PATH/summary.json" "$OUT_DIR/stats/summary.json" 2>/dev/null || true
    
    # Download raw data
    log_info "Downloading raw data..."
    # Use gsutil rsync or cp without trailing slash to avoid "InvalidUrlError"
    if ! gsutil -m cp "$GCS_PATH/raw/run.jsonl" "$OUT_DIR/raw/run.jsonl" 2>&1; then
        # Fallback: try with rsync if direct copy fails
        if ! gsutil -m rsync -r "$GCS_PATH/raw" "$OUT_DIR/raw" 2>&1; then
            log_warn "Some raw data files may have failed to download"
        fi
    fi
    
    log_success "Download complete"
fi

# Verify we have data to analyze and validate integrity
if [[ ! -f "$OUT_DIR/merged/merged.jsonl" ]]; then
    # Check for raw data
    RAW_COUNT=$(find "$OUT_DIR/raw" -name "*.jsonl" -type f 2>/dev/null | wc -l)
    if [[ $RAW_COUNT -eq 0 ]]; then
        log_error "No data files found!"
        exit 1
    fi
    
    log_info "Found $RAW_COUNT raw JSONL file(s)"
    
    # Validate integrity of downloaded files
    INVALID_COUNT=0
    for jsonl_file in "$OUT_DIR/raw"/*.jsonl; do
        [[ ! -f "$jsonl_file" ]] && continue
        
        FILE_SIZE=$(stat -f%z "$jsonl_file" 2>/dev/null || stat -c%s "$jsonl_file" 2>/dev/null || echo 0)
        if [[ $FILE_SIZE -eq 0 ]]; then
            log_error "Invalid file (0 bytes): $(basename "$jsonl_file")"
            INVALID_COUNT=$((INVALID_COUNT + 1))
            continue
        fi
        
        # Check for error messages
        FIRST_LINE=$(head -1 "$jsonl_file" 2>/dev/null || echo "")
        if [[ -n "$FIRST_LINE" ]] && [[ "$FIRST_LINE" =~ ^error: ]]; then
            log_error "Invalid file (contains error message): $(basename "$jsonl_file")"
            log_error "  First line: ${FIRST_LINE:0:80}..."
            INVALID_COUNT=$((INVALID_COUNT + 1))
            continue
        fi
        
        # Validate JSONL format
        # Use Python's json module directly for more robust validation
        # This handles long lines and special characters better than json.tool
        if [[ -n "$FIRST_LINE" ]]; then
            if ! $PYTHON_CMD -c "import json, sys; json.loads(sys.stdin.read())" <<< "$FIRST_LINE" >/dev/null 2>&1; then
                log_error "Invalid file (not valid JSONL): $(basename "$jsonl_file")"
                log_error "  First line: ${FIRST_LINE:0:80}..."
                INVALID_COUNT=$((INVALID_COUNT + 1))
                continue
            fi
        fi
    done
    
    if [[ $INVALID_COUNT -gt 0 ]]; then
        log_error "Found $INVALID_COUNT invalid file(s) out of $RAW_COUNT"
        log_error "These files may need to be re-downloaded or re-run"
        exit 1
    fi
    
    log_success "All $RAW_COUNT file(s) validated successfully"
fi

# =============================================================================
# Step 4: Run analysis pipeline
# =============================================================================
log_step "Step 4/4: Running analysis pipeline"

if [[ "$SKIP_ANALYSIS" == "true" ]]; then
    log_warn "Skipping analysis (--skip-analysis)"
else
    log_info "Running analysis pipeline..."
    
    # Determine input path
    if [[ -f "$OUT_DIR/merged/merged.jsonl" ]]; then
        INPUT_PATH="$OUT_DIR/merged"
    else
        INPUT_PATH="$OUT_DIR/raw"
    fi
    
    # Run the full pipeline if available
    if [[ -f "$SCRIPT_DIR/analysis/run_full_pipeline.sh" ]]; then
        bash "$SCRIPT_DIR/analysis/run_full_pipeline.sh" "$EXP_ID" "$INPUT_PATH" 2>&1 | while read -r line; do
            echo "  $line"
        done || log_warn "Analysis pipeline completed with warnings"
    else
        log_info "Running individual analysis scripts..."
        
        # Merge if needed
        if [[ ! -f "$OUT_DIR/merged/merged.jsonl" ]]; then
            log_info "Merging JSONL files..."
            $PYTHON_CMD "$SCRIPT_DIR/analysis/scripts/merge_jsonl.py" \
                --input "$OUT_DIR/raw" \
                --output "$OUT_DIR/merged" 2>/dev/null || true
        fi
        
        # Compute statistics
        INPUT_FILE="$OUT_DIR/merged/merged.parquet"
        [[ ! -f "$INPUT_FILE" ]] && INPUT_FILE="$OUT_DIR/merged/merged.jsonl"
        
        log_info "Computing statistics..."
        $PYTHON_CMD "$SCRIPT_DIR/analysis/scripts/compute_statistics.py" \
            --input "$INPUT_FILE" \
            --output "$OUT_DIR/stats" \
            --experiment-id "$EXP_ID" 2>/dev/null || \
        $PYTHON_CMD "$SCRIPT_DIR/analysis/scripts/compute_stats.py" \
            --input "$INPUT_FILE" \
            --output "$OUT_DIR/stats" \
            --experiment-id "$EXP_ID" 2>/dev/null || true
        
        # Generate plots with cloud suffix
        log_info "Generating latency CDF plot..."
        $PYTHON_CMD "$SCRIPT_DIR/analysis/scripts/plot_ecdf.py" \
            --input "$INPUT_FILE" \
            --output "$OUT_DIR/figures" \
            --experiment-id "$EXP_ID" \
            --suffix "_cloud" 2>/dev/null || \
        $PYTHON_CMD "$SCRIPT_DIR/analysis/scripts/plot_latency.py" \
            --input "$INPUT_FILE" \
            --output "$OUT_DIR/figures" \
            --experiment-id "$EXP_ID" 2>/dev/null || true
        
        # Copy and rename for cloud versions
        if [[ -f "$OUT_DIR/figures/latency_cdf.png" ]]; then
            cp "$OUT_DIR/figures/latency_cdf.png" "$OUT_DIR/figures/latency_cdf_cloud.png"
        fi
        
        log_info "Generating throughput plot..."
        $PYTHON_CMD "$SCRIPT_DIR/analysis/scripts/plot_throughput.py" \
            --input "$INPUT_FILE" \
            --output "$OUT_DIR/figures" \
            --experiment-id "$EXP_ID" \
            --suffix "_cloud" 2>/dev/null || \
        $PYTHON_CMD "$SCRIPT_DIR/analysis/scripts/plot_throughput.py" \
            --input "$INPUT_FILE" \
            --output "$OUT_DIR/figures" \
            --experiment-id "$EXP_ID" 2>/dev/null || true
        
        # Copy and rename for cloud versions
        if [[ -f "$OUT_DIR/figures/throughput.png" ]]; then
            cp "$OUT_DIR/figures/throughput.png" "$OUT_DIR/figures/throughput_cloud.png"
        fi
    fi
    
    log_success "Analysis complete"
fi

# =============================================================================
# Summary
# =============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║               FETCH & ANALYZE COMPLETE                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

log_info "Experiment ID: $EXP_ID"
log_info "Duration: ${ELAPSED}s"
echo ""

log_info "Output files:"
[[ -f "$OUT_DIR/merged/merged.jsonl" ]] && echo "  Merged JSONL:        $OUT_DIR/merged/merged.jsonl"
[[ -f "$OUT_DIR/merged/merged.parquet" ]] && echo "  Parquet:             $OUT_DIR/merged/merged.parquet"
[[ -f "$OUT_DIR/stats/summary.json" ]] && echo "  Stats:               $OUT_DIR/stats/summary.json"
[[ -f "$OUT_DIR/figures/latency_cdf_cloud.png" ]] && echo "  Latency CDF (cloud): $OUT_DIR/figures/latency_cdf_cloud.png"
[[ -f "$OUT_DIR/figures/throughput_cloud.png" ]] && echo "  Throughput (cloud):  $OUT_DIR/figures/throughput_cloud.png"
[[ -f "$OUT_DIR/manifest.json" ]] && echo "  Manifest:            $OUT_DIR/manifest.json"
[[ -f "$OUT_DIR/provenance.json" ]] && echo "  Provenance:          $OUT_DIR/provenance.json"
echo ""

log_info "To compare all environments:"
echo "  python analysis/compare_all_environments.py \\"
echo "    --native results/exp1/stats/summary.json \\"
echo "    --minikube results/exp2/stats/summary.json \\"
echo "    --gcp $OUT_DIR/stats/summary.json"
echo ""

log_success "Done!"

exit 0

