#!/usr/bin/env bash
# Download all GCP experiment results from GCS to local results/gcp directory
# Preserves existing data and ensures correct directory structure

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results/gcp"
BUCKET="${BUCKET:-pqc-benchmark}"
FETCH_SCRIPT="$SCRIPT_DIR/fetch_and_analyse_from_gcs.sh"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Check prerequisites
if ! command -v gsutil &> /dev/null; then
    log_error "gsutil not found. Please install Google Cloud SDK."
    exit 1
fi

if [[ ! -f "$FETCH_SCRIPT" ]]; then
    log_error "fetch_and_analyse_from_gcs.sh not found at $FETCH_SCRIPT"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

log_info "Downloading all GCP results from gs://${BUCKET}/experiments/"
log_info "Output directory: $RESULTS_DIR"
log_info ""

# Get all experiments from GCS
log_info "Listing experiments in GCS..."
GCS_EXPERIMENTS=$(gsutil ls "gs://${BUCKET}/experiments/" 2>/dev/null | sed 's|gs://[^/]*/experiments/||' | sed 's|/$||' | grep -v '^$' || true)

if [[ -z "$GCS_EXPERIMENTS" ]]; then
    log_error "No experiments found in GCS bucket: $BUCKET"
    exit 1
fi

TOTAL_EXPERIMENTS=$(echo "$GCS_EXPERIMENTS" | wc -l)
log_info "Found $TOTAL_EXPERIMENTS experiments in GCS"
log_info ""

# Process each experiment
DOWNLOADED=0
SKIPPED=0
FAILED=0
TOTAL_RUNS=0

while IFS= read -r exp_id; do
    [[ -z "$exp_id" ]] && continue
    
    # Extract base ID and run number
    # Format: <base_id>_run<N> or <base_id>_run<N>_r<replicas>
    if [[ "$exp_id" =~ ^(.+?)_run([0-9]+)(_r[0-9]+)?$ ]]; then
        base_id="${BASH_REMATCH[1]}"
        run_num="${BASH_REMATCH[2]}"
    else
        # Single run experiment (no _run suffix)
        base_id="$exp_id"
        run_num="1"
    fi
    
    # Determine local output directory
    output_dir="$RESULTS_DIR/$base_id/run-${run_num}"
    local_file="$output_dir/raw/run.jsonl"
    
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    
    # Check if already downloaded
    if [[ -f "$local_file" ]] && [[ -s "$local_file" ]]; then
        SKIPPED=$((SKIPPED + 1))
        if [[ $((TOTAL_RUNS % 50)) -eq 0 ]]; then
            log_info "Progress: $TOTAL_RUNS processed (downloaded: $DOWNLOADED, skipped: $SKIPPED, failed: $FAILED)"
        fi
        continue
    fi
    
    # Download the experiment
    log_info "Downloading: $exp_id -> $output_dir"
    mkdir -p "$output_dir"
    
    # Download directly from GCS to avoid validation issues
    GCS_PATH="gs://${BUCKET}/experiments/${exp_id}"
    
    # Create subdirectories
    mkdir -p "$output_dir/raw"
    mkdir -p "$output_dir/merged"
    mkdir -p "$output_dir/stats"
    mkdir -p "$output_dir/figures"
    
    # Download files
    download_success=true
    
    # Download raw data
    if gsutil -q cp "$GCS_PATH/raw/run.jsonl" "$output_dir/raw/run.jsonl" 2>/dev/null; then
        : # Success
    else
        log_warn "  Raw data not found in GCS"
        download_success=false
    fi
    
    # Download metadata files (non-critical)
    gsutil -q cp "$GCS_PATH/manifest.json" "$output_dir/manifest.json" 2>/dev/null || true
    gsutil -q cp "$GCS_PATH/provenance.json" "$output_dir/provenance.json" 2>/dev/null || true
    gsutil -q cp "$GCS_PATH/cloud_metadata.json" "$output_dir/cloud_metadata.json" 2>/dev/null || true
    gsutil -q cp "$GCS_PATH/merged.jsonl" "$output_dir/merged/merged.jsonl" 2>/dev/null || true
    gsutil -q cp "$GCS_PATH/summary.json" "$output_dir/stats/summary.json" 2>/dev/null || true
    
    if [[ "$download_success" == "true" ]]; then
        
        # Verify download
        if [[ -f "$local_file" ]] && [[ -s "$local_file" ]]; then
            file_size=$(stat -f%z "$local_file" 2>/dev/null || stat -c%s "$local_file" 2>/dev/null || echo 0)
            line_count=$(wc -l < "$local_file" 2>/dev/null || echo 0)
            DOWNLOADED=$((DOWNLOADED + 1))
            log_success "  Downloaded: $file_size bytes, $line_count events"
        else
            FAILED=$((FAILED + 1))
            log_error "  Download failed: file not found or empty"
        fi
    else
        FAILED=$((FAILED + 1))
        log_error "  Download failed for: $exp_id"
    fi
    
    # Progress update every 10 downloads
    if [[ $DOWNLOADED -gt 0 ]] && [[ $((DOWNLOADED % 10)) -eq 0 ]]; then
        log_info "Progress: $TOTAL_RUNS processed (downloaded: $DOWNLOADED, skipped: $SKIPPED, failed: $FAILED)"
    fi
    
done <<< "$GCS_EXPERIMENTS"

# Final summary
log_info ""
log_info "=" | tr -d '\n'
for i in {1..70}; do echo -n "="; done
echo ""
log_info "DOWNLOAD SUMMARY"
log_info "=" | tr -d '\n'
for i in {1..70}; do echo -n "="; done
echo ""
log_info "  Total runs in GCS: $TOTAL_RUNS"
log_info "  Downloaded: $DOWNLOADED"
log_info "  Skipped (already exists): $SKIPPED"
log_info "  Failed: $FAILED"
log_info ""

if [[ $FAILED -gt 0 ]]; then
    log_warn "Some downloads failed. Check the logs above for details."
    exit 1
else
    log_success "All downloads completed successfully!"
    exit 0
fi

