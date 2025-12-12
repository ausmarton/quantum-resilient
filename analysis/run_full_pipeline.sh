#!/usr/bin/env bash
#
# Quantum-Resilient Analysis Pipeline
#
# Usage:
#   ./run_full_pipeline.sh <experiment-id> <uri-or-path>
#
# Examples:
#   ./run_full_pipeline.sh exp_2025_01_01_001 gs://qr-results/exp_2025_01_01_001
#   ./run_full_pipeline.sh exp_local ./results/exp_local/raw
#   ./run_full_pipeline.sh exp_local file:///path/to/results
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Get project root (parent of analysis directory)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Determine Python command (containerized if available, fallback to host Python)
# Use containerized Python by default to ensure consistent dependencies
PYTHON_CMD="python3"
if [[ -f "$PROJECT_ROOT/scripts/lib/run-python-container.sh" ]] && \
   [[ "${QR_USE_CONTAINER:-true}" != "false" ]]; then
    PYTHON_CMD="$PROJECT_ROOT/scripts/lib/run-python-container.sh"
    echo -e "${BLUE}Using containerized analysis environment${NC}"
fi

# Check arguments
if [[ $# -lt 2 ]]; then
    echo -e "${RED}Error: Missing arguments${NC}"
    echo "Usage: $0 <experiment-id> <uri-or-path>"
    echo ""
    echo "Examples:"
    echo "  $0 exp_2025_01_01_001 gs://qr-results/exp_2025_01_01_001"
    echo "  $0 exp_local ./results/exp_local/raw"
    echo "  $0 exp_s3 s3://bucket/prefix"
    exit 1
fi

EXPERIMENT_ID="$1"
URI="$2"

# Determine if URI is a local path
IS_LOCAL=false
if [[ -d "$URI" ]]; then
    IS_LOCAL=true
    RAW_DIR="$URI"
elif [[ "$URI" == file://* ]]; then
    IS_LOCAL=true
    RAW_DIR="${URI#file://}"
fi

# Directories - for local runs, output next to raw data
if [[ "$IS_LOCAL" == "true" ]]; then
    # If raw dir ends with /raw, use parent
    if [[ "$RAW_DIR" == */raw ]]; then
        BASE_DIR="$(dirname "$RAW_DIR")"
    else
        BASE_DIR="$RAW_DIR"
    fi
    DATA_DIR="$BASE_DIR"
    FIGURES_DIR="$BASE_DIR/figures"
else
    DATA_DIR="$SCRIPT_DIR/data/$EXPERIMENT_ID"
    FIGURES_DIR="$SCRIPT_DIR/figures/$EXPERIMENT_ID"
fi

SCRIPTS_DIR="$SCRIPT_DIR/scripts"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Quantum-Resilient Analysis Pipeline${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "Experiment: ${GREEN}$EXPERIMENT_ID${NC}"
echo -e "URI:        ${GREEN}$URI${NC}"
echo -e "Local:      ${GREEN}$IS_LOCAL${NC}"
echo ""

# Step 1: Fetch results (skip for local)
if [[ "$IS_LOCAL" == "true" ]]; then
    echo -e "${YELLOW}[1/6] Using local data...${NC}"
    # Ensure raw directory exists
    if [[ "$RAW_DIR" == */raw ]]; then
        RAW_PATH="$RAW_DIR"
    elif [[ -d "$RAW_DIR/raw" ]]; then
        RAW_PATH="$RAW_DIR/raw"
    else
        RAW_PATH="$RAW_DIR"
    fi
    
    if [[ ! -d "$RAW_PATH" ]]; then
        echo -e "${RED}Error: Raw data directory not found: $RAW_PATH${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}[1/6] Fetching results...${NC}"
    mkdir -p "$DATA_DIR"
    $PYTHON_CMD "$SCRIPTS_DIR/fetch_results.py" \
        --experiment-id "$EXPERIMENT_ID" \
        --uri "$URI" \
        --out "$DATA_DIR" \
        --parallel 8
    RAW_PATH="$DATA_DIR/raw"
fi

if [[ ! -d "$RAW_PATH" ]]; then
    echo -e "${RED}Error: No data at $RAW_PATH${NC}"
    exit 1
fi

RAW_COUNT=$(find "$RAW_PATH" -name "*.jsonl" 2>/dev/null | wc -l || echo 0)
echo -e "${GREEN}  Found $RAW_COUNT JSONL files in $RAW_PATH${NC}"
echo ""

# Step 2: Merge JSONL files
echo -e "${YELLOW}[2/6] Merging JSONL files...${NC}"
MERGED_DIR="$DATA_DIR/merged"
$PYTHON_CMD "$SCRIPTS_DIR/merge_jsonl.py" \
    --input "$RAW_PATH" \
    --output "$MERGED_DIR"

if [[ ! -f "$MERGED_DIR/merged.jsonl" ]] && [[ ! -f "$MERGED_DIR/merged.parquet" ]]; then
    echo -e "${RED}Error: Merge failed${NC}"
    exit 1
fi

if [[ -f "$MERGED_DIR/merged.jsonl" ]]; then
    MERGED_LINES=$(wc -l < "$MERGED_DIR/merged.jsonl")
    echo -e "${GREEN}  Merged $MERGED_LINES events${NC}"
fi
echo ""

# Step 3: Compute statistics
echo -e "${YELLOW}[3/6] Computing statistics...${NC}"
STATS_DIR="$DATA_DIR/stats"

# Use parquet if available, otherwise jsonl
if [[ -f "$MERGED_DIR/merged.parquet" ]]; then
    INPUT_FILE="$MERGED_DIR/merged.parquet"
else
    INPUT_FILE="$MERGED_DIR/merged.jsonl"
fi

$PYTHON_CMD "$SCRIPTS_DIR/compute_statistics.py" \
    --input "$INPUT_FILE" \
    --output "$STATS_DIR" \
    --experiment-id "$EXPERIMENT_ID" 2>/dev/null || \
$PYTHON_CMD "$SCRIPTS_DIR/compute_stats.py" \
    --input "$INPUT_FILE" \
    --output "$STATS_DIR" \
    --experiment-id "$EXPERIMENT_ID" 2>/dev/null || \
echo -e "${YELLOW}  Warning: Statistics computation script not found or failed${NC}"

if [[ -f "$STATS_DIR/summary.json" ]]; then
    echo -e "${GREEN}  Generated summary${NC}"
else
    echo -e "${YELLOW}  Warning: summary.json not generated${NC}"
fi
echo ""

# Step 4: Generate plots
echo -e "${YELLOW}[4/6] Generating plots...${NC}"
mkdir -p "$FIGURES_DIR"

# Try ECDF plot first
$PYTHON_CMD "$SCRIPTS_DIR/plot_ecdf.py" \
    --input "$INPUT_FILE" \
    --output "$FIGURES_DIR" \
    --experiment-id "$EXPERIMENT_ID" 2>/dev/null || \
$PYTHON_CMD "$SCRIPTS_DIR/plot_latency.py" \
    --input "$INPUT_FILE" \
    --output "$FIGURES_DIR" \
    --experiment-id "$EXPERIMENT_ID" 2>/dev/null || \
echo -e "${YELLOW}  Warning: Latency plot script not found${NC}"

$PYTHON_CMD "$SCRIPTS_DIR/plot_throughput.py" \
    --input "$INPUT_FILE" \
    --output "$FIGURES_DIR" \
    --experiment-id "$EXPERIMENT_ID" 2>/dev/null || \
echo -e "${YELLOW}  Warning: Throughput plot script not found${NC}"

# Optional plots
$PYTHON_CMD "$SCRIPTS_DIR/plot_queue_delay.py" \
    --input "$INPUT_FILE" \
    --output "$FIGURES_DIR" \
    --experiment-id "$EXPERIMENT_ID" 2>/dev/null || true

PLOT_COUNT=$(find "$FIGURES_DIR" -name "*.png" 2>/dev/null | wc -l || echo 0)
echo -e "${GREEN}  Generated $PLOT_COUNT plots${NC}"
echo ""

# Step 5: Export dataset (optional)
echo -e "${YELLOW}[5/6] Exporting dataset...${NC}"
EXPORTS_DIR="$DATA_DIR/exports"
if [[ -f "$SCRIPTS_DIR/export_dataset.py" ]]; then
    $PYTHON_CMD "$SCRIPTS_DIR/export_dataset.py" \
        --input "$MERGED_DIR" \
        --output "$EXPORTS_DIR" \
        --experiment-id "$EXPERIMENT_ID" \
        --formats parquet csv summary 2>/dev/null || \
    echo -e "${YELLOW}  Warning: Export failed${NC}"
    echo -e "${GREEN}  Exported datasets${NC}"
else
    echo -e "${YELLOW}  Skipping (export_dataset.py not found)${NC}"
fi
echo ""

# Step 6: Generate summary
echo -e "${YELLOW}[6/6] Pipeline complete!${NC}"
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Results Summary${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "Data directory:    $DATA_DIR"
echo "Figures directory: $FIGURES_DIR"
echo ""
echo "Generated files:"
[[ -f "$MERGED_DIR/merged.jsonl" ]] && echo "  - $MERGED_DIR/merged.jsonl"
[[ -f "$MERGED_DIR/merged.parquet" ]] && echo "  - $MERGED_DIR/merged.parquet"
[[ -f "$STATS_DIR/summary.json" ]] && echo "  - $STATS_DIR/summary.json"
[[ -f "$FIGURES_DIR/latency_cdf.png" ]] && echo "  - $FIGURES_DIR/latency_cdf.png"
[[ -f "$FIGURES_DIR/throughput.png" ]] && echo "  - $FIGURES_DIR/throughput.png"
echo ""

# Print summary stats
if [[ -f "$STATS_DIR/summary.json" ]]; then
    echo "Quick Statistics:"
    python3 -c "
import json
try:
    with open('$STATS_DIR/summary.json') as f:
        s = json.load(f)
    # Use latency_ns for nanosecond precision, fallback to latency (microseconds) for old data
    if 'latency_ns' in s:
        lat_ns = s['latency_ns']
        print(f\"  Latency (ns): mean={lat_ns.get('mean', 0):.0f}, p50={lat_ns.get('p50', 0):.0f}, p99={lat_ns.get('p99', 0):.0f}\")
    elif 'latency' in s:
        lat = s['latency']
        # Convert microseconds to nanoseconds
        mean_ns = lat.get('mean', 0) * 1000
        p50_ns = lat.get('p50', 0) * 1000
        p99_ns = lat.get('p99', 0) * 1000
        print(f\"  Latency (ns): mean={mean_ns:.0f}, p50={p50_ns:.0f}, p99={p99_ns:.0f}\")
    if 'throughput' in s:
        tput = s['throughput']
        if 'mean_msgs_per_sec' in tput:
            print(f\"  Throughput: {tput['mean_msgs_per_sec']:.0f} msg/s\")
    print(f\"  Total events: {s.get('total_events', 'N/A')}\")
except Exception as e:
    print(f'  Could not read summary: {e}')
" 2>/dev/null || echo "  (Could not read summary)"
fi

echo ""
echo -e "${GREEN}Pipeline completed successfully!${NC}"
