#!/usr/bin/env bash
#
# Quantum-Resilient Analysis Pipeline
#
# Usage:
#   ./run_full_pipeline.sh <experiment-id> <uri>
#
# Example:
#   ./run_full_pipeline.sh exp_2025_01_01_001 gs://qr-results/exp_2025_01_01_001
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

# Check arguments
if [[ $# -lt 2 ]]; then
    echo -e "${RED}Error: Missing arguments${NC}"
    echo "Usage: $0 <experiment-id> <uri>"
    echo ""
    echo "Examples:"
    echo "  $0 exp_2025_01_01_001 gs://qr-results/exp_2025_01_01_001"
    echo "  $0 exp_local file:///path/to/results"
    echo "  $0 exp_s3 s3://bucket/prefix"
    exit 1
fi

EXPERIMENT_ID="$1"
URI="$2"

# Directories
DATA_DIR="$SCRIPT_DIR/data/$EXPERIMENT_ID"
FIGURES_DIR="$SCRIPT_DIR/figures/$EXPERIMENT_ID"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Quantum-Resilient Analysis Pipeline${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "Experiment: ${GREEN}$EXPERIMENT_ID${NC}"
echo -e "URI:        ${GREEN}$URI${NC}"
echo ""

# Step 1: Fetch results
echo -e "${YELLOW}[1/6] Fetching results...${NC}"
python "$SCRIPTS_DIR/fetch_results.py" \
    --experiment-id "$EXPERIMENT_ID" \
    --uri "$URI" \
    --out "$DATA_DIR" \
    --parallel 8

if [[ ! -d "$DATA_DIR/raw" ]]; then
    echo -e "${RED}Error: No data fetched${NC}"
    exit 1
fi

RAW_COUNT=$(find "$DATA_DIR/raw" -name "*.jsonl" | wc -l)
echo -e "${GREEN}  Fetched $RAW_COUNT JSONL files${NC}"
echo ""

# Step 2: Merge JSONL files
echo -e "${YELLOW}[2/6] Merging JSONL files...${NC}"
python "$SCRIPTS_DIR/merge_jsonl.py" \
    --input "$DATA_DIR/raw" \
    --output "$DATA_DIR/merged"

if [[ ! -f "$DATA_DIR/merged/merged.jsonl" ]]; then
    echo -e "${RED}Error: Merge failed${NC}"
    exit 1
fi

MERGED_LINES=$(wc -l < "$DATA_DIR/merged/merged.jsonl")
echo -e "${GREEN}  Merged $MERGED_LINES events${NC}"
echo ""

# Step 3: Compute statistics
echo -e "${YELLOW}[3/6] Computing statistics...${NC}"
python "$SCRIPTS_DIR/compute_statistics.py" \
    --input "$DATA_DIR/merged/merged.jsonl" \
    --output "$DATA_DIR/stats" \
    --experiment-id "$EXPERIMENT_ID"

if [[ ! -f "$DATA_DIR/stats/summary.json" ]]; then
    echo -e "${RED}Error: Statistics computation failed${NC}"
    exit 1
fi

echo -e "${GREEN}  Generated summary and histograms${NC}"
echo ""

# Step 4: Generate plots
echo -e "${YELLOW}[4/6] Generating plots...${NC}"
mkdir -p "$FIGURES_DIR"

python "$SCRIPTS_DIR/plot_latency.py" \
    --input "$DATA_DIR/merged/merged.jsonl" \
    --output "$FIGURES_DIR" \
    --experiment-id "$EXPERIMENT_ID"

python "$SCRIPTS_DIR/plot_throughput.py" \
    --input "$DATA_DIR/merged/merged.jsonl" \
    --output "$FIGURES_DIR" \
    --experiment-id "$EXPERIMENT_ID"

python "$SCRIPTS_DIR/plot_queue_delay.py" \
    --input "$DATA_DIR/merged/merged.jsonl" \
    --output "$FIGURES_DIR" \
    --experiment-id "$EXPERIMENT_ID"

PLOT_COUNT=$(find "$FIGURES_DIR" -name "*.png" | wc -l)
echo -e "${GREEN}  Generated $PLOT_COUNT plots${NC}"
echo ""

# Step 5: Export dataset
echo -e "${YELLOW}[5/6] Exporting dataset...${NC}"
python "$SCRIPTS_DIR/export_dataset.py" \
    --input "$DATA_DIR/merged" \
    --output "$DATA_DIR/exports" \
    --experiment-id "$EXPERIMENT_ID" \
    --formats parquet csv summary

echo -e "${GREEN}  Exported to Parquet and CSV${NC}"
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
echo "  - $DATA_DIR/merged/merged.jsonl"
echo "  - $DATA_DIR/merged/merged.parquet"
echo "  - $DATA_DIR/stats/summary.json"
echo "  - $DATA_DIR/stats/latency_hist.png"
echo "  - $DATA_DIR/stats/queue_hist.png"
echo "  - $DATA_DIR/stats/throughput_curve.png"
echo "  - $DATA_DIR/exports/${EXPERIMENT_ID}.parquet"
echo "  - $DATA_DIR/exports/${EXPERIMENT_ID}.csv"
echo ""

# Print summary stats
if [[ -f "$DATA_DIR/stats/summary.json" ]]; then
    echo "Quick Statistics:"
    python3 -c "
import json
with open('$DATA_DIR/stats/summary.json') as f:
    s = json.load(f)
if 'latency' in s:
    lat = s['latency']
    print(f\"  Latency (μs): mean={lat['mean']:.1f}, p50={lat['p50']:.0f}, p99={lat['p99']:.0f}\")
if 'throughput' in s:
    tput = s['throughput']
    if 'mean_msgs_per_sec' in tput:
        print(f\"  Throughput: {tput['mean_msgs_per_sec']:.0f} msg/s\")
print(f\"  Total events: {s.get('total_events', 'N/A')}\")
"
fi

echo ""
echo -e "${GREEN}Pipeline completed successfully!${NC}"
