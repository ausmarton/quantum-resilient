#!/usr/bin/env bash
# =============================================================================
# generate_missing_summaries.sh - Generate summary.json for experiments missing them
#
# This script identifies experiments with raw data but no summary.json files
# and generates the missing summaries using compute_statistics.py
#
# Usage:
#   ./scripts/generate_missing_summaries.sh [--env <env>]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

ENV_FILTER=""

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Generate missing summary.json files for experiments.

OPTIONS:
    --env ENV        Process only specific environment (native, minikube, gcp)
    -h, --help       Show this help message
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV_FILTER="$2"
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

echo "Finding experiments missing summary.json files..."

# Find experiments with raw data but no summary.json
python3 <<'PYTHON_SCRIPT'
import json
from pathlib import Path

with open('final-results/index.json') as f:
    index = json.load(f)

missing = []
for entry in index.get('experiments', []):
    if entry.get('status') not in ['success', 'cached']:
        continue
    
    output_dir = Path(entry['output_dir'])
    
    # Check if summary.json exists
    summary_paths = [
        output_dir / 'stats' / 'summary.json',
        output_dir / 'merged' / 'stats' / 'summary.json',
        output_dir / 'summary.json'
    ]
    
    has_summary = any(p.exists() for p in summary_paths)
    
    # Check if raw data exists
    raw_file = output_dir / 'raw' / 'run.jsonl'
    has_raw = raw_file.exists() and raw_file.stat().st_size > 0
    
    if has_raw and not has_summary:
        missing.append({
            'scenario_id': entry.get('scenario_id'),
            'environment': entry.get('environment'),
            'output_dir': str(output_dir),
            'raw_file': str(raw_file)
        })

print(json.dumps(missing, indent=2))
PYTHON_SCRIPT

# Generate summaries
echo ""
echo "Generating missing summary files..."

python3 <<'PYTHON_SCRIPT'
import json
import subprocess
from pathlib import Path

with open('final-results/index.json') as f:
    index = json.load(f)

missing = []
for entry in index.get('experiments', []):
    if entry.get('status') not in ['success', 'cached']:
        continue
    
    output_dir = Path(entry['output_dir'])
    
    # Check if summary.json exists
    summary_paths = [
        output_dir / 'stats' / 'summary.json',
        output_dir / 'merged' / 'stats' / 'summary.json',
        output_dir / 'summary.json'
    ]
    
    has_summary = any(p.exists() for p in summary_paths)
    
    # Check if raw data exists
    raw_file = output_dir / 'raw' / 'run.jsonl'
    has_raw = raw_file.exists() and raw_file.stat().st_size > 0
    
    if has_raw and not has_summary:
        # Check if merged data exists
        merged_file = output_dir / 'merged' / 'merged.jsonl'
        if not merged_file.exists():
            # Try to merge raw data first
            print(f"Merging raw data for {entry.get('scenario_id')}...")
            subprocess.run([
                'python3', 'analysis/scripts/merge_jsonl.py',
                '--input', str(output_dir / 'raw'),
                '--output', str(output_dir / 'merged')
            ], check=False)
        
        # Generate summary
        if merged_file.exists():
            print(f"Generating summary for {entry.get('scenario_id')}...")
            stats_dir = output_dir / 'merged' / 'stats'
            stats_dir.mkdir(parents=True, exist_ok=True)
            
            subprocess.run([
                'python3', 'analysis/scripts/compute_statistics.py',
                '--input', str(merged_file),
                '--output', str(stats_dir),
                '--experiment-id', entry.get('scenario_id', '')
            ], check=False)
        elif (output_dir / 'raw' / 'run.jsonl').exists():
            # Use raw data directly
            print(f"Generating summary from raw data for {entry.get('scenario_id')}...")
            stats_dir = output_dir / 'stats'
            stats_dir.mkdir(parents=True, exist_ok=True)
            
            subprocess.run([
                'python3', 'analysis/scripts/compute_statistics.py',
                '--input', str(output_dir / 'raw' / 'run.jsonl'),
                '--output', str(stats_dir),
                '--experiment-id', entry.get('scenario_id', '')
            ], check=False)

print("Done!")
PYTHON_SCRIPT

echo ""
echo "Summary generation complete!"
echo "Re-run aggregation: python3 analysis/aggregate_results.py --index final-results/index.json --output final-results"

