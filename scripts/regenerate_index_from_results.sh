#!/usr/bin/env bash
# =============================================================================
# regenerate_index_from_results.sh - Regenerate index.json from existing results
#
# When data collection is done separately for each environment, this script
# regenerates a combined index.json from all existing results directories.
#
# Usage:
#   ./scripts/regenerate_index_from_results.sh \
#     --matrix orchestration/experiment_matrix.yaml \
#     --output final-results/
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MATRIX="$SCRIPT_DIR/orchestration/experiment_matrix.yaml"
OUTPUT_DIR="$SCRIPT_DIR/final-results"
ENVS="native,minikube,gcp"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Regenerate index.json from existing results directories.

OPTIONS:
    --matrix PATH    Experiment matrix YAML (default: orchestration/experiment_matrix.yaml)
    --output DIR     Output directory for index.json (default: final-results/)
    --envs LIST      Comma-separated environments (default: native,minikube,gcp)
    -h, --help       Show this help message
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --matrix)
            MATRIX="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --envs)
            ENVS="$2"
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

mkdir -p "$OUTPUT_DIR"

python3 <<EOF
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

script_dir = Path("$SCRIPT_DIR")
output_dir = Path("$OUTPUT_DIR")
matrix_file = "$MATRIX"
envs = "$ENVS".split(",")

index = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "matrix_file": matrix_file,
    "environments": envs,
    "total_scenarios": 0,
    "completed_scenarios": 0,
    "failed_scenarios": 0,
    "experiments": []
}

# Count total scenarios from matrix
try:
    import yaml
    with open(matrix_file) as f:
        matrix = yaml.safe_load(f)
    
    experiments = matrix.get("experiments", [])
    payload_sizes = set()
    rates = set()
    
    for exp in experiments:
        for payload in exp.get("payload_sizes", []):
            payload_sizes.add(payload)
        for rate in exp.get("rates", []):
            rates.add(rate)
    
    total_scenarios = len(experiments) * len(payload_sizes) * len(rates)
    index["total_scenarios"] = total_scenarios
except Exception as e:
    print(f"Warning: Could not count scenarios from matrix: {e}", file=sys.stderr)

# Scan results directories
for env in envs:
    env_results_dir = script_dir / "results" / env
    if not env_results_dir.exists():
        continue
    
    for exp_dir in sorted(env_results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        scenario_id = exp_dir.name
        
        # Check for data files
        merged_file = exp_dir / "merged" / "merged.jsonl"
        stats_file = exp_dir / "stats" / "summary.json"
        raw_file = exp_dir / "raw" / "run.jsonl"
        
        has_data = merged_file.exists() or stats_file.exists() or raw_file.exists()
        
        if not has_data:
            continue
        
        # Try to extract algorithm, payload, rate from scenario ID
        # Format: <algorithm>_p<payload>_r<rate>_run<N>_<hash>
        # Or: <algorithm>-smoketest-p<payload>-r<rate>
        algorithm = "unknown"
        payload = 0
        rate = 0
        replicas = 1
        
        if "-smoketest-" in scenario_id:
            parts = scenario_id.split("-smoketest-")
            algorithm = parts[0]
            if len(parts) > 1:
                rest = parts[1]
                if rest.startswith("p") and "r" in rest:
                    p_part, r_part = rest.split("-r")
                    payload = int(p_part[1:]) if p_part[1:].isdigit() else 0
                    rate = int(r_part) if r_part.isdigit() else 0
        else:
            # Try to parse full format
            if "_p" in scenario_id and "_r" in scenario_id:
                parts = scenario_id.split("_")
                algorithm = parts[0]
                for part in parts:
                    if part.startswith("p") and part[1:].isdigit():
                        payload = int(part[1:])
                    elif part.startswith("r") and not part.startswith("run") and part[1:].isdigit():
                        rate = int(part[1:])
                    elif part.startswith("r") and len(part) > 4 and part[4:].isdigit():
                        replicas = int(part[4:])
        
        # Determine status
        if stats_file.exists() or merged_file.exists():
            status = "success"
            index["completed_scenarios"] += 1
        else:
            status = "failed"
            index["failed_scenarios"] += 1
        
        experiment_entry = {
            "scenario_id": scenario_id,
            "environment": env,
            "algorithm": algorithm,
            "payload_size": payload,
            "rate": rate,
            "replicas": replicas,
            "output_dir": str(exp_dir),
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        index["experiments"].append(experiment_entry)

# Write index
index_file = output_dir / "index.json"
with open(index_file, 'w') as f:
    json.dump(index, f, indent=2)

print(f"Index regenerated: {index_file}")
print(f"Total experiments: {len(index['experiments'])}")
print(f"Completed: {index['completed_scenarios']}")
print(f"Failed: {index['failed_scenarios']}")
EOF

echo ""
echo "Index regenerated successfully!"
echo "You can now run analysis:"
echo "  ./run_all_experiments.sh --skip-generation --skip-native --skip-minikube --skip-gcp"

