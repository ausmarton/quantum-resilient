#!/usr/bin/env bash
# =============================================================================
# check_progress.sh - Check progress of data collection across all environments
#
# Shows current progress for:
# - Individual environments (native, minikube, gcp)
# - Overall progress across all environments
# - Estimated time remaining
# - What's completed, in progress, and remaining
#
# Usage:
#   ./scripts/check_progress.sh [--env ENV] [--matrix PATH]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATRIX="$SCRIPT_DIR/orchestration/experiment_matrix.yaml"
ENV=""
RESULTS_BASE="$SCRIPT_DIR/results"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Check progress of data collection across environments.

OPTIONS:
    --env ENV              Check specific environment: native, minikube, or gcp
    --matrix PATH          Experiment matrix YAML (default: orchestration/experiment_matrix.yaml)
    -h, --help             Show this help message

EXAMPLES:
    # Check all environments
    ./scripts/check_progress.sh

    # Check specific environment
    ./scripts/check_progress.sh --env native
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV="$2"
            shift 2
            ;;
        --matrix)
            MATRIX="$2"
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
echo -e "${CYAN}  Data Collection Progress Report${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Calculate expected scenarios from matrix
python3 <<EOF
import yaml
import sys
from pathlib import Path

matrix_file = Path("$MATRIX")
if not matrix_file.exists():
    print("Error: Matrix file not found: $MATRIX")
    sys.exit(1)

with open(matrix_file) as f:
    matrix = yaml.safe_load(f)

experiments = matrix.get('experiments', [])
total_scenarios = 0

for exp in experiments:
    algorithms = exp.get('algorithm', [])
    if isinstance(algorithms, str):
        algorithms = [algorithms]
    
    payload_sizes = exp.get('payload_sizes', [])
    rates = exp.get('rates', [])
    runs = exp.get('runs', 1)
    
    total_scenarios += len(algorithms) * len(payload_sizes) * len(rates) * runs

print(f"Expected scenarios per environment: {total_scenarios}")
sys.exit(0)
EOF

EXPECTED_PER_ENV=$?
if [[ $EXPECTED_PER_ENV -ne 0 ]]; then
    EXPECTED_PER_ENV=459  # Default fallback (includes baseline 300 + quick wins 159)
else
    EXPECTED_PER_ENV=$(python3 <<EOF
import yaml
from pathlib import Path

with open(Path("$MATRIX")) as f:
    matrix = yaml.safe_load(f)

experiments = matrix.get('experiments', [])
total = 0

for exp in experiments:
    algorithms = exp.get('algorithm', [])
    if isinstance(algorithms, str):
        algorithms = [algorithms]
    
    payload_sizes = exp.get('payload_sizes', [])
    rates = exp.get('rates', [])
    runs = exp.get('runs', 1)
    
    total += len(algorithms) * len(payload_sizes) * len(rates) * runs

print(total)
EOF
)
fi

TOTAL_EXPECTED=$((EXPECTED_PER_ENV * ${#ENVS[@]}))
TOTAL_COMPLETED=0
TOTAL_INCOMPLETE=0
TOTAL_MISSING=0

echo ""
echo -e "${MAGENTA}Per-Environment Status:${NC}"
echo ""

for env in "${ENVS[@]}"; do
    ENV_RESULTS_DIR="$RESULTS_BASE/$env"
    
    if [[ ! -d "$ENV_RESULTS_DIR" ]]; then
        echo -e "${YELLOW}${env^^}:${NC} No results directory found"
        echo "  Status: Not started"
        echo "  Completed: 0/$EXPECTED_PER_ENV (0%)"
        echo ""
        TOTAL_MISSING=$((TOTAL_MISSING + EXPECTED_PER_ENV))
        continue
    fi
    
    # Generate expected scenario IDs from matrix (same logic as validate_data_collection.sh)
    read -r COMPLETED INCOMPLETE MISSING TOTAL_FOUND PERCENTAGE EXTRA_COUNT <<< $(python3 <<EOF
import yaml
import hashlib
from pathlib import Path

matrix_file = Path("$MATRIX")
env_results_dir = Path("$ENV_RESULTS_DIR")

# Load matrix
with open(matrix_file) as f:
    matrix = yaml.safe_load(f)

experiments = matrix.get('experiments', [])
defaults = matrix.get('defaults', {})

# Generate expected scenario IDs
expected_scenario_ids = set()

for exp in experiments:
    algorithm = exp["algorithm"]
    payload_sizes = exp.get("payload_sizes", [1024])
    rates = exp.get("rates", [500])
    runs = exp.get("runs", defaults.get("runs", 5))
    
    for payload in payload_sizes:
        for rate in rates:
            for run_index in range(1, runs + 1):
                # Generate scenario ID (matching generate_scenarios.py logic)
                seed_str = f"{algorithm}:{payload}:{rate}:{run_index}"
                hash_suffix = hashlib.sha256(seed_str.encode()).hexdigest()[:8]
                scenario_id = f"{algorithm}_p{payload}_r{rate}_run{run_index}_{hash_suffix}"
                expected_scenario_ids.add(scenario_id)

expected = len(expected_scenario_ids)

# Check actual results - only count expected scenarios
completed = 0
incomplete = 0
extra_dirs = []

if env_results_dir.exists():
    existing_dirs = [d for d in env_results_dir.iterdir() if d.is_dir()]
    
    for exp_dir in existing_dirs:
        scenario_id = exp_dir.name
        
        # Check if this is an expected scenario
        if scenario_id in expected_scenario_ids:
            merged_file = exp_dir / "merged" / "merged.jsonl"
            stats_file = exp_dir / "stats" / "summary.json"
            
            has_merged = merged_file.exists() and merged_file.stat().st_size > 0
            has_stats = stats_file.exists() and stats_file.stat().st_size > 0
            
            if has_merged or has_stats:
                completed += 1
            else:
                incomplete += 1
        else:
            # This is an extra/unexpected directory
            extra_dirs.append(scenario_id)

missing = max(0, expected - completed - incomplete)
total_found = completed + incomplete
pct = int((completed / expected) * 100) if expected > 0 else 0

print(f"{completed} {incomplete} {missing} {total_found} {pct} {len(extra_dirs)}")
EOF
)
    
    TOTAL_COMPLETED=$((TOTAL_COMPLETED + COMPLETED))
    TOTAL_INCOMPLETE=$((TOTAL_INCOMPLETE + INCOMPLETE))
    TOTAL_MISSING=$((TOTAL_MISSING + MISSING))
    
    # Status indicator
    if [[ $PERCENTAGE -eq 100 ]]; then
        STATUS_COLOR="$GREEN"
        STATUS="✓ Complete"
    elif [[ $PERCENTAGE -ge 50 ]]; then
        STATUS_COLOR="$YELLOW"
        STATUS="⏳ In Progress"
    elif [[ $PERCENTAGE -gt 0 ]]; then
        STATUS_COLOR="$YELLOW"
        STATUS="⏳ Started"
    else
        STATUS_COLOR="$RED"
        STATUS="✗ Not Started"
    fi
    
    echo -e "${STATUS_COLOR}${env^^}:${NC} ${STATUS}"
    echo "  Completed: $COMPLETED/$EXPECTED_PER_ENV ($PERCENTAGE%)"
    if [[ $INCOMPLETE -gt 0 ]]; then
        echo "  Incomplete: $INCOMPLETE"
    fi
    if [[ $MISSING -gt 0 ]]; then
        echo "  Remaining: $MISSING"
    fi
    if [[ $EXTRA_COUNT -gt 0 ]]; then
        echo -e "  ${YELLOW}Extra directories (not in matrix): $EXTRA_COUNT${NC}"
        echo "    (These may be from smoke tests or previous runs)"
    fi
    echo ""
done

# Overall summary
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}Overall Progress:${NC}"
echo ""

if [[ $TOTAL_EXPECTED -gt 0 ]]; then
    OVERALL_PCT=$((TOTAL_COMPLETED * 100 / TOTAL_EXPECTED))
else
    OVERALL_PCT=0
fi

echo "  Total Expected: $TOTAL_EXPECTED scenarios (${#ENVS[@]} environments × $EXPECTED_PER_ENV)"
echo "  Total Completed: $TOTAL_COMPLETED ($OVERALL_PCT%)"
if [[ $TOTAL_INCOMPLETE -gt 0 ]]; then
    echo "  Incomplete: $TOTAL_INCOMPLETE"
fi
if [[ $TOTAL_MISSING -gt 0 ]]; then
    echo "  Remaining: $TOTAL_MISSING"
fi

# Progress bar (using ASCII characters for better terminal compatibility)
BAR_WIDTH=50
FILLED=$((OVERALL_PCT * BAR_WIDTH / 100))
EMPTY=$((BAR_WIDTH - FILLED))

# Use # for filled and - for empty (more compatible than Unicode blocks)
printf "  Progress: ["
if [[ $FILLED -gt 0 ]]; then
    printf "%${FILLED}s" | tr ' ' '#'
fi
if [[ $EMPTY -gt 0 ]]; then
    printf "%${EMPTY}s" | tr ' ' '-'
fi
printf "] %d%%\n" "$OVERALL_PCT"

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"

# Next steps
if [[ $TOTAL_COMPLETED -lt $TOTAL_EXPECTED ]]; then
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    
    for env in "${ENVS[@]}"; do
        ENV_RESULTS_DIR="$RESULTS_BASE/$env"
        if [[ ! -d "$ENV_RESULTS_DIR" ]]; then
            echo "  - Run: ./run_full_scale_data_collection.sh --env $env"
            continue
        fi
        
        # Quick check for this env
        ENV_COMPLETED=$(find "$ENV_RESULTS_DIR" -name "summary.json" -o -name "merged.jsonl" 2>/dev/null | wc -l)
        if [[ $ENV_COMPLETED -lt $EXPECTED_PER_ENV ]]; then
            echo "  - Continue: ./run_full_scale_data_collection.sh --env $env"
        fi
    done
    
    if [[ $TOTAL_COMPLETED -eq $TOTAL_EXPECTED ]]; then
        echo "  - All data collected! Run analysis:"
        echo "    ./run_all_experiments.sh --skip-generation --skip-native --skip-minikube --skip-gcp"
    fi
fi

echo ""

