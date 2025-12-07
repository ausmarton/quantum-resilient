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
# Note: Native has 468 scenarios (no scaling), Minikube/GCP have 495 (468 baseline + 27 scaling)
BASELINE_EXPECTED=$(python3 <<EOF
import yaml
from pathlib import Path

with open(Path("$MATRIX")) as f:
    matrix = yaml.safe_load(f)

experiments = matrix.get('experiments', [])
defaults = matrix.get('defaults', {})

total = 0
for exp in experiments:
    payload_sizes = exp.get('payload_sizes', [1024])
    rates = exp.get('rates', [500])
    runs = exp.get('runs', defaults.get('runs', 5))
    total += len(payload_sizes) * len(rates) * runs

print(total)
EOF
)

# Calculate scaling experiments (replicas 2,4,8)
SCALING_EXPECTED=$(python3 <<EOF
import yaml
from pathlib import Path

with open(Path("$MATRIX")) as f:
    matrix = yaml.safe_load(f)

experiments = matrix.get('experiments', [])
defaults = matrix.get('defaults', {})
scaling_config = matrix.get('scaling', {})
replicas = scaling_config.get('replicas', [1, 2, 4, 8])
scaling_replicas = [r for r in replicas if r > 1]  # [2, 4, 8]

# Count scaling experiments (those with scaling_experiment: true)
scaling_count = 0
for exp in experiments:
    if exp.get('scaling_experiment', False):
        payload_sizes = exp.get('payload_sizes', [1024])
        rates = exp.get('rates', [500])
        runs = exp.get('runs', defaults.get('runs', 5))
        scaling_count += len(payload_sizes) * len(rates) * runs

# Scaling experiments run with replicas 2, 4, 8 (3 replica counts)
total_scaling = scaling_count * len(scaling_replicas)
print(total_scaling)
EOF
)

# Expected counts per environment
NATIVE_EXPECTED=$BASELINE_EXPECTED  # 468 (no scaling)
MINIKUBE_EXPECTED=$((BASELINE_EXPECTED + SCALING_EXPECTED))  # 495 (468 + 27)
GCP_EXPECTED=$((BASELINE_EXPECTED + SCALING_EXPECTED))  # 495 (468 + 27)

# Calculate total expected across all environments
TOTAL_EXPECTED=0
for env in "${ENVS[@]}"; do
    if [[ "$env" == "native" ]]; then
        TOTAL_EXPECTED=$((TOTAL_EXPECTED + NATIVE_EXPECTED))
    else
        TOTAL_EXPECTED=$((TOTAL_EXPECTED + MINIKUBE_EXPECTED))
    fi
done
TOTAL_COMPLETED=0
TOTAL_INCOMPLETE=0
TOTAL_MISSING=0

echo ""
echo -e "${MAGENTA}Per-Environment Status:${NC}"
echo ""

for env in "${ENVS[@]}"; do
    ENV_RESULTS_DIR="$RESULTS_BASE/$env"
    
    # Determine expected count for this environment
    if [[ "$env" == "native" ]]; then
        ENV_EXPECTED=$NATIVE_EXPECTED
    else
        ENV_EXPECTED=$MINIKUBE_EXPECTED
    fi
    
    if [[ ! -d "$ENV_RESULTS_DIR" ]]; then
        echo -e "${YELLOW}${env^^}:${NC} No results directory found"
        echo "  Status: Not started"
        echo "  Completed: 0/$ENV_EXPECTED (0%)"
        echo ""
        TOTAL_MISSING=$((TOTAL_MISSING + ENV_EXPECTED))
        continue
    fi
    
    # Generate expected scenario IDs from matrix (matching generate_scenarios.py logic exactly)
    # Also include scaling experiments with _r2, _r4, _r8 suffixes for Minikube/GCP
    read -r COMPLETED INCOMPLETE MISSING TOTAL_FOUND PERCENTAGE EXTRA_COUNT <<< $(python3 <<EOF
import yaml
import hashlib
from pathlib import Path

matrix_file = Path("$MATRIX")
env_results_dir = Path("$ENV_RESULTS_DIR")
env_name = "$env"

# Load matrix
with open(matrix_file) as f:
    matrix = yaml.safe_load(f)

experiments = matrix.get('experiments', [])
defaults = matrix.get('defaults', {})
scaling_config = matrix.get('scaling', {})
replicas = scaling_config.get('replicas', [1, 2, 4, 8])
scaling_replicas = [r for r in replicas if r > 1]  # [2, 4, 8]

# Generate expected scenario IDs (matching generate_scenarios.py exactly)
expected_scenario_ids = set()

def compute_scenario_hash(algorithm, payload, rate, run, pattern="constant", duration=None, is_scaling=False):
    """Match generate_scenarios.py logic exactly"""
    # Always include pattern, duration, and scaling flag in hash
    seed_parts = [algorithm, str(payload), str(rate), str(run), pattern]
    if duration and duration != 30:
        seed_parts.append(str(duration))
    if is_scaling:
        seed_parts.append("scaling")
    seed_str = ":".join(seed_parts)
    return hashlib.sha256(seed_str.encode()).hexdigest()[:8]

def generate_scenario_id(algorithm, payload, rate, run, pattern="constant", duration=None, is_scaling=False):
    """Match generate_scenarios.py logic exactly"""
    hash_suffix = compute_scenario_hash(algorithm, payload, rate, run, pattern, duration, is_scaling)
    
    parts = [algorithm, f"p{payload}", f"r{rate}"]
    
    if pattern and pattern != "constant":
        parts.append(pattern)
    
    if duration and duration != 30:
        if duration == 300:
            parts.append("5m")
        else:
            parts.append(f"{duration}s")
    
    if is_scaling:
        parts.append("scaling")
    
    parts.append(f"run{run}")
    parts.append(hash_suffix)
    
    return "_".join(parts)

for exp in experiments:
    algorithm = exp["algorithm"]
    payload_sizes = exp.get("payload_sizes", [1024])
    rates = exp.get("rates", [500])
    runs = exp.get("runs", defaults.get("runs", 5))
    pattern = exp.get("workload_pattern", "constant")
    duration = exp.get("duration_sec", defaults.get("duration_sec", 30))
    is_scaling = exp.get("scaling_experiment", False)
    
    for payload in payload_sizes:
        for rate in rates:
            for run_index in range(1, runs + 1):
                # Generate baseline scenario ID (replica=1)
                scenario_id = generate_scenario_id(algorithm, payload, rate, run_index, pattern, duration, is_scaling)
                expected_scenario_ids.add(scenario_id)
                
                # For Minikube/GCP, also expect scaling experiments (replicas 2,4,8)
                if env_name != "native" and is_scaling:
                    for replica_count in scaling_replicas:
                        scaling_id = f"{scenario_id}_r{replica_count}"
                        expected_scenario_ids.add(scaling_id)

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
    echo "  Completed: $COMPLETED/$ENV_EXPECTED ($PERCENTAGE%)"
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

echo "  Total Expected: $TOTAL_EXPECTED scenarios"
echo "    - Native: $NATIVE_EXPECTED (baseline only, no scaling)"
echo "    - Minikube: $MINIKUBE_EXPECTED ($BASELINE_EXPECTED baseline + $SCALING_EXPECTED scaling)"
echo "    - GCP: $GCP_EXPECTED ($BASELINE_EXPECTED baseline + $SCALING_EXPECTED scaling)"
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
        
        # Determine expected count for this environment
        if [[ "$env" == "native" ]]; then
            ENV_EXPECTED_CHECK=$NATIVE_EXPECTED
        else
            ENV_EXPECTED_CHECK=$MINIKUBE_EXPECTED
        fi
        
        # Quick check for this env
        ENV_COMPLETED=$(find "$ENV_RESULTS_DIR" -name "summary.json" -o -name "merged.jsonl" 2>/dev/null | wc -l)
        if [[ $ENV_COMPLETED -lt $ENV_EXPECTED_CHECK ]]; then
            echo "  - Continue: ./run_full_scale_data_collection.sh --env $env"
        fi
    done
    
    if [[ $TOTAL_COMPLETED -eq $TOTAL_EXPECTED ]]; then
        echo "  - All data collected! Run analysis:"
        echo "    ./run_all_experiments.sh --skip-generation --skip-native --skip-minikube --skip-gcp"
    fi
fi

echo ""

