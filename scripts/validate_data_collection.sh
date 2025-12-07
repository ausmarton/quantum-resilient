#!/usr/bin/env bash
# =============================================================================
# validate_data_collection.sh - Validate that all required raw data is collected
#
# Checks that all experiments from the matrix have been run and have the
# required raw data files before proceeding to analysis.
#
# Usage:
#   ./scripts/validate_data_collection.sh [OPTIONS]
#
# Options:
#   --matrix PATH        Experiment matrix YAML (default: orchestration/experiment_matrix.yaml)
#   --results-dir DIR    Results directory (default: results/)
#   --envs LIST          Comma-separated environments to check (default: native,minikube,gcp)
#   --strict             Fail if any experiment is missing (default: warn only)
#   --output FILE        Write validation report to file
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MATRIX="$SCRIPT_DIR/orchestration/experiment_matrix.yaml"
RESULTS_DIR="$SCRIPT_DIR/results"
ENVS="native,minikube,gcp"
STRICT=false
OUTPUT_FILE=""

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

Validate that all required raw data is collected before analysis.

OPTIONS:
    --matrix PATH        Experiment matrix YAML (default: orchestration/experiment_matrix.yaml)
    --results-dir DIR    Results directory (default: results/)
    --envs LIST          Comma-separated environments (default: native,minikube,gcp)
    --strict             Exit with error if any experiment is missing
    --output FILE        Write validation report to JSON file
    -h, --help           Show this help message

EXAMPLES:
    # Basic validation
    ./scripts/validate_data_collection.sh

    # Check specific environments
    ./scripts/validate_data_collection.sh --envs native,minikube

    # Strict mode (fail on missing data)
    ./scripts/validate_data_collection.sh --strict

    # Save report to file
    ./scripts/validate_data_collection.sh --output validation-report.json
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --matrix)
            MATRIX="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --envs)
            ENVS="$2"
            shift 2
            ;;
        --strict)
            STRICT=true
            shift
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
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

if [[ ! -f "$MATRIX" ]]; then
    log_error "Matrix file not found: $MATRIX"
    exit 1
fi

log_info "Validating data collection..."
log_info "Matrix: $MATRIX"
log_info "Results: $RESULTS_DIR"
log_info "Environments: $ENVS"
echo ""

# Generate expected scenarios and validate
python3 <<EOF
import json
import sys
import yaml
from pathlib import Path
from collections import defaultdict

script_dir = Path("$SCRIPT_DIR")
matrix_file = Path("$MATRIX")
results_dir = Path("$RESULTS_DIR")
envs = "$ENVS".split(",")
strict = "$STRICT" == "true"
output_file = "$OUTPUT_FILE"

# Load matrix
with open(matrix_file) as f:
    matrix = yaml.safe_load(f)

defaults = matrix.get("defaults", {})
experiments = matrix.get("experiments", [])

# Generate expected scenario IDs
expected_scenarios = defaultdict(lambda: defaultdict(set))  # env -> algorithm -> set of scenario_ids

# Scenario ID generation functions (matching generate_scenarios.py exactly)
def compute_scenario_hash(algorithm, payload, rate, run, pattern="constant", duration=None, is_scaling=False):
    """Match generate_scenarios.py logic exactly - backward compatible"""
    # IMPORTANT: Only include pattern if NOT "constant" for backward compatibility
    seed_parts = [algorithm, str(payload), str(rate), str(run)]
    if pattern and pattern != "constant":
        seed_parts.append(pattern)
    if duration and duration != 30:
        seed_parts.append(str(duration))
    if is_scaling:
        seed_parts.append("scaling")
    seed_str = ":".join(seed_parts)
    import hashlib
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
                # Generate scenario ID using exact same logic as generate_scenarios.py
                scenario_id = generate_scenario_id(algorithm, payload, rate, run_index, pattern, duration, is_scaling)
                
                for env in envs:
                    expected_scenarios[env][algorithm].add(scenario_id)

# Check actual results
validation_results = {
    "matrix_file": str(matrix_file),
    "results_dir": str(results_dir),
    "environments": envs,
    "validation_timestamp": None,
    "environments_status": {},
    "summary": {
        "total_expected": 0,
        "total_found": 0,
        "total_missing": 0,
        "total_incomplete": 0,
        "all_complete": False
    },
    "missing_experiments": [],
    "incomplete_experiments": []
}

total_expected = 0
total_found = 0
total_missing = 0
total_incomplete = 0

for env in envs:
    env_results_dir = results_dir / env
    env_expected = sum(len(scenarios) for scenarios in expected_scenarios[env].values())
    total_expected += env_expected
    
    env_found = 0
    env_missing = 0
    env_incomplete = 0
    missing = []
    incomplete = []
    
    if not env_results_dir.exists():
        print(f"\n{env.upper()}: Results directory not found")
        env_missing = env_expected
        total_missing += env_missing
        for algorithm, scenarios in expected_scenarios[env].items():
            for scenario_id in scenarios:
                missing.append({
                    "scenario_id": scenario_id,
                    "algorithm": algorithm,
                    "reason": "results directory not found"
                })
    else:
        print(f"\n{env.upper()}: Checking {env_expected} expected experiments...")
        
        # Check each expected scenario
        for algorithm, scenarios in expected_scenarios[env].items():
            for scenario_id in scenarios:
                exp_dir = env_results_dir / scenario_id
                
                # Check for required files
                merged_file = exp_dir / "merged" / "merged.jsonl"
                stats_file = exp_dir / "stats" / "summary.json"
                raw_file = exp_dir / "raw" / "run.jsonl"
                
                has_merged = merged_file.exists() and merged_file.stat().st_size > 0
                has_stats = stats_file.exists()
                has_raw = raw_file.exists() and raw_file.stat().st_size > 0
                
                if has_merged or (has_stats and has_raw):
                    env_found += 1
                    total_found += 1
                elif exp_dir.exists() and (has_raw or has_stats):
                    # Partially complete
                    env_incomplete += 1
                    total_incomplete += 1
                    incomplete.append({
                        "scenario_id": scenario_id,
                        "algorithm": algorithm,
                        "has_raw": has_raw,
                        "has_stats": has_stats,
                        "has_merged": has_merged,
                        "path": str(exp_dir)
                    })
                else:
                    env_missing += 1
                    total_missing += 1
                    missing.append({
                        "scenario_id": scenario_id,
                        "algorithm": algorithm,
                        "reason": "experiment not found",
                        "path": str(exp_dir)
                    })
    
    validation_results["environments_status"][env] = {
        "expected": env_expected,
        "found": env_found,
        "missing": env_missing,
        "incomplete": env_incomplete,
        "completion_rate": round((env_found / env_expected * 100) if env_expected > 0 else 0, 1)
    }
    
    validation_results["missing_experiments"].extend(missing)
    validation_results["incomplete_experiments"].extend(incomplete)
    
    # Print summary
    if env_missing == 0 and env_incomplete == 0:
        print(f"  ✓ All {env_found} experiments complete")
    else:
        print(f"  Found: {env_found}/{env_expected}")
        if env_incomplete > 0:
            print(f"  ⚠️  Incomplete: {env_incomplete}")
        if env_missing > 0:
            print(f"  ✗ Missing: {env_missing}")

validation_results["summary"]["total_expected"] = total_expected
validation_results["summary"]["total_found"] = total_found
validation_results["summary"]["total_missing"] = total_missing
validation_results["summary"]["total_incomplete"] = total_incomplete
validation_results["summary"]["all_complete"] = (total_missing == 0 and total_incomplete == 0)

from datetime import datetime, timezone
validation_results["validation_timestamp"] = datetime.now(timezone.utc).isoformat()

# Print overall summary
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)
print(f"Total expected: {total_expected}")
print(f"Total found: {total_found} ({total_found/total_expected*100:.1f}%)" if total_expected > 0 else "Total found: 0")
if total_incomplete > 0:
    print(f"Incomplete: {total_incomplete}")
if total_missing > 0:
    print(f"Missing: {total_missing}")

if validation_results["summary"]["all_complete"]:
    print("\n✓ All required data is present - ready for analysis!")
    exit_code = 0
else:
    print(f"\n⚠️  Data collection incomplete:")
    if total_missing > 0:
        print(f"   - {total_missing} experiments not found (need to run)")
    if total_incomplete > 0:
        print(f"   - {total_incomplete} experiments incomplete (have raw data, missing merged/stats)")
        print(f"\n   For incomplete experiments, you can:")
        print(f"     1. Complete analysis only (faster):")
        print(f"        ./scripts/complete_incomplete_experiments.sh --env <env>")
        print(f"     2. Re-run experiments (will skip if already have data):")
        print(f"        ./run_full_scale_data_collection.sh --env <env>")
    if total_missing > 0:
        print(f"\n   For missing experiments, re-run data collection:")
        print(f"      ./run_full_scale_data_collection.sh --env <env>")
    exit_code = 1 if strict else 0

# Write output file if requested
if output_file:
    with open(output_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    print(f"\nValidation report written: {output_file}")

sys.exit(exit_code)
EOF

VALIDATION_EXIT=$?

if [[ $VALIDATION_EXIT -eq 0 ]]; then
    log_success "Validation passed - all required data is present"
    echo ""
    log_info "You can now proceed to analysis:"
    echo "  ./run_all_experiments.sh --skip-generation --skip-native --skip-minikube --skip-gcp"
else
    if [[ "$STRICT" == "true" ]]; then
        log_error "Validation failed - missing or incomplete data"
        exit 1
    else
        log_warn "Validation found missing or incomplete data (use --strict to fail on errors)"
        exit 0
    fi
fi

