#!/usr/bin/env bash
# =============================================================================
# complete_incomplete_experiments.sh - Complete analysis for incomplete experiments
#
# For experiments that have raw data but are missing merged/stats files,
# this script runs the analysis step (merge + compute statistics) without
# re-running the benchmark.
#
# Usage:
#   ./scripts/complete_incomplete_experiments.sh [OPTIONS]
#
# Options:
#   --env ENV              Environment to process (native, minikube, gcp)
#   --results-dir DIR      Results directory (default: results/)
#   --dry-run              Show what would be done without doing it
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RESULTS_DIR="$SCRIPT_DIR/results"
ENV=""
DRY_RUN=false

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

Complete analysis for experiments that have raw data but missing merged/stats.

OPTIONS:
    --env ENV              Environment to process: native, minikube, or gcp
    --results-dir DIR      Results directory (default: results/)
    --dry-run              Show what would be done without doing it
    -h, --help             Show this help message

EXAMPLES:
    # Complete all incomplete native experiments
    ./scripts/complete_incomplete_experiments.sh --env native

    # Dry run to see what would be processed
    ./scripts/complete_incomplete_experiments.sh --env native --dry-run
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
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

if [[ -z "$ENV" ]]; then
    log_error "Must specify --env"
    usage
fi

ENV_RESULTS_DIR="$RESULTS_DIR/$ENV"

if [[ ! -d "$ENV_RESULTS_DIR" ]]; then
    log_error "Results directory not found: $ENV_RESULTS_DIR"
    exit 1
fi

log_info "Completing analysis for incomplete experiments in: $ENV_RESULTS_DIR"
echo ""

# Find incomplete experiments
python3 <<EOF
import json
import subprocess
import sys
from pathlib import Path

script_dir = Path("$SCRIPT_DIR")
env_results_dir = Path("$ENV_RESULTS_DIR")
dry_run = "$DRY_RUN" == "true"

incomplete = []

for exp_dir in sorted(env_results_dir.iterdir()):
    if not exp_dir.is_dir():
        continue
    
    # Check for required files
    merged_file = exp_dir / "merged" / "merged.jsonl"
    stats_file = exp_dir / "stats" / "summary.json"
    raw_file = exp_dir / "raw" / "run.jsonl"
    
    has_merged = merged_file.exists() and merged_file.stat().st_size > 0
    has_stats = stats_file.exists()
    has_raw = raw_file.exists() and raw_file.stat().st_size > 0
    
    # Incomplete if has raw but missing merged/stats
    if has_raw and not (has_merged or has_stats):
        incomplete.append({
            "scenario_id": exp_dir.name,
            "path": str(exp_dir),
            "has_raw": True,
            "raw_size": raw_file.stat().st_size
        })

print(f"Found {len(incomplete)} incomplete experiments")
print("")

if dry_run:
    print("DRY RUN - Would process:")
    for exp in incomplete[:10]:  # Show first 10
        print(f"  - {exp['scenario_id']}")
    if len(incomplete) > 10:
        print(f"  ... and {len(incomplete) - 10} more")
    sys.exit(0)

# Determine Python command (use container wrapper if available, else venv, else system)
container_wrapper = script_dir / "scripts" / "lib" / "run-python-container.sh"
venv_python = script_dir / "analysis" / "venv" / "bin" / "python3"

if container_wrapper.exists():
    # Use container wrapper for consistent environment
    python_cmd = str(container_wrapper.absolute())
    use_container = True
elif venv_python.exists():
    # Use absolute path to venv python
    python_cmd = str(venv_python.absolute())
    use_container = False
else:
    python_cmd = sys.executable
    use_container = False

print(f"Using Python: {python_cmd}")
print("")

# Process each incomplete experiment
completed = 0
failed = 0

for i, exp in enumerate(incomplete, 1):
    exp_dir = Path(exp["path"])
    scenario_id = exp["scenario_id"]
    
    print(f"[{i}/{len(incomplete)}] Processing: {scenario_id}")
    
    # Step 1: Merge JSONL
    raw_dir = exp_dir / "raw"
    merged_dir = exp_dir / "merged"
    merged_dir.mkdir(exist_ok=True)
    
    try:
        # Build command: container wrapper or direct python
        if use_container:
            cmd = [
                python_cmd,
                str(script_dir / "analysis" / "scripts" / "merge_jsonl.py"),
                "--input", str(raw_dir),
                "--output", str(merged_dir),
            ]
        else:
            cmd = [
                python_cmd,
                str(script_dir / "analysis" / "scripts" / "merge_jsonl.py"),
                "--input", str(raw_dir),
                "--output", str(merged_dir),
            ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"  ⚠️  Merge failed: {result.stderr[:100]}")
            failed += 1
            continue
    except Exception as e:
        print(f"  ✗ Merge error: {e}")
        failed += 1
        continue
    
    # Step 2: Compute statistics
    merged_file = merged_dir / "merged.jsonl"
    if not merged_file.exists():
        merged_file = merged_dir / "merged.parquet"
    
    if not merged_file.exists():
        print(f"  ✗ Merged file not created")
        failed += 1
        continue
    
    stats_dir = exp_dir / "stats"
    stats_dir.mkdir(exist_ok=True)
    
    try:
        # Build command: container wrapper or direct python
        if use_container:
            cmd = [
                python_cmd,
                str(script_dir / "analysis" / "scripts" / "compute_statistics.py"),
                "--input", str(merged_file),
                "--output", str(stats_dir),
                "--experiment-id", scenario_id,
            ]
        else:
            cmd = [
                python_cmd,
                str(script_dir / "analysis" / "scripts" / "compute_statistics.py"),
                "--input", str(merged_file),
                "--output", str(stats_dir),
                "--experiment-id", scenario_id,
            ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            # Try alternative script
            if use_container:
                alt_cmd = [
                    python_cmd,
                    str(script_dir / "analysis" / "scripts" / "compute_stats.py"),
                    "--input", str(merged_file),
                    "--output", str(stats_dir),
                    "--experiment-id", scenario_id,
                ]
            else:
                alt_cmd = [
                    python_cmd,
                    str(script_dir / "analysis" / "scripts" / "compute_stats.py"),
                    "--input", str(merged_file),
                    "--output", str(stats_dir),
                    "--experiment-id", scenario_id,
                ]
            
            result = subprocess.run(
                alt_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
        
        if result.returncode != 0:
            print(f"  ⚠️  Statistics failed: {result.stderr[:100]}")
            failed += 1
            continue
        
        print(f"  ✓ Completed")
        completed += 1
    except Exception as e:
        print(f"  ✗ Statistics error: {e}")
        failed += 1
        continue

print("")
print("=" * 70)
print("COMPLETION SUMMARY")
print("=" * 70)
print(f"Total incomplete: {len(incomplete)}")
print(f"Completed: {completed}")
print(f"Failed: {failed}")
print("")

if completed > 0:
    print(f"✓ Successfully completed analysis for {completed} experiments")
if failed > 0:
    print(f"⚠️  Failed to complete {failed} experiments (check logs above)")

sys.exit(0 if failed == 0 else 1)
EOF

EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    log_success "Analysis completion finished"
else
    log_warn "Some experiments failed to complete (see output above)"
fi

echo ""
log_info "Re-run validation to check status:"
echo "  ./scripts/validate_data_collection.sh --envs $ENV"
echo ""

exit $EXIT_CODE

