#!/usr/bin/env bash
# =============================================================================
# remove_unusable_data.sh - Identify and remove unusable experiment data
#
# Identifies experiments with errors or insufficient runs and removes them
# so they can be re-run. Supports dry-run mode to preview what would be removed.
#
# Usage:
#   ./scripts/remove_unusable_data.sh [OPTIONS]
#
# Options:
#   --env ENV              Check specific environment (native, minikube, gcp)
#   --results-dir DIR      Results directory (default: results/)
#   --dry-run              Show what would be removed without actually removing
#   --error-only           Only remove experiments with errors (keep incomplete runs)
#   --incomplete-only      Only remove experiments with insufficient runs (keep errors)
#   --confirm              Require confirmation before removing (default: true)
#   --force                Remove without confirmation (use with caution)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
ENV_FILTER=""
DRY_RUN=false
ERROR_ONLY=false
INCOMPLETE_ONLY=false
CONFIRM=true
FORCE=false

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

Identify and remove unusable experiment data.

OPTIONS:
    --env ENV              Check specific environment (native, minikube, gcp)
    --results-dir DIR      Results directory (default: results/)
    --dry-run              Show what would be removed without actually removing
    --error-only           Only remove experiments with errors (keep incomplete runs)
    --incomplete-only      Only remove experiments with insufficient runs (keep errors)
    --confirm              Require confirmation before removing (default: true)
    --force                Remove without confirmation (use with caution)
    -h, --help             Show this help message

EXAMPLES:
    # Dry run to see what would be removed
    ./scripts/remove_unusable_data.sh --env gcp --dry-run

    # Remove only experiments with errors
    ./scripts/remove_unusable_data.sh --env gcp --error-only

    # Remove all unusable data (errors + incomplete)
    ./scripts/remove_unusable_data.sh --env gcp --force
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV_FILTER="$2"
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
        --error-only)
            ERROR_ONLY=true
            shift
            ;;
        --incomplete-only)
            INCOMPLETE_ONLY=true
            shift
            ;;
        --confirm)
            CONFIRM=true
            shift
            ;;
        --force)
            FORCE=true
            CONFIRM=false
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

# Python script to identify unusable experiments
PYTHON_SCRIPT=$(cat <<'PYTHON_EOF'
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

def find_unusable_experiments(results_dir: Path, env_filter: str = "",
                              error_only: bool = False, incomplete_only: bool = False) -> list:
    """Find experiments that should be removed."""
    
    unusable = []
    
    # Find all raw JSONL files
    if env_filter:
        search_path = results_dir / env_filter
    else:
        search_path = results_dir
    
    # Support both directory structures
    raw_files = []
    seen_files = set()
    for pattern in ["*/raw/run.jsonl", "*/run-*/raw/run.jsonl"]:
        for f in search_path.rglob(pattern):
            if f not in seen_files:
                raw_files.append(f)
                seen_files.add(f)
    
    # Group by experiment
    experiments = defaultdict(lambda: {
        "runs": [],
        "has_errors": False,
        "error_types": Counter(),
        "total_events": 0
    })
    
    seen_runs = defaultdict(set)
    
    for jsonl_file in raw_files:
        # Extract experiment name
        if "run-" in str(jsonl_file):
            exp_name = jsonl_file.parent.parent.parent.name
            run_name = jsonl_file.parent.parent.name
            exp_dir = jsonl_file.parent.parent.parent
            run_dir = jsonl_file.parent.parent
        else:
            exp_name = jsonl_file.parent.parent.name
            run_name = "run-1"
            exp_dir = jsonl_file.parent.parent
            run_dir = jsonl_file.parent.parent
        
        # Avoid duplicates
        run_key = f"{exp_name}:{run_name}"
        if run_key not in seen_runs[exp_name]:
            seen_runs[exp_name].add(run_key)
            experiments[exp_name]["runs"].append({
                "name": run_name,
                "file": str(jsonl_file),
                "dir": str(run_dir),
                "exp_dir": str(exp_dir)
            })
        
        # Check for errors
        try:
            with open(jsonl_file) as f:
                first_line = f.readline()
                if first_line.strip():
                    event = json.loads(first_line)
                    if event.get("error"):
                        experiments[exp_name]["has_errors"] = True
                        experiments[exp_name]["error_types"][event["error"]] += 1
                    
                    # Count events
                    f.seek(0)
                    experiments[exp_name]["total_events"] += len([l for l in f if l.strip()])
        except:
            pass
    
    # Identify unusable experiments
    for exp_name, data in experiments.items():
        reasons = []
        
        # Check for errors
        if data["has_errors"] and not incomplete_only:
            reasons.append(f"Has errors: {dict(data['error_types'])}")
        
        # Check for insufficient runs
        run_count = len(data["runs"])
        if run_count < 3 and not error_only:
            reasons.append(f"Insufficient runs: {run_count} (need at least 3)")
        elif run_count < 5 and not error_only:
            # Check if this is a scaling experiment (3 runs expected) or regular (5 runs expected)
            # For now, flag anything with < 5 runs as potentially incomplete
            if "scaling" not in exp_name.lower():
                reasons.append(f"Incomplete runs: {run_count} (expected 5)")
        
        if reasons:
            unusable.append({
                "experiment": exp_name,
                "experiment_dir": data["runs"][0]["exp_dir"],
                "runs": [r["name"] for r in data["runs"]],
                "run_dirs": [r["dir"] for r in data["runs"]],
                "reasons": reasons,
                "has_errors": data["has_errors"],
                "error_types": dict(data["error_types"]),
                "total_events": data["total_events"]
            })
    
    return unusable

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Find unusable experiments")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--env", type=str, default="")
    parser.add_argument("--error-only", action="store_true", default=False)
    parser.add_argument("--incomplete-only", action="store_true", default=False)
    parser.add_argument("--output", type=str, default="")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    unusable = find_unusable_experiments(results_dir, args.env, args.error_only, args.incomplete_only)
    
    # Convert Path objects to strings for JSON serialization
    for exp in unusable:
        exp["experiment_dir"] = str(exp["experiment_dir"])
        for run in exp.get("runs", []):
            if isinstance(run, dict):
                run["file"] = str(run.get("file", ""))
                run["dir"] = str(run.get("dir", ""))
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(unusable, f, indent=2, default=str)
        print(f"Found {len(unusable)} unusable experiments", file=sys.stderr)
    else:
        print(json.dumps(unusable, indent=2, default=str))
PYTHON_EOF
)

# Find unusable experiments
log_info "Identifying unusable experiments..."

TMP_SCRIPT=$(mktemp)
TMP_OUTPUT=$(mktemp)
echo "$PYTHON_SCRIPT" > "$TMP_SCRIPT"

# Get JSON output to temp file
python3 "$TMP_SCRIPT" \
    --results-dir "$RESULTS_DIR" \
    --env "$ENV_FILTER" \
    $([[ "$ERROR_ONLY" == "true" ]] && echo "--error-only") \
    $([[ "$INCOMPLETE_ONLY" == "true" ]] && echo "--incomplete-only") \
    --output "$TMP_OUTPUT" 2>&1

# Wait a moment for file to be written
sleep 0.1

# Read JSON from temp file and validate
if [[ -f "$TMP_OUTPUT" ]] && [[ -s "$TMP_OUTPUT" ]]; then
    # Validate and get count
    UNUSABLE_COUNT=$(python3 << PYINLINE
import json
try:
    with open('$TMP_OUTPUT') as f:
        data = json.load(f)
    print(len(data) if isinstance(data, list) else 0)
except Exception:
    print(0)
PYINLINE
)
    
    if [[ "$UNUSABLE_COUNT" -gt 0 ]]; then
        # Read JSON directly from file
        UNUSABLE_JSON=$(cat "$TMP_OUTPUT")
    else
        UNUSABLE_JSON="[]"
        UNUSABLE_COUNT=0
        log_warn "JSON file exists but count is 0 - file may be invalid"
    fi
else
    UNUSABLE_JSON="[]"
    UNUSABLE_COUNT=0
    if [[ -f "$TMP_OUTPUT" ]]; then
        FILE_SIZE=$(wc -c < "$TMP_OUTPUT" 2>/dev/null || echo "0")
        log_warn "Output file exists but is empty (size: $FILE_SIZE bytes)"
    else
        log_warn "Output file was not created"
    fi
fi

# Cleanup temp script (keep output file for now, will remove later)
rm -f "$TMP_SCRIPT"

if [[ "$UNUSABLE_COUNT" -eq 0 ]]; then
    log_success "No unusable experiments found!"
    rm -f "$TMP_OUTPUT" 2>/dev/null || true
    exit 0
fi

log_warn "Found $UNUSABLE_COUNT unusable experiment(s)"

# Display what will be removed
echo ""
echo "=================================================================================="
echo "EXPERIMENTS TO BE REMOVED"
echo "=================================================================================="
echo ""

if [[ "$UNUSABLE_COUNT" -gt 0 ]]; then
    # Read directly from file to avoid shell variable issues
    python3 << PYTHON_EOF
import json
import sys

try:
    with open('$TMP_OUTPUT') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        data = []
    
    for i, exp in enumerate(data, 1):
        print(f"{i}. {exp['experiment']}")
        print(f"   Reasons: {', '.join(exp['reasons'])}")
        print(f"   Runs: {len(exp['runs'])}")
        print(f"   Total events: {exp['total_events']:,}")
        if exp.get('has_errors'):
            print(f"   Errors: {exp.get('error_types', {})}")
        print(f"   Directory: {exp['experiment_dir']}")
        print("")
except Exception as e:
    print(f"Error displaying experiments: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
PYTHON_EOF
else
    echo "No experiments to remove."
fi

if [[ "$DRY_RUN" == "true" ]]; then
    log_info "DRY RUN - No data will be removed"
    rm -f "$TMP_OUTPUT" 2>/dev/null || true
    exit 0
fi

# Confirm removal
if [[ "$CONFIRM" == "true" ]] && [[ "$FORCE" != "true" ]]; then
    echo ""
    read -p "Remove these $UNUSABLE_COUNT experiment(s)? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removal cancelled"
        rm -f "$TMP_OUTPUT" 2>/dev/null || true
        exit 0
    fi
fi

# Remove experiments
log_info "Removing unusable experiments..."

# Read directly from file
python3 << PYTHON_EOF
import json
import sys
import shutil
from pathlib import Path

try:
    with open('$TMP_OUTPUT') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        data = []
    
    removed_count = 0
    for exp in data:
        exp_dir = Path(exp['experiment_dir'])
        if exp_dir.exists():
            # Calculate size before removal
            try:
                size = sum(f.stat().st_size for f in exp_dir.rglob('*') if f.is_file())
            except:
                size = 0
            
            # Remove directory
            shutil.rmtree(exp_dir)
            
            print(f"Removed: {exp['experiment']} ({size / 1024 / 1024:.2f} MB)")
            removed_count += 1
        else:
            print(f"Warning: Directory not found: {exp_dir}")
    
    print(f"\nTotal removed: {removed_count} experiment(s)")
except Exception as e:
    print(f"Error removing experiments: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
PYTHON_EOF

if [[ $? -eq 0 ]]; then
    log_success "Removed unusable experiment(s)"
    log_info "You can now re-run these experiments"
    rm -f "$TMP_OUTPUT" 2>/dev/null || true
else
    log_error "Failed to remove experiments"
    rm -f "$TMP_OUTPUT" 2>/dev/null || true
    exit 1
fi
