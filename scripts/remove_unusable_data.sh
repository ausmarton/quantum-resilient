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
#   --all-unusable         Remove ALL unusable data (errors + incomplete) [DEFAULT]
#   --confirm              Require confirmation before removing (default: true)
#   --force                Remove without confirmation (use with caution)
#   --validate-first       Run validation script first to verify (default: true)
#   --backup                Create backup before removing (default: false)
#   --backup-dir DIR       Backup directory (default: results_backup_TIMESTAMP)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
ENV_FILTER=""
DRY_RUN=false
ERROR_ONLY=false
INCOMPLETE_ONLY=false
ALL_UNUSABLE=false  # Default behavior when no flags specified
CONFIRM=true
FORCE=false
VALIDATE_FIRST=true
BACKUP=false
BACKUP_DIR=""

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
    # Dry run to see what would be removed (all unusable by default)
    ./scripts/remove_unusable_data.sh --env gcp --dry-run

    # Remove only experiments with errors (with backup)
    ./scripts/remove_unusable_data.sh --env gcp --error-only --backup

    # Remove only incomplete experiments
    ./scripts/remove_unusable_data.sh --env gcp --incomplete-only --backup

    # Remove ALL unusable data (errors + incomplete) - DEFAULT behavior
    ./scripts/remove_unusable_data.sh --env gcp --all-unusable --backup

    # Remove without validation (use with caution)
    ./scripts/remove_unusable_data.sh --env gcp --no-validate --force
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
            ALL_UNUSABLE=false
            shift
            ;;
        --incomplete-only)
            INCOMPLETE_ONLY=true
            ALL_UNUSABLE=false
            shift
            ;;
        --all-unusable)
            ALL_UNUSABLE=true
            ERROR_ONLY=false
            INCOMPLETE_ONLY=false
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
        --validate-first)
            VALIDATE_FIRST=true
            shift
            ;;
        --no-validate)
            VALIDATE_FIRST=false
            shift
            ;;
        --backup)
            BACKUP=true
            shift
            ;;
        --backup-dir)
            BACKUP_DIR="$2"
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
    # Determine expected runs: 5 for baseline, 3 for scaling (matching validation script)
    expected_runs_baseline = 5
    expected_runs_scaling = 3
    
    for exp_name, data in experiments.items():
        reasons = []
        is_scaling = "scaling" in exp_name.lower()
        expected_runs = expected_runs_scaling if is_scaling else expected_runs_baseline
        run_count = len(data["runs"])
        
        # Check for errors
        if data["has_errors"] and not incomplete_only:
            reasons.append(f"Has errors: {dict(data['error_types'])}")
        
        # Check for insufficient runs (matching validation script logic)
        if run_count < expected_runs and not error_only:
            reasons.append(f"Insufficient runs: {run_count}/{expected_runs} (missing {expected_runs - run_count})")
        
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

# Safety: Validate first if requested
if [[ "$VALIDATE_FIRST" == "true" ]] && [[ "$DRY_RUN" != "true" ]]; then
    log_info "Running validation first to ensure data integrity..."
    VALIDATION_SCRIPT="$SCRIPT_DIR/scripts/validate_dissertation_data.sh"
    
    if [[ ! -f "$VALIDATION_SCRIPT" ]]; then
        log_error "Validation script not found: $VALIDATION_SCRIPT"
        exit 1
    fi
    
    # Run validation and capture output
    VALIDATION_OUTPUT=$(mktemp)
    if bash "$VALIDATION_SCRIPT" \
        --results-dir "$RESULTS_DIR" \
        --env "$ENV_FILTER" \
        --output "$VALIDATION_OUTPUT" 2>&1; then
        log_success "Validation completed successfully"
        
        # Check if validation found issues
        VALIDATION_FIT=$(python3 -c "
import json
try:
    with open('$VALIDATION_OUTPUT') as f:
        data = json.load(f)
    print('YES' if data.get('summary', {}).get('fit_for_purpose', False) else 'NO')
except:
    print('UNKNOWN')
" 2>/dev/null || echo "UNKNOWN")
        
        if [[ "$VALIDATION_FIT" == "NO" ]]; then
            log_warn "Validation found issues - proceeding with removal of unusable data"
        elif [[ "$VALIDATION_FIT" == "YES" ]]; then
            log_warn "Validation shows data is fit for purpose - are you sure you want to remove data?"
            if [[ "$FORCE" != "true" ]]; then
                read -p "Continue anyway? [y/N] " -n 1 -r
                echo ""
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    log_info "Removal cancelled"
                    rm -f "$VALIDATION_OUTPUT" 2>/dev/null || true
                    exit 0
                fi
            fi
        fi
    else
        log_error "Validation failed - aborting removal for safety"
        rm -f "$VALIDATION_OUTPUT" 2>/dev/null || true
        exit 1
    fi
    rm -f "$VALIDATION_OUTPUT" 2>/dev/null || true
fi

# Find unusable experiments
log_info "Identifying unusable experiments..."

TMP_SCRIPT=$(mktemp)
TMP_OUTPUT=$(mktemp)
echo "$PYTHON_SCRIPT" > "$TMP_SCRIPT"

# Determine removal mode: if neither flag is set, remove all unusable (default behavior)
if [[ "$ERROR_ONLY" != "true" ]] && [[ "$INCOMPLETE_ONLY" != "true" ]]; then
    ALL_UNUSABLE=true
fi

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

# Calculate total size to be removed
TOTAL_SIZE=$(python3 << PYTHON_EOF
import json
from pathlib import Path

try:
    with open('$TMP_OUTPUT') as f:
        data = json.load(f)
    
    total_size = 0
    for exp in data:
        exp_dir = Path(exp['experiment_dir'])
        if exp_dir.exists():
            try:
                size = sum(f.stat().st_size for f in exp_dir.rglob('*') if f.is_file())
                total_size += size
            except:
                pass
    
    print(f"{total_size / 1024 / 1024:.2f}")
except:
    print("0.00")
PYTHON_EOF
)

# Verify experiments exist and match expected patterns
log_info "Verifying experiments before removal..."
VERIFICATION_FAILED=0

python3 << PYTHON_EOF
import json
import sys
from pathlib import Path

try:
    with open('$TMP_OUTPUT') as f:
        data = json.load(f)
    
    failed = []
    for exp in data:
        exp_dir = Path(exp['experiment_dir'])
        
        # Check if directory exists
        if not exp_dir.exists():
            print(f"⚠️  Warning: Directory does not exist: {exp_dir}", file=sys.stderr)
            failed.append(exp['experiment'])
            continue
        
        # Check if it's actually in the results directory
        results_dir = Path('$RESULTS_DIR')
        try:
            exp_dir.resolve().relative_to(results_dir.resolve())
        except ValueError:
            print(f"⚠️  Warning: Directory outside results dir: {exp_dir}", file=sys.stderr)
            failed.append(exp['experiment'])
            continue
        
        # Check if it matches expected experiment name pattern
        if exp_dir.name != exp['experiment']:
            print(f"⚠️  Warning: Directory name mismatch: {exp_dir.name} != {exp['experiment']}", file=sys.stderr)
            failed.append(exp['experiment'])
    
    if failed:
        print(f"Verification failed for {len(failed)} experiments", file=sys.stderr)
        sys.exit(1)
    else:
        print("✅ All experiments verified successfully", file=sys.stderr)
        sys.exit(0)
except Exception as e:
    print(f"Verification error: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF

VERIFICATION_FAILED=$?

if [[ $VERIFICATION_FAILED -ne 0 ]]; then
    log_error "Verification failed - aborting removal for safety"
    log_error "Some experiments don't match expected patterns or don't exist"
    rm -f "$TMP_OUTPUT" 2>/dev/null || true
    exit 1
fi

# Create backup if requested
if [[ "$BACKUP" == "true" ]]; then
    if [[ -z "$BACKUP_DIR" ]]; then
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        BACKUP_DIR="${RESULTS_DIR}_backup_${TIMESTAMP}"
    fi
    
    log_info "Creating backup to: $BACKUP_DIR"
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    
    # Copy experiments to backup
    python3 << PYTHON_EOF
import json
import shutil
from pathlib import Path

try:
    with open('$TMP_OUTPUT') as f:
        data = json.load(f)
    
    backup_base = Path('$BACKUP_DIR')
    results_dir = Path('$RESULTS_DIR')
    
    for exp in data:
        exp_dir = Path(exp['experiment_dir'])
        if exp_dir.exists():
            # Maintain directory structure in backup
            rel_path = exp_dir.relative_to(results_dir)
            backup_path = backup_base / rel_path
            
            # Create parent directories
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy directory
            shutil.copytree(exp_dir, backup_path, dirs_exist_ok=True)
            print(f"Backed up: {exp['experiment']}")
except Exception as e:
    print(f"Backup error: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF
    
    if [[ $? -eq 0 ]]; then
        log_success "Backup created successfully"
    else
        log_error "Backup failed - aborting removal"
        rm -f "$TMP_OUTPUT" 2>/dev/null || true
        exit 1
    fi
fi

# Show summary before removal
echo ""
echo "=================================================================================="
echo "REMOVAL SUMMARY"
echo "=================================================================================="
echo "  Experiments to remove: $UNUSABLE_COUNT"
echo "  Total size: ${TOTAL_SIZE} MB"
if [[ "$BACKUP" == "true" ]] && [[ -n "$BACKUP_DIR" ]]; then
    echo "  Backup location: $BACKUP_DIR"
fi
echo "=================================================================================="
echo ""

# Confirm removal
if [[ "$CONFIRM" == "true" ]] && [[ "$FORCE" != "true" ]]; then
    echo "⚠️  WARNING: This will permanently delete $UNUSABLE_COUNT experiment(s) (${TOTAL_SIZE} MB)"
    if [[ "$BACKUP" != "true" ]]; then
        echo "⚠️  WARNING: No backup will be created!"
        echo "   Use --backup to create a backup first"
    fi
    echo ""
    read -p "Are you sure you want to proceed? Type 'DELETE' to confirm: " -r
    echo ""
    if [[ "$REPLY" != "DELETE" ]]; then
        log_info "Removal cancelled (expected 'DELETE' but got: '$REPLY')"
        rm -f "$TMP_OUTPUT" 2>/dev/null || true
        exit 0
    fi
fi

# Remove experiments
log_info "Removing unusable experiments..."

# Create removal log
REMOVAL_LOG=$(mktemp)
REMOVAL_LOG_FINAL="${RESULTS_DIR}/.removal_log_$(date +%Y%m%d_%H%M%S).txt"

# Read directly from file and remove
python3 << PYTHON_EOF
import json
import sys
import shutil
from pathlib import Path
from datetime import datetime

removal_log = []

try:
    with open('$TMP_OUTPUT') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        data = []
    
    removed_count = 0
    failed_count = 0
    total_size_removed = 0
    
    for exp in data:
        exp_dir = Path(exp['experiment_dir'])
        if exp_dir.exists():
            # Double-check it's in results directory (safety)
            results_dir = Path('$RESULTS_DIR')
            try:
                exp_dir.resolve().relative_to(results_dir.resolve())
            except ValueError:
                print(f"⚠️  SKIPPED (outside results dir): {exp['experiment']}", file=sys.stderr)
                failed_count += 1
                continue
            
            # Calculate size before removal
            try:
                size = sum(f.stat().st_size for f in exp_dir.rglob('*') if f.is_file())
            except:
                size = 0
            
            # Remove directory
            try:
                shutil.rmtree(exp_dir)
                print(f"✅ Removed: {exp['experiment']} ({size / 1024 / 1024:.2f} MB)")
                
                removal_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'experiment': exp['experiment'],
                    'directory': str(exp_dir),
                    'size_mb': size / 1024 / 1024,
                    'reasons': exp.get('reasons', []),
                    'status': 'removed'
                })
                
                removed_count += 1
                total_size_removed += size
            except Exception as e:
                print(f"❌ Failed to remove: {exp['experiment']} - {e}", file=sys.stderr)
                failed_count += 1
        else:
            print(f"⚠️  Directory not found: {exp_dir}")
            failed_count += 1
    
    print(f"\n{'=' * 60}")
    print(f"Removal Summary:")
    print(f"  Successfully removed: {removed_count} experiment(s)")
    print(f"  Failed: {failed_count} experiment(s)")
    print(f"  Total size removed: {total_size_removed / 1024 / 1024:.2f} MB")
    print(f"{'=' * 60}")
    
    # Write removal log
    with open('$REMOVAL_LOG', 'w') as f:
        f.write(f"Removal log - {datetime.now().isoformat()}\n")
        f.write(f"Total removed: {removed_count}\n")
        f.write(f"Total failed: {failed_count}\n")
        f.write(f"Total size: {total_size_removed / 1024 / 1024:.2f} MB\n\n")
        f.write("Removed experiments:\n")
        for entry in removal_log:
            f.write(f"\n{json.dumps(entry, indent=2)}\n")
    
    if failed_count > 0:
        sys.exit(1)
except Exception as e:
    print(f"Error removing experiments: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
PYTHON_EOF

REMOVAL_EXIT=$?

# Move removal log to results directory
if [[ -f "$REMOVAL_LOG" ]]; then
    mkdir -p "$RESULTS_DIR"
    mv "$REMOVAL_LOG" "$REMOVAL_LOG_FINAL"
    log_info "Removal log saved to: $REMOVAL_LOG_FINAL"
fi

if [[ $REMOVAL_EXIT -eq 0 ]]; then
    log_success "Successfully removed unusable experiment(s)"
    log_info "You can now re-run these experiments"
    if [[ "$BACKUP" == "true" ]] && [[ -n "$BACKUP_DIR" ]]; then
        log_info "Backup available at: $BACKUP_DIR"
    fi
    rm -f "$TMP_OUTPUT" 2>/dev/null || true
else
    log_error "Some experiments failed to remove - check output above"
    log_error "Removal log: $REMOVAL_LOG_FINAL"
    rm -f "$TMP_OUTPUT" 2>/dev/null || true
    exit 1
fi
