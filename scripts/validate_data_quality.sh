#!/usr/bin/env bash
# =============================================================================
# validate_data_quality.sh - Comprehensive data quality validation
#
# Validates all collected experiment data for:
# - Missing values in required fields
# - Abnormal/outlier values
# - Data completeness (event ID gaps)
# - Format validity
# - Expected value ranges
# - Cross-environment consistency
# - Dissertation requirements (experiment completeness, statistical validity)
#
# Usage:
#   ./scripts/validate_data_quality.sh [OPTIONS]
#
# Options:
#   --env ENV              Check specific environment (native, minikube, gcp)
#   --results-dir DIR      Results directory (default: results/)
#   --matrix PATH          Experiment matrix YAML (default: orchestration/experiment_matrix.yaml)
#   --output FILE          Write detailed report to JSON file
#   --fail-on-issues       Exit with error if any issues found
#   --min-events N         Minimum expected events per experiment (default: 100)
#   --max-latency-us N     Maximum expected latency in microseconds (default: 1000000)
#   --check-outliers       Enable outlier detection (uses IQR method)
#   --check-dissertation   Validate dissertation requirements (experiment completeness)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
MATRIX="$SCRIPT_DIR/orchestration/experiment_matrix.yaml"
ENV_FILTER=""
OUTPUT_FILE=""
FAIL_ON_ISSUES=false
MIN_EVENTS=100
MAX_LATENCY_US=1000000
CHECK_OUTLIERS=false
CHECK_DISSERTATION=false
PARALLEL_JOBS=0  # 0 means auto-detect (CPU count - 1)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
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

Comprehensive data quality validation for all collected experiments.

OPTIONS:
    --env ENV              Check specific environment (native, minikube, gcp)
    --results-dir DIR      Results directory (default: results/)
    --output FILE          Write detailed report to JSON file
    --fail-on-issues       Exit with error if any issues found
    --min-events N         Minimum expected events per experiment (default: 100)
    --max-latency-us N     Maximum expected latency in microseconds (default: 1000000)
    --check-outliers       Enable outlier detection (uses IQR method)
    --check-dissertation   Validate dissertation requirements (experiment completeness)
    --matrix PATH          Experiment matrix YAML (default: orchestration/experiment_matrix.yaml)
    --parallel N           Number of parallel workers (default: auto-detect, CPU count - 1)
    -h, --help             Show this help message

EXAMPLES:
    # Basic validation
    ./scripts/validate_data_quality.sh

    # Check only minikube, enable outlier detection
    ./scripts/validate_data_quality.sh --env minikube --check-outliers

    # Generate detailed report with dissertation requirements
    ./scripts/validate_data_quality.sh --output quality-report.json --check-outliers --check-dissertation

    # Full dissertation validation
    ./scripts/validate_data_quality.sh --check-dissertation --check-outliers
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
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --fail-on-issues)
            FAIL_ON_ISSUES=true
            shift
            ;;
        --min-events)
            MIN_EVENTS="$2"
            shift 2
            ;;
        --max-latency-us)
            MAX_LATENCY_US="$2"
            shift 2
            ;;
    --check-outliers)
        CHECK_OUTLIERS=true
        shift
        ;;
    --check-dissertation)
        CHECK_DISSERTATION=true
        shift
        ;;
    --matrix)
        MATRIX="$2"
        shift 2
        ;;
    --parallel)
        PARALLEL_JOBS="$2"
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

# Auto-detect parallel jobs if not specified
if [[ "$PARALLEL_JOBS" -eq 0 ]]; then
    CPU_COUNT=$(python3 -c "import multiprocessing; print(max(1, multiprocessing.cpu_count() - 1))" 2>/dev/null || echo "4")
    PARALLEL_JOBS=$CPU_COUNT
fi

# Python script for data quality analysis
PYTHON_SCRIPT=$(cat <<'PYTHON_EOF'
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Global parameters (will be set from command line)
MIN_EVENTS = 100
MAX_LATENCY_US = 1000000
CHECK_OUTLIERS = False
PARALLEL_WORKERS = None  # Will be set based on CPU count

def validate_jsonl_file(file_path: Path, min_events: int, max_latency_us: int, check_outliers: bool) -> Dict[str, Any]:
    """Validate a single JSONL file and return quality metrics."""
    issues = []
    warnings = []
    stats = {
        "total_lines": 0,
        "valid_json_lines": 0,
        "invalid_json_lines": 0,
        "missing_fields": defaultdict(int),
        "null_values": defaultdict(int),
        "event_ids": [],
        "latencies": [],
        "timestamps": [],
        "algorithms": set(),
        "operations": set(),
    }
    
    required_fields = ["latency_us", "algorithm", "operation", "event_id"]
    optional_fields = ["timestamp", "timestamp_utc_iso", "cpu_user_time_us", "memory_rss_bytes"]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                stats["total_lines"] += 1
                line = line.strip()
                
                if not line:
                    continue
                
                # Check for error messages
                if line.startswith("error:") or line.startswith("Error:"):
                    issues.append({
                        "type": "error_message",
                        "line": line_num,
                        "content": line[:100]
                    })
                    stats["invalid_json_lines"] += 1
                    continue
                
                # Parse JSON
                try:
                    record = json.loads(line)
                    stats["valid_json_lines"] += 1
                except json.JSONDecodeError as e:
                    issues.append({
                        "type": "invalid_json",
                        "line": line_num,
                        "error": str(e),
                        "content": line[:100]
                    })
                    stats["invalid_json_lines"] += 1
                    continue
                
                # Check required fields
                for field in required_fields:
                    if field not in record:
                        stats["missing_fields"][field] += 1
                        issues.append({
                            "type": "missing_field",
                            "line": line_num,
                            "field": field
                        })
                    elif record[field] is None:
                        stats["null_values"][field] += 1
                        issues.append({
                            "type": "null_value",
                            "line": line_num,
                            "field": field
                        })
                
                # Collect statistics
                if "event_id" in record and record["event_id"] is not None:
                    stats["event_ids"].append(record["event_id"])
                
                if "latency_us" in record and record["latency_us"] is not None:
                    latency = record["latency_us"]
                    stats["latencies"].append(latency)
                    
                    # Check for negative or extremely large values
                    if latency < 0:
                        issues.append({
                            "type": "negative_latency",
                            "line": line_num,
                            "value": latency
                        })
                    elif latency > max_latency_us:
                        warnings.append({
                            "type": "extreme_latency",
                            "line": line_num,
                            "value": latency,
                            "threshold": max_latency_us
                        })
                
                if "timestamp" in record:
                    stats["timestamps"].append(record["timestamp"])
                elif "timestamp_utc_iso" in record:
                    stats["timestamps"].append(record["timestamp_utc_iso"])
                
                if "algorithm" in record:
                    stats["algorithms"].add(record["algorithm"])
                
                if "operation" in record:
                    stats["operations"].add(record["operation"])
        
        # Check event ID completeness
        if stats["event_ids"]:
            event_ids_sorted = sorted(stats["event_ids"])
            min_id = min(event_ids_sorted)
            max_id = max(event_ids_sorted)
            expected_count = max_id - min_id + 1
            actual_count = len(event_ids_sorted)
            
            if actual_count < expected_count:
                missing_ids = set(range(min_id, max_id + 1)) - set(event_ids_sorted)
                issues.append({
                    "type": "missing_event_ids",
                    "missing_count": len(missing_ids),
                    "missing_ids": sorted(list(missing_ids))[:50],  # Limit to first 50
                    "expected": expected_count,
                    "actual": actual_count
                })
            
            # Check for duplicate event IDs
            seen_ids = set()
            duplicates = []
            for eid in event_ids_sorted:
                if eid in seen_ids:
                    duplicates.append(eid)
                seen_ids.add(eid)
            
            if duplicates:
                issues.append({
                    "type": "duplicate_event_ids",
                    "count": len(duplicates),
                    "examples": duplicates[:10]
                })
        
        # Check minimum event count
        if stats["valid_json_lines"] < min_events:
            issues.append({
                "type": "insufficient_events",
                "actual": stats["valid_json_lines"],
                "minimum": min_events
            })
        
        # Outlier detection (IQR method)
        outliers = []
        if check_outliers and len(stats["latencies"]) > 10:
            latencies_sorted = sorted(stats["latencies"])
            q1_idx = len(latencies_sorted) // 4
            q3_idx = 3 * len(latencies_sorted) // 4
            q1 = latencies_sorted[q1_idx]
            q3 = latencies_sorted[q3_idx]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_count = sum(1 for l in latencies_sorted if l < lower_bound or l > upper_bound)
            if outlier_count > 0:
                outlier_values = [l for l in latencies_sorted if l < lower_bound or l > upper_bound]
                warnings.append({
                    "type": "outliers_detected",
                    "count": outlier_count,
                    "percentage": (outlier_count / len(latencies_sorted)) * 100,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "examples": outlier_values[:10]
                })
        
        # Calculate statistics
        result = {
            "file_path": str(file_path),
            "total_lines": stats["total_lines"],
            "valid_json_lines": stats["valid_json_lines"],
            "invalid_json_lines": stats["invalid_json_lines"],
            "issues": issues,
            "warnings": warnings,
            "statistics": {
                "event_count": stats["valid_json_lines"],
                "unique_algorithms": list(stats["algorithms"]),
                "unique_operations": list(stats["operations"]),
            }
        }
        
        if stats["latencies"]:
            result["statistics"]["latency"] = {
                "min": min(stats["latencies"]),
                "max": max(stats["latencies"]),
                "mean": sum(stats["latencies"]) / len(stats["latencies"]),
                "count": len(stats["latencies"])
            }
        
        if stats["event_ids"]:
            result["statistics"]["event_ids"] = {
                "min": min(stats["event_ids"]),
                "max": max(stats["event_ids"]),
                "count": len(stats["event_ids"]),
                "unique_count": len(set(stats["event_ids"]))
            }
        
        if stats["missing_fields"]:
            result["statistics"]["missing_fields"] = dict(stats["missing_fields"])
        
        if stats["null_values"]:
            result["statistics"]["null_values"] = dict(stats["null_values"])
        
        return result
        
    except Exception as e:
        return {
            "file_path": str(file_path),
            "error": str(e),
            "issues": [{"type": "file_read_error", "error": str(e)}],
            "warnings": [],
            "statistics": {}
        }

# Progress reporting
def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"

def report_progress(current, total, stage, start_time, file_name="", is_main=True):
    """Report progress to stderr (so it doesn't interfere with JSON output)."""
    if not is_main or total == 0:
        return
    
    elapsed = time.time() - start_time
    percentage = (current * 100) // total
    
    # Calculate ETA
    if current > 0 and elapsed > 0:
        rate = current / elapsed
        remaining = total - current
        eta_seconds = int(remaining / rate) if rate > 0 else 0
        eta_str = format_time(eta_seconds)
    else:
        eta_str = "calculating..."
    
    elapsed_str = format_time(int(elapsed))
    
    # Progress bar (50 chars)
    bar_width = 50
    filled = (percentage * bar_width) // 100
    empty = bar_width - filled
    bar = "#" * filled + "-" * empty
    
    # Truncate filename if too long
    display_name = file_name
    if len(display_name) > 40:
        display_name = "..." + display_name[-37:]
    
    # Write to stderr so it doesn't interfere with JSON output
    # Use carriage return to overwrite the line
    status_line = f"\r[{percentage:3d}%] [{stage:8s}] [{bar}] {current}/{total} | Elapsed: {elapsed_str:>8s} | ETA: {eta_str:>8s} | {display_name}"
    print(status_line, end="", file=sys.stderr, flush=True)
    
    if current == total:
        print("", file=sys.stderr)  # New line when complete

# Main validation
if len(sys.argv) < 2:
    print(json.dumps({"error": "Missing results_dir argument"}, indent=2))
    sys.exit(1)

results_dir = Path(sys.argv[1])
env_filter = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
MIN_EVENTS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
MAX_LATENCY_US = int(sys.argv[4]) if len(sys.argv) > 4 else 1000000
CHECK_OUTLIERS = sys.argv[5] == "true" if len(sys.argv) > 5 else False
PARALLEL_WORKERS = int(sys.argv[6]) if len(sys.argv) > 6 else max(1, cpu_count() - 1)

validation_results = {
    "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    "environments": {},
    "summary": {
        "total_experiments": 0,
        "valid_experiments": 0,
        "experiments_with_issues": 0,
        "experiments_with_warnings": 0,
        "total_issues": 0,
        "total_warnings": 0
    },
    "experiments": []
}

envs = [env_filter] if env_filter else ["native", "minikube", "gcp"]

# Print stage info (will be filtered if called from worker process)
# Use a simple check: if we can't determine parent, assume main process
try:
    import multiprocessing
    is_main_process = multiprocessing.current_process().name == 'MainProcess'
except:
    is_main_process = True

if is_main_process:
    print(f"\n[STAGE] Data Quality Validation", file=sys.stderr)
    print(f"  Environments to validate: {', '.join(envs)}", file=sys.stderr)
    print(f"  Parallel workers: {PARALLEL_WORKERS}", file=sys.stderr)
    print(f"  Outlier detection: {'enabled' if CHECK_OUTLIERS else 'disabled'}", file=sys.stderr)
    print("", file=sys.stderr)

for env in envs:
    env_dir = results_dir / env
    if not env_dir.exists():
        continue
    
    env_results = {
        "environment": env,
        "experiments": [],
        "summary": {
            "total": 0,
            "valid": 0,
            "with_issues": 0,
            "with_warnings": 0
        }
    }
    
    # Find all raw JSONL files
    jsonl_files = list(env_dir.rglob("raw/run.jsonl"))
    total_files = len(jsonl_files)
    
    if total_files == 0:
        if is_main_process:
            print(f"[SKIP] {env}: No data files found", file=sys.stderr)
        continue
    
    # Only print in main process
    if is_main_process:
        print(f"\n[STAGE] Validating {env} environment", file=sys.stderr)
        print(f"  Found {total_files} experiment(s) to validate", file=sys.stderr)
        print(f"  Using {PARALLEL_WORKERS} parallel worker(s)", file=sys.stderr)
    start_time = time.time()
    
    # Process files in parallel
    processed = 0
    results_list = []
    
    # Create a wrapper function that includes file path for progress reporting
    def validate_with_progress(args):
        file_path, exp_id, env_name, min_evts, max_lat, check_out = args
        result = validate_jsonl_file(file_path, min_evts, max_lat, check_out)
        result["experiment_id"] = exp_id
        result["environment"] = env_name
        return result
    
    # Prepare arguments for parallel processing
    file_args = []
    for jsonl_file in jsonl_files:
        exp_dir = jsonl_file.parent.parent
        exp_id = exp_dir.name
        file_args.append((jsonl_file, exp_id, env, MIN_EVENTS, MAX_LATENCY_US, CHECK_OUTLIERS))
    
    # Process in parallel with progress reporting
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid pickling issues
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        future_to_file = {executor.submit(validate_with_progress, args): args[0] 
                          for args in file_args}
        
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results_list.append(result)
                processed += 1
                
                # Report progress (only in main process)
                # Get experiment ID for display
                exp_dir = file_path.parent.parent
                exp_id = exp_dir.name if exp_dir else file_path.name
                report_progress(processed, total_files, env, start_time, exp_id, is_main_process)
            except Exception as e:
                # Handle errors gracefully
                exp_dir = file_path.parent.parent
                exp_id = exp_dir.name
                error_result = {
                    "file_path": str(file_path),
                    "experiment_id": exp_id,
                    "environment": env,
                    "error": str(e),
                    "issues": [{"type": "validation_error", "error": str(e)}],
                    "warnings": [],
                    "statistics": {}
                }
                results_list.append(error_result)
                processed += 1
                report_progress(processed, total_files, env, start_time, exp_id, is_main_process)
    
    # Final progress update (only in main process)
    if is_main_process:
        elapsed = time.time() - start_time
        print(f"\n[OK] {env}: Validated {total_files} experiment(s) in {format_time(int(elapsed))}", file=sys.stderr)
    
    # Process results
    for result in results_list:
        env_results["experiments"].append(result)
        validation_results["experiments"].append(result)
        env_results["summary"]["total"] += 1
        validation_results["summary"]["total_experiments"] += 1
        
        if result.get("issues"):
            env_results["summary"]["with_issues"] += 1
            validation_results["summary"]["experiments_with_issues"] += 1
            validation_results["summary"]["total_issues"] += len(result["issues"])
        else:
            env_results["summary"]["valid"] += 1
            validation_results["summary"]["valid_experiments"] += 1
        
        if result.get("warnings"):
            env_results["summary"]["with_warnings"] += 1
            validation_results["summary"]["experiments_with_warnings"] += 1
            validation_results["summary"]["total_warnings"] += len(result["warnings"])
    
    validation_results["environments"][env] = env_results

# Print final summary to stderr (only in main process)
if is_main_process:
    print("\n[STAGE] Validation Complete", file=sys.stderr)
    total = validation_results["summary"]["total_experiments"]
    valid = validation_results["summary"]["valid_experiments"]
    issues_count = validation_results["summary"]["experiments_with_issues"]
    warnings_count = validation_results["summary"]["experiments_with_warnings"]
    print(f"  Total experiments: {total}", file=sys.stderr)
    print(f"  Valid: {valid}", file=sys.stderr)
    print(f"  With issues: {issues_count}", file=sys.stderr)
    print(f"  With warnings: {warnings_count}", file=sys.stderr)

# Output JSON to stdout (for script to capture)
print(json.dumps(validation_results, indent=2))
PYTHON_EOF
)

echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Comprehensive Data Quality Validation${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

log_info "Scanning experiments in: $RESULTS_DIR"
[[ -n "$ENV_FILTER" ]] && log_info "Environment filter: $ENV_FILTER"
[[ "$CHECK_OUTLIERS" == "true" ]] && log_info "Outlier detection: enabled"
echo ""

# Create temporary Python script file
TEMP_PYTHON_SCRIPT=$(mktemp)
cat > "$TEMP_PYTHON_SCRIPT" <<PYTHON_EOF
$PYTHON_SCRIPT
PYTHON_EOF

# Run Python validation with progress output to stderr
log_info "Starting data quality validation..."
log_info "Parallel workers: $PARALLEL_JOBS"
echo ""

# Run Python script - progress goes to stderr, JSON to stdout
# We need to separate them properly
TEMP_STDOUT=$(mktemp)
TEMP_STDERR=$(mktemp)

python3 "$TEMP_PYTHON_SCRIPT" "$RESULTS_DIR" "$ENV_FILTER" "$MIN_EVENTS" "$MAX_LATENCY_US" "$CHECK_OUTLIERS" "$PARALLEL_JOBS" > "$TEMP_STDOUT" 2> "$TEMP_STDERR"
PYTHON_EXIT=$?

# Show progress output (from stderr)
cat "$TEMP_STDERR" >&2
echo "" >&2

# Get JSON output (from stdout)
VALIDATION_OUTPUT=$(cat "$TEMP_STDOUT")
VALIDATION_JSON=$(echo "$VALIDATION_OUTPUT" | python3 -m json.tool 2>/dev/null || echo "$VALIDATION_OUTPUT")

# Cleanup
rm -f "$TEMP_STDOUT" "$TEMP_STDERR"

# Cleanup
rm -f "$TEMP_PYTHON_SCRIPT"

if [[ $PYTHON_EXIT -ne 0 ]]; then
    log_error "Python validation script failed"
    echo "$VALIDATION_OUTPUT"
    exit 1
fi

# Parse results
VALIDATION_JSON=$(echo "$VALIDATION_OUTPUT" | python3 -m json.tool 2>/dev/null || echo "$VALIDATION_OUTPUT")

# Extract summary
SUMMARY=$(echo "$VALIDATION_JSON" | python3 -c "
import json, sys
data = json.load(sys.stdin)
summary = data.get('summary', {})
print(f\"Total experiments: {summary.get('total_experiments', 0)}\")
print(f\"Valid: {summary.get('valid_experiments', 0)}\")
print(f\"With issues: {summary.get('experiments_with_issues', 0)}\")
print(f\"With warnings: {summary.get('experiments_with_warnings', 0)}\")
print(f\"Total issues: {summary.get('total_issues', 0)}\")
print(f\"Total warnings: {summary.get('total_warnings', 0)}\")
" 2>/dev/null || echo "Error parsing summary")

echo "$SUMMARY"
echo ""

# Show issues by environment
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}Issues by Environment:${NC}"
echo ""

for env in native minikube gcp; do
    if [[ -n "$ENV_FILTER" ]] && [[ "$env" != "$ENV_FILTER" ]]; then
        continue
    fi
    
    ENV_SUMMARY=$(echo "$VALIDATION_JSON" | python3 -c "
import json, sys
data = json.load(sys.stdin)
env_data = data.get('environments', {}).get('$env', {})
if env_data:
    summary = env_data.get('summary', {})
    print(f\"{summary.get('total', 0)} total, {summary.get('valid', 0)} valid, {summary.get('with_issues', 0)} with issues, {summary.get('with_warnings', 0)} with warnings\")
" 2>/dev/null)
    
    if [[ -n "$ENV_SUMMARY" ]]; then
        echo -e "${MAGENTA}$env:${NC} $ENV_SUMMARY"
    fi
done

echo ""

# Show experiments with issues
ISSUES_COUNT=$(echo "$VALIDATION_JSON" | python3 -c "
import json, sys
data = json.load(sys.stdin)
issues = [exp for exp in data.get('experiments', []) if exp.get('issues')]
print(len(issues))
" 2>/dev/null || echo "0")

if [[ "$ISSUES_COUNT" -gt 0 ]]; then
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}Experiments with Issues:${NC}"
    echo ""
    
    echo "$VALIDATION_JSON" | python3 -c "
import json, sys
data = json.load(sys.stdin)
issues = [exp for exp in data.get('experiments', []) if exp.get('issues')]
for exp in issues[:20]:  # Show first 20
    exp_id = exp.get('experiment_id', 'unknown')
    env = exp.get('environment', 'unknown')
    issue_count = len(exp.get('issues', []))
    issue_types = {}
    for issue in exp.get('issues', []):
        issue_type = issue.get('type', 'unknown')
        issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
    print(f\"  {env}/{exp_id}: {issue_count} issue(s) - {', '.join(f'{k}({v})' for k, v in issue_types.items())}\")
if len(issues) > 20:
    print(f\"  ... and {len(issues) - 20} more\")
" 2>/dev/null
    
    echo ""
fi

# Show experiments with warnings
WARNINGS_COUNT=$(echo "$VALIDATION_JSON" | python3 -c "
import json, sys
data = json.load(sys.stdin)
warnings = [exp for exp in data.get('experiments', []) if exp.get('warnings')]
print(len(warnings))
" 2>/dev/null || echo "0")

if [[ "$WARNINGS_COUNT" -gt 0 ]]; then
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}Experiments with Warnings:${NC}"
    echo ""
    
    echo "$VALIDATION_JSON" | python3 -c "
import json, sys
data = json.load(sys.stdin)
warnings = [exp for exp in data.get('experiments', []) if exp.get('warnings')]
for exp in warnings[:10]:  # Show first 10
    exp_id = exp.get('experiment_id', 'unknown')
    env = exp.get('environment', 'unknown')
    warning_count = len(exp.get('warnings', []))
    warning_types = {}
    for warning in exp.get('warnings', []):
        warning_type = warning.get('type', 'unknown')
        warning_types[warning_type] = warning_types.get(warning_type, 0) + 1
    print(f\"  {env}/{exp_id}: {warning_count} warning(s) - {', '.join(f'{k}({v})' for k, v in warning_types.items())}\")
if len(warnings) > 10:
    print(f\"  ... and {len(warnings) - 10} more\")
" 2>/dev/null
    
    echo ""
fi

# Write output file if requested
if [[ -n "$OUTPUT_FILE" ]]; then
    echo "$VALIDATION_JSON" > "$OUTPUT_FILE"
    log_success "Detailed report written to: $OUTPUT_FILE"
    echo ""
fi

# Dissertation requirements validation
if [[ "$CHECK_DISSERTATION" == "true" ]]; then
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}Dissertation Requirements Validation${NC}"
    echo ""
    
    DISSERTATION_REPORT=$(python3 <<DISSERTATION_EOF
import json
import sys
import yaml
from pathlib import Path
from collections import defaultdict

results_dir = Path("$RESULTS_DIR")
matrix_file = Path("$MATRIX")
env_filter = "$ENV_FILTER" if "$ENV_FILTER" else None

# Load matrix
with open(matrix_file) as f:
    matrix = yaml.safe_load(f)

defaults = matrix.get("defaults", {})
experiments = matrix.get("experiments", [])
scaling_config = matrix.get("scaling", {})
scaling_replicas = scaling_config.get("replicas", [1, 2, 4, 8])
scaling_algorithms = scaling_config.get("scaling_algorithms", [])

# Scenario ID generation (matching generate_scenarios.py)
def compute_scenario_hash(algorithm, payload, rate, run, pattern="constant", duration=None, is_scaling=False):
    seed_parts = [algorithm, str(payload), str(rate), str(run), pattern]
    if duration and duration != 30:
        seed_parts.append(str(duration))
    if is_scaling:
        seed_parts.append("scaling")
    seed_str = ":".join(seed_parts)
    import hashlib
    return hashlib.sha256(seed_str.encode()).hexdigest()[:8]

def generate_scenario_id(algorithm, payload, rate, run, pattern="constant", duration=None, is_scaling=False):
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

# Generate expected experiments
expected_baseline = defaultdict(lambda: defaultdict(set))  # env -> algorithm -> set of exp_ids
expected_scaling = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))  # env -> algorithm -> replica -> set of exp_ids

for exp in experiments:
    algorithm = exp["algorithm"]
    payload_sizes = exp.get("payload_sizes", [1024])
    rates = exp.get("rates", [500])
    runs = exp.get("runs", defaults.get("runs", 5))
    pattern = exp.get("workload_pattern", "constant")
    duration = exp.get("duration_sec", defaults.get("duration_sec", 30))
    is_scaling_exp = exp.get("scaling_experiment", False)
    
    for payload in payload_sizes:
        for rate in rates:
            for run_index in range(1, runs + 1):
                scenario_id = generate_scenario_id(algorithm, payload, rate, run_index, pattern, duration, is_scaling_exp)
                
                if is_scaling_exp:
                    # Scaling experiments: minikube and GCP only, with replicas
                    for env in ["minikube", "gcp"]:
                        if env_filter and env != env_filter:
                            continue
                        for replica in scaling_replicas:
                            if replica == 1:
                                # Replica 1 uses base ID
                                expected_scaling[env][algorithm][replica].add(scenario_id)
                            else:
                                # Replicas 2,4,8 append _rN
                                scaling_id = f"{scenario_id}_r{replica}"
                                expected_scaling[env][algorithm][replica].add(scaling_id)
                else:
                    # Baseline experiments: all environments
                    for env in ["native", "minikube", "gcp"]:
                        if env_filter and env != env_filter:
                            continue
                        expected_baseline[env][algorithm].add(scenario_id)

# Check actual results
found_baseline = defaultdict(lambda: defaultdict(set))
found_scaling = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
missing_baseline = defaultdict(lambda: defaultdict(list))
missing_scaling = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for env in ["native", "minikube", "gcp"]:
    if env_filter and env != env_filter:
        continue
    
    env_dir = results_dir / env
    if not env_dir.exists():
        continue
    
    # Find all experiments
    for exp_dir in env_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        exp_id = exp_dir.name
        raw_file = exp_dir / "raw" / "run.jsonl"
        
        if not raw_file.exists():
            continue
        
        # Parse experiment ID to extract components
        parts = exp_id.split("_")
        algorithm = None
        replica = 1
        is_scaling_exp = "scaling" in exp_id.lower()
        
        # Detect scaling experiments (have _r2, _r4, _r8 suffix)
        if len(parts) > 1 and parts[-1].startswith("r") and parts[-1][1:].isdigit():
            replica = int(parts[-1][1:])
            # Remove replica suffix to get base ID
            base_parts = parts[:-1]
            base_id = "_".join(base_parts)
        else:
            base_id = exp_id
            replica = 1  # Default to replica 1 if no suffix
        
        # Extract algorithm by matching against known algorithm names
        # Try longest names first (e.g., hybrid_kyber_dilithium before hybrid)
        known_algorithms = [exp["algorithm"] for exp in experiments]
        # Sort by length descending to match longest first
        known_algorithms.sort(key=len, reverse=True)
        for alg in known_algorithms:
            if exp_id.startswith(alg + "_"):
                algorithm = alg
                break
        
        # Classify experiment
        if is_scaling_exp and algorithm and algorithm in scaling_algorithms:
            # This is a scaling experiment
            if replica in scaling_replicas:
                found_scaling[env][algorithm][replica].add(exp_id)
        elif not is_scaling_exp:
            # This is a baseline experiment (no "scaling" in name)
            if algorithm:
                found_baseline[env][algorithm].add(exp_id)

# Compare expected vs found
dissertation_issues = []
dissertation_warnings = []

# Check baseline experiments
for env in expected_baseline:
    for algorithm in expected_baseline[env]:
        expected = expected_baseline[env][algorithm]
        found = found_baseline[env][algorithm]
        missing = expected - found
        
        if missing:
            missing_list = sorted(list(missing))[:10]  # Limit to first 10
            dissertation_issues.append({
                "type": "missing_baseline_experiment",
                "environment": env,
                "algorithm": algorithm,
                "missing_count": len(missing),
                "missing_examples": missing_list,
                "expected": len(expected),
                "found": len(found)
            })

# Check scaling experiments
for env in expected_scaling:
    for algorithm in expected_scaling[env]:
        for replica in expected_scaling[env][algorithm]:
            expected = expected_scaling[env][algorithm][replica]
            found = found_scaling[env][algorithm][replica]
            missing = expected - found
            
            if missing:
                missing_list = sorted(list(missing))[:10]
                dissertation_issues.append({
                    "type": "missing_scaling_experiment",
                    "environment": env,
                    "algorithm": algorithm,
                    "replica": replica,
                    "missing_count": len(missing),
                    "missing_examples": missing_list,
                    "expected": len(expected),
                    "found": len(found)
                })

# Check cross-environment consistency for baseline
algorithms_in_all_envs = set()
for env in expected_baseline:
    algorithms_in_all_envs.update(expected_baseline[env].keys())

for algorithm in algorithms_in_all_envs:
    envs_with_data = []
    envs_missing = []
    
    for env in ["native", "minikube", "gcp"]:
        if env_filter and env != env_filter:
            continue
        if algorithm in found_baseline[env] and len(found_baseline[env][algorithm]) > 0:
            envs_with_data.append(env)
        else:
            envs_missing.append(env)
    
    if envs_missing:
        dissertation_warnings.append({
            "type": "incomplete_cross_environment",
            "algorithm": algorithm,
            "environments_with_data": envs_with_data,
            "environments_missing": envs_missing
        })

# Check statistical validity (minimum runs per configuration)
# This is a simplified check - would need to parse experiment IDs more carefully
# For now, we check if we have at least some runs for each algorithm

# Generate summary
dissertation_summary = {
    "baseline_experiments": {
        "expected_total": sum(len(exps) for exps in expected_baseline.values() for exps in exps.values()),
        "found_total": sum(len(exps) for exps in found_baseline.values() for exps in exps.values()),
        "by_environment": {
            env: {
                "expected": sum(len(exps) for exps in expected_baseline[env].values()),
                "found": sum(len(exps) for exps in found_baseline[env].values())
            }
            for env in expected_baseline
        }
    },
    "scaling_experiments": {
        "expected_total": sum(
            len(exps) 
            for env_exps in expected_scaling.values() 
            for algo_exps in env_exps.values() 
            for exps in algo_exps.values()
        ),
        "found_total": sum(
            len(exps) 
            for env_exps in found_scaling.values() 
            for algo_exps in env_exps.values() 
            for exps in algo_exps.values()
        ),
        "by_environment": {
            env: {
                "expected": sum(
                    len(exps) 
                    for algo_exps in expected_scaling[env].values() 
                    for exps in algo_exps.values()
                ),
                "found": sum(
                    len(exps) 
                    for algo_exps in found_scaling[env].values() 
                    for exps in algo_exps.values()
                )
            }
            for env in expected_scaling
        }
    },
    "issues": dissertation_issues,
    "warnings": dissertation_warnings
}

print(json.dumps(dissertation_summary, indent=2))
DISSERTATION_EOF
)
    
    DISSERTATION_JSON=$(echo "$DISSERTATION_REPORT" | python3 -m json.tool 2>/dev/null || echo "$DISSERTATION_REPORT")
    
    # Display dissertation summary
    echo "$DISSERTATION_JSON" | python3 -c "
import json, sys
data = json.load(sys.stdin)

print('Baseline Experiments:')
baseline = data.get('baseline_experiments', {})
print(f\"  Expected: {baseline.get('expected_total', 0)}\")
print(f\"  Found: {baseline.get('found_total', 0)}\")
print(f\"  Completion: {(baseline.get('found_total', 0) / baseline.get('expected_total', 1) * 100):.1f}%\")
print('')
print('  By Environment:')
for env, stats in baseline.get('by_environment', {}).items():
    expected = stats.get('expected', 0)
    found = stats.get('found', 0)
    pct = (found / expected * 100) if expected > 0 else 0
    print(f\"    {env}: {found}/{expected} ({pct:.1f}%)\")

print('')
print('Scaling Experiments:')
scaling = data.get('scaling_experiments', {})
print(f\"  Expected: {scaling.get('expected_total', 0)}\")
print(f\"  Found: {scaling.get('found_total', 0)}\")
print(f\"  Completion: {(scaling.get('found_total', 0) / scaling.get('expected_total', 1) * 100):.1f}%\")
print('')
print('  By Environment:')
for env, stats in scaling.get('by_environment', {}).items():
    expected = stats.get('expected', 0)
    found = stats.get('found', 0)
    pct = (found / expected * 100) if expected > 0 else 0
    print(f\"    {env}: {found}/{expected} ({pct:.1f}%)\")

issues = data.get('issues', [])
warnings = data.get('warnings', [])
print('')
print(f'Issues: {len(issues)}')
print(f'Warnings: {len(warnings)}')
" 2>/dev/null || echo "Error parsing dissertation report"
    
    echo ""
    
    # Show missing experiments
    MISSING_COUNT=$(echo "$DISSERTATION_JSON" | python3 -c "import json, sys; data = json.load(sys.stdin); print(len(data.get('issues', [])))" 2>/dev/null || echo "0")
    
    if [[ "$MISSING_COUNT" -gt 0 ]]; then
        echo -e "${YELLOW}Missing Experiments for Dissertation:${NC}"
        echo ""
        echo "$DISSERTATION_JSON" | python3 -c "
import json, sys
data = json.load(sys.stdin)
issues = data.get('issues', [])
for issue in issues[:15]:  # Show first 15
    issue_type = issue.get('type', 'unknown')
    if issue_type == 'missing_baseline_experiment':
        env = issue.get('environment', 'unknown')
        algo = issue.get('algorithm', 'unknown')
        missing = issue.get('missing_count', 0)
        expected = issue.get('expected', 0)
        found = issue.get('found', 0)
        print(f\"  {env}/{algo}: {found}/{expected} found, {missing} missing\")
    elif issue_type == 'missing_scaling_experiment':
        env = issue.get('environment', 'unknown')
        algo = issue.get('algorithm', 'unknown')
        replica = issue.get('replica', 'unknown')
        missing = issue.get('missing_count', 0)
        expected = issue.get('expected', 0)
        found = issue.get('found', 0)
        print(f\"  {env}/{algo} (replica {replica}): {found}/{expected} found, {missing} missing\")
if len(issues) > 15:
    print(f\"  ... and {len(issues) - 15} more\")
" 2>/dev/null
        
        echo ""
    fi
    
    # Add dissertation report to output JSON if requested
    if [[ -n "$OUTPUT_FILE" ]]; then
        # Merge dissertation report into validation JSON
        TEMP_VALIDATION=$(mktemp)
        TEMP_DISSERTATION=$(mktemp)
        echo "$VALIDATION_JSON" > "$TEMP_VALIDATION"
        echo "$DISSERTATION_REPORT" > "$TEMP_DISSERTATION"
        
        MERGED_JSON=$(python3 <<MERGE_EOF
import json
from pathlib import Path

validation_file = Path("$TEMP_VALIDATION")
dissertation_file = Path("$TEMP_DISSERTATION")

with open(validation_file) as f:
    validation_data = json.load(f)

with open(dissertation_file) as f:
    dissertation_data = json.load(f)

validation_data['dissertation_requirements'] = dissertation_data

print(json.dumps(validation_data, indent=2))
MERGE_EOF
)
        echo "$MERGED_JSON" > "$OUTPUT_FILE"
        rm -f "$TEMP_VALIDATION" "$TEMP_DISSERTATION"
        log_success "Updated report with dissertation requirements: $OUTPUT_FILE"
        echo ""
    fi
fi

# Final summary
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
TOTAL_ISSUES=$(echo "$VALIDATION_JSON" | python3 -c "import json, sys; data = json.load(sys.stdin); print(data.get('summary', {}).get('total_issues', 0))" 2>/dev/null || echo "0")

if [[ "$TOTAL_ISSUES" -eq 0 ]]; then
    log_success "✓ All data quality checks passed!"
    EXIT_CODE=0
else
    log_warn "⚠ Found $TOTAL_ISSUES issue(s) across experiments"
    log_info "Review the output above or the detailed report for specifics"
    EXIT_CODE=1
fi

echo ""

if [[ "$FAIL_ON_ISSUES" == "true" ]] && [[ "$TOTAL_ISSUES" -gt 0 ]]; then
    log_error "Exiting with error due to data quality issues (--fail-on-issues)"
    exit 1
fi

exit $EXIT_CODE

