#!/usr/bin/env bash
# =============================================================================
# validate_dissertation_data.sh - Validate experiment data for analysis
#
# Validates all collected experiment data against project requirements:
# - Data format compliance (all required fields)
# - Data quality (no errors, valid values)
# - Completeness (all runs, all experiments)
# - Statistical validity (sufficient runs per configuration)
# - Readiness for analysis (all required data points present)
#
# Usage:
#   ./scripts/validate_dissertation_data.sh [OPTIONS]
#
# Options:
#   --env ENV              Check specific environment (native, minikube, gcp)
#   --results-dir DIR      Results directory (default: results/)
#   --matrix PATH          Experiment matrix YAML (default: orchestration/experiment_matrix.yaml)
#   --output FILE          Write detailed report to JSON file
#   --fail-on-issues       Exit with error if any issues found
#   --list-unusable        List experiments that need to be re-run
#   --check-claims         Validate data supports all required analysis (default: true)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
MATRIX="$SCRIPT_DIR/orchestration/experiment_matrix.yaml"
ENV_FILTER=""
OUTPUT_FILE=""
FAIL_ON_ISSUES=false
LIST_UNUSABLE=false
CHECK_CLAIMS=true

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

Validate experiment data completeness and quality for analysis.

OPTIONS:
    --env ENV              Check specific environment (native, minikube, gcp)
    --results-dir DIR      Results directory (default: results/)
    --matrix PATH          Experiment matrix YAML (default: orchestration/experiment_matrix.yaml)
    --output FILE          Write detailed report to JSON file
    --fail-on-issues       Exit with error if any issues found
    --list-unusable        List experiments that need to be re-run
    --check-claims         Validate data supports all required analysis (default: true)
    -h, --help             Show this help message

EXAMPLES:
    # Full validation
    ./scripts/validate_dissertation_data.sh

    # Check only GCP data
    ./scripts/validate_dissertation_data.sh --env gcp

    # Generate report and list unusable experiments
    ./scripts/validate_dissertation_data.sh --output report.json --list-unusable
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
        --list-unusable)
            LIST_UNUSABLE=true
            shift
            ;;
        --check-claims)
            CHECK_CLAIMS=true
            shift
            ;;
        --matrix)
            MATRIX="$2"
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

# Python script for comprehensive validation
PYTHON_SCRIPT=$(cat <<'PYTHON_EOF'
import json
import yaml
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional
import statistics

# Required fields from REQUIREMENTS_SPECIFICATION.md
REQUIRED_FIELDS = [
    "run_id", "scenario_id", "event_id",
    "timestamp_utc_iso", "timestamp_monotonic_ns",
    "operation", "algorithm",
    "latency_ns", "payload_size_bytes",
    "cpu_user_seconds", "memory_rss_bytes",
    "rng_seed"
]

# Optional but important fields
OPTIONAL_FIELDS = [
    "queue_delay_ns", "worker_id",
    "ciphertext_size_bytes", "signature_size_bytes",
    "error"
]

def validate_dissertation_data(results_dir: Path, matrix_file: Path, env_filter: str = "", 
                                check_claims: bool = True) -> Dict[str, Any]:
    """Validate experiment data completeness and quality for analysis."""
    
    report = {
        "timestamp": "",
        "environment": env_filter or "all",
        "summary": {
            "total_experiments": 0,
            "total_runs": 0,
            "total_events": 0,
            "experiments_with_errors": 0,
            "experiments_complete": 0,
            "experiments_incomplete": 0,
            "fit_for_purpose": False
        },
        "field_coverage": {},
        "error_analysis": {},
        "algorithm_coverage": {},
        "completeness": {},
        "statistical_validity": {},
        "claims_support": {},
        "issues": [],
        "unusable_experiments": []
    }
    
    # Load experiment matrix
    try:
        with open(matrix_file) as f:
            matrix = yaml.safe_load(f)
    except Exception as e:
        report["issues"].append(f"Failed to load experiment matrix: {e}")
        return report
    
    # Find all raw JSONL files
    if env_filter:
        search_path = results_dir / env_filter
    else:
        search_path = results_dir
    
    # Support both directory structures:
    # - results/env/experiment/raw/run.jsonl (native)
    # - results/env/experiment/run-X/raw/run.jsonl (GCP)
    raw_files = []
    # Find all run.jsonl files, avoiding duplicates
    seen_files = set()
    for pattern in ["*/raw/run.jsonl", "*/run-*/raw/run.jsonl"]:
        for f in search_path.rglob(pattern):
            if f not in seen_files:
                raw_files.append(f)
                seen_files.add(f)
    
    if not raw_files:
        report["issues"].append(f"No raw JSONL files found in {search_path}")
        return report
    
    # Group experiments
    experiments = defaultdict(lambda: {
        "runs": [],
        "total_events": 0,
        "events_with_errors": 0,
        "events_total": 0,
        "has_errors": False,
        "algorithms": set(),
        "operations": set(),
        "sample": None,
        "field_coverage": Counter(),
        "error_types": Counter(),
        "run_details": []  # Track each run's status
    })
    
    field_coverage = Counter()
    error_patterns = Counter()
    all_algorithms = Counter()
    all_operations = Counter()
    total_events_all = 0
    total_events_with_errors = 0
    
    for jsonl_file in raw_files:
        # Extract experiment name
        if "run-" in str(jsonl_file):
            # GCP structure: experiment/run-X/raw/run.jsonl
            exp_name = jsonl_file.parent.parent.parent.name
            run_name = jsonl_file.parent.parent.name
        else:
            # Native structure: experiment/raw/run.jsonl
            exp_name = jsonl_file.parent.parent.name
            run_name = "run-1"  # Default for native
        
        # Avoid duplicate runs
        if run_name not in experiments[exp_name]["runs"]:
            experiments[exp_name]["runs"].append(run_name)
        
        try:
            with open(jsonl_file) as f:
                # Check first event
                first_line = f.readline()
                if not first_line.strip():
                    continue
                    
                event = json.loads(first_line)
                
                if experiments[exp_name]["sample"] is None:
                    experiments[exp_name]["sample"] = event
                    experiments[exp_name]["algorithms"].add(event.get("algorithm", "unknown"))
                    experiments[exp_name]["operations"].add(event.get("operation", "unknown"))
                
                # Check required fields
                for field in REQUIRED_FIELDS:
                    if field in event:
                        field_coverage[field] += 1
                        experiments[exp_name]["field_coverage"][field] += 1
                    else:
                        report["issues"].append(f"Missing required field '{field}' in {exp_name}")
                
                # Count all events and check for errors
                f.seek(0)
                events_in_file = []
                events_with_errors_in_file = 0
                for line in f:
                    if line.strip():
                        try:
                            evt = json.loads(line)
                            events_in_file.append(evt)
                            error = evt.get("error")
                            if error:
                                experiments[exp_name]["has_errors"] = True
                                experiments[exp_name]["events_with_errors"] += 1
                                experiments[exp_name]["error_types"][error] += 1
                                error_patterns[error] += 1
                                events_with_errors_in_file += 1
                                total_events_with_errors += 1
                        except:
                            pass
                
                event_count = len(events_in_file)
                experiments[exp_name]["total_events"] += event_count
                experiments[exp_name]["events_total"] += event_count
                total_events_all += event_count
                
                # Track run details
                run_has_errors = events_with_errors_in_file > 0
                experiments[exp_name]["run_details"].append({
                    "run": run_name,
                    "events": event_count,
                    "events_with_errors": events_with_errors_in_file,
                    "has_errors": run_has_errors,
                    "error_rate": (events_with_errors_in_file / event_count * 100) if event_count > 0 else 0
                })
                
                # Track algorithms and operations
                if "algorithm" in event:
                    all_algorithms[event["algorithm"]] += 1
                if "operation" in event:
                    all_operations[event["operation"]] += 1
                
        except json.JSONDecodeError as e:
            report["issues"].append(f"Invalid JSON in {jsonl_file}: {e}")
        except Exception as e:
            report["issues"].append(f"Error processing {jsonl_file}: {e}")
    
    # Field coverage
    report["field_coverage"] = {
        field: {
            "present": field_coverage[field],
            "total": len(raw_files),
            "coverage_pct": (field_coverage[field] / len(raw_files) * 100) if raw_files else 0,
            "status": "✅" if field_coverage[field] == len(raw_files) else "⚠️"
        }
        for field in REQUIRED_FIELDS
    }
    
    # Error analysis with detailed statistics
    exps_with_errors = {k: v for k, v in experiments.items() if v["has_errors"]}
    exps_without_errors = {k: v for k, v in experiments.items() if not v["has_errors"]}
    
    # Calculate error rates
    total_runs_all = len(raw_files)
    total_runs_with_errors = sum(
        sum(1 for r in v["run_details"] if r["has_errors"]) 
        for v in experiments.values()
    )
    
    report["error_analysis"] = {
        "total_experiments_with_errors": len(exps_with_errors),
        "total_experiments_without_errors": len(exps_without_errors),
        "experiment_error_rate_pct": (len(exps_with_errors) / len(experiments) * 100) if experiments else 0,
        "total_runs_with_errors": total_runs_with_errors,
        "total_runs_without_errors": total_runs_all - total_runs_with_errors,
        "run_error_rate_pct": (total_runs_with_errors / total_runs_all * 100) if total_runs_all > 0 else 0,
        "total_events_with_errors": total_events_with_errors,
        "total_events_all": total_events_all,
        "event_error_rate_pct": (total_events_with_errors / total_events_all * 100) if total_events_all > 0 else 0,
        "error_patterns": dict(error_patterns),
        "error_patterns_detailed": {
            error: {
                "count": count,
                "experiments_affected": len([exp for exp, data in experiments.items() 
                                            if data["has_errors"] and error in data["error_types"]]),
                "experiments": [exp for exp, data in experiments.items() 
                               if data["has_errors"] and error in data["error_types"]]
            }
            for error, count in error_patterns.items()
        },
        "experiments_by_error": {
            error: [exp for exp, data in experiments.items() 
                   if data["has_errors"] and error in data["error_types"]]
            for error in error_patterns.keys()
        }
    }
    
    # Algorithm/operation coverage
    report["algorithm_coverage"] = {
        algo: {
            "runs": count,
            "experiments": len([e for e, d in experiments.items() 
                               if algo in d["algorithms"]])
        }
        for algo, count in sorted(all_algorithms.items())
    }
    
    report["operation_coverage"] = {
        op: {
            "runs": count,
            "experiments": len([e for e, d in experiments.items() 
                               if op in d["operations"]])
        }
        for op, count in sorted(all_operations.items())
    }
    
    # Completeness analysis
    if check_claims:
        # Build expected experiments from matrix
        expected_configs = defaultdict(lambda: {"runs": 0, "experiments": set()})
        
        if "experiments" in matrix:
            for exp in matrix["experiments"]:
                algo = exp.get("algorithm", "unknown")
                operation = exp.get("operation", "unknown")
                payload_sizes = exp.get("payload_sizes", [])
                rates = exp.get("rates", [])
                runs = exp.get("runs", 5)
                
                patterns = exp.get("workload_patterns", ["constant"])
                
                for payload in payload_sizes:
                    for rate in rates:
                        for pattern in patterns:
                            config_key = f"{algo}_{operation}_p{payload}_r{rate}_{pattern}"
                            expected_configs[algo]["runs"] += runs
                            expected_configs[algo]["experiments"].add(config_key)
        
        # Check scaling experiments
        if "scaling" in matrix:
            scaling = matrix["scaling"]
            for algo in scaling.get("scaling_algorithms", []):
                replicas = scaling.get("replicas", [])
                runs = scaling.get("scaling_runs", 3)
                expected_configs[algo]["runs"] += len(replicas) * runs
        
        # Compare expected vs actual
        actual_configs = defaultdict(lambda: {"runs": 0, "experiments": set()})
        for exp_name, data in experiments.items():
            parts = exp_name.split("_")
            if len(parts) >= 2:
                algo = parts[0]
                actual_configs[algo]["runs"] += len(data["runs"])
                actual_configs[algo]["experiments"].add(exp_name)
        
        report["completeness"] = {}
        for algo in set(list(expected_configs.keys()) + list(actual_configs.keys())):
            exp_runs = expected_configs.get(algo, {}).get("runs", 0)
            act_runs = actual_configs.get(algo, {}).get("runs", 0)
            exp_exps = len(expected_configs.get(algo, {}).get("experiments", set()))
            act_exps = len(actual_configs.get(algo, {}).get("experiments", set()))
            
            if exp_runs > 0:
                report["completeness"][algo] = {
                    "expected_runs": exp_runs,
                    "actual_runs": act_runs,
                    "coverage_pct": (act_runs / exp_runs * 100) if exp_runs > 0 else 0,
                    "expected_experiments": exp_exps,
                    "actual_experiments": act_exps,
                    "status": "✅" if act_runs >= exp_runs else "⚠️"
                }
    
    # Statistical validity with expected run counts
    # Determine expected runs: 5 for baseline, 3 for scaling and 5-minute sustained load
    expected_runs_baseline = 5
    expected_runs_scaling = 3
    expected_runs_5m = 3  # 5-minute sustained load experiments
    
    experiments_by_run_count = defaultdict(list)
    incomplete_experiments = []
    complete_experiments = []
    
    for exp_name, exp_data in experiments.items():
        run_count = len(exp_data["runs"])
        experiments_by_run_count[run_count].append(exp_name)
        
        # Determine if scaling, 5-minute sustained load, or baseline
        is_scaling = "scaling" in exp_name.lower()
        is_5m = "_5m_" in exp_name or exp_name.endswith("_5m") or "5m_" in exp_name
        if is_scaling or is_5m:
            expected_runs = expected_runs_scaling
        else:
            expected_runs = expected_runs_baseline
        
        if run_count < expected_runs:
            incomplete_experiments.append({
                "experiment": exp_name,
                "actual_runs": run_count,
                "expected_runs": expected_runs,
                "missing_runs": expected_runs - run_count,
                "is_scaling": is_scaling,
                "has_errors": exp_data["has_errors"],
                "runs": exp_data["runs"]
            })
        elif run_count >= expected_runs and not exp_data["has_errors"]:
            complete_experiments.append(exp_name)
    
    run_counts = Counter(len(e["runs"]) for e in experiments.values())
    
    report["statistical_validity"] = {
        "run_distribution": dict(run_counts),
        "experiments_by_run_count": {str(k): v for k, v in experiments_by_run_count.items()},
        "experiments_with_5_runs": sum(1 for e in experiments.values() if len(e["runs"]) == 5),
        "experiments_with_3_runs": sum(1 for e in experiments.values() if len(e["runs"]) == 3),
        "experiments_incomplete": len(incomplete_experiments),
        "experiments_complete": len(complete_experiments),
        "total_experiments": len(experiments),
        "expected_runs_baseline": expected_runs_baseline,
        "expected_runs_scaling": expected_runs_scaling,
        "incomplete_experiments": incomplete_experiments
    }
    
    # Claims support analysis
    if check_claims:
        report["claims_support"] = {
            "algorithm_comparison": {
                "supported": len([e for e, d in experiments.items() 
                                if not d["has_errors"] and "sign" in d["operations"] or "kem_aead_encrypt" in d["operations"]]),
                "unsupported": len([e for e, d in experiments.items() if d["has_errors"]]),
                "algorithms_working": [algo for algo, data in report["algorithm_coverage"].items() 
                                      if data["runs"] > 0 and algo not in [e for err_list in report["error_analysis"]["experiments_by_error"].values() 
                                                                          for e in err_list]]
            },
            "statistical_rigor": {
                "sufficient_runs": report["statistical_validity"]["experiments_with_5_runs"],
                "incomplete_runs": report["statistical_validity"]["experiments_incomplete"]
            },
            "resource_utilization": {
                "cpu_data_available": all(report["field_coverage"][f]["status"] == "✅" 
                                         for f in ["cpu_user_seconds"]),
                "memory_data_available": all(report["field_coverage"][f]["status"] == "✅" 
                                            for f in ["memory_rss_bytes"])
            },
            "queue_delay_analysis": {
                "queue_delay_field_present": "queue_delay_ns" in [f for e in experiments.values() 
                                                                  if e["sample"] and "queue_delay_ns" in e["sample"]]
            }
        }
    
    # Identify unusable experiments (with detailed reasons)
    report["unusable_experiments"] = []
    unusable_by_reason = defaultdict(list)
    
    for exp_name, data in experiments.items():
        reasons = []
        # Determine if scaling, 5-minute sustained load, or baseline
        is_scaling = "scaling" in exp_name.lower()
        is_5m = "_5m_" in exp_name or exp_name.endswith("_5m") or "5m_" in exp_name
        if is_scaling or is_5m:
            expected_runs = expected_runs_scaling
        else:
            expected_runs = expected_runs_baseline
        run_count = len(data["runs"])
        
        if data["has_errors"]:
            error_rate = (data["events_with_errors"] / data["events_total"] * 100) if data["events_total"] > 0 else 0
            reasons.append(f"Has errors ({error_rate:.1f}% error rate)")
            unusable_by_reason["has_errors"].append(exp_name)
        
        if run_count < expected_runs:
            reasons.append(f"Insufficient runs ({run_count}/{expected_runs})")
            if not data["has_errors"]:
                unusable_by_reason["insufficient_runs"].append(exp_name)
        
        if reasons:
            report["unusable_experiments"].append({
                "experiment": exp_name,
                "runs": data["runs"],
                "run_count": run_count,
                "expected_runs": expected_runs,
                "is_scaling": is_scaling,
                "error_types": dict(data["error_types"]),
                "events_with_errors": data["events_with_errors"],
                "events_total": data["events_total"],
                "error_rate_pct": (data["events_with_errors"] / data["events_total"] * 100) if data["events_total"] > 0 else 0,
                "total_events": data["total_events"],
                "reasons": reasons,
                "primary_reason": reasons[0]
            })
    
    report["unusable_by_reason"] = dict(unusable_by_reason)
    
    # Enhanced summary with percentages
    total_exps = len(experiments)
    exps_with_errors_count = len(exps_with_errors)
    exps_complete_count = len(complete_experiments)
    exps_incomplete_count = len(incomplete_experiments)
    unusable_count = len(report["unusable_experiments"])
    
    report["summary"] = {
        "total_experiments": total_exps,
        "total_runs": len(raw_files),
        "total_events": sum(e["total_events"] for e in experiments.values()),
        "experiments_with_errors": exps_with_errors_count,
        "experiments_without_errors": len(exps_without_errors),
        "experiments_complete": exps_complete_count,
        "experiments_incomplete": exps_incomplete_count,
        "experiments_unusable": unusable_count,
        "experiments_usable": total_exps - unusable_count,
        "experiment_success_rate_pct": ((total_exps - exps_with_errors_count) / total_exps * 100) if total_exps > 0 else 0,
        "experiment_completeness_pct": (exps_complete_count / total_exps * 100) if total_exps > 0 else 0,
        "experiment_usability_pct": ((total_exps - unusable_count) / total_exps * 100) if total_exps > 0 else 0,
        "fit_for_purpose": (
            len(exps_with_errors) == 0 and
            all(f["coverage_pct"] == 100 for f in report["field_coverage"].values()) and
            exps_incomplete_count == 0
        )
    }
    
    return report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate experiment data for analysis")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--matrix", type=str, default="orchestration/experiment_matrix.yaml")
    parser.add_argument("--env", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--check-claims", action="store_true", default=True)
    parser.add_argument("--list-unusable", action="store_true", default=False)
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    matrix_file = Path(args.matrix)
    
    report = validate_dissertation_data(results_dir, matrix_file, args.env, args.check_claims)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.output}")
    else:
        # Print detailed summary
        print("=" * 80)
        print("DATA VALIDATION REPORT")
        print("=" * 80)
        
        summary = report['summary']
        print(f"\n📊 OVERALL STATISTICS")
        print(f"  Total Experiments: {summary['total_experiments']}")
        print(f"  Total Runs: {summary['total_runs']}")
        print(f"  Total Events: {summary['total_events']:,}")
        
        print(f"\n✅ SUCCESS RATES")
        success_rate = summary.get('experiment_success_rate_pct', 0)
        error_rate = 100 - success_rate
        print(f"  Experiments without errors: {summary['experiments_without_errors']} ({success_rate:.1f}% success)")
        print(f"  Experiments with errors: {summary['experiments_with_errors']} ({error_rate:.1f}% error rate)")
        print(f"  Usable experiments: {summary['experiments_usable']} ({summary.get('experiment_usability_pct', 0):.1f}% usable)")
        print(f"  Unusable experiments: {summary['experiments_unusable']} ({100 - summary.get('experiment_usability_pct', 0):.1f}% unusable)")
        
        print(f"\n📈 COMPLETENESS")
        print(f"  Complete experiments: {summary['experiments_complete']} ({summary.get('experiment_completeness_pct', 0):.1f}%)")
        print(f"  Incomplete experiments: {summary['experiments_incomplete']} ({100 - summary.get('experiment_completeness_pct', 0):.1f}%)")
        
        # Error analysis
        error_analysis = report.get('error_analysis', {})
        if error_analysis:
            print(f"\n🔍 ERROR ANALYSIS")
            print(f"  Event error rate: {error_analysis.get('event_error_rate_pct', 0):.2f}% ({error_analysis.get('total_events_with_errors', 0):,} / {error_analysis.get('total_events_all', 0):,} events)")
            print(f"  Run error rate: {error_analysis.get('run_error_rate_pct', 0):.2f}% ({error_analysis.get('total_runs_with_errors', 0)} / {error_analysis.get('total_runs_with_errors', 0) + error_analysis.get('total_runs_without_errors', 0)} runs)")
            
            error_patterns = error_analysis.get('error_patterns_detailed', {})
            if error_patterns:
                print(f"\n  Error Types:")
                for error_type, details in sorted(error_patterns.items(), key=lambda x: x[1]['count'], reverse=True):
                    print(f"    • {error_type}:")
                    print(f"      - Occurrences: {details['count']}")
                    print(f"      - Experiments affected: {details['experiments_affected']}")
                    if details['experiments_affected'] <= 10:
                        for exp in details['experiments'][:5]:
                            print(f"        - {exp}")
                        if len(details['experiments']) > 5:
                            print(f"        ... and {len(details['experiments']) - 5} more")
        
        # Statistical validity
        stat_validity = report.get('statistical_validity', {})
        if stat_validity:
            print(f"\n📊 STATISTICAL VALIDITY")
            print(f"  Experiments with 5 runs: {stat_validity.get('experiments_with_5_runs', 0)} (baseline expected)")
            print(f"  Experiments with 3 runs: {stat_validity.get('experiments_with_3_runs', 0)} (scaling expected)")
            print(f"  Incomplete experiments: {stat_validity.get('experiments_incomplete', 0)}")
            
            run_dist = stat_validity.get('run_distribution', {})
            if run_dist:
                print(f"\n  Run Distribution:")
                for run_count in sorted(run_dist.keys(), reverse=True):
                    count = run_dist[run_count]
                    print(f"    {run_count} runs: {count} experiments")
        
        # Incomplete experiments
        incomplete = stat_validity.get('incomplete_experiments', [])
        if incomplete:
            print(f"\n⚠️  INCOMPLETE EXPERIMENTS ({len(incomplete)})")
            for exp in incomplete[:10]:
                print(f"  • {exp['experiment']}: {exp['actual_runs']}/{exp['expected_runs']} runs (missing {exp['missing_runs']})")
                if exp.get('has_errors'):
                    print(f"    Also has errors!")
            if len(incomplete) > 10:
                print(f"  ... and {len(incomplete) - 10} more incomplete experiments")
        
        # Algorithm coverage
        algo_coverage = report.get('algorithm_coverage', {})
        if algo_coverage:
            print(f"\n🔬 ALGORITHM COVERAGE")
            for algo, data in sorted(algo_coverage.items()):
                print(f"  {algo}: {data.get('experiments', 0)} experiments, {data.get('runs', 0)} runs")
        
        # Operation coverage
        op_coverage = report.get('operation_coverage', {})
        if op_coverage:
            print(f"\n⚙️  OPERATION COVERAGE")
            for op, data in sorted(op_coverage.items()):
                print(f"  {op}: {data.get('experiments', 0)} experiments, {data.get('runs', 0)} runs")
        
        # Field coverage summary
        field_cov = report.get('field_coverage', {})
        if field_cov:
            missing_fields = [f for f, data in field_cov.items() if data.get('coverage_pct', 0) < 100]
            if missing_fields:
                print(f"\n⚠️  MISSING FIELDS")
                for field in missing_fields:
                    data = field_cov[field]
                    print(f"  {field}: {data.get('coverage_pct', 0):.1f}% coverage ({data.get('present', 0)}/{data.get('total', 0)})")
            else:
                print(f"\n✅ FIELD COVERAGE: 100% (all required fields present)")
        
        # Unusable experiments
        unusable = report.get('unusable_experiments', [])
        if unusable:
            print(f"\n❌ UNUSABLE EXPERIMENTS ({len(unusable)})")
            print("=" * 80)
            
            # Group by reason
            by_reason = {}
            for exp in unusable:
                reason = exp.get('primary_reason', 'Unknown')
                if reason not in by_reason:
                    by_reason[reason] = []
                by_reason[reason].append(exp)
            
            for reason, exps in sorted(by_reason.items(), key=lambda x: len(x[1]), reverse=True):
                print(f"\n{reason} ({len(exps)} experiments):")
                for exp in exps[:15]:
                    print(f"  • {exp['experiment']}")
                    print(f"    Runs: {exp.get('run_count', len(exp.get('runs', [])))}/{exp.get('expected_runs', '?')} | Events: {exp.get('total_events', 0):,}")
                    if exp.get('error_rate_pct', 0) > 0:
                        print(f"    Error rate: {exp['error_rate_pct']:.1f}% ({exp.get('events_with_errors', 0):,}/{exp.get('events_total', 0):,} events)")
                    if exp.get('error_types'):
                        print(f"    Error types: {', '.join(exp['error_types'].keys())}")
                if len(exps) > 15:
                    print(f"  ... and {len(exps) - 15} more")
        else:
            print(f"\n✅ NO UNUSABLE EXPERIMENTS")
        
        print(f"\n{'=' * 80}")
        print(f"FIT FOR PURPOSE: {'✅ YES' if report['summary']['fit_for_purpose'] else '❌ NO'}")
        print(f"{'=' * 80}")
        
        # Recommendations
        if not report['summary']['fit_for_purpose']:
            print(f"\n📋 RECOMMENDATIONS:")
            if summary['experiments_with_errors'] > 0:
                print(f"  1. Fix code issues causing errors and re-run {summary['experiments_with_errors']} experiments with errors")
            if summary['experiments_incomplete'] > 0:
                print(f"  2. Complete {summary['experiments_incomplete']} incomplete experiments (missing runs)")
            if missing_fields:
                print(f"  3. Investigate missing fields: {', '.join(missing_fields)}")
            print(f"\n  Use --list-unusable to see detailed list of experiments needing re-run")
            print(f"  Use remove_unusable_data.sh to remove unusable data before re-running")
        
        # Exit with error if issues found
        if not report['summary']['fit_for_purpose']:
            sys.exit(1)
PYTHON_EOF
)

# Run validation
log_info "Starting data validation..."
log_info "Results directory: $RESULTS_DIR"
log_info "Experiment matrix: $MATRIX"

if [[ -n "$ENV_FILTER" ]]; then
    log_info "Environment filter: $ENV_FILTER"
fi

# Create temporary Python script
TMP_SCRIPT=$(mktemp)
echo "$PYTHON_SCRIPT" > "$TMP_SCRIPT"

# Run Python validation
if [[ -n "$OUTPUT_FILE" ]]; then
    python3 "$TMP_SCRIPT" \
        --results-dir "$RESULTS_DIR" \
        --matrix "$MATRIX" \
        --env "$ENV_FILTER" \
        --check-claims \
        --output "$OUTPUT_FILE" \
        $([[ "$LIST_UNUSABLE" == "true" ]] && echo "--list-unusable")
    VALIDATION_EXIT=$?
else
    python3 "$TMP_SCRIPT" \
        --results-dir "$RESULTS_DIR" \
        --matrix "$MATRIX" \
        --env "$ENV_FILTER" \
        --check-claims \
        $([[ "$LIST_UNUSABLE" == "true" ]] && echo "--list-unusable")
    VALIDATION_EXIT=$?
fi

# Cleanup
rm -f "$TMP_SCRIPT"

if [[ "$FAIL_ON_ISSUES" == "true" ]] && [[ $VALIDATION_EXIT -ne 0 ]]; then
    log_error "Validation failed - issues found"
    exit 1
fi

exit $VALIDATION_EXIT
