#!/usr/bin/env python3
"""
Regenerate index.json from existing results directories.

When data collection is done separately for each environment, this script
regenerates a combined index.json from all existing results directories.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

def main():
    # Handle script_dir: if absolute path, convert to relative for container
    # Container mounts project root as /workspace, so absolute paths need conversion
    script_dir_arg = sys.argv[1]
    if Path(script_dir_arg).is_absolute():
        # If we're in a container, the project root is /workspace
        # Try to detect if we're in container by checking if /workspace exists
        if Path("/workspace").exists():
            # We're in container - use /workspace as base
            script_dir = Path("/workspace")
        else:
            # We're on host - use the provided path
            script_dir = Path(script_dir_arg)
    else:
        # Relative path - resolve relative to current working directory
        script_dir = Path.cwd() / script_dir_arg
    
    output_dir = Path(sys.argv[2])
    matrix_file = sys.argv[3]
    envs = sys.argv[4].split(",")
    
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
        # Convert absolute path to relative if needed (for container)
        matrix_path = Path(matrix_file)
        if not matrix_path.is_absolute() or not matrix_path.exists():
            # Try relative to script_dir
            matrix_path = script_dir / matrix_file if not Path(matrix_file).is_absolute() else Path(matrix_file)
        with open(matrix_path) as f:
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
            # Support multiple structures:
            # 1. Direct raw file: exp_dir/raw/run.jsonl
            # 2. Run-based structure: exp_dir/run-*/raw/run.jsonl
            merged_file = exp_dir / "merged" / "merged.jsonl"
            stats_file = exp_dir / "stats" / "summary.json"
            raw_file = exp_dir / "raw" / "run.jsonl"
            
            # Check for run-based structure
            has_run_data = False
            for run_dir in exp_dir.glob("run-*/raw/run.jsonl"):
                if run_dir.exists():
                    has_run_data = True
                    break
            
            has_data = merged_file.exists() or stats_file.exists() or raw_file.exists() or has_run_data
            
            if not has_data:
                continue
            
            # Try to extract algorithm, payload, rate from experiment ID
            # Format: <algorithm>_p<payload>_r<rate>_<hash> (base experiment ID, without run_index)
            # Or: <algorithm>_p<payload>_r<rate>_run<N>_<hash> (legacy scenario ID with run_index)
            # Or: <algorithm>-smoketest-p<payload>-r<rate> (smoke test format)
            # Note: Output directories now use base experiment IDs (without run_index)
            # Each experiment handles multiple runs internally
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
            # For multi-run experiments, also check for aggregated stats or run-1 data
            aggregated_file = exp_dir / "aggregated_stats.json"
            run1_raw = exp_dir / "run-1" / "raw" / "run.jsonl"
            
            # Check for any run data
            has_any_run_data = False
            for run_dir in exp_dir.glob("run-*/raw/run.jsonl"):
                if run_dir.exists():
                    has_any_run_data = True
                    break
            
            has_stats = stats_file.exists() or merged_file.exists()
            has_aggregated = aggregated_file.exists()
            has_run1 = run1_raw.exists()
            
            if has_stats or has_aggregated or has_run1 or has_any_run_data or raw_file.exists():
                status = "success"
                index["completed_scenarios"] += 1
            else:
                status = "failed"
                index["failed_scenarios"] += 1
            
            experiment_entry = {
                "scenario_id": scenario_id,  # Note: This is now a base experiment ID (without run_index)
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
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
    index_file = output_dir / "index.json"
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"Index regenerated: {index_file}")
    print(f"Total experiments: {len(index['experiments'])}")
    print(f"Completed: {index['completed_scenarios']}")
    print(f"Failed: {index['failed_scenarios']}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: regenerate_index.py <script_dir> <output_dir> <matrix_file> <envs>", file=sys.stderr)
        sys.exit(1)
    main()
