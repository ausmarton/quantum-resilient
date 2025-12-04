#!/usr/bin/env python3
"""
Execute experiments multiple times for reproducibility analysis.

This script runs the same experiment N times via the orchestrator API,
storing results for subsequent variance and stability analysis.

Usage:
    python runner.py --scenario scenario.yaml --runs 20 --replicas 30 --exp-prefix kyber_test
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


class OrchestratorClient:
    """Client for the orchestrator API."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def create_experiment(self, scenario: dict, experiment_id: str) -> dict:
        """Create a new experiment."""
        response = self.session.post(
            f"{self.base_url}/experiment",
            json={"scenario": scenario, "experiment_id": experiment_id}
        )
        response.raise_for_status()
        return response.json()
    
    def start_experiment(self, experiment_id: str) -> dict:
        """Start an experiment."""
        response = self.session.post(
            f"{self.base_url}/experiment/{experiment_id}/start"
        )
        response.raise_for_status()
        return response.json()
    
    def get_experiment_status(self, experiment_id: str) -> dict:
        """Get experiment status."""
        response = self.session.get(
            f"{self.base_url}/experiment/{experiment_id}/status"
        )
        response.raise_for_status()
        return response.json()
    
    def collect_results(self, experiment_id: str) -> dict:
        """Collect experiment results."""
        response = self.session.post(
            f"{self.base_url}/experiment/{experiment_id}/collect"
        )
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(
        self,
        experiment_id: str,
        timeout: int = 3600,
        poll_interval: int = 10,
    ) -> bool:
        """Wait for experiment to complete."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_experiment_status(experiment_id)
            if status.get("status") == "completed":
                return True
            if status.get("status") == "failed":
                return False
            time.sleep(poll_interval)
        return False


def generate_experiment_id(prefix: str, run_index: int, timestamp: str) -> str:
    """Generate deterministic experiment ID."""
    return f"{prefix}_run{run_index:03d}_{timestamp}"


def load_scenario(scenario_path: Path) -> dict:
    """Load scenario from YAML file."""
    with open(scenario_path) as f:
        return yaml.safe_load(f)


def run_single_experiment(
    client: OrchestratorClient,
    scenario: dict,
    experiment_id: str,
    output_dir: Path,
    timeout: int = 3600,
) -> dict:
    """Execute a single experiment run."""
    result = {
        "experiment_id": experiment_id,
        "status": "pending",
        "start_time": datetime.now(timezone.utc).isoformat(),
    }
    
    try:
        # Create experiment
        client.create_experiment(scenario, experiment_id)
        result["created"] = True
        
        # Start experiment
        client.start_experiment(experiment_id)
        result["started"] = True
        
        # Wait for completion
        completed = client.wait_for_completion(experiment_id, timeout=timeout)
        
        if completed:
            # Collect results
            collect_result = client.collect_results(experiment_id)
            result["status"] = "completed"
            result["results_path"] = collect_result.get("path")
        else:
            result["status"] = "timeout"
        
    except requests.RequestException as e:
        result["status"] = "failed"
        result["error"] = str(e)
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    result["end_time"] = datetime.now(timezone.utc).isoformat()
    
    # Save result metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(result, f, indent=2)
    
    return result


def run_local_analysis(
    run_index: int,
    data_dir: Path,
    output_dir: Path,
    analysis_scripts: Path,
) -> bool:
    """Run local analysis pipeline on collected data."""
    try:
        # Merge JSONL
        merged_dir = output_dir / "merged"
        merged_dir.mkdir(exist_ok=True)
        
        subprocess.run(
            [
                sys.executable,
                str(analysis_scripts / "merge_jsonl.py"),
                "--input", str(data_dir),
                "--output", str(merged_dir),
            ],
            check=True,
            capture_output=True,
        )
        
        # Compute statistics
        stats_dir = output_dir / "stats"
        stats_dir.mkdir(exist_ok=True)
        
        merged_file = merged_dir / "merged.jsonl"
        if not merged_file.exists():
            merged_file = merged_dir / "merged.parquet"
        
        subprocess.run(
            [
                sys.executable,
                str(analysis_scripts / "compute_statistics.py"),
                "--input", str(merged_file),
                "--output", str(stats_dir),
                "--experiment-id", f"run_{run_index:03d}",
            ],
            check=True,
            capture_output=True,
        )
        
        return True
    except subprocess.CalledProcessError:
        return False


def run_experiments(
    scenario_path: Path,
    num_runs: int,
    replicas: int,
    exp_prefix: str,
    output_base: Path,
    orchestrator_url: str = "http://localhost:8080",
    parallel: int = 1,
    timeout: int = 3600,
    retry_failed: bool = True,
    max_retries: int = 2,
) -> dict:
    """Run multiple experiment iterations."""
    console.print(f"[bold blue]Reproducibility Test Suite[/bold blue]")
    console.print(f"  Scenario: {scenario_path}")
    console.print(f"  Runs: {num_runs}")
    console.print(f"  Replicas: {replicas}")
    console.print(f"  Prefix: {exp_prefix}")
    console.print(f"  Output: {output_base}")
    
    # Load scenario
    scenario = load_scenario(scenario_path)
    scenario["replicas"] = replicas
    scenario_name = scenario_path.stem
    
    # Generate timestamp for this batch
    batch_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    batch_id = f"{exp_prefix}_{batch_timestamp}"
    
    output_dir = output_base / batch_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save batch metadata
    batch_metadata = {
        "batch_id": batch_id,
        "scenario": str(scenario_path),
        "scenario_name": scenario_name,
        "num_runs": num_runs,
        "replicas": replicas,
        "exp_prefix": exp_prefix,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "runs": [],
    }
    
    # Check if orchestrator is available
    client = OrchestratorClient(orchestrator_url)
    use_orchestrator = True
    
    try:
        requests.get(f"{orchestrator_url}/health", timeout=5)
    except requests.RequestException:
        console.print("[yellow]Orchestrator not available, running in local mode[/yellow]")
        use_orchestrator = False
    
    results = []
    
    if use_orchestrator and parallel > 1:
        # Parallel execution via orchestrator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running experiments...", total=num_runs)
            
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {}
                
                for i in range(num_runs):
                    exp_id = generate_experiment_id(exp_prefix, i, batch_timestamp)
                    run_output = output_dir / f"run_{i:03d}"
                    
                    future = executor.submit(
                        run_single_experiment,
                        client,
                        scenario,
                        exp_id,
                        run_output,
                        timeout,
                    )
                    futures[future] = i
                
                for future in as_completed(futures):
                    run_index = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append({
                            "run_index": run_index,
                            "status": "error",
                            "error": str(e),
                        })
                    progress.update(task, advance=1)
    else:
        # Sequential execution
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running experiments...", total=num_runs)
            
            for i in range(num_runs):
                exp_id = generate_experiment_id(exp_prefix, i, batch_timestamp)
                run_output = output_dir / f"run_{i:03d}"
                progress.update(task, description=f"Run {i+1}/{num_runs}")
                
                if use_orchestrator:
                    result = run_single_experiment(
                        client, scenario, exp_id, run_output, timeout
                    )
                else:
                    # Local/simulation mode - just create directories
                    run_output.mkdir(parents=True, exist_ok=True)
                    result = {
                        "run_index": i,
                        "experiment_id": exp_id,
                        "status": "simulated",
                        "output_dir": str(run_output),
                    }
                    with open(run_output / "run_metadata.json", "w") as f:
                        json.dump(result, f, indent=2)
                
                results.append(result)
                
                # Retry failed runs
                if result.get("status") == "failed" and retry_failed:
                    for retry in range(max_retries):
                        console.print(f"  [yellow]Retry {retry + 1} for run {i}[/yellow]")
                        result = run_single_experiment(
                            client, scenario, exp_id, run_output, timeout
                        )
                        if result.get("status") == "completed":
                            results[-1] = result
                            break
                
                progress.update(task, advance=1)
    
    # Update batch metadata
    batch_metadata["runs"] = results
    batch_metadata["end_time"] = datetime.now(timezone.utc).isoformat()
    batch_metadata["completed"] = sum(1 for r in results if r.get("status") == "completed")
    batch_metadata["failed"] = sum(1 for r in results if r.get("status") in ["failed", "error"])
    
    # Save batch metadata
    with open(output_dir / "batch_metadata.json", "w") as f:
        json.dump(batch_metadata, f, indent=2)
    
    # Summary
    completed = batch_metadata["completed"]
    failed = batch_metadata["failed"]
    
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Completed: {completed}/{num_runs}")
    if failed > 0:
        console.print(f"  [red]Failed: {failed}[/red]")
    console.print(f"  Output: {output_dir}")
    
    return batch_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Run reproducibility experiments"
    )
    parser.add_argument(
        "--scenario",
        required=True,
        type=Path,
        help="Path to scenario YAML file"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of experiment runs (default: 10)"
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=10,
        help="Number of worker replicas per run (default: 10)"
    )
    parser.add_argument(
        "--exp-prefix",
        default="repro",
        help="Experiment ID prefix"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reproducibility/output"),
        help="Output directory"
    )
    parser.add_argument(
        "--orchestrator-url",
        default="http://localhost:8080",
        help="Orchestrator API URL"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel runs (default: 1)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout per run in seconds (default: 3600)"
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Don't retry failed runs"
    )
    
    args = parser.parse_args()
    
    result = run_experiments(
        scenario_path=args.scenario,
        num_runs=args.runs,
        replicas=args.replicas,
        exp_prefix=args.exp_prefix,
        output_base=args.out,
        orchestrator_url=args.orchestrator_url,
        parallel=args.parallel,
        timeout=args.timeout,
        retry_failed=not args.no_retry,
    )
    
    sys.exit(0 if result["failed"] == 0 else 1)


if __name__ == "__main__":
    main()

