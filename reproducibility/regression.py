#!/usr/bin/env python3
"""
Regression detection for benchmark results.

Compares current results against:
- Previous run batches
- Reference baselines
- Algorithm comparisons

Signals regression if metrics exceed thresholds.

Usage:
    python regression.py --current batch_002 --baseline batch_001 --out analysis/
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console

console = Console()


# Default thresholds (as percentages)
DEFAULT_THRESHOLDS = {
    "latency_mean": 10.0,      # 10% increase
    "latency_p50": 10.0,
    "latency_p99": 15.0,       # Allow more variance in tail
    "latency_p999": 20.0,
    "throughput_mean": -10.0,  # 10% decrease (negative = bad if lower)
    "throughput_max": -15.0,
    "variance": 50.0,          # 50% increase in variance
}


def load_run_statistics(run_dir: Path) -> Optional[dict]:
    """Load statistics from a single run."""
    stats_path = run_dir / "stats" / "summary.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    return None


def load_batch_summary(batch_dir: Path) -> Optional[dict]:
    """Load summary from a batch analysis."""
    summary_path = batch_dir / "analysis" / "variance_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    
    # Try to compute from runs
    metrics_list = []
    for run_dir in sorted(batch_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        
        run_stats = load_run_statistics(run_dir)
        if run_stats:
            metrics_list.append({
                "latency_mean": run_stats.get("latency", {}).get("mean", 0),
                "latency_p50": run_stats.get("latency", {}).get("p50", 0),
                "latency_p99": run_stats.get("latency", {}).get("p99", 0),
                "latency_p999": run_stats.get("latency", {}).get("p999", 0),
                "throughput_mean": run_stats.get("throughput", {}).get("mean_msgs_per_sec", 0),
                "throughput_max": run_stats.get("throughput", {}).get("max_msgs_per_sec", 0),
            })
    
    if not metrics_list:
        return None
    
    df = pd.DataFrame(metrics_list)
    
    summary = {"metrics": {}}
    for col in df.columns:
        summary["metrics"][col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }
    
    return summary


def compare_metrics(
    current: dict,
    baseline: dict,
    thresholds: Optional[dict] = None,
) -> dict:
    """Compare current metrics against baseline."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    
    results = {
        "comparisons": [],
        "failures": [],
        "warnings": [],
    }
    
    current_metrics = current.get("metrics", {})
    baseline_metrics = baseline.get("metrics", {})
    
    for metric, threshold in thresholds.items():
        if metric == "variance":
            continue  # Handle separately
        
        current_val = current_metrics.get(metric, {}).get("mean", 0)
        baseline_val = baseline_metrics.get(metric, {}).get("mean", 0)
        
        if baseline_val == 0:
            continue
        
        change_pct = ((current_val - baseline_val) / baseline_val) * 100
        
        comparison = {
            "metric": metric,
            "baseline": baseline_val,
            "current": current_val,
            "change_pct": change_pct,
            "threshold": threshold,
        }
        
        # Check for regression
        if threshold >= 0:
            # Higher is worse (latency)
            is_regression = change_pct > threshold
        else:
            # Lower is worse (throughput)
            is_regression = change_pct < threshold
        
        comparison["regression"] = is_regression
        results["comparisons"].append(comparison)
        
        if is_regression:
            results["failures"].append(comparison)
        elif abs(change_pct) > abs(threshold) * 0.5:
            results["warnings"].append(comparison)
    
    # Check variance regression
    variance_threshold = thresholds.get("variance", 50.0)
    for metric in ["latency_mean", "latency_p99"]:
        current_std = current_metrics.get(metric, {}).get("std", 0)
        baseline_std = baseline_metrics.get(metric, {}).get("std", 0)
        
        if baseline_std == 0:
            continue
        
        variance_change = ((current_std - baseline_std) / baseline_std) * 100
        
        if variance_change > variance_threshold:
            results["failures"].append({
                "metric": f"{metric}_variance",
                "baseline": baseline_std,
                "current": current_std,
                "change_pct": variance_change,
                "threshold": variance_threshold,
                "regression": True,
            })
    
    results["regression_detected"] = len(results["failures"]) > 0
    
    return results


def detect_regression(
    current_dir: Path,
    baseline_dir: Path,
    output_dir: Path,
    thresholds: Optional[dict] = None,
) -> dict:
    """Detect regressions between current and baseline batches."""
    console.print(f"[bold blue]Regression Detection[/bold blue]")
    console.print(f"  Current: {current_dir}")
    console.print(f"  Baseline: {baseline_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load summaries
    console.print("[cyan]Loading batch summaries...[/cyan]")
    current = load_batch_summary(current_dir)
    baseline = load_batch_summary(baseline_dir)
    
    if not current:
        console.print("[red]Could not load current batch summary![/red]")
        return {"error": "no current data"}
    
    if not baseline:
        console.print("[red]Could not load baseline batch summary![/red]")
        return {"error": "no baseline data"}
    
    # Compare
    console.print("[cyan]Comparing metrics...[/cyan]")
    results = compare_metrics(current, baseline, thresholds)
    results["current_batch"] = str(current_dir)
    results["baseline_batch"] = str(baseline_dir)
    
    # Save report
    report_path = output_dir / "regression_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"  Saved: {report_path}")
    
    # Save failures to text file
    if results["failures"]:
        failures_path = output_dir / "regression_failures.txt"
        with open(failures_path, "w") as f:
            f.write("REGRESSION FAILURES DETECTED\n")
            f.write("=" * 50 + "\n\n")
            
            for failure in results["failures"]:
                f.write(f"Metric: {failure['metric']}\n")
                f.write(f"  Baseline: {failure['baseline']:.2f}\n")
                f.write(f"  Current:  {failure['current']:.2f}\n")
                f.write(f"  Change:   {failure['change_pct']:+.2f}%\n")
                f.write(f"  Threshold: {failure['threshold']}%\n")
                f.write("\n")
        
        console.print(f"  Saved: {failures_path}")
    
    # Summary output
    console.print(f"\n[bold]Regression Summary:[/bold]")
    console.print(f"  Comparisons: {len(results['comparisons'])}")
    console.print(f"  Warnings: {len(results['warnings'])}")
    
    if results["regression_detected"]:
        console.print(f"  [red]Failures: {len(results['failures'])}[/red]")
        for failure in results["failures"]:
            console.print(
                f"    ✗ {failure['metric']}: {failure['change_pct']:+.2f}% "
                f"(threshold: {failure['threshold']}%)"
            )
    else:
        console.print("[green]✓ No regressions detected[/green]")
    
    return results


def detect_against_reference(
    batch_dir: Path,
    reference_file: Path,
    output_dir: Path,
    thresholds: Optional[dict] = None,
) -> dict:
    """Detect regressions against a reference baseline file."""
    console.print(f"[bold blue]Regression Detection (Reference)[/bold blue]")
    console.print(f"  Batch: {batch_dir}")
    console.print(f"  Reference: {reference_file}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load current
    current = load_batch_summary(batch_dir)
    if not current:
        console.print("[red]Could not load batch summary![/red]")
        return {"error": "no current data"}
    
    # Load reference
    if not reference_file.exists():
        console.print("[red]Reference file not found![/red]")
        return {"error": "no reference file"}
    
    with open(reference_file) as f:
        reference = json.load(f)
    
    # Convert reference format if needed
    if "metrics" not in reference:
        # Assume it's a flat format
        reference = {"metrics": {k: {"mean": v} for k, v in reference.items()}}
    
    # Compare
    results = compare_metrics(current, reference, thresholds)
    results["current_batch"] = str(batch_dir)
    results["reference_file"] = str(reference_file)
    
    # Save
    report_path = output_dir / "regression_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Detect performance regressions")
    parser.add_argument("--current", required=True, type=Path, help="Current batch directory")
    parser.add_argument("--baseline", type=Path, help="Baseline batch directory")
    parser.add_argument("--reference", type=Path, help="Reference baseline JSON file")
    parser.add_argument("--out", type=Path, help="Output directory")
    parser.add_argument("--latency-threshold", type=float, default=10.0, help="Latency regression threshold (%)")
    parser.add_argument("--throughput-threshold", type=float, default=-10.0, help="Throughput regression threshold (%)")
    
    args = parser.parse_args()
    
    if not args.baseline and not args.reference:
        console.print("[red]Either --baseline or --reference must be provided[/red]")
        exit(1)
    
    output_dir = args.out or (args.current / "analysis")
    
    thresholds = {
        **DEFAULT_THRESHOLDS,
        "latency_mean": args.latency_threshold,
        "latency_p50": args.latency_threshold,
        "throughput_mean": args.throughput_threshold,
    }
    
    if args.baseline:
        result = detect_regression(args.current, args.baseline, output_dir, thresholds)
    else:
        result = detect_against_reference(args.current, args.reference, output_dir, thresholds)
    
    exit(1 if result.get("regression_detected") else 0)


if __name__ == "__main__":
    main()

