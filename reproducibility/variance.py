#!/usr/bin/env python3
"""
Variance analysis across multiple experiment runs.

Computes:
- Mean and variance of key metrics
- Coefficient of variation
- Run-to-run drift detection
- Variance ratio between algorithms

Usage:
    python variance.py --input reproducibility/output/batch_001 --out reproducibility/output/batch_001/analysis
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from scipy import stats

console = Console()


def load_run_statistics(run_dir: Path) -> Optional[dict]:
    """Load statistics from a single run."""
    stats_path = run_dir / "stats" / "summary.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    return None


def load_merged_data(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load merged data from a single run."""
    parquet_path = run_dir / "merged" / "merged.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    
    jsonl_path = run_dir / "merged" / "merged.jsonl"
    if jsonl_path.exists():
        return pd.read_json(jsonl_path, lines=True)
    
    return None


def collect_run_metrics(batch_dir: Path) -> pd.DataFrame:
    """Collect metrics from all runs in a batch."""
    metrics = []
    
    for run_dir in sorted(batch_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        
        run_index = int(run_dir.name.split("_")[1])
        stats = load_run_statistics(run_dir)
        
        if stats:
            metric = {
                "run_index": run_index,
                "total_events": stats.get("total_events", 0),
                "latency_mean": stats.get("latency", {}).get("mean", 0),
                "latency_std": stats.get("latency", {}).get("std", 0),
                "latency_p50": stats.get("latency", {}).get("p50", 0),
                "latency_p90": stats.get("latency", {}).get("p90", 0),
                "latency_p99": stats.get("latency", {}).get("p99", 0),
                "latency_p999": stats.get("latency", {}).get("p999", 0),
                "throughput_mean": stats.get("throughput", {}).get("mean_msgs_per_sec", 0),
                "throughput_max": stats.get("throughput", {}).get("max_msgs_per_sec", 0),
                "duration_sec": stats.get("throughput", {}).get("total_duration_sec", 0),
            }
            
            # Add queue delay if available
            if "queue_delay" in stats:
                metric["queue_delay_mean"] = stats["queue_delay"].get("mean", 0)
                metric["queue_delay_p99"] = stats["queue_delay"].get("p99", 0)
            
            metrics.append(metric)
    
    return pd.DataFrame(metrics)


def compute_variance_summary(df: pd.DataFrame) -> dict:
    """Compute variance summary for all metrics."""
    summary = {}
    
    # Metrics to analyze
    metrics = [
        "latency_mean", "latency_p50", "latency_p99", "latency_p999",
        "throughput_mean", "throughput_max",
    ]
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        values = df[metric].values
        
        if len(values) == 0 or np.all(values == 0):
            continue
        
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        cv = std_val / mean_val if mean_val > 0 else 0
        
        summary[metric] = {
            "mean": float(mean_val),
            "std": float(std_val),
            "variance": float(np.var(values, ddof=1)),
            "cv": float(cv),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
            "n": len(values),
        }
        
        # Standard error
        summary[metric]["se"] = float(std_val / np.sqrt(len(values)))
    
    return summary


def detect_drift(df: pd.DataFrame, metric: str = "latency_mean") -> dict:
    """Detect drift over time between runs."""
    if metric not in df.columns:
        return {"detected": False, "reason": "metric not found"}
    
    values = df[metric].values
    n = len(values)
    
    if n < 4:
        return {"detected": False, "reason": "insufficient runs"}
    
    # Split into first half and second half
    first_half = values[:n//2]
    second_half = values[n//2:]
    
    # T-test between halves
    t_stat, p_value = stats.ttest_ind(first_half, second_half)
    
    # Linear regression for trend
    x = np.arange(n)
    slope, intercept, r_value, p_trend, std_err = stats.linregress(x, values)
    
    # Detect significant drift
    drift_detected = p_value < 0.05 or p_trend < 0.05
    
    return {
        "detected": drift_detected,
        "t_statistic": float(t_stat),
        "t_pvalue": float(p_value),
        "trend_slope": float(slope),
        "trend_pvalue": float(p_trend),
        "trend_r2": float(r_value ** 2),
        "first_half_mean": float(np.mean(first_half)),
        "second_half_mean": float(np.mean(second_half)),
        "change_pct": float((np.mean(second_half) - np.mean(first_half)) / np.mean(first_half) * 100)
        if np.mean(first_half) > 0 else 0,
    }


def compute_variance_ratio(df: pd.DataFrame, groupby: str = "algorithm") -> dict:
    """Compute variance ratio between groups (e.g., algorithms)."""
    if groupby not in df.columns:
        return {}
    
    ratios = {}
    groups = df[groupby].unique()
    
    if len(groups) < 2:
        return {}
    
    for i, g1 in enumerate(groups):
        for g2 in groups[i+1:]:
            v1 = df[df[groupby] == g1]["latency_mean"].var()
            v2 = df[df[groupby] == g2]["latency_mean"].var()
            
            if v2 > 0:
                ratio = v1 / v2
                f_stat, f_pval = stats.levene(
                    df[df[groupby] == g1]["latency_mean"],
                    df[df[groupby] == g2]["latency_mean"]
                )
                
                ratios[f"{g1}_vs_{g2}"] = {
                    "variance_ratio": float(ratio),
                    "levene_statistic": float(f_stat),
                    "levene_pvalue": float(f_pval),
                    "variances_equal": float(f_pval) > 0.05,
                }
    
    return ratios


def plot_variance(df: pd.DataFrame, output_path: Path) -> None:
    """Generate variance visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Latency across runs
    ax = axes[0, 0]
    ax.errorbar(
        df["run_index"],
        df["latency_mean"],
        yerr=df["latency_std"] if "latency_std" in df.columns else None,
        fmt="o-",
        capsize=3,
        color="#2196F3",
    )
    ax.axhline(df["latency_mean"].mean(), color="red", linestyle="--", alpha=0.7)
    ax.fill_between(
        df["run_index"],
        df["latency_mean"].mean() - df["latency_mean"].std(),
        df["latency_mean"].mean() + df["latency_mean"].std(),
        alpha=0.2,
        color="red",
    )
    ax.set_xlabel("Run Index")
    ax.set_ylabel("Mean Latency (μs)")
    ax.set_title("Latency Across Runs")
    ax.grid(True, alpha=0.3)
    
    # Latency distribution boxplot
    ax = axes[0, 1]
    latency_cols = ["latency_mean", "latency_p50", "latency_p99"]
    available_cols = [c for c in latency_cols if c in df.columns]
    if available_cols:
        df[available_cols].boxplot(ax=ax)
    ax.set_ylabel("Latency (μs)")
    ax.set_title("Latency Metric Distributions")
    ax.grid(True, alpha=0.3)
    
    # Throughput across runs
    ax = axes[1, 0]
    if "throughput_mean" in df.columns:
        ax.plot(df["run_index"], df["throughput_mean"], "o-", color="#4CAF50")
        ax.axhline(df["throughput_mean"].mean(), color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Run Index")
    ax.set_ylabel("Mean Throughput (ops/s)")
    ax.set_title("Throughput Across Runs")
    ax.grid(True, alpha=0.3)
    
    # Coefficient of Variation bar chart
    ax = axes[1, 1]
    metrics = ["latency_mean", "latency_p99", "throughput_mean"]
    cvs = []
    labels = []
    for m in metrics:
        if m in df.columns and df[m].mean() > 0:
            cv = df[m].std() / df[m].mean()
            cvs.append(cv * 100)
            labels.append(m.replace("_", " ").title())
    
    if cvs:
        colors = ["#4CAF50" if cv < 10 else "#FFC107" if cv < 25 else "#F44336" for cv in cvs]
        ax.bar(labels, cvs, color=colors)
        ax.axhline(10, color="green", linestyle="--", alpha=0.7, label="Low (10%)")
        ax.axhline(25, color="orange", linestyle="--", alpha=0.7, label="Moderate (25%)")
    ax.set_ylabel("Coefficient of Variation (%)")
    ax.set_title("Variability by Metric")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def analyze_variance(
    batch_dir: Path,
    output_dir: Path,
) -> dict:
    """Perform complete variance analysis."""
    console.print(f"[bold blue]Variance Analysis[/bold blue]")
    console.print(f"  Input: {batch_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect metrics
    console.print("[cyan]Collecting run metrics...[/cyan]")
    df = collect_run_metrics(batch_dir)
    
    if df.empty:
        console.print("[red]No run data found![/red]")
        return {"error": "no data"}
    
    console.print(f"  Found {len(df)} runs")
    
    # Compute variance summary
    console.print("[cyan]Computing variance summary...[/cyan]")
    variance_summary = compute_variance_summary(df)
    
    # Detect drift
    console.print("[cyan]Detecting drift...[/cyan]")
    drift = detect_drift(df, "latency_mean")
    
    # Assess overall stability
    latency_cv = variance_summary.get("latency_mean", {}).get("cv", 0)
    throughput_cv = variance_summary.get("throughput_mean", {}).get("cv", 0)
    p99_cv = variance_summary.get("latency_p99", {}).get("cv", 0)
    
    overall_stable = latency_cv < 0.15 and throughput_cv < 0.15 and not drift.get("detected", False)
    
    result = {
        "batch_dir": str(batch_dir),
        "num_runs": len(df),
        "metrics": variance_summary,
        "drift": drift,
        "latency_cv": float(latency_cv),
        "throughput_cv": float(throughput_cv),
        "p99_cv": float(p99_cv),
        "overall_stable": overall_stable,
    }
    
    # Save summary
    summary_path = output_dir / "variance_summary.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    console.print(f"  Saved: {summary_path}")
    
    # Generate plots
    console.print("[cyan]Generating plots...[/cyan]")
    plot_path = output_dir / "variance_plots.png"
    plot_variance(df, plot_path)
    console.print(f"  Saved: {plot_path}")
    
    # Save raw metrics
    df.to_csv(output_dir / "run_metrics.csv", index=False)
    
    # Summary output
    console.print(f"\n[bold]Variance Summary:[/bold]")
    console.print(f"  Latency CV: {latency_cv*100:.2f}%")
    console.print(f"  Throughput CV: {throughput_cv*100:.2f}%")
    console.print(f"  Drift detected: {drift.get('detected', False)}")
    
    if overall_stable:
        console.print("[green]✓ Results are stable[/green]")
    else:
        console.print("[yellow]⚠ Variability detected[/yellow]")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Variance analysis across runs")
    parser.add_argument("--input", required=True, type=Path, help="Batch directory")
    parser.add_argument("--out", type=Path, help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = args.out or (args.input / "analysis")
    
    result = analyze_variance(args.input, output_dir)
    
    if "error" in result:
        exit(1)


if __name__ == "__main__":
    main()

