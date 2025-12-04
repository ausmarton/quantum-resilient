#!/usr/bin/env python3
"""
Compute confidence intervals for experiment metrics.

Supports:
- Bootstrap CI
- Percentile CI
- Normal-approximation CI
- Bias-corrected and accelerated (BCa) bootstrap

Usage:
    python confidence.py --input reproducibility/output/batch_001 --out reproducibility/output/batch_001/analysis
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from rich.console import Console
from scipy import stats

console = Console()


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    method: str = "percentile",
) -> dict:
    """
    Compute bootstrap confidence interval.
    
    Methods:
    - percentile: Simple percentile method
    - basic: Basic bootstrap (reflection)
    - bca: Bias-corrected and accelerated
    """
    n = len(data)
    if n == 0:
        return {"estimate": 0, "lower": 0, "upper": 0, "method": method}
    
    # Point estimate
    estimate = statistic(data)
    
    # Bootstrap samples
    bootstrap_stats = np.array([
        statistic(np.random.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    
    alpha = 1 - confidence
    
    if method == "percentile":
        lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
        upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    
    elif method == "basic":
        lower = 2 * estimate - np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
        upper = 2 * estimate - np.percentile(bootstrap_stats, alpha / 2 * 100)
    
    elif method == "bca":
        # Bias correction
        z0 = stats.norm.ppf(np.mean(bootstrap_stats < estimate))
        
        # Acceleration (jackknife)
        jackknife_stats = np.array([
            statistic(np.delete(data, i))
            for i in range(n)
        ])
        jackknife_mean = np.mean(jackknife_stats)
        
        num = np.sum((jackknife_mean - jackknife_stats) ** 3)
        denom = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)
        a = num / denom if denom != 0 else 0
        
        # Adjusted percentiles
        z_alpha_lower = stats.norm.ppf(alpha / 2)
        z_alpha_upper = stats.norm.ppf(1 - alpha / 2)
        
        p_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
        p_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))
        
        lower = np.percentile(bootstrap_stats, p_lower * 100)
        upper = np.percentile(bootstrap_stats, p_upper * 100)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        "estimate": float(estimate),
        "lower": float(lower),
        "upper": float(upper),
        "se": float(np.std(bootstrap_stats)),
        "method": f"bootstrap_{method}",
        "n_bootstrap": n_bootstrap,
    }


def normal_ci(
    data: np.ndarray,
    confidence: float = 0.95,
) -> dict:
    """Compute normal approximation confidence interval."""
    n = len(data)
    if n == 0:
        return {"estimate": 0, "lower": 0, "upper": 0, "method": "normal"}
    
    mean = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)
    
    # t-distribution for small samples
    if n < 30:
        t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    else:
        t_crit = stats.norm.ppf((1 + confidence) / 2)
    
    margin = t_crit * se
    
    return {
        "estimate": float(mean),
        "lower": float(mean - margin),
        "upper": float(mean + margin),
        "se": float(se),
        "method": "normal",
    }


def percentile_ci(
    data: np.ndarray,
    confidence: float = 0.95,
) -> dict:
    """Compute percentile-based confidence interval for the median."""
    n = len(data)
    if n == 0:
        return {"estimate": 0, "lower": 0, "upper": 0, "method": "percentile"}
    
    median = np.median(data)
    alpha = 1 - confidence
    
    lower = np.percentile(data, alpha / 2 * 100)
    upper = np.percentile(data, (1 - alpha / 2) * 100)
    
    return {
        "estimate": float(median),
        "lower": float(lower),
        "upper": float(upper),
        "method": "percentile",
    }


def compute_all_cis(
    data: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 5000,
) -> dict:
    """Compute all types of confidence intervals."""
    return {
        "normal": normal_ci(data, confidence),
        "bootstrap_percentile": bootstrap_ci(data, np.mean, confidence, n_bootstrap, "percentile"),
        "bootstrap_bca": bootstrap_ci(data, np.mean, confidence, n_bootstrap, "bca"),
        "percentile": percentile_ci(data, confidence),
    }


def load_run_statistics(run_dir: Path) -> Optional[dict]:
    """Load statistics from a single run."""
    stats_path = run_dir / "stats" / "summary.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    return None


def collect_run_metrics(batch_dir: Path) -> pd.DataFrame:
    """Collect metrics from all runs in a batch."""
    metrics = []
    
    for run_dir in sorted(batch_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        
        run_index = int(run_dir.name.split("_")[1])
        run_stats = load_run_statistics(run_dir)
        
        if run_stats:
            metric = {
                "run_index": run_index,
                "latency_mean": run_stats.get("latency", {}).get("mean", 0),
                "latency_p50": run_stats.get("latency", {}).get("p50", 0),
                "latency_p99": run_stats.get("latency", {}).get("p99", 0),
                "latency_p999": run_stats.get("latency", {}).get("p999", 0),
                "throughput_mean": run_stats.get("throughput", {}).get("mean_msgs_per_sec", 0),
                "throughput_max": run_stats.get("throughput", {}).get("max_msgs_per_sec", 0),
            }
            
            # Queue delay
            if "queue_delay" in run_stats:
                metric["queue_delay_mean"] = run_stats["queue_delay"].get("mean", 0)
            
            # Worker jitter
            if "worker_jitter" in run_stats:
                metric["jitter_mean"] = run_stats["worker_jitter"].get("mean_ms", 0)
            
            metrics.append(metric)
    
    return pd.DataFrame(metrics)


def analyze_confidence_intervals(
    batch_dir: Path,
    output_dir: Path,
    confidence: float = 0.95,
    n_bootstrap: int = 5000,
    method: str = "bca",
) -> dict:
    """Compute confidence intervals for all metrics."""
    console.print(f"[bold blue]Confidence Interval Analysis[/bold blue]")
    console.print(f"  Input: {batch_dir}")
    console.print(f"  Confidence level: {confidence*100}%")
    console.print(f"  Method: {method}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect metrics
    console.print("[cyan]Collecting run metrics...[/cyan]")
    df = collect_run_metrics(batch_dir)
    
    if df.empty:
        console.print("[red]No run data found![/red]")
        return {"error": "no data"}
    
    console.print(f"  Found {len(df)} runs")
    
    # Metrics to analyze
    metrics_to_analyze = [
        ("latency_mean", "Mean Latency (μs)"),
        ("latency_p50", "p50 Latency (μs)"),
        ("latency_p99", "p99 Latency (μs)"),
        ("latency_p999", "p99.9 Latency (μs)"),
        ("throughput_mean", "Mean Throughput (ops/s)"),
        ("throughput_max", "Max Throughput (ops/s)"),
        ("queue_delay_mean", "Queue Delay (μs)"),
        ("jitter_mean", "Start Jitter (ms)"),
    ]
    
    results = {
        "confidence_level": confidence,
        "n_runs": len(df),
        "method": method,
        "intervals": {},
    }
    
    console.print("[cyan]Computing confidence intervals...[/cyan]")
    
    for metric, description in metrics_to_analyze:
        if metric not in df.columns:
            continue
        
        data = df[metric].dropna().values
        if len(data) == 0:
            continue
        
        if method == "bca":
            ci = bootstrap_ci(data, np.mean, confidence, n_bootstrap, "bca")
        elif method == "percentile":
            ci = bootstrap_ci(data, np.mean, confidence, n_bootstrap, "percentile")
        elif method == "normal":
            ci = normal_ci(data, confidence)
        else:
            ci = bootstrap_ci(data, np.mean, confidence, n_bootstrap, method)
        
        ci["description"] = description
        ci["n"] = len(data)
        
        results["intervals"][metric] = ci
        
        console.print(f"  {description}: {ci['estimate']:.2f} [{ci['lower']:.2f}, {ci['upper']:.2f}]")
    
    # Save results
    output_path = output_dir / "confidence_intervals.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]Saved: {output_path}[/green]")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute confidence intervals")
    parser.add_argument("--input", required=True, type=Path, help="Batch directory")
    parser.add_argument("--out", type=Path, help="Output directory")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level")
    parser.add_argument("--n-bootstrap", type=int, default=5000, help="Bootstrap iterations")
    parser.add_argument(
        "--method",
        choices=["bca", "percentile", "basic", "normal"],
        default="bca",
        help="CI computation method"
    )
    
    args = parser.parse_args()
    
    output_dir = args.out or (args.input / "analysis")
    
    result = analyze_confidence_intervals(
        args.input,
        output_dir,
        confidence=args.confidence,
        n_bootstrap=args.n_bootstrap,
        method=args.method,
    )
    
    if "error" in result:
        exit(1)


if __name__ == "__main__":
    main()

