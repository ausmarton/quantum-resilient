#!/usr/bin/env python3
"""
Stability analysis across experiment runs.

Analyzes:
- Distribution consistency (KS tests)
- Tail stability (p99/p99.9)
- Pairwise distribution comparisons
- Multi-run deviation heatmaps

Usage:
    python stability.py --input reproducibility/output/batch_001 --out reproducibility/output/batch_001/analysis
"""

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from scipy import stats

console = Console()


def load_merged_data(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load merged data from a single run."""
    parquet_path = run_dir / "merged" / "merged.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    
    jsonl_path = run_dir / "merged" / "merged.jsonl"
    if jsonl_path.exists():
        return pd.read_json(jsonl_path, lines=True)
    
    return None


def load_run_statistics(run_dir: Path) -> Optional[dict]:
    """Load statistics from a single run."""
    stats_path = run_dir / "stats" / "summary.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    return None


def ks_test(data1: np.ndarray, data2: np.ndarray) -> dict:
    """Perform Kolmogorov-Smirnov test between two samples."""
    stat, pval = stats.ks_2samp(data1, data2)
    return {
        "statistic": float(stat),
        "pvalue": float(pval),
        "significant": pval < 0.05,
    }


def wasserstein_distance(data1: np.ndarray, data2: np.ndarray) -> float:
    """Compute Wasserstein (Earth Mover's) distance."""
    return float(stats.wasserstein_distance(data1, data2))


def compare_runs(run1_data: np.ndarray, run2_data: np.ndarray) -> dict:
    """Compare two runs statistically."""
    ks = ks_test(run1_data, run2_data)
    wd = wasserstein_distance(run1_data, run2_data)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((run1_data.var() + run2_data.var()) / 2)
    cohens_d = (run1_data.mean() - run2_data.mean()) / pooled_std if pooled_std > 0 else 0
    
    return {
        "ks_stat": ks["statistic"],
        "ks_pval": ks["pvalue"],
        "wasserstein": wd,
        "cohens_d": float(cohens_d),
        "stable": not ks["significant"] and abs(cohens_d) < 0.5,
    }


def analyze_pairwise_stability(
    batch_dir: Path,
    metric: str = "latency_us",
    sample_size: int = 10000,
) -> list[dict]:
    """Analyze pairwise stability between all runs."""
    run_dirs = sorted(batch_dir.glob("run_*"))
    
    run_data = {}
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        
        run_index = int(run_dir.name.split("_")[1])
        df = load_merged_data(run_dir)
        
        if df is not None and metric in df.columns:
            data = df[metric].dropna().values
            # Sample if too large
            if len(data) > sample_size:
                data = np.random.choice(data, size=sample_size, replace=False)
            run_data[run_index] = data
    
    pairwise_results = []
    
    for (i, data_i), (j, data_j) in combinations(run_data.items(), 2):
        comparison = compare_runs(data_i, data_j)
        comparison["run_a"] = i
        comparison["run_b"] = j
        pairwise_results.append(comparison)
    
    return pairwise_results


def analyze_tail_stability(batch_dir: Path) -> dict:
    """Analyze stability of tail latencies (p99, p99.9)."""
    p99_values = []
    p999_values = []
    
    for run_dir in sorted(batch_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        
        run_stats = load_run_statistics(run_dir)
        if run_stats and "latency" in run_stats:
            p99 = run_stats["latency"].get("p99", 0)
            p999 = run_stats["latency"].get("p999", 0)
            
            if p99 > 0:
                p99_values.append(p99)
            if p999 > 0:
                p999_values.append(p999)
    
    result = {}
    
    if p99_values:
        p99_mean = np.mean(p99_values)
        p99_std = np.std(p99_values)
        p99_cv = p99_std / p99_mean if p99_mean > 0 else 0
        p99_max_dev = (max(p99_values) - p99_mean) / p99_mean if p99_mean > 0 else 0
        
        result["p99"] = {
            "mean": float(p99_mean),
            "std": float(p99_std),
            "cv": float(p99_cv),
            "max_deviation": float(p99_max_dev),
            "stable": p99_cv < 0.2,
        }
    
    if p999_values:
        p999_mean = np.mean(p999_values)
        p999_std = np.std(p999_values)
        p999_cv = p999_std / p999_mean if p999_mean > 0 else 0
        p999_max_dev = (max(p999_values) - p999_mean) / p999_mean if p999_mean > 0 else 0
        
        result["p999"] = {
            "mean": float(p999_mean),
            "std": float(p999_std),
            "cv": float(p999_cv),
            "max_deviation": float(p999_max_dev),
            "stable": p999_cv < 0.3,
        }
    
    return result


def plot_stability_matrix(pairwise_results: list[dict], output_path: Path) -> None:
    """Plot stability matrix heatmap."""
    if not pairwise_results:
        return
    
    # Get unique runs
    runs = set()
    for p in pairwise_results:
        runs.add(p["run_a"])
        runs.add(p["run_b"])
    runs = sorted(runs)
    n = len(runs)
    
    if n < 2:
        return
    
    # Create matrices
    ks_matrix = np.zeros((n, n))
    wd_matrix = np.zeros((n, n))
    
    run_to_idx = {r: i for i, r in enumerate(runs)}
    
    for p in pairwise_results:
        i, j = run_to_idx[p["run_a"]], run_to_idx[p["run_b"]]
        ks_matrix[i, j] = ks_matrix[j, i] = p["ks_stat"]
        wd_matrix[i, j] = wd_matrix[j, i] = p["wasserstein"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # KS statistic heatmap
    ax = axes[0]
    im = ax.imshow(ks_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"Run {r}" for r in runs], rotation=45, ha="right")
    ax.set_yticklabels([f"Run {r}" for r in runs])
    ax.set_title("KS Statistic Between Runs")
    plt.colorbar(im, ax=ax)
    
    # Wasserstein distance heatmap
    ax = axes[1]
    im = ax.imshow(wd_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"Run {r}" for r in runs], rotation=45, ha="right")
    ax.set_yticklabels([f"Run {r}" for r in runs])
    ax.set_title("Wasserstein Distance Between Runs")
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def assess_overall_stability(pairwise: list[dict], tails: dict) -> dict:
    """Assess overall stability and generate recommendations."""
    stable_pairs = sum(1 for p in pairwise if p.get("stable", False))
    total_pairs = len(pairwise)
    
    pairwise_stable = stable_pairs / total_pairs > 0.8 if total_pairs > 0 else True
    
    p99_stable = tails.get("p99", {}).get("stable", True)
    p999_stable = tails.get("p999", {}).get("stable", True)
    
    overall_stable = pairwise_stable and p99_stable and p999_stable
    
    recommendations = []
    
    if not pairwise_stable:
        recommendations.append("Run distributions show significant variation. Consider increasing sample size or investigating sources of variance.")
    
    if not p99_stable:
        recommendations.append("p99 tail latency is unstable. Check for GC pauses, resource contention, or cold-start effects.")
    
    if not p999_stable:
        recommendations.append("p99.9 tail latency shows high variance. This may be expected for extreme percentiles but warrants investigation.")
    
    # Calculate max deviations
    p99_max_dev = tails.get("p99", {}).get("max_deviation", 0)
    p999_max_dev = tails.get("p999", {}).get("max_deviation", 0)
    
    return {
        "overall_stable": overall_stable,
        "pairwise_stable": pairwise_stable,
        "stable_pairs": stable_pairs,
        "total_pairs": total_pairs,
        "p99_stable": p99_stable,
        "p999_stable": p999_stable,
        "p99_max_deviation": p99_max_dev,
        "p999_max_deviation": p999_max_dev,
        "recommendations": recommendations,
    }


def analyze_stability(
    batch_dir: Path,
    output_dir: Path,
    metric: str = "latency_us",
    sample_size: int = 10000,
) -> dict:
    """Perform complete stability analysis."""
    console.print(f"[bold blue]Stability Analysis[/bold blue]")
    console.print(f"  Input: {batch_dir}")
    console.print(f"  Metric: {metric}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pairwise stability
    console.print("[cyan]Analyzing pairwise stability...[/cyan]")
    pairwise = analyze_pairwise_stability(batch_dir, metric, sample_size)
    console.print(f"  Compared {len(pairwise)} run pairs")
    
    # Tail stability
    console.print("[cyan]Analyzing tail stability...[/cyan]")
    tails = analyze_tail_stability(batch_dir)
    
    # Overall assessment
    console.print("[cyan]Assessing overall stability...[/cyan]")
    assessment = assess_overall_stability(pairwise, tails)
    
    result = {
        "batch_dir": str(batch_dir),
        "metric": metric,
        "pairwise": pairwise,
        "tails": tails,
        **assessment,
    }
    
    # Save results
    summary_path = output_dir / "stability_summary.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    console.print(f"  Saved: {summary_path}")
    
    # Generate matrix plot
    console.print("[cyan]Generating stability matrix plot...[/cyan]")
    matrix_path = output_dir / "stability_matrix.png"
    plot_stability_matrix(pairwise, matrix_path)
    console.print(f"  Saved: {matrix_path}")
    
    # Summary output
    console.print(f"\n[bold]Stability Summary:[/bold]")
    console.print(f"  Stable pairs: {assessment['stable_pairs']}/{assessment['total_pairs']}")
    console.print(f"  p99 stable: {assessment['p99_stable']}")
    console.print(f"  p99.9 stable: {assessment['p999_stable']}")
    
    if assessment["overall_stable"]:
        console.print("[green]✓ System is stable[/green]")
    else:
        console.print("[yellow]⚠ Stability concerns detected[/yellow]")
        for rec in assessment["recommendations"]:
            console.print(f"  → {rec}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Stability analysis across runs")
    parser.add_argument("--input", required=True, type=Path, help="Batch directory")
    parser.add_argument("--out", type=Path, help="Output directory")
    parser.add_argument("--metric", default="latency_us", help="Metric to analyze")
    parser.add_argument("--sample-size", type=int, default=10000, help="Sample size for comparison")
    
    args = parser.parse_args()
    
    output_dir = args.out or (args.input / "analysis")
    
    result = analyze_stability(
        args.input,
        output_dir,
        metric=args.metric,
        sample_size=args.sample_size,
    )
    
    if "error" in result:
        exit(1)


if __name__ == "__main__":
    main()

