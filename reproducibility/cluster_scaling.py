#!/usr/bin/env python3
"""
Cluster scaling analysis.

Studies how performance scales with cluster size:
- Throughput scaling curves
- Latency scaling curves
- Efficiency analysis
- Saturation point detection

Usage:
    python cluster_scaling.py --input scaling_experiments/ --out analysis/
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from scipy import optimize
from scipy import stats

console = Console()


def load_run_statistics(run_dir: Path) -> Optional[dict]:
    """Load statistics from a single run."""
    stats_path = run_dir / "stats" / "summary.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    return None


def load_scaling_data(base_dir: Path, cluster_sizes: Optional[list[int]] = None) -> pd.DataFrame:
    """Load data from scaling experiments."""
    data = []
    
    # Try to auto-detect cluster sizes from directory names
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        # Try to extract cluster size from directory name
        name = subdir.name
        size = None
        
        # Try various naming patterns
        for pattern in ["pods_", "replicas_", "workers_", "size_", "n"]:
            if pattern in name.lower():
                try:
                    size_str = name.lower().split(pattern)[-1].split("_")[0]
                    size = int(size_str)
                    break
                except ValueError:
                    continue
        
        if size is None and cluster_sizes:
            # Try to match by index
            idx = list(sorted(base_dir.iterdir())).index(subdir)
            if idx < len(cluster_sizes):
                size = cluster_sizes[idx]
        
        if size is None:
            continue
        
        # Load statistics
        run_stats = load_run_statistics(subdir)
        
        if run_stats:
            data.append({
                "cluster_size": size,
                "throughput_mean": run_stats.get("throughput", {}).get("mean_msgs_per_sec", 0),
                "throughput_max": run_stats.get("throughput", {}).get("max_msgs_per_sec", 0),
                "latency_mean": run_stats.get("latency", {}).get("mean", 0),
                "latency_p50": run_stats.get("latency", {}).get("p50", 0),
                "latency_p99": run_stats.get("latency", {}).get("p99", 0),
                "total_events": run_stats.get("total_events", 0),
                "duration_sec": run_stats.get("throughput", {}).get("total_duration_sec", 0),
                "directory": str(subdir),
            })
    
    return pd.DataFrame(data).sort_values("cluster_size")


def fit_scaling_model(sizes: np.ndarray, values: np.ndarray) -> dict:
    """Fit various scaling models and return the best fit."""
    if len(sizes) < 3:
        return {"model": "insufficient_data"}
    
    results = {}
    
    # Linear model: y = a * x + b
    try:
        slope, intercept, r_linear, p_linear, se = stats.linregress(sizes, values)
        results["linear"] = {
            "params": {"slope": float(slope), "intercept": float(intercept)},
            "r2": float(r_linear ** 2),
            "predict": lambda x: slope * x + intercept,
        }
    except Exception:
        pass
    
    # Logarithmic model: y = a * log(x) + b
    try:
        log_sizes = np.log(sizes)
        slope, intercept, r_log, p_log, se = stats.linregress(log_sizes, values)
        results["logarithmic"] = {
            "params": {"a": float(slope), "b": float(intercept)},
            "r2": float(r_log ** 2),
            "predict": lambda x: slope * np.log(x) + intercept,
        }
    except Exception:
        pass
    
    # Power model: y = a * x^b
    try:
        def power_func(x, a, b):
            return a * np.power(x, b)
        
        popt, pcov = optimize.curve_fit(power_func, sizes, values, p0=[1, 1], maxfev=5000)
        pred = power_func(sizes, *popt)
        ss_res = np.sum((values - pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r2_power = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        results["power"] = {
            "params": {"a": float(popt[0]), "b": float(popt[1])},
            "r2": float(r2_power),
            "predict": lambda x, a=popt[0], b=popt[1]: a * np.power(x, b),
        }
    except Exception:
        pass
    
    # Saturation model: y = a * (1 - exp(-x/b))
    try:
        def saturation_func(x, a, b):
            return a * (1 - np.exp(-x / b))
        
        # Initial guess: a = max(values) * 1.2, b = mean(sizes)
        p0 = [max(values) * 1.2, np.mean(sizes)]
        popt, pcov = optimize.curve_fit(saturation_func, sizes, values, p0=p0, maxfev=5000)
        pred = saturation_func(sizes, *popt)
        ss_res = np.sum((values - pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r2_sat = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        results["saturation"] = {
            "params": {"asymptote": float(popt[0]), "rate": float(popt[1])},
            "r2": float(r2_sat),
            "predict": lambda x, a=popt[0], b=popt[1]: a * (1 - np.exp(-x / b)),
        }
    except Exception:
        pass
    
    # Select best model
    if not results:
        return {"model": "fit_failed"}
    
    best_model = max(results.items(), key=lambda x: x[1]["r2"])
    
    return {
        "model": best_model[0],
        "r2": best_model[1]["r2"],
        "params": best_model[1]["params"],
        "all_models": {k: {"r2": v["r2"], "params": v["params"]} for k, v in results.items()},
    }


def detect_saturation_point(df: pd.DataFrame, throughput_col: str = "throughput_mean") -> Optional[int]:
    """Detect saturation point where scaling efficiency drops."""
    if len(df) < 3:
        return None
    
    df = df.sort_values("cluster_size")
    sizes = df["cluster_size"].values
    throughputs = df[throughput_col].values
    
    # Calculate efficiency (throughput per pod)
    efficiency = throughputs / sizes
    
    # Find where efficiency drops below 80% of initial
    initial_efficiency = efficiency[0]
    
    for i, eff in enumerate(efficiency):
        if eff < initial_efficiency * 0.8:
            return int(sizes[i])
    
    return None


def analyze_scaling(
    base_dir: Path,
    output_dir: Path,
    cluster_sizes: Optional[list[int]] = None,
) -> dict:
    """Perform complete cluster scaling analysis."""
    console.print(f"[bold blue]Cluster Scaling Analysis[/bold blue]")
    console.print(f"  Input: {base_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    console.print("[cyan]Loading scaling data...[/cyan]")
    df = load_scaling_data(base_dir, cluster_sizes)
    
    if df.empty:
        console.print("[red]No scaling data found![/red]")
        return {"error": "no data"}
    
    console.print(f"  Found {len(df)} cluster sizes: {sorted(df['cluster_size'].tolist())}")
    
    sizes = df["cluster_size"].values
    
    # Throughput scaling
    console.print("[cyan]Analyzing throughput scaling...[/cyan]")
    throughput_model = fit_scaling_model(sizes, df["throughput_mean"].values)
    
    # Latency scaling
    console.print("[cyan]Analyzing latency scaling...[/cyan]")
    latency_model = fit_scaling_model(sizes, df["latency_mean"].values)
    
    # Efficiency calculation
    df["efficiency"] = df["throughput_mean"] / df["cluster_size"]
    initial_efficiency = df["efficiency"].iloc[0]
    df["efficiency_relative"] = df["efficiency"] / initial_efficiency
    
    # Saturation point
    console.print("[cyan]Detecting saturation point...[/cyan]")
    saturation_point = detect_saturation_point(df)
    
    # Calculate scaling factor
    if len(sizes) >= 2:
        throughput_ratio = df["throughput_mean"].iloc[-1] / df["throughput_mean"].iloc[0]
        size_ratio = sizes[-1] / sizes[0]
        scaling_factor = throughput_ratio / size_ratio
    else:
        scaling_factor = 1.0
    
    result = {
        "cluster_sizes": sorted(df["cluster_size"].tolist()),
        "throughput_scaling": {
            "best_model": throughput_model.get("model"),
            "r2_score": throughput_model.get("r2"),
            "params": throughput_model.get("params", {}),
            "all_models": throughput_model.get("all_models", {}),
        },
        "latency_scaling": {
            "best_model": latency_model.get("model"),
            "r2_score": latency_model.get("r2"),
            "params": latency_model.get("params", {}),
        },
        "scaling_factor": float(scaling_factor),
        "saturation_point": saturation_point,
        "throughput_curve": df[["cluster_size", "throughput_mean", "efficiency_relative"]].rename(
            columns={"cluster_size": "size", "throughput_mean": "throughput", "efficiency_relative": "efficiency"}
        ).to_dict(orient="records"),
        "latency_curve": df[["cluster_size", "latency_mean", "latency_p99"]].rename(
            columns={"cluster_size": "size", "latency_mean": "mean_latency", "latency_p99": "p99_latency"}
        ).to_dict(orient="records"),
    }
    
    # Save summary
    summary_path = output_dir / "scaling_summary.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    console.print(f"  Saved: {summary_path}")
    
    # Generate plots
    console.print("[cyan]Generating scaling plots...[/cyan]")
    plot_scaling_curves(df, output_dir)
    
    # Summary output
    console.print(f"\n[bold]Scaling Summary:[/bold]")
    console.print(f"  Best throughput model: {throughput_model.get('model')}")
    console.print(f"  R² score: {throughput_model.get('r2', 0):.4f}")
    console.print(f"  Scaling factor: {scaling_factor:.3f}")
    
    if saturation_point:
        console.print(f"  [yellow]Saturation detected at: {saturation_point} pods[/yellow]")
    else:
        console.print("  [green]No saturation detected[/green]")
    
    return result


def plot_scaling_curves(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate scaling curve plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    sizes = df["cluster_size"].values
    
    # Throughput scaling
    ax = axes[0, 0]
    ax.plot(sizes, df["throughput_mean"], "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.fill_between(sizes, df["throughput_mean"] * 0.95, df["throughput_mean"] * 1.05, alpha=0.2)
    ax.set_xlabel("Cluster Size (pods)")
    ax.set_ylabel("Mean Throughput (ops/s)")
    ax.set_title("Throughput Scaling")
    ax.grid(True, alpha=0.3)
    
    # Throughput with ideal linear scaling
    ax = axes[0, 1]
    ax.plot(sizes, df["throughput_mean"], "o-", color="#2196F3", linewidth=2, markersize=8, label="Actual")
    ideal_throughput = df["throughput_mean"].iloc[0] * (sizes / sizes[0])
    ax.plot(sizes, ideal_throughput, "--", color="gray", linewidth=1, label="Ideal Linear")
    ax.set_xlabel("Cluster Size (pods)")
    ax.set_ylabel("Mean Throughput (ops/s)")
    ax.set_title("Throughput vs Ideal Scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Latency scaling
    ax = axes[1, 0]
    ax.plot(sizes, df["latency_mean"], "o-", color="#4CAF50", linewidth=2, markersize=8, label="Mean")
    ax.plot(sizes, df["latency_p99"], "s--", color="#FF9800", linewidth=2, markersize=6, label="p99")
    ax.set_xlabel("Cluster Size (pods)")
    ax.set_ylabel("Latency (μs)")
    ax.set_title("Latency Scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Efficiency
    ax = axes[1, 1]
    efficiency = df["throughput_mean"] / sizes
    relative_efficiency = efficiency / efficiency.iloc[0] * 100
    
    colors = ["#4CAF50" if e >= 80 else "#FFC107" if e >= 60 else "#F44336" for e in relative_efficiency]
    ax.bar(range(len(sizes)), relative_efficiency, color=colors)
    ax.axhline(100, color="gray", linestyle="--", linewidth=1)
    ax.axhline(80, color="orange", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Cluster Size (pods)")
    ax.set_ylabel("Relative Efficiency (%)")
    ax.set_title("Scaling Efficiency")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_dir / "scaling_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    console.print(f"  Saved: {output_dir / 'scaling_curve.png'}")


def main():
    parser = argparse.ArgumentParser(description="Cluster scaling analysis")
    parser.add_argument("--input", required=True, type=Path, help="Base directory with scaling experiments")
    parser.add_argument("--out", type=Path, help="Output directory")
    parser.add_argument(
        "--cluster-sizes",
        type=int,
        nargs="+",
        help="Cluster sizes (if not auto-detected)"
    )
    
    args = parser.parse_args()
    
    output_dir = args.out or (args.input / "analysis")
    
    result = analyze_scaling(
        args.input,
        output_dir,
        cluster_sizes=args.cluster_sizes,
    )
    
    if "error" in result:
        exit(1)


if __name__ == "__main__":
    main()

