#!/usr/bin/env python3
"""
Generate ECDF (Empirical Cumulative Distribution Function) plots for latency.

Produces publication-quality CDF and violin plots.

Usage:
    python plot_ecdf.py --input merged.parquet --output figures/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console

console = Console()


def compute_ecdf(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute ECDF from data."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    ecdf_y = np.arange(1, n + 1) / n
    return sorted_data, ecdf_y


def plot_latency_cdf(
    df: pd.DataFrame,
    output_path: Path,
    experiment_id: str = "",
    dpi: int = 300,
    log_scale: bool = True,
) -> None:
    """Generate latency CDF plot."""
    console.print("[cyan]Generating latency CDF plot...[/cyan]")
    
    # Determine latency column
    latency_col = None
    for col in ["latency_us", "latency_µs", "latency_ms"]:
        if col in df.columns:
            latency_col = col
            break
    
    if latency_col is None:
        console.print("[red]No latency column found![/red]")
        return
    
    latency = df[latency_col].dropna().values
    
    if len(latency) == 0:
        console.print("[red]No latency data![/red]")
        return
    
    # Compute ECDF
    x, y = compute_ecdf(latency)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot ECDF
    ax.plot(x, y, linewidth=2, color='#2E86AB')
    
    # Add percentile markers
    percentiles = [50, 90, 95, 99, 99.9]
    for p in percentiles:
        val = np.percentile(latency, p)
        idx = np.searchsorted(x, val)
        if idx < len(y):
            ax.axhline(y=p/100, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.axvline(x=val, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.annotate(f'p{p}: {val:.0f}', xy=(val, p/100), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
    
    # Labels and title
    unit = "μs" if "us" in latency_col or "µs" in latency_col else "ms"
    ax.set_xlabel(f"Latency ({unit})", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    
    title = "Latency ECDF"
    if experiment_id:
        title = f"{title} - {experiment_id}"
    ax.set_title(title, fontsize=14)
    
    # Log scale for x-axis (optional)
    if log_scale and latency.max() / latency.min() > 10:
        ax.set_xscale('log')
    
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.02)
    
    # Add statistics annotation
    stats_text = f"n={len(latency):,}\nMean: {np.mean(latency):.1f} {unit}\nMedian: {np.median(latency):.1f} {unit}\np99: {np.percentile(latency, 99):.1f} {unit}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    output_file = output_path / "latency_cdf.png"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]  Saved: {output_file}[/green]")


def plot_latency_violin(
    df: pd.DataFrame,
    output_path: Path,
    experiment_id: str = "",
    dpi: int = 300,
) -> None:
    """Generate latency violin plot by algorithm."""
    console.print("[cyan]Generating latency violin plot...[/cyan]")
    
    # Determine latency column
    latency_col = None
    for col in ["latency_us", "latency_µs", "latency_ms"]:
        if col in df.columns:
            latency_col = col
            break
    
    if latency_col is None:
        console.print("[yellow]No latency column found, skipping violin plot[/yellow]")
        return
    
    # Check for algorithm column
    if "algorithm" not in df.columns:
        console.print("[yellow]No algorithm column, skipping violin plot[/yellow]")
        return
    
    algorithms = df["algorithm"].unique()
    if len(algorithms) < 2:
        console.print("[yellow]Only one algorithm, skipping violin plot[/yellow]")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for violin plot
    data_by_algo = [df[df["algorithm"] == algo][latency_col].dropna().values for algo in algorithms]
    
    parts = ax.violinplot(data_by_algo, showmeans=True, showmedians=True)
    
    # Style violins
    for pc in parts['bodies']:
        pc.set_facecolor('#2E86AB')
        pc.set_alpha(0.7)
    
    ax.set_xticks(range(1, len(algorithms) + 1))
    ax.set_xticklabels(algorithms)
    
    unit = "μs" if "us" in latency_col or "µs" in latency_col else "ms"
    ax.set_ylabel(f"Latency ({unit})", fontsize=12)
    ax.set_xlabel("Algorithm", fontsize=12)
    
    title = "Latency Distribution by Algorithm"
    if experiment_id:
        title = f"{title} - {experiment_id}"
    ax.set_title(title, fontsize=14)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = output_path / "latency_violin.png"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]  Saved: {output_file}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Generate ECDF plots for latency")
    parser.add_argument("--input", required=True, help="Input parquet or JSONL file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--experiment-id", default="", help="Experiment ID for title")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output images")
    parser.add_argument("--no-log", action="store_true", help="Disable log scale on x-axis")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    console.print(f"[cyan]Loading data from {input_path}...[/cyan]")
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_json(input_path, lines=True)
    
    console.print(f"  Loaded {len(df)} events")
    
    # Generate plots
    plot_latency_cdf(df, output_path, args.experiment_id, args.dpi, not args.no_log)
    plot_latency_violin(df, output_path, args.experiment_id, args.dpi)
    
    console.print("[bold green]ECDF plots complete![/bold green]")


if __name__ == "__main__":
    main()

