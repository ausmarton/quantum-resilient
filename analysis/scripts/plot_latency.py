#!/usr/bin/env python3
"""
Generate latency plots (CDF, PDF, log-scale tail).

Usage:
    python plot_latency.py --input ./data/exp_001/merged/merged.jsonl --output ./figures/exp_001/
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console

console = Console()

# Academic/publication-quality style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})


def load_data(filepath: Path) -> pd.DataFrame:
    """Load data from JSONL or Parquet file."""
    if filepath.suffix == ".parquet":
        return pd.read_parquet(filepath)
    else:
        return pd.read_json(filepath, lines=True)


def plot_cdf(
    data: np.ndarray,
    output_path: Path,
    title: str = "Latency CDF",
    xlabel: str = "Latency (μs)",
    experiment_id: Optional[str] = None,
    adapter: Optional[str] = None,
    operation: Optional[str] = None,
) -> None:
    """Plot Cumulative Distribution Function."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort data for CDF
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    ax.plot(sorted_data, cdf, linewidth=2, color="#2c3e50")

    # Add percentile markers
    percentiles = [50, 90, 95, 99]
    colors = ["#27ae60", "#f39c12", "#e74c3c", "#8e44ad"]
    for p, color in zip(percentiles, colors):
        val = np.percentile(data, p)
        ax.axvline(val, linestyle="--", color=color, alpha=0.8, label=f"p{p}: {val:.0f}")
        ax.axhline(p / 100, linestyle=":", color=color, alpha=0.4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative Probability")

    # Build title
    title_parts = [title]
    if experiment_id:
        title_parts.append(f"Experiment: {experiment_id}")
    if adapter:
        title_parts.append(f"Adapter: {adapter}")
    if operation:
        title_parts.append(f"Operation: {operation}")
    ax.set_title("\n".join(title_parts))

    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    console.print(f"[green]  Saved {output_path}[/green]")


def plot_pdf(
    data: np.ndarray,
    output_path: Path,
    title: str = "Latency PDF",
    xlabel: str = "Latency (μs)",
    experiment_id: Optional[str] = None,
    adapter: Optional[str] = None,
    operation: Optional[str] = None,
) -> None:
    """Plot Probability Density Function."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram with KDE
    ax.hist(data, bins=100, density=True, alpha=0.6, color="#3498db", edgecolor="white", label="Histogram")

    # KDE overlay
    from scipy import stats
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 500)
    ax.plot(x_range, kde(x_range), linewidth=2, color="#e74c3c", label="KDE")

    # Add mean and median lines
    ax.axvline(np.mean(data), color="#27ae60", linestyle="--", linewidth=1.5, label=f"Mean: {np.mean(data):.0f}")
    ax.axvline(np.median(data), color="#f39c12", linestyle="--", linewidth=1.5, label=f"Median: {np.median(data):.0f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")

    # Build title
    title_parts = [title]
    if experiment_id:
        title_parts.append(f"Experiment: {experiment_id}")
    if adapter:
        title_parts.append(f"Adapter: {adapter}")
    if operation:
        title_parts.append(f"Operation: {operation}")
    ax.set_title("\n".join(title_parts))

    ax.legend()
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    console.print(f"[green]  Saved {output_path}[/green]")


def plot_tail_log(
    data: np.ndarray,
    output_path: Path,
    title: str = "Latency Tail Distribution",
    xlabel: str = "Latency (μs)",
    experiment_id: Optional[str] = None,
    adapter: Optional[str] = None,
    operation: Optional[str] = None,
) -> None:
    """Plot log-scale tail distribution (1 - CDF)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort data for tail distribution
    sorted_data = np.sort(data)
    # Survival function (1 - CDF)
    sf = 1 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    # Avoid log(0)
    sf = np.maximum(sf, 1e-10)

    ax.semilogy(sorted_data, sf, linewidth=2, color="#2c3e50")

    # Add percentile markers
    percentiles = [90, 95, 99, 99.9]
    colors = ["#27ae60", "#f39c12", "#e74c3c", "#8e44ad"]
    for p, color in zip(percentiles, colors):
        val = np.percentile(data, p)
        ax.axvline(val, linestyle="--", color=color, alpha=0.8, label=f"p{p}: {val:.0f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("P(Latency > x) [log scale]")

    # Build title
    title_parts = [title]
    if experiment_id:
        title_parts.append(f"Experiment: {experiment_id}")
    if adapter:
        title_parts.append(f"Adapter: {adapter}")
    if operation:
        title_parts.append(f"Operation: {operation}")
    ax.set_title("\n".join(title_parts))

    ax.legend(loc="upper right")
    ax.set_ylim(bottom=1e-5)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    console.print(f"[green]  Saved {output_path}[/green]")


def plot_latency(
    input_path: Path,
    output_dir: Path,
    column: str = "latency_us",
    experiment_id: Optional[str] = None,
) -> None:
    """Generate all latency plots."""
    console.print(f"[bold blue]Generating latency plots[/bold blue]")
    console.print(f"  Input: {input_path}")
    console.print(f"  Output: {output_dir}")

    # Load data
    df = load_data(input_path)

    if column not in df.columns:
        console.print(f"[red]Column '{column}' not found in data![/red]")
        return

    data = df[column].dropna().values
    console.print(f"  Loaded {len(data)} samples")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get metadata for titles
    exp_id = experiment_id or input_path.parent.parent.name
    adapter = df["algorithm"].iloc[0] if "algorithm" in df.columns else None
    operation = df["operation"].iloc[0] if "operation" in df.columns else None

    # Generate plots
    console.print("[cyan]Generating CDF plot...[/cyan]")
    plot_cdf(
        data,
        output_dir / "latency_cdf.png",
        experiment_id=exp_id,
        adapter=adapter,
        operation=operation,
    )

    console.print("[cyan]Generating PDF plot...[/cyan]")
    plot_pdf(
        data,
        output_dir / "latency_pdf.png",
        experiment_id=exp_id,
        adapter=adapter,
        operation=operation,
    )

    console.print("[cyan]Generating tail distribution plot...[/cyan]")
    plot_tail_log(
        data,
        output_dir / "latency_tail.png",
        experiment_id=exp_id,
        adapter=adapter,
        operation=operation,
    )

    # Per-algorithm plots if available
    if "algorithm" in df.columns and df["algorithm"].nunique() > 1:
        console.print("[cyan]Generating per-algorithm comparison...[/cyan]")
        plot_algorithm_comparison(df, output_dir, column=column)

    console.print("[bold green]Latency plots complete![/bold green]")


def plot_algorithm_comparison(
    df: pd.DataFrame,
    output_dir: Path,
    column: str = "latency_us",
) -> None:
    """Plot latency comparison across algorithms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    algorithms = df["algorithm"].unique()

    # CDF comparison
    ax1 = axes[0]
    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
    for algo, color in zip(algorithms, colors):
        data = df[df["algorithm"] == algo][column].dropna().values
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax1.plot(sorted_data, cdf, linewidth=2, label=algo, color=color)

    ax1.set_xlabel(f"{column}")
    ax1.set_ylabel("Cumulative Probability")
    ax1.set_title("Latency CDF by Algorithm")
    ax1.legend()

    # Box plot comparison
    ax2 = axes[1]
    data_by_algo = [df[df["algorithm"] == algo][column].dropna().values for algo in algorithms]
    bp = ax2.boxplot(data_by_algo, labels=algorithms, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_xlabel("Algorithm")
    ax2.set_ylabel(f"{column}")
    ax2.set_title("Latency Distribution by Algorithm")

    plt.tight_layout()
    plt.savefig(output_dir / "latency_algorithm_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    console.print(f"[green]  Saved {output_dir / 'latency_algorithm_comparison.png'}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Generate latency plots")
    parser.add_argument("--input", required=True, help="Input JSONL or Parquet file")
    parser.add_argument("--output", required=True, help="Output directory for plots")
    parser.add_argument("--column", default="latency_us", help="Column to plot")
    parser.add_argument("--experiment-id", help="Experiment identifier for titles")

    args = parser.parse_args()

    plot_latency(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        column=args.column,
        experiment_id=args.experiment_id,
    )


if __name__ == "__main__":
    main()
