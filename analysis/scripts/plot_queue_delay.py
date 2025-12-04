#!/usr/bin/env python3
"""
Generate queue delay analysis plots.

Usage:
    python plot_queue_delay.py --input ./data/exp_001/merged/merged.jsonl --output ./figures/exp_001/
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def plot_queue_delay_distribution(
    df: pd.DataFrame,
    output_path: Path,
    column: str = "queue_delay_us",
    experiment_id: Optional[str] = None,
) -> None:
    """Plot queue delay distribution."""
    if column not in df.columns:
        console.print(f"[yellow]Column '{column}' not found, skipping[/yellow]")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    data = df[column].dropna().values

    # Linear scale histogram
    ax1 = axes[0]
    ax1.hist(data, bins=100, edgecolor="white", alpha=0.7, color="#9b59b6")

    mean_val = np.mean(data)
    median_val = np.median(data)
    ax1.axvline(mean_val, color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.0f}")
    ax1.axvline(median_val, color="#27ae60", linestyle="--", linewidth=2, label=f"Median: {median_val:.0f}")

    ax1.set_xlabel(f"{column}")
    ax1.set_ylabel("Frequency")
    title = "Queue Delay Distribution"
    if experiment_id:
        title += f"\nExperiment: {experiment_id}"
    ax1.set_title(title)
    ax1.legend()

    # CDF
    ax2 = axes[1]
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax2.plot(sorted_data, cdf, linewidth=2, color="#9b59b6")

    # Percentile markers
    for p, color in [(50, "#27ae60"), (90, "#f39c12"), (99, "#e74c3c")]:
        val = np.percentile(data, p)
        ax2.axvline(val, linestyle="--", color=color, label=f"p{p}: {val:.0f}")

    ax2.set_xlabel(f"{column}")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("Queue Delay CDF")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    console.print(f"[green]  Saved {output_path}[/green]")


def plot_queue_delay_vs_load(
    df: pd.DataFrame,
    output_path: Path,
    experiment_id: Optional[str] = None,
) -> None:
    """Plot queue delay correlation with load."""
    if "queue_delay_us" not in df.columns:
        console.print("[yellow]No queue_delay_us column, skipping load correlation plot[/yellow]")
        return

    if "timestamp" not in df.columns:
        if "timestamp_utc_iso" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp_utc_iso"])
        else:
            console.print("[yellow]No timestamp column, skipping load correlation plot[/yellow]")
            return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Calculate load (messages per second)
    df["second"] = df["timestamp"].dt.floor("S")
    load_per_second = df.groupby("second").size()
    df["load"] = df["second"].map(load_per_second)

    # Scatter plot: queue delay vs load
    ax1 = axes[0, 0]
    sample_size = min(10000, len(df))
    sample = df.sample(sample_size)
    ax1.scatter(sample["load"], sample["queue_delay_us"], alpha=0.3, s=10, color="#9b59b6")
    ax1.set_xlabel("Load (messages/second)")
    ax1.set_ylabel("Queue Delay (μs)")
    title = "Queue Delay vs Load"
    if experiment_id:
        title += f" - {experiment_id}"
    ax1.set_title(title)

    # Binned average
    ax2 = axes[0, 1]
    df["load_bin"] = pd.cut(df["load"], bins=20)
    binned_mean = df.groupby("load_bin")["queue_delay_us"].mean()
    binned_std = df.groupby("load_bin")["queue_delay_us"].std()

    x = range(len(binned_mean))
    ax2.bar(x, binned_mean.values, yerr=binned_std.values, alpha=0.7, color="#9b59b6", capsize=3)
    ax2.set_xlabel("Load Bin")
    ax2.set_ylabel("Mean Queue Delay (μs)")
    ax2.set_title("Average Queue Delay by Load Bin")
    ax2.set_xticks([])

    # Queue delay over time
    ax3 = axes[1, 0]
    queue_delay_per_second = df.groupby("second")["queue_delay_us"].mean()
    time_seconds = (queue_delay_per_second.index - queue_delay_per_second.index.min()).total_seconds()
    ax3.plot(time_seconds, queue_delay_per_second.values, linewidth=1.5, color="#9b59b6", alpha=0.7)
    ax3.fill_between(time_seconds, queue_delay_per_second.values, alpha=0.3, color="#9b59b6")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Mean Queue Delay (μs)")
    ax3.set_title("Queue Delay Over Time")

    # Load over time with queue delay overlay
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()

    ax4.plot(time_seconds, load_per_second.values, linewidth=1.5, color="#3498db", label="Load")
    ax4_twin.plot(time_seconds, queue_delay_per_second.values, linewidth=1.5, color="#e74c3c", label="Queue Delay")

    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel("Load (messages/second)", color="#3498db")
    ax4_twin.set_ylabel("Queue Delay (μs)", color="#e74c3c")
    ax4.set_title("Load and Queue Delay Over Time")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    console.print(f"[green]  Saved {output_path}[/green]")


def plot_queue_delay_by_worker(
    df: pd.DataFrame,
    output_path: Path,
    experiment_id: Optional[str] = None,
) -> None:
    """Plot queue delay by worker."""
    if "queue_delay_us" not in df.columns or "worker_id" not in df.columns:
        console.print("[yellow]Missing required columns, skipping per-worker plot[/yellow]")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot by worker
    ax1 = axes[0]
    workers = sorted(df["worker_id"].unique())
    data_by_worker = [df[df["worker_id"] == w]["queue_delay_us"].dropna().values for w in workers]
    colors = plt.cm.Set2(np.linspace(0, 1, len(workers)))

    bp = ax1.boxplot(data_by_worker, labels=workers, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax1.set_xlabel("Worker ID")
    ax1.set_ylabel("Queue Delay (μs)")
    title = "Queue Delay Distribution by Worker"
    if experiment_id:
        title += f"\n{experiment_id}"
    ax1.set_title(title)

    # Mean queue delay per worker
    ax2 = axes[1]
    mean_by_worker = df.groupby("worker_id")["queue_delay_us"].mean()
    std_by_worker = df.groupby("worker_id")["queue_delay_us"].std()

    ax2.bar(mean_by_worker.index, mean_by_worker.values, yerr=std_by_worker.values,
            alpha=0.7, color=colors, capsize=3)
    ax2.axhline(mean_by_worker.mean(), color="#e74c3c", linestyle="--", label=f"Overall mean: {mean_by_worker.mean():.0f}")
    ax2.set_xlabel("Worker ID")
    ax2.set_ylabel("Mean Queue Delay (μs)")
    ax2.set_title("Mean Queue Delay by Worker")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    console.print(f"[green]  Saved {output_path}[/green]")


def plot_queue_delay(
    input_path: Path,
    output_dir: Path,
    experiment_id: Optional[str] = None,
) -> None:
    """Generate all queue delay plots."""
    console.print(f"[bold blue]Generating queue delay plots[/bold blue]")
    console.print(f"  Input: {input_path}")
    console.print(f"  Output: {output_dir}")

    # Load data
    df = load_data(input_path)
    console.print(f"  Loaded {len(df)} events")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_id = experiment_id or input_path.parent.parent.name

    # Generate plots
    console.print("[cyan]Generating distribution plot...[/cyan]")
    plot_queue_delay_distribution(df, output_dir / "queue_delay_distribution.png", experiment_id=exp_id)

    console.print("[cyan]Generating load correlation plot...[/cyan]")
    plot_queue_delay_vs_load(df, output_dir / "queue_delay_vs_load.png", experiment_id=exp_id)

    console.print("[cyan]Generating per-worker plot...[/cyan]")
    plot_queue_delay_by_worker(df, output_dir / "queue_delay_by_worker.png", experiment_id=exp_id)

    console.print("[bold green]Queue delay plots complete![/bold green]")


def main():
    parser = argparse.ArgumentParser(description="Generate queue delay plots")
    parser.add_argument("--input", required=True, help="Input JSONL or Parquet file")
    parser.add_argument("--output", required=True, help="Output directory for plots")
    parser.add_argument("--experiment-id", help="Experiment identifier for titles")

    args = parser.parse_args()

    plot_queue_delay(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        experiment_id=args.experiment_id,
    )


if __name__ == "__main__":
    main()
