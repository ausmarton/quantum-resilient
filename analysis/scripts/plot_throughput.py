#!/usr/bin/env python3
"""
Generate throughput plots over time.

Usage:
    python plot_throughput.py --input ./data/exp_001/merged/merged.jsonl --output ./figures/exp_001/
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


def plot_throughput_timeseries(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Throughput Over Time",
    experiment_id: Optional[str] = None,
) -> None:
    """Plot throughput timeseries."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Ensure timestamp column
    if "timestamp" not in df.columns:
        if "timestamp_utc_iso" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp_utc_iso"])
        else:
            console.print("[red]No timestamp column found![/red]")
            return

    # Group by second
    df["second"] = df["timestamp"].dt.floor("S")
    throughput = df.groupby("second").size()

    # Normalize time to start at 0
    time_seconds = (throughput.index - throughput.index.min()).total_seconds()

    # Raw throughput
    ax1 = axes[0]
    ax1.plot(time_seconds, throughput.values, linewidth=1.5, color="#2c3e50", alpha=0.7)
    ax1.fill_between(time_seconds, throughput.values, alpha=0.3, color="#3498db")

    mean_tput = throughput.mean()
    ax1.axhline(mean_tput, color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: {mean_tput:.0f} msg/s")

    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Messages/second")
    title_str = title
    if experiment_id:
        title_str += f"\nExperiment: {experiment_id}"
    ax1.set_title(title_str)
    ax1.legend(loc="upper right")

    # Rolling average with confidence band
    ax2 = axes[1]
    window = min(10, len(throughput) // 4)
    if window > 1:
        rolling_mean = throughput.rolling(window=window, center=True).mean()
        rolling_std = throughput.rolling(window=window, center=True).std()

        ax2.plot(time_seconds, rolling_mean.values, linewidth=2, color="#2c3e50", label=f"Rolling mean ({window}s)")
        ax2.fill_between(
            time_seconds,
            (rolling_mean - rolling_std).values,
            (rolling_mean + rolling_std).values,
            alpha=0.3,
            color="#3498db",
            label="±1 std dev",
        )

    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Messages/second")
    ax2.set_title(f"Smoothed Throughput ({window}s Rolling Window)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    console.print(f"[green]  Saved {output_path}[/green]")


def plot_throughput_distribution(
    df: pd.DataFrame,
    output_path: Path,
    experiment_id: Optional[str] = None,
) -> None:
    """Plot throughput distribution histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Ensure timestamp column
    if "timestamp" not in df.columns:
        if "timestamp_utc_iso" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp_utc_iso"])
        else:
            return

    # Group by second
    df["second"] = df["timestamp"].dt.floor("S")
    throughput = df.groupby("second").size()

    ax.hist(throughput.values, bins=50, edgecolor="white", alpha=0.7, color="#3498db")

    # Add statistics
    mean_val = throughput.mean()
    median_val = throughput.median()
    std_val = throughput.std()

    ax.axvline(mean_val, color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.0f}")
    ax.axvline(median_val, color="#27ae60", linestyle="--", linewidth=2, label=f"Median: {median_val:.0f}")

    ax.set_xlabel("Throughput (messages/second)")
    ax.set_ylabel("Frequency (seconds)")

    title = "Throughput Distribution"
    if experiment_id:
        title += f"\nExperiment: {experiment_id}"
    ax.set_title(title)
    ax.legend()

    # Add text box with stats
    textstr = f"Mean: {mean_val:.0f}\nStd: {std_val:.0f}\nCV: {std_val/mean_val:.2%}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right", bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    console.print(f"[green]  Saved {output_path}[/green]")


def plot_worker_throughput(
    df: pd.DataFrame,
    output_path: Path,
    experiment_id: Optional[str] = None,
) -> None:
    """Plot per-worker throughput comparison."""
    if "worker_id" not in df.columns:
        console.print("[yellow]No worker_id column, skipping per-worker plot[/yellow]")
        return

    if "timestamp" not in df.columns:
        if "timestamp_utc_iso" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp_utc_iso"])
        else:
            return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Group by worker and second
    df["second"] = df["timestamp"].dt.floor("S")

    # Per-worker total events
    ax1 = axes[0]
    events_per_worker = df.groupby("worker_id").size()
    colors = plt.cm.Set2(np.linspace(0, 1, len(events_per_worker)))
    bars = ax1.bar(events_per_worker.index, events_per_worker.values, color=colors)
    ax1.axhline(events_per_worker.mean(), color="#e74c3c", linestyle="--", label=f"Mean: {events_per_worker.mean():.0f}")
    ax1.set_xlabel("Worker ID")
    ax1.set_ylabel("Total Events")
    ax1.set_title("Events per Worker")
    ax1.legend()

    # Per-worker throughput over time
    ax2 = axes[1]
    for worker_id in sorted(df["worker_id"].unique()):
        worker_df = df[df["worker_id"] == worker_id]
        throughput = worker_df.groupby("second").size()
        time_seconds = (throughput.index - df["second"].min()).total_seconds()
        ax2.plot(time_seconds, throughput.values, linewidth=1, alpha=0.7, label=f"Worker {worker_id}")

    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Messages/second")
    ax2.set_title("Per-Worker Throughput Over Time")
    if len(df["worker_id"].unique()) <= 10:
        ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    console.print(f"[green]  Saved {output_path}[/green]")


def plot_throughput(
    input_path: Path,
    output_dir: Path,
    experiment_id: Optional[str] = None,
) -> None:
    """Generate all throughput plots."""
    console.print(f"[bold blue]Generating throughput plots[/bold blue]")
    console.print(f"  Input: {input_path}")
    console.print(f"  Output: {output_dir}")

    # Load data
    df = load_data(input_path)
    console.print(f"  Loaded {len(df)} events")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_id = experiment_id or input_path.parent.parent.name

    # Generate plots
    console.print("[cyan]Generating timeseries plot...[/cyan]")
    plot_throughput_timeseries(df, output_dir / "throughput_timeseries.png", experiment_id=exp_id)

    console.print("[cyan]Generating distribution plot...[/cyan]")
    plot_throughput_distribution(df, output_dir / "throughput_distribution.png", experiment_id=exp_id)

    console.print("[cyan]Generating per-worker plot...[/cyan]")
    plot_worker_throughput(df, output_dir / "throughput_per_worker.png", experiment_id=exp_id)

    console.print("[bold green]Throughput plots complete![/bold green]")


def main():
    parser = argparse.ArgumentParser(description="Generate throughput plots")
    parser.add_argument("--input", required=True, help="Input JSONL or Parquet file")
    parser.add_argument("--output", required=True, help="Output directory for plots")
    parser.add_argument("--experiment-id", help="Experiment identifier for titles")

    args = parser.parse_args()

    plot_throughput(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        experiment_id=args.experiment_id,
    )


if __name__ == "__main__":
    main()
