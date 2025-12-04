#!/usr/bin/env python3
"""
Compute statistical summaries from experiment data.

Computes:
- Latency percentiles (p50, p90, p95, p99)
- Mean, std, variance, count
- Throughput time series
- Per-second event counts

Usage:
    python compute_stats.py --input merged.parquet --output stats/ --experiment-id exp1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

console = Console()


def compute_latency_stats(df: pd.DataFrame) -> dict:
    """Compute latency statistics."""
    # Find latency column
    latency_col = None
    for col in ["latency_us", "latency_µs", "latency_ms", "latency"]:
        if col in df.columns:
            latency_col = col
            break
    
    if latency_col is None:
        return {}
    
    latency = df[latency_col].dropna().values
    
    if len(latency) == 0:
        return {}
    
    return {
        "count": len(latency),
        "mean": float(np.mean(latency)),
        "std": float(np.std(latency)),
        "variance": float(np.var(latency)),
        "min": float(np.min(latency)),
        "max": float(np.max(latency)),
        "p50": float(np.percentile(latency, 50)),
        "p90": float(np.percentile(latency, 90)),
        "p95": float(np.percentile(latency, 95)),
        "p99": float(np.percentile(latency, 99)),
        "p999": float(np.percentile(latency, 99.9)),
    }


def compute_throughput_series(df: pd.DataFrame) -> dict:
    """Compute throughput time series (events per second)."""
    # Find timestamp column
    ts_col = None
    for col in ["timestamp_monotonic_ns", "timestamp_utc_iso", "timestamp"]:
        if col in df.columns:
            ts_col = col
            break
    
    if ts_col is None:
        return {}
    
    # Convert to seconds from start
    if ts_col == "timestamp_monotonic_ns":
        ts = df[ts_col].values / 1e9  # ns to seconds
    else:
        ts = pd.to_datetime(df[ts_col])
        ts = (ts - ts.min()).dt.total_seconds().values
    
    # Bucket into 1-second intervals
    buckets = np.floor(ts).astype(int)
    max_bucket = int(buckets.max()) + 1
    
    throughput = np.zeros(max_bucket)
    for b in buckets:
        throughput[b] += 1
    
    # Compute statistics
    nonzero = throughput[throughput > 0]
    
    return {
        "total_duration_sec": float(max_bucket),
        "total_events": int(len(df)),
        "mean_msgs_per_sec": float(np.mean(nonzero)) if len(nonzero) > 0 else 0,
        "max_msgs_per_sec": float(np.max(throughput)),
        "min_msgs_per_sec": float(np.min(nonzero)) if len(nonzero) > 0 else 0,
        "std_msgs_per_sec": float(np.std(nonzero)) if len(nonzero) > 0 else 0,
        "time_series": throughput.tolist(),
    }


def compute_queue_delay_stats(df: pd.DataFrame) -> dict:
    """Compute queue delay statistics."""
    if "queue_delay_us" not in df.columns:
        return {}
    
    delay = df["queue_delay_us"].dropna().values
    
    if len(delay) == 0:
        return {}
    
    return {
        "count": len(delay),
        "mean": float(np.mean(delay)),
        "std": float(np.std(delay)),
        "p50": float(np.percentile(delay, 50)),
        "p99": float(np.percentile(delay, 99)),
        "max": float(np.max(delay)),
    }


def compute_stats(
    input_path: Path,
    output_dir: Path,
    experiment_id: str = "",
) -> dict:
    """Compute all statistics."""
    console.print(f"[bold blue]Computing Statistics[/bold blue]")
    console.print(f"  Input: {input_path}")
    console.print(f"  Output: {output_dir}")
    
    # Load data
    console.print("[cyan]Loading data...[/cyan]")
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_json(input_path, lines=True)
    
    console.print(f"  Loaded {len(df)} events")
    
    # Compute statistics
    console.print("[cyan]Computing latency statistics...[/cyan]")
    latency_stats = compute_latency_stats(df)
    
    console.print("[cyan]Computing throughput series...[/cyan]")
    throughput_stats = compute_throughput_series(df)
    
    console.print("[cyan]Computing queue delay statistics...[/cyan]")
    queue_stats = compute_queue_delay_stats(df)
    
    # Compile summary
    summary = {
        "experiment_id": experiment_id,
        "total_events": len(df),
        "latency": latency_stats,
        "throughput": throughput_stats,
        "queue_delay": queue_stats,
    }
    
    # Add algorithm info if available
    if "algorithm" in df.columns:
        summary["algorithms"] = df["algorithm"].unique().tolist()
    
    if "operation" in df.columns:
        summary["operations"] = df["operation"].unique().tolist()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary JSON (without time series for compact version)
    summary_compact = {k: v for k, v in summary.items() if k != "throughput"}
    summary_compact["throughput"] = {k: v for k, v in throughput_stats.items() if k != "time_series"}
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_compact, f, indent=2)
    console.print(f"[green]  Saved: {summary_path}[/green]")
    
    # Save full summary with time series
    full_summary_path = output_dir / "summary_full.json"
    with open(full_summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    console.print(f"[green]  Saved: {full_summary_path}[/green]")
    
    # Print summary
    console.print("\n[bold]Statistics Summary:[/bold]")
    if latency_stats:
        console.print(f"  Latency (μs): mean={latency_stats['mean']:.1f}, p50={latency_stats['p50']:.0f}, p99={latency_stats['p99']:.0f}")
    if throughput_stats:
        console.print(f"  Throughput: {throughput_stats.get('mean_msgs_per_sec', 0):.0f} events/sec (avg)")
    console.print(f"  Total events: {len(df)}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compute statistics from experiment data")
    parser.add_argument("--input", required=True, help="Input parquet or JSONL file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--experiment-id", default="", help="Experiment ID")
    
    args = parser.parse_args()
    
    compute_stats(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        experiment_id=args.experiment_id,
    )
    
    console.print("[bold green]Statistics computation complete![/bold green]")


if __name__ == "__main__":
    main()

