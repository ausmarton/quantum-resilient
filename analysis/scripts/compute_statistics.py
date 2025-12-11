#!/usr/bin/env python3
"""
Compute statistical summaries from merged experiment data.

Usage:
    python compute_statistics.py --input ./data/exp_001/merged/merged.jsonl --output ./data/exp_001/stats/
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console
from scipy import stats

console = Console()

# Set style for plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def load_data(filepath: Path) -> pd.DataFrame:
    """Load data from JSONL or Parquet file."""
    if filepath.suffix == ".parquet":
        return pd.read_parquet(filepath)
    else:
        # Try to load JSONL with error handling for malformed lines
        try:
        return pd.read_json(filepath, lines=True)
        except ValueError as e:
            # If pandas fails, try reading line by line and skipping bad lines
            console.print(f"[yellow]Warning: pandas read_json failed, trying line-by-line parsing: {e}[/yellow]")
            import json
            records = []
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as je:
                        console.print(f"[yellow]Warning: Skipping malformed JSON at line {line_num}: {je}[/yellow]")
                        continue
            if not records:
                raise ValueError(f"No valid JSON records found in {filepath}")
            return pd.DataFrame(records)


def convert_to_python_types(obj):
    """Convert numpy/pandas types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj


def compute_percentiles(series: pd.Series) -> dict:
    """Compute standard percentiles for a series."""
    return {
        "p50": float(series.quantile(0.50)),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "p999": float(series.quantile(0.999)),
    }


def compute_basic_stats(series: pd.Series) -> dict:
    """Compute basic statistics for a series."""
    return {
        "count": int(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "var": float(series.var()),
        "min": float(series.min()),
        "max": float(series.max()),
        **compute_percentiles(series),
    }


def compute_throughput_stats(df: pd.DataFrame) -> dict:
    """Compute throughput statistics over time."""
    if "timestamp" not in df.columns:
        if "timestamp_utc_iso" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp_utc_iso"])
        else:
            return {"error": "No timestamp column found"}

    # Group by second and count events
    df["second"] = df["timestamp"].dt.floor("S")
    throughput_per_second = df.groupby("second").size()

    return {
        "mean_msgs_per_sec": float(throughput_per_second.mean()),
        "max_msgs_per_sec": float(throughput_per_second.max()),
        "min_msgs_per_sec": float(throughput_per_second.min()),
        "std_msgs_per_sec": float(throughput_per_second.std()),
        "total_duration_sec": float(len(throughput_per_second)),
        "total_messages": int(len(df)),
    }


def detect_drift(df: pd.DataFrame, threshold_ms: float = 100.0) -> dict:
    """Detect start-time synchronization drift between workers."""
    if "worker_id" not in df.columns or "timestamp" not in df.columns:
        return {"warning": "Cannot detect drift without worker_id and timestamp"}

    # Get first timestamp per worker
    first_timestamps = df.groupby("worker_id")["timestamp"].min()

    if len(first_timestamps) < 2:
        return {"workers": 1, "drift_detected": False}

    # Calculate drift
    drift_ms = (first_timestamps.max() - first_timestamps.min()).total_seconds() * 1000

    return {
        "workers": len(first_timestamps),
        "max_drift_ms": float(drift_ms),
        "drift_detected": bool(drift_ms > threshold_ms),
        "threshold_ms": threshold_ms,
        "first_timestamps": {
            int(k): str(v) for k, v in first_timestamps.to_dict().items()
        },
    }


def detect_worker_skew(df: pd.DataFrame) -> dict:
    """Detect event count skew between workers."""
    if "worker_id" not in df.columns:
        return {"warning": "No worker_id column"}

    events_per_worker = df.groupby("worker_id").size()

    mean_events = events_per_worker.mean()
    std_events = events_per_worker.std()
    cv = std_events / mean_events if mean_events > 0 else 0

    return {
        "events_per_worker": {int(k): int(v) for k, v in events_per_worker.to_dict().items()},
        "mean_events": float(mean_events),
        "std_events": float(std_events),
        "coefficient_of_variation": float(cv),
        "skew_detected": cv > 0.1,  # More than 10% variation
    }


def plot_latency_histogram(
    df: pd.DataFrame,
    output_path: Path,
    column: str = "latency_us",
    title: str = "Latency Distribution",
) -> None:
    """Plot latency histogram."""
    # Expect latency_ns to be present, convert to microseconds for plotting
    if column == "latency_us" and "latency_ns" in df.columns:
        df = df.copy()
        df["latency_us"] = df["latency_ns"] / 1000.0
    
    if column not in df.columns:
        console.print(f"[yellow]Column {column} not found, skipping histogram[/yellow]")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    data = df[column].dropna()

    # Linear scale histogram
    ax1 = axes[0]
    ax1.hist(data, bins=100, edgecolor="black", alpha=0.7, color="#3498db")
    ax1.set_xlabel(f"{column}")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"{title} (Linear Scale)")
    ax1.axvline(data.median(), color="red", linestyle="--", label=f"Median: {data.median():.0f}")
    ax1.axvline(data.mean(), color="orange", linestyle="--", label=f"Mean: {data.mean():.0f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Log scale histogram (for tail analysis)
    ax2 = axes[1]
    ax2.hist(data, bins=100, edgecolor="black", alpha=0.7, color="#e74c3c")
    ax2.set_xlabel(f"{column}")
    ax2.set_ylabel("Frequency (log scale)")
    ax2.set_yscale("log")
    ax2.set_title(f"{title} (Log Scale)")
    ax2.axvline(
        data.quantile(0.99),
        color="purple",
        linestyle="--",
        label=f"p99: {data.quantile(0.99):.0f}",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"[green]  Saved {output_path}[/green]")


def plot_queue_histogram(df: pd.DataFrame, output_path: Path) -> None:
    """Plot queue delay histogram."""
    plot_latency_histogram(
        df,
        output_path,
        column="queue_delay_us",
        title="Queue Delay Distribution",
    )


def plot_throughput_curve(df: pd.DataFrame, output_path: Path) -> None:
    """Plot throughput over time."""
    if "timestamp" not in df.columns:
        if "timestamp_utc_iso" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp_utc_iso"])
        else:
            console.print("[yellow]No timestamp column, skipping throughput plot[/yellow]")
            return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Group by second
    df["second"] = df["timestamp"].dt.floor("S")
    throughput = df.groupby("second").size()

    # Raw throughput
    ax1 = axes[0]
    ax1.plot(range(len(throughput)), throughput.values, linewidth=1.5, color="#2ecc71")
    ax1.fill_between(range(len(throughput)), throughput.values, alpha=0.3, color="#2ecc71")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Messages/second")
    ax1.set_title("Throughput Over Time")
    ax1.axhline(
        throughput.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {throughput.mean():.0f} msg/s",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Rolling average
    ax2 = axes[1]
    window = min(10, len(throughput) // 4)
    if window > 0:
        rolling = throughput.rolling(window=window, center=True).mean()
        ax2.plot(range(len(rolling)), rolling.values, linewidth=2, color="#9b59b6")
        ax2.fill_between(
            range(len(rolling)),
            rolling.values,
            alpha=0.3,
            color="#9b59b6",
        )
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Messages/second (rolling avg)")
    ax2.set_title(f"Throughput (Rolling {window}s Average)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"[green]  Saved {output_path}[/green]")


def compute_statistics(
    input_path: Path,
    output_dir: Path,
    experiment_id: Optional[str] = None,
) -> dict:
    """Compute statistics and generate plots."""
    console.print(f"[bold blue]Computing statistics[/bold blue]")
    console.print(f"  Input: {input_path}")
    console.print(f"  Output: {output_dir}")

    # Load data
    console.print("[cyan]Loading data...[/cyan]")
    df = load_data(input_path)
    console.print(f"  Loaded {len(df)} events")

    if df.empty:
        console.print("[red]No data to analyze![/red]")
        return {}

    # Ensure timestamp column
    if "timestamp" not in df.columns and "timestamp_utc_iso" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp_utc_iso"])

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute statistics
    console.print("[cyan]Computing statistics...[/cyan]")

    summary = {
        "experiment_id": experiment_id or input_path.parent.parent.name,
        "total_events": len(df),
    }

    # Latency stats - handle both nanosecond precision (new) and microsecond precision (old) formats
    if "latency_ns" in df.columns:
        # New format: nanosecond precision
    # Convert nanoseconds to microseconds for analysis
    df["latency_us"] = df["latency_ns"] / 1000.0
    summary["latency"] = compute_basic_stats(df["latency_us"])
    summary["latency_ns"] = compute_basic_stats(df["latency_ns"])  # Store nanosecond stats
    elif "latency_us" in df.columns:
        # Old format: microsecond precision only (backward compatibility)
        # Convert microseconds to nanoseconds for consistency (approximate)
        df["latency_ns"] = df["latency_us"] * 1000.0
        summary["latency"] = compute_basic_stats(df["latency_us"])
        summary["latency_ns"] = compute_basic_stats(df["latency_ns"])  # Approximate nanosecond stats
        summary["_note"] = "Data in legacy microsecond format - latency_ns is approximate"
    else:
        raise ValueError("Missing required column: latency_ns or latency_us. Data must include latency information.")

    # Queue delay stats - expect queue_delay_ns
    if "queue_delay_ns" in df.columns:
        df["queue_delay_us"] = df["queue_delay_ns"] / 1000.0
        summary["queue_delay"] = compute_basic_stats(df["queue_delay_us"])
        summary["queue_delay_ns"] = compute_basic_stats(df["queue_delay_ns"])
    elif "queue_delay_us" in df.columns:
        # Legacy fallback (shouldn't happen with new data)
        summary["queue_delay"] = compute_basic_stats(df["queue_delay_us"])
    else:
        summary["queue_delay"] = {"count": len(df), "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    # Crypto latency stats
    if "crypto_latency_us" in df.columns:
        summary["crypto_latency"] = compute_basic_stats(df["crypto_latency_us"])

    # Throughput stats
    summary["throughput"] = compute_throughput_stats(df)

    # Memory utilization stats
    if "memory_rss_bytes" in df.columns:
        memory_series = df["memory_rss_bytes"]
        summary["memory"] = {
            "mean_rss_bytes": float(memory_series.mean()),
            "max_rss_bytes": int(memory_series.max()),
            "min_rss_bytes": int(memory_series.min()),
            "std_rss_bytes": float(memory_series.std()),
            "p50_rss_bytes": float(memory_series.quantile(0.50)),
            "p95_rss_bytes": float(memory_series.quantile(0.95)),
            "p99_rss_bytes": float(memory_series.quantile(0.99)),
            "mean_rss_mb": float(memory_series.mean() / 1_000_000.0),
            "max_rss_mb": float(memory_series.max() / 1_000_000.0),
        }
    else:
        summary["memory"] = {"note": "Memory data not available"}

    # CPU utilization stats (if CPU data is valid)
    if "cpu_user_seconds" in df.columns and "timestamp" in df.columns:
        # Calculate CPU delta between events
        df = df.copy()  # Avoid SettingWithCopyWarning
        df["cpu_delta"] = df["cpu_user_seconds"].diff()
        df["time_delta"] = df["timestamp"].diff().dt.total_seconds()
        
        # Filter out invalid deltas (first row, negative values, zero time deltas)
        valid_mask = (df["cpu_delta"] > 0) & (df["time_delta"] > 0)
        df_valid = df[valid_mask].copy()
        
        if len(df_valid) > 0:
            # Calculate CPU utilization (CPU time / wall time)
            df_valid["cpu_utilization"] = df_valid["cpu_delta"] / df_valid["time_delta"]
            
            # Calculate total CPU time and CPU per operation
            total_cpu_seconds = float(df["cpu_user_seconds"].iloc[-1])
            total_events = len(df)
            
            summary["cpu"] = {
                "mean_utilization": float(df_valid["cpu_utilization"].mean()),
                "max_utilization": float(df_valid["cpu_utilization"].max()),
                "min_utilization": float(df_valid["cpu_utilization"].min()),
                "std_utilization": float(df_valid["cpu_utilization"].std()),
                "p50_utilization": float(df_valid["cpu_utilization"].quantile(0.50)),
                "p95_utilization": float(df_valid["cpu_utilization"].quantile(0.95)),
                "p99_utilization": float(df_valid["cpu_utilization"].quantile(0.99)),
                "cpu_per_operation_seconds": float(total_cpu_seconds / total_events) if total_events > 0 else 0.0,
                "total_cpu_seconds": total_cpu_seconds,
                "total_events": total_events,
                "valid_delta_count": len(df_valid),
            }
        else:
            # All zeros or no valid deltas - document limitation
            summary["cpu"] = {
                "note": "CPU data unavailable (operations too fast for sampling or all zeros)",
                "all_zeros": bool((df["cpu_user_seconds"] == 0.0).all()),
                "total_cpu_seconds": float(df["cpu_user_seconds"].iloc[-1]) if len(df) > 0 else 0.0,
                "total_events": int(len(df)),
            }
    else:
        summary["cpu"] = {"note": "CPU data not available"}

    # Worker skew detection
    summary["worker_skew"] = detect_worker_skew(df)

    # Drift detection
    summary["drift"] = detect_drift(df)

    # Per-algorithm stats
    if "algorithm" in df.columns:
        summary["per_algorithm"] = {}
        for algo in df["algorithm"].unique():
            algo_df = df[df["algorithm"] == algo]
            algo_stats = {}
            
            # Handle both nanosecond (new) and microsecond (old) formats
            if "latency_ns" in algo_df.columns:
                algo_df = algo_df.copy()
                algo_df["latency_us"] = algo_df["latency_ns"] / 1000.0
                algo_stats["latency"] = compute_basic_stats(algo_df["latency_us"])
            elif "latency_us" in algo_df.columns:
                # Old format: backward compatibility
                algo_df = algo_df.copy()
                algo_df["latency_ns"] = algo_df["latency_us"] * 1000.0
                algo_stats["latency"] = compute_basic_stats(algo_df["latency_us"])
            else:
                raise ValueError(f"Missing latency_ns or latency_us column for algorithm {algo}")
            
            # Per-algorithm memory stats
            if "memory_rss_bytes" in algo_df.columns:
                memory_series = algo_df["memory_rss_bytes"]
                algo_stats["memory"] = {
                    "mean_rss_bytes": float(memory_series.mean()),
                    "max_rss_bytes": int(memory_series.max()),
                    "mean_rss_mb": float(memory_series.mean() / 1_000_000.0),
                }
            
            # Per-algorithm CPU stats (if available)
            if "cpu_user_seconds" in algo_df.columns and "timestamp" in algo_df.columns:
                algo_df = algo_df.copy()
                algo_df["cpu_delta"] = algo_df["cpu_user_seconds"].diff()
                algo_df["time_delta"] = algo_df["timestamp"].diff().dt.total_seconds()
                valid_mask = (algo_df["cpu_delta"] > 0) & (algo_df["time_delta"] > 0)
                algo_df_valid = algo_df[valid_mask].copy()
                
                if len(algo_df_valid) > 0:
                    algo_df_valid["cpu_utilization"] = algo_df_valid["cpu_delta"] / algo_df_valid["time_delta"]
                    total_cpu = float(algo_df["cpu_user_seconds"].iloc[-1])
                    algo_stats["cpu"] = {
                        "mean_utilization": float(algo_df_valid["cpu_utilization"].mean()),
                        "max_utilization": float(algo_df_valid["cpu_utilization"].max()),
                        "cpu_per_operation_seconds": float(total_cpu / len(algo_df)) if len(algo_df) > 0 else 0.0,
                        "total_cpu_seconds": total_cpu,
                    }
            
            summary["per_algorithm"][algo] = algo_stats

    # Per-operation stats
    if "operation" in df.columns:
        summary["per_operation"] = {}
        for op in df["operation"].unique():
            op_df = df[df["operation"] == op]
            # Handle both nanosecond (new) and microsecond (old) formats
            if "latency_ns" in op_df.columns:
                op_df = op_df.copy()
                op_df["latency_us"] = op_df["latency_ns"] / 1000.0
                summary["per_operation"][op] = compute_basic_stats(op_df["latency_us"])
            elif "latency_us" in op_df.columns:
                # Old format: backward compatibility
                op_df = op_df.copy()
                op_df["latency_ns"] = op_df["latency_us"] * 1000.0
                summary["per_operation"][op] = compute_basic_stats(op_df["latency_us"])
            else:
                raise ValueError(f"Missing latency_ns or latency_us column for operation {op}")

    # Save summary
    # Convert numpy/pandas types to Python native types for JSON serialization
    summary_clean = convert_to_python_types(summary)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_clean, f, indent=2)
    console.print(f"[green]  Saved {summary_path}[/green]")

    # Generate plots (non-critical - continue even if plots fail)
    console.print("[cyan]Generating plots...[/cyan]")

    try:
    plot_latency_histogram(df, output_dir / "latency_hist.png")
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to generate latency histogram: {e}[/yellow]")
    
    try:
    plot_queue_histogram(df, output_dir / "queue_hist.png")
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to generate queue histogram: {e}[/yellow]")
    
    try:
    plot_throughput_curve(df, output_dir / "throughput_curve.png")
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to generate throughput curve: {e}[/yellow]")

    # Print summary
    console.print("\n[bold]Summary:[/bold]")
    if "latency" in summary:
        lat = summary["latency"]
        console.print(f"  Latency (μs): mean={lat['mean']:.1f}, p50={lat['p50']:.1f}, p99={lat['p99']:.1f}")
    if "throughput" in summary and "mean_msgs_per_sec" in summary["throughput"]:
        tput = summary["throughput"]
        console.print(f"  Throughput: {tput['mean_msgs_per_sec']:.0f} msg/s (mean)")
    if "memory" in summary and "mean_rss_mb" in summary["memory"]:
        mem = summary["memory"]
        console.print(f"  Memory: mean={mem['mean_rss_mb']:.2f} MB, max={mem['max_rss_mb']:.2f} MB")
    if "cpu" in summary and "mean_utilization" in summary["cpu"]:
        cpu = summary["cpu"]
        console.print(f"  CPU: mean_util={cpu['mean_utilization']:.2%}, max_util={cpu['max_utilization']:.2%}, cpu_per_op={cpu['cpu_per_operation_seconds']*1e6:.2f} μs")
    elif "cpu" in summary and "note" in summary["cpu"]:
        cpu = summary["cpu"]
        console.print(f"  CPU: {cpu['note']}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compute statistics from experiment data")
    parser.add_argument("--input", required=True, help="Input JSONL or Parquet file")
    parser.add_argument("--output", required=True, help="Output directory for stats and plots")
    parser.add_argument("--experiment-id", help="Experiment identifier")

    args = parser.parse_args()

    summary = compute_statistics(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        experiment_id=args.experiment_id,
    )

    if not summary:
        console.print("[red]Statistics computation failed![/red]")
        return 1

    console.print("[bold green]Statistics complete![/bold green]")
    return 0


if __name__ == "__main__":
    exit(main())
