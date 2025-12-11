#!/usr/bin/env python3
"""
Merge multiple worker JSONL files into a single sorted file.

Usage:
    python merge_jsonl.py --input ./data/exp_001/raw/ --output ./data/exp_001/merged/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd
from rich.console import Console
from tqdm import tqdm

console = Console()


def load_jsonl(filepath: Path) -> Iterator[dict]:
    """Load JSONL file line by line."""
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def load_all_jsonl(input_dir: Path) -> pd.DataFrame:
    """Load all JSONL files from directory into DataFrame."""
    console.print(f"[cyan]Loading JSONL files from {input_dir}...[/cyan]")

    files = list(input_dir.glob("*.jsonl"))
    if not files:
        console.print("[red]No JSONL files found![/red]")
        return pd.DataFrame()

    console.print(f"  Found {len(files)} files")

    all_events = []
    for filepath in tqdm(files, desc="Loading"):
        # Try to extract worker_id from filename
        worker_id = None
        if "worker_" in filepath.name:
            try:
                worker_id = int(filepath.stem.split("_")[1])
            except (IndexError, ValueError):
                pass

        for event in load_jsonl(filepath):
            # Add worker_id if not present and we extracted it from filename
            if "worker_id" not in event and worker_id is not None:
                event["worker_id"] = worker_id
            all_events.append(event)

    console.print(f"  Loaded {len(all_events)} events")

    df = pd.DataFrame(all_events)
    return df


def compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived columns for analysis."""
    console.print("[cyan]Computing derived columns...[/cyan]")

    # Ensure timestamp columns exist
    if "timestamp_utc_iso" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp_utc_iso"])

    # Expect nanosecond precision format (latency_ns, queue_delay_ns)
    if "latency_ns" not in df.columns:
        console.print("[red]Error: latency_ns column not found. Data must be in nanosecond precision format.[/red]")
        raise ValueError("Missing required column: latency_ns")
    
    # Ensure queue_delay_ns exists
    if "queue_delay_ns" not in df.columns:
        console.print("[yellow]Warning: queue_delay_ns not found, setting to 0[/yellow]")
        df["queue_delay_ns"] = 0

    # Compute end-to-end latency (same as latency_ns)
    df["end_to_end_latency_ns"] = df["latency_ns"]

    # Crypto latency = latency - queue_delay (in nanoseconds)
    df["crypto_latency_ns"] = df["latency_ns"] - df["queue_delay_ns"].fillna(0)

    # Add convenience columns for display (convert to microseconds and milliseconds)
    df["latency_us"] = df["latency_ns"] / 1000.0
    df["queue_delay_us"] = df["queue_delay_ns"] / 1000.0
    df["crypto_latency_us"] = df["crypto_latency_ns"] / 1000.0
    df["latency_ms"] = df["latency_ns"] / 1_000_000.0

    return df


def validate_uniqueness(df: pd.DataFrame) -> bool:
    """Validate that event IDs are unique within each run."""
    if "run_id" not in df.columns or "event_id" not in df.columns:
        console.print("[yellow]Cannot validate uniqueness: missing run_id or event_id[/yellow]")
        return True

    duplicates = df.groupby(["run_id", "event_id"]).size()
    duplicates = duplicates[duplicates > 1]

    if len(duplicates) > 0:
        console.print(f"[red]Found {len(duplicates)} duplicate event IDs![/red]")
        console.print(duplicates.head(10))
        return False

    console.print("[green]  Event IDs are unique[/green]")
    return True


def merge_jsonl(
    input_dir: Path,
    output_dir: Path,
    sort_by: str = "timestamp_monotonic_ns",
) -> pd.DataFrame:
    """Merge JSONL files into a single sorted file."""
    console.print(f"[bold blue]Merging JSONL files[/bold blue]")
    console.print(f"  Input: {input_dir}")
    console.print(f"  Output: {output_dir}")

    # Load all files
    df = load_all_jsonl(input_dir)
    if df.empty:
        return df

    # Compute derived columns
    df = compute_derived_columns(df)

    # Sort by timestamp
    if sort_by in df.columns:
        console.print(f"[cyan]Sorting by {sort_by}...[/cyan]")
        df = df.sort_values(sort_by).reset_index(drop=True)
    elif "timestamp" in df.columns:
        console.print("[cyan]Sorting by timestamp...[/cyan]")
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Validate uniqueness
    validate_uniqueness(df)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSONL
    jsonl_path = output_dir / "merged.jsonl"
    console.print(f"[cyan]Writing {jsonl_path}...[/cyan]")

    # Convert timestamp to string for JSON serialization
    df_export = df.copy()
    if "timestamp" in df_export.columns:
        df_export["timestamp"] = df_export["timestamp"].astype(str)

    with open(jsonl_path, "w") as f:
        for _, row in tqdm(df_export.iterrows(), total=len(df_export), desc="Writing JSONL"):
            # Remove NaN values
            record = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
            f.write(json.dumps(record) + "\n")

    console.print(f"[green]  Written {len(df)} events to {jsonl_path}[/green]")

    # Save as Parquet
    parquet_path = output_dir / "merged.parquet"
    console.print(f"[cyan]Writing {parquet_path}...[/cyan]")
    df.to_parquet(parquet_path, index=False)
    console.print(f"[green]  Written {len(df)} events to {parquet_path}[/green]")

    # Print summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Total events: {len(df)}")
    if "worker_id" in df.columns:
        console.print(f"  Workers: {df['worker_id'].nunique()}")
    if "algorithm" in df.columns:
        console.print(f"  Algorithms: {df['algorithm'].unique().tolist()}")
    if "operation" in df.columns:
        console.print(f"  Operations: {df['operation'].unique().tolist()}")
    if "timestamp" in df.columns:
        duration = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
        console.print(f"  Duration: {duration:.2f} seconds")

    return df


def main():
    parser = argparse.ArgumentParser(description="Merge JSONL files from distributed workers")
    parser.add_argument("--input", required=True, help="Input directory with raw JSONL files")
    parser.add_argument("--output", required=True, help="Output directory for merged files")
    parser.add_argument(
        "--sort-by",
        default="timestamp_monotonic_ns",
        help="Column to sort by (default: timestamp_monotonic_ns)",
    )

    args = parser.parse_args()

    df = merge_jsonl(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        sort_by=args.sort_by,
    )

    if df.empty:
        console.print("[red]No data merged![/red]")
        sys.exit(1)

    console.print("[bold green]Merge complete![/bold green]")


if __name__ == "__main__":
    main()
