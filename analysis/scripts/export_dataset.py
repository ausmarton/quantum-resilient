#!/usr/bin/env python3
"""
Export processed data to various formats for sharing/archival.

Usage:
    python export_dataset.py --input ./data/exp_001/merged/ --output ./exports/exp_001/
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from rich.console import Console

console = Console()


def load_data(filepath: Path) -> pd.DataFrame:
    """Load data from JSONL or Parquet file."""
    if filepath.suffix == ".parquet":
        return pd.read_parquet(filepath)
    else:
        return pd.read_json(filepath, lines=True)


def export_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Export to Parquet format."""
    df.to_parquet(output_path, index=False, compression="snappy")
    console.print(f"[green]  Exported Parquet: {output_path}[/green]")


def export_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Export to CSV format."""
    df.to_csv(output_path, index=False)
    console.print(f"[green]  Exported CSV: {output_path}[/green]")


def export_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    """Export to JSONL format."""
    df.to_json(output_path, orient="records", lines=True)
    console.print(f"[green]  Exported JSONL: {output_path}[/green]")


def export_summary_stats(df: pd.DataFrame, output_path: Path) -> None:
    """Export summary statistics."""
    summary = {
        "total_events": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    # Numeric column statistics
    numeric_cols = df.select_dtypes(include=["number"]).columns
    summary["numeric_stats"] = {}
    for col in numeric_cols:
        summary["numeric_stats"][col] = {
            "count": int(df[col].count()),
            "mean": float(df[col].mean()) if not df[col].isna().all() else None,
            "std": float(df[col].std()) if not df[col].isna().all() else None,
            "min": float(df[col].min()) if not df[col].isna().all() else None,
            "max": float(df[col].max()) if not df[col].isna().all() else None,
            "p50": float(df[col].quantile(0.50)) if not df[col].isna().all() else None,
            "p95": float(df[col].quantile(0.95)) if not df[col].isna().all() else None,
            "p99": float(df[col].quantile(0.99)) if not df[col].isna().all() else None,
        }

    # Categorical column statistics
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    summary["categorical_stats"] = {}
    for col in cat_cols:
        value_counts = df[col].value_counts().head(10).to_dict()
        summary["categorical_stats"][col] = {
            "unique_count": int(df[col].nunique()),
            "top_values": {str(k): int(v) for k, v in value_counts.items()},
        }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    console.print(f"[green]  Exported summary: {output_path}[/green]")


def export_for_dissertation(
    df: pd.DataFrame,
    output_dir: Path,
    experiment_id: str,
) -> None:
    """Export in format suitable for dissertation appendix."""
    # Create a condensed summary table
    summary_rows = []

    if "algorithm" in df.columns:
        for algo in df["algorithm"].unique():
            algo_df = df[df["algorithm"] == algo]
            if "latency_us" in algo_df.columns:
                lat = algo_df["latency_us"]
                row = {
                    "Algorithm": algo,
                    "N": len(lat),
                    "Mean (μs)": f"{lat.mean():.1f}",
                    "Std (μs)": f"{lat.std():.1f}",
                    "p50 (μs)": f"{lat.quantile(0.50):.1f}",
                    "p95 (μs)": f"{lat.quantile(0.95):.1f}",
                    "p99 (μs)": f"{lat.quantile(0.99):.1f}",
                }
                summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        # Export as LaTeX table
        latex_path = output_dir / f"{experiment_id}_summary.tex"
        latex_content = summary_df.to_latex(index=False, escape=False)
        with open(latex_path, "w") as f:
            f.write(latex_content)
        console.print(f"[green]  Exported LaTeX table: {latex_path}[/green]")

        # Export as markdown table
        md_path = output_dir / f"{experiment_id}_summary.md"
        md_content = summary_df.to_markdown(index=False)
        with open(md_path, "w") as f:
            f.write(md_content)
        console.print(f"[green]  Exported Markdown table: {md_path}[/green]")


def export_dataset(
    input_path: Path,
    output_dir: Path,
    experiment_id: Optional[str] = None,
    formats: Optional[list[str]] = None,
) -> None:
    """Export dataset to various formats."""
    console.print(f"[bold blue]Exporting dataset[/bold blue]")
    console.print(f"  Input: {input_path}")
    console.print(f"  Output: {output_dir}")

    # Determine input file
    if input_path.is_dir():
        # Look for merged files
        parquet_file = input_path / "merged.parquet"
        jsonl_file = input_path / "merged.jsonl"
        if parquet_file.exists():
            input_file = parquet_file
        elif jsonl_file.exists():
            input_file = jsonl_file
        else:
            console.print("[red]No merged.parquet or merged.jsonl found![/red]")
            return
    else:
        input_file = input_path

    # Load data
    console.print("[cyan]Loading data...[/cyan]")
    df = load_data(input_file)
    console.print(f"  Loaded {len(df)} events")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_id = experiment_id or input_path.parent.name

    # Default formats
    if formats is None:
        formats = ["parquet", "csv", "summary"]

    # Export
    console.print("[cyan]Exporting...[/cyan]")

    if "parquet" in formats:
        export_parquet(df, output_dir / f"{exp_id}.parquet")

    if "csv" in formats:
        export_csv(df, output_dir / f"{exp_id}.csv")

    if "jsonl" in formats:
        export_jsonl(df, output_dir / f"{exp_id}.jsonl")

    if "summary" in formats:
        export_summary_stats(df, output_dir / f"{exp_id}_stats.json")

    if "dissertation" in formats:
        export_for_dissertation(df, output_dir, exp_id)

    console.print("[bold green]Export complete![/bold green]")


def main():
    parser = argparse.ArgumentParser(description="Export dataset to various formats")
    parser.add_argument("--input", required=True, help="Input directory or file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--experiment-id", help="Experiment identifier")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["parquet", "csv", "summary"],
        choices=["parquet", "csv", "jsonl", "summary", "dissertation"],
        help="Output formats",
    )

    args = parser.parse_args()

    export_dataset(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        experiment_id=args.experiment_id,
        formats=args.formats,
    )


if __name__ == "__main__":
    main()
