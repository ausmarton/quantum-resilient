#!/usr/bin/env python3
"""
Compute effect size metrics between two experiments.

Metrics:
- Cohen's d
- Hedge's g
- Glass delta
- Cliff's Delta
- Kolmogorov-Smirnov distance
- Wasserstein distance (Earth Mover's Distance)

Usage:
    python effect_sizes.py \
        --exp-a path/to/merged_A.jsonl \
        --exp-b path/to/merged_B.jsonl \
        --metric latency_us \
        --out comparisons/A_vs_B.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy import stats

console = Console()


def load_metric(filepath: Path, metric: str) -> np.ndarray:
    """Load a specific metric from a JSONL or Parquet file."""
    if filepath.suffix == ".parquet":
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_json(filepath, lines=True)

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in {filepath}")

    return df[metric].dropna().values


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.

    Cohen's d = (M1 - M2) / pooled_std

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    n1, n2 = len(a), len(b)
    var1, var2 = a.var(ddof=1), b.var(ddof=1)

    # Pooled standard deviation
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((a.mean() - b.mean()) / pooled_std)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Hedge's g (bias-corrected Cohen's d).

    Better for small samples.
    """
    d = cohens_d(a, b)
    n = len(a) + len(b)

    # Correction factor
    correction = 1 - (3 / (4 * n - 9))

    return float(d * correction)


def glass_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Glass's delta.

    Uses only the control group's (b) standard deviation.
    Useful when the experimental treatment affects variance.
    """
    std_b = b.std(ddof=1)
    if std_b == 0:
        return 0.0
    return float((a.mean() - b.mean()) / std_b)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cliff's Delta (non-parametric effect size).

    Range: [-1, 1]
    - |d| < 0.147: negligible
    - 0.147 <= |d| < 0.33: small
    - 0.33 <= |d| < 0.474: medium
    - |d| >= 0.474: large
    """
    n1, n2 = len(a), len(b)

    # Count dominance pairs
    greater = 0
    less = 0

    for x in a:
        greater += np.sum(x > b)
        less += np.sum(x < b)

    return float((greater - less) / (n1 * n2))


def ks_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """
    Compute Kolmogorov-Smirnov statistic and p-value.

    The KS statistic measures the maximum distance between the CDFs.
    """
    statistic, pvalue = stats.ks_2samp(a, b)
    return float(statistic), float(pvalue)


def wasserstein_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Wasserstein distance (Earth Mover's Distance).

    Measures the amount of "work" needed to transform one distribution into another.
    """
    return float(stats.wasserstein_distance(a, b))


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def interpret_cliffs_delta(d: float) -> str:
    """Interpret Cliff's delta effect size."""
    abs_d = abs(d)
    if abs_d < 0.147:
        return "negligible"
    elif abs_d < 0.33:
        return "small"
    elif abs_d < 0.474:
        return "medium"
    else:
        return "large"


def compute_effect_sizes(
    exp_a_path: Path,
    exp_b_path: Path,
    metric: str,
    output_path: Optional[Path] = None,
    exp_a_name: Optional[str] = None,
    exp_b_name: Optional[str] = None,
) -> dict:
    """Compute all effect size metrics between two experiments."""
    console.print(f"[bold blue]Computing Effect Sizes[/bold blue]")
    console.print(f"  Experiment A: {exp_a_path}")
    console.print(f"  Experiment B: {exp_b_path}")
    console.print(f"  Metric: {metric}")

    # Load data
    console.print("[cyan]Loading data...[/cyan]")
    a = load_metric(exp_a_path, metric)
    b = load_metric(exp_b_path, metric)

    console.print(f"  Experiment A: {len(a)} samples")
    console.print(f"  Experiment B: {len(b)} samples")

    # Compute effect sizes
    console.print("[cyan]Computing effect sizes...[/cyan]")

    d = cohens_d(a, b)
    g = hedges_g(a, b)
    delta_glass = glass_delta(a, b)
    delta_cliff = cliffs_delta(a, b)
    ks_stat, ks_pvalue = ks_distance(a, b)
    wasserstein = wasserstein_distance(a, b)

    results = {
        "experiment_a": exp_a_name or exp_a_path.stem,
        "experiment_b": exp_b_name or exp_b_path.stem,
        "metric": metric,
        "sample_sizes": {
            "a": int(len(a)),
            "b": int(len(b)),
        },
        "descriptive_stats": {
            "a": {
                "mean": float(a.mean()),
                "std": float(a.std()),
                "median": float(np.median(a)),
            },
            "b": {
                "mean": float(b.mean()),
                "std": float(b.std()),
                "median": float(np.median(b)),
            },
        },
        "effect_sizes": {
            "cohens_d": d,
            "hedges_g": g,
            "glass_delta": delta_glass,
            "cliffs_delta": delta_cliff,
        },
        "distribution_distances": {
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            "wasserstein_distance": wasserstein,
        },
        "interpretation": {
            "cohens_d": interpret_cohens_d(d),
            "cliffs_delta": interpret_cliffs_delta(delta_cliff),
            "ks_significant": ks_pvalue < 0.05,
        },
    }

    # Print results table
    table = Table(title="Effect Size Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Interpretation", style="green")

    table.add_row("Cohen's d", f"{d:.4f}", interpret_cohens_d(d))
    table.add_row("Hedge's g", f"{g:.4f}", interpret_cohens_d(g))
    table.add_row("Glass's Δ", f"{delta_glass:.4f}", interpret_cohens_d(delta_glass))
    table.add_row("Cliff's δ", f"{delta_cliff:.4f}", interpret_cliffs_delta(delta_cliff))
    table.add_row("KS statistic", f"{ks_stat:.4f}", "significant" if ks_pvalue < 0.05 else "not significant")
    table.add_row("KS p-value", f"{ks_pvalue:.4e}", "")
    table.add_row("Wasserstein", f"{wasserstein:.2f}", f"({metric} units)")

    console.print(table)

    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"[green]Results saved to {output_path}[/green]")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute effect sizes between experiments")
    parser.add_argument("--exp-a", required=True, help="Path to experiment A (JSONL or Parquet)")
    parser.add_argument("--exp-b", required=True, help="Path to experiment B (JSONL or Parquet)")
    parser.add_argument("--metric", required=True, help="Metric column to compare (e.g., latency_us)")
    parser.add_argument("--out", required=True, help="Output JSON file path")
    parser.add_argument("--name-a", help="Name for experiment A")
    parser.add_argument("--name-b", help="Name for experiment B")

    args = parser.parse_args()

    results = compute_effect_sizes(
        exp_a_path=Path(args.exp_a),
        exp_b_path=Path(args.exp_b),
        metric=args.metric,
        output_path=Path(args.out),
        exp_a_name=args.name_a,
        exp_b_name=args.name_b,
    )

    console.print("[bold green]Effect size computation complete![/bold green]")


if __name__ == "__main__":
    main()
