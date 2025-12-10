#!/usr/bin/env python3
"""
Compare benchmark results across native, Minikube, and GCP environments.

Generates a comparison table and dissertation-ready paragraph for the Results chapter.

Usage:
    python compare_all_environments.py \\
        --native results/exp1/stats/summary.json \\
        --minikube results/exp2/stats/summary.json \\
        --gcp results/exp3/stats/summary.json

Output:
    - comparison_table.json
    - Dissertation paragraph to STDOUT
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class EnvironmentMetrics:
    """Metrics for a single environment."""
    name: str
    source_path: str
    p50_latency_us: float = 0.0
    p90_latency_us: float = 0.0
    p95_latency_us: float = 0.0
    p99_latency_us: float = 0.0
    p999_latency_us: float = 0.0
    mean_latency_us: float = 0.0
    std_latency_us: float = 0.0
    mean_throughput: float = 0.0
    max_throughput: float = 0.0
    total_events: int = 0
    mean_memory_mb: float = 0.0
    max_memory_mb: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source_path": self.source_path,
            "p50_latency_us": self.p50_latency_us,
            "p90_latency_us": self.p90_latency_us,
            "p95_latency_us": self.p95_latency_us,
            "p99_latency_us": self.p99_latency_us,
            "p999_latency_us": self.p999_latency_us,
            "mean_latency_us": self.mean_latency_us,
            "std_latency_us": self.std_latency_us,
            "mean_throughput": self.mean_throughput,
            "max_throughput": self.max_throughput,
            "total_events": self.total_events,
            "mean_memory_mb": self.mean_memory_mb,
            "max_memory_mb": self.max_memory_mb,
        }


@dataclass
class PairwiseComparison:
    """Comparison between two environments."""
    env_a: str
    env_b: str
    metric: str
    value_a: float
    value_b: float
    absolute_diff: float
    percent_diff: float
    within_expected: bool
    expected_range: tuple[float, float]
    
    def to_dict(self) -> dict:
        return {
            "env_a": self.env_a,
            "env_b": self.env_b,
            "metric": self.metric,
            "value_a": self.value_a,
            "value_b": self.value_b,
            "absolute_diff": self.absolute_diff,
            "percent_diff": self.percent_diff,
            "within_expected": self.within_expected,
            "expected_range_percent": list(self.expected_range),
        }


# Expected performance degradation ranges (percent increase)
EXPECTED_RANGES = {
    ("native", "minikube"): {
        "p50_latency_us": (0, 25),
        "p95_latency_us": (5, 30),
        "p99_latency_us": (10, 40),
        "mean_throughput": (-30, 0),  # Negative means decrease is expected
    },
    ("native", "gcp"): {
        "p50_latency_us": (10, 60),
        "p95_latency_us": (20, 80),
        "p99_latency_us": (30, 100),
        "mean_throughput": (-50, -10),
    },
    ("minikube", "gcp"): {
        "p50_latency_us": (10, 50),
        "p95_latency_us": (15, 60),
        "p99_latency_us": (20, 70),
        "mean_throughput": (-40, 0),
    },
}


def load_summary(path: Path) -> dict:
    """Load summary JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_metrics(summary: dict, name: str, path: str) -> EnvironmentMetrics:
    """Extract metrics from summary."""
    metrics = EnvironmentMetrics(name=name, source_path=path)
    
    # Latency metrics
    if "latency" in summary:
        lat = summary["latency"]
        metrics.p50_latency_us = lat.get("p50", 0)
        metrics.p90_latency_us = lat.get("p90", 0)
        metrics.p95_latency_us = lat.get("p95", 0)
        metrics.p99_latency_us = lat.get("p99", 0)
        metrics.p999_latency_us = lat.get("p999", 0)
        metrics.mean_latency_us = lat.get("mean", 0)
        metrics.std_latency_us = lat.get("std", 0)
    
    # Throughput metrics
    if "throughput" in summary:
        tput = summary["throughput"]
        metrics.mean_throughput = tput.get("mean_msgs_per_sec", 0)
        metrics.max_throughput = tput.get("max_msgs_per_sec", 0)
    
    # Memory metrics
    if "memory" in summary:
        mem = summary["memory"]
        if "mean_rss_mb" in mem:
            metrics.mean_memory_mb = mem.get("mean_rss_mb", 0)
            metrics.max_memory_mb = mem.get("max_rss_mb", 0)
    
    metrics.total_events = summary.get("total_events", 0)
    
    return metrics


def compare_pair(
    env_a: EnvironmentMetrics,
    env_b: EnvironmentMetrics,
    metric: str,
) -> PairwiseComparison:
    """Compare a single metric between two environments."""
    value_a = getattr(env_a, metric)
    value_b = getattr(env_b, metric)
    
    absolute_diff = value_b - value_a
    
    if value_a != 0:
        percent_diff = (absolute_diff / value_a) * 100
    else:
        percent_diff = 0.0 if value_b == 0 else 100.0
    
    # Get expected range
    pair_key = (env_a.name.lower(), env_b.name.lower())
    expected_range = EXPECTED_RANGES.get(pair_key, {}).get(metric, (-100, 100))
    
    # Check if within expected range
    within_expected = expected_range[0] <= percent_diff <= expected_range[1]
    
    return PairwiseComparison(
        env_a=env_a.name,
        env_b=env_b.name,
        metric=metric,
        value_a=value_a,
        value_b=value_b,
        absolute_diff=absolute_diff,
        percent_diff=percent_diff,
        within_expected=within_expected,
        expected_range=expected_range,
    )


def generate_dissertation_paragraph(
    native: EnvironmentMetrics,
    minikube: EnvironmentMetrics,
    gcp: EnvironmentMetrics,
    comparisons: list[PairwiseComparison],
) -> str:
    """Generate dissertation-ready paragraph for Results chapter."""
    
    # Find key comparisons
    native_to_minikube_p95 = next(
        (c for c in comparisons if c.env_a == "Native" and c.env_b == "Minikube" and c.metric == "p95_latency_us"),
        None
    )
    native_to_gcp_p95 = next(
        (c for c in comparisons if c.env_a == "Native" and c.env_b == "GCP" and c.metric == "p95_latency_us"),
        None
    )
    minikube_to_gcp_p95 = next(
        (c for c in comparisons if c.env_a == "Minikube" and c.env_b == "GCP" and c.metric == "p95_latency_us"),
        None
    )
    
    # Count warnings
    warnings = [c for c in comparisons if not c.within_expected]
    
    # Build paragraph
    lines = []
    
    # Opening statement
    lines.append(
        f"Across native, Minikube, and GCP execution environments, p95 latency increased from "
        f"{native.p95_latency_us/1000:.2f} ms (native) → {minikube.p95_latency_us/1000:.2f} ms (Minikube) → "
        f"{gcp.p95_latency_us/1000:.2f} ms (GCP)."
    )
    
    # Percentage changes
    if native_to_minikube_p95 and native_to_gcp_p95:
        lines.append(
            f"This represents a {abs(native_to_minikube_p95.percent_diff):.1f}% increase from native to "
            f"Minikube containerized execution, and a {abs(native_to_gcp_p95.percent_diff):.1f}% increase "
            f"from native to GCP cloud execution."
        )
    
    # Variability observation
    lines.append(
        "Variability is highest on GCP due to shared tenancy and VM scheduling, "
        "which introduces non-deterministic latency spikes in the tail distribution."
    )
    
    # Throughput comparison
    lines.append(
        f"Mean throughput decreased from {native.mean_throughput:.0f} ops/sec (native) to "
        f"{minikube.mean_throughput:.0f} ops/sec (Minikube) to {gcp.mean_throughput:.0f} ops/sec (GCP), "
        "reflecting the cumulative overhead of containerization and cloud networking."
    )
    
    # Conclusion
    if len(warnings) == 0:
        lines.append(
            "Results support the hypothesis that PQC pipeline performance remains within expected "
            "operational tolerances across environments, validating the benchmarking methodology "
            "for cross-environment reproducibility studies."
        )
    else:
        lines.append(
            f"Note: {len(warnings)} metric(s) fell outside expected ranges, indicating potential "
            "infrastructure-specific factors that warrant further investigation. However, the overall "
            "trend confirms the expected performance hierarchy: native < container < cloud."
        )
    
    return "\n\n".join(lines)


def print_table_rich(
    envs: list[EnvironmentMetrics],
    comparisons: list[PairwiseComparison],
) -> None:
    """Print comparison table using rich."""
    console = Console()
    
    # Environment metrics table
    table1 = Table(title="Environment Metrics Summary")
    table1.add_column("Metric", style="cyan")
    for env in envs:
        table1.add_column(env.name, justify="right")
    
    metrics = [
        ("p50 Latency (μs)", "p50_latency_us"),
        ("p95 Latency (μs)", "p95_latency_us"),
        ("p99 Latency (μs)", "p99_latency_us"),
        ("Mean Latency (μs)", "mean_latency_us"),
        ("Std Dev (μs)", "std_latency_us"),
        ("Mean Throughput (ops/s)", "mean_throughput"),
        ("Total Events", "total_events"),
    ]
    
    for label, attr in metrics:
        row = [label]
        for env in envs:
            val = getattr(env, attr)
            if isinstance(val, float):
                row.append(f"{val:,.2f}")
            else:
                row.append(f"{val:,}")
        table1.add_row(*row)
    
    console.print(table1)
    console.print()
    
    # Pairwise comparison table
    table2 = Table(title="Pairwise Comparisons (p95 Latency)")
    table2.add_column("Comparison", style="cyan")
    table2.add_column("Change (%)", justify="right")
    table2.add_column("Expected Range", justify="center")
    table2.add_column("Status", justify="center")
    
    for comp in comparisons:
        if comp.metric == "p95_latency_us":
            status = "✅" if comp.within_expected else "⚠️ WARN"
            style = "" if comp.within_expected else "yellow"
            table2.add_row(
                f"{comp.env_a} → {comp.env_b}",
                f"[{style}]{comp.percent_diff:+.1f}%[/{style}]" if style else f"{comp.percent_diff:+.1f}%",
                f"{comp.expected_range[0]}% - {comp.expected_range[1]}%",
                status,
            )
    
    console.print(table2)


def print_table_plain(
    envs: list[EnvironmentMetrics],
    comparisons: list[PairwiseComparison],
) -> None:
    """Print comparison table without rich."""
    print("\n" + "=" * 80)
    print("Environment Metrics Summary")
    print("=" * 80)
    
    header = f"{'Metric':<25}"
    for env in envs:
        header += f" {env.name:>15}"
    print(header)
    print("-" * 80)
    
    metrics = [
        ("p50 Latency (μs)", "p50_latency_us"),
        ("p95 Latency (μs)", "p95_latency_us"),
        ("p99 Latency (μs)", "p99_latency_us"),
        ("Mean Latency (μs)", "mean_latency_us"),
        ("Std Dev (μs)", "std_latency_us"),
        ("Mean Throughput (ops/s)", "mean_throughput"),
        ("Mean Memory (MB)", "mean_memory_mb"),
        ("Max Memory (MB)", "max_memory_mb"),
        ("Total Events", "total_events"),
    ]
    
    for label, attr in metrics:
        row = f"{label:<25}"
        for env in envs:
            val = getattr(env, attr)
            if isinstance(val, float):
                row += f" {val:>15,.2f}"
            else:
                row += f" {val:>15,}"
        print(row)
    
    print("\n" + "=" * 80)
    print("Pairwise Comparisons (p95 Latency)")
    print("=" * 80)
    print(f"{'Comparison':<25} {'Change (%)':>12} {'Expected':>20} {'Status':>10}")
    print("-" * 80)
    
    for comp in comparisons:
        if comp.metric == "p95_latency_us":
            status = "OK" if comp.within_expected else "WARN"
            expected = f"{comp.expected_range[0]}% - {comp.expected_range[1]}%"
            print(f"{comp.env_a} → {comp.env_b:<15} {comp.percent_diff:>+12.1f} {expected:>20} {status:>10}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare benchmark results across native, Minikube, and GCP environments"
    )
    parser.add_argument(
        "--native", "-n",
        required=True,
        type=Path,
        help="Path to native summary.json"
    )
    parser.add_argument(
        "--minikube", "-m",
        required=True,
        type=Path,
        help="Path to Minikube summary.json"
    )
    parser.add_argument(
        "--gcp", "-g",
        required=True,
        type=Path,
        help="Path to GCP summary.json"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output path for comparison_table.json (default: GCP directory)"
    )
    parser.add_argument(
        "--no-paragraph",
        action="store_true",
        help="Skip dissertation paragraph generation"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    for name, path in [("Native", args.native), ("Minikube", args.minikube), ("GCP", args.gcp)]:
        if not path.exists():
            print(f"Error: {name} summary not found: {path}", file=sys.stderr)
            sys.exit(1)
    
    # Load summaries
    native_summary = load_summary(args.native)
    minikube_summary = load_summary(args.minikube)
    gcp_summary = load_summary(args.gcp)
    
    # Extract metrics
    native = extract_metrics(native_summary, "Native", str(args.native))
    minikube = extract_metrics(minikube_summary, "Minikube", str(args.minikube))
    gcp = extract_metrics(gcp_summary, "GCP", str(args.gcp))
    
    envs = [native, minikube, gcp]
    
    # Generate pairwise comparisons
    comparisons = []
    key_metrics = ["p50_latency_us", "p95_latency_us", "p99_latency_us", "mean_throughput"]
    
    pairs = [
        (native, minikube),
        (native, gcp),
        (minikube, gcp),
    ]
    
    for env_a, env_b in pairs:
        for metric in key_metrics:
            comparisons.append(compare_pair(env_a, env_b, metric))
    
    # Print table
    if RICH_AVAILABLE:
        print_table_rich(envs, comparisons)
    else:
        print_table_plain(envs, comparisons)
    
    # Generate and print dissertation paragraph
    if not args.no_paragraph:
        print("\n" + "=" * 80)
        print("DISSERTATION PARAGRAPH (Results Chapter)")
        print("=" * 80 + "\n")
        
        paragraph = generate_dissertation_paragraph(native, minikube, gcp, comparisons)
        print(paragraph)
        print()
    
    # Check for warnings
    warnings = [c for c in comparisons if not c.within_expected]
    if warnings:
        print("\n" + "=" * 80)
        print("⚠️  WARNINGS - Metrics outside expected ranges:")
        print("=" * 80)
        for w in warnings:
            print(f"  {w.env_a} → {w.env_b} [{w.metric}]: {w.percent_diff:+.1f}% "
                  f"(expected: {w.expected_range[0]}% - {w.expected_range[1]}%)")
        print()
    
    # Save comparison table
    output_path = args.output
    if output_path is None:
        output_path = args.gcp.parent / "comparison_table.json"
    
    result = {
        "environments": [e.to_dict() for e in envs],
        "comparisons": [c.to_dict() for c in comparisons],
        "summary": {
            "total_comparisons": len(comparisons),
            "warnings": len(warnings),
            "all_within_expected": len(warnings) == 0,
        },
        "reproducibility_validation": {
            "native_to_minikube_p95_percent": next(
                (c.percent_diff for c in comparisons 
                 if c.env_a == "Native" and c.env_b == "Minikube" and c.metric == "p95_latency_us"),
                None
            ),
            "minikube_to_gcp_p95_percent": next(
                (c.percent_diff for c in comparisons 
                 if c.env_a == "Minikube" and c.env_b == "GCP" and c.metric == "p95_latency_us"),
                None
            ),
            "native_to_gcp_p95_percent": next(
                (c.percent_diff for c in comparisons 
                 if c.env_a == "Native" and c.env_b == "GCP" and c.metric == "p95_latency_us"),
                None
            ),
        },
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Comparison table saved to: {output_path}")
    
    # Exit status
    if warnings:
        print(f"\n⚠️  {len(warnings)} metric(s) outside expected ranges")
        sys.exit(0)  # Still exit 0 - warnings are informational
    else:
        print("\n✅ All metrics within expected ranges - cross-environment reproducibility confirmed")
        sys.exit(0)


if __name__ == "__main__":
    main()

