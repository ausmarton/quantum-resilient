#!/usr/bin/env python3
"""
Generate LaTeX and Markdown tables from experiment statistics.

Produces:
- Latency quantiles table
- Throughput summary table
- Queue delay stats
- Adapter comparison table
- Effect size comparison table
- Cluster scaling table

Usage:
    python generate_tables.py --exp-id exp_001 --stats-file analysis/data/exp_001/stats/summary.json --out research/output/exp_001/tables/
"""

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from tabulate import tabulate


def format_number(value: Any, precision: int = 1) -> str:
    """Format number with given precision."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{precision}f}"
    except (ValueError, TypeError):
        return str(value)


def generate_latency_table(stats: dict, experiment_id: str) -> tuple[str, str]:
    """Generate latency quantiles table in LaTeX and Markdown."""
    latency = stats.get("latency", {})
    
    rows = [
        ["Count", format_number(latency.get("count"), 0)],
        ["Mean", format_number(latency.get("mean"), 1)],
        ["Std Dev", format_number(latency.get("std"), 1)],
        ["Min", format_number(latency.get("min"), 1)],
        ["p50 (Median)", format_number(latency.get("p50"), 1)],
        ["p90", format_number(latency.get("p90"), 1)],
        ["p95", format_number(latency.get("p95"), 1)],
        ["p99", format_number(latency.get("p99"), 1)],
        ["p99.9", format_number(latency.get("p999"), 1)],
        ["Max", format_number(latency.get("max"), 1)],
    ]
    
    # Markdown
    md = tabulate(rows, headers=["Statistic", "Value (μs)"], tablefmt="pipe")
    
    # LaTeX
    tex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Latency Distribution ({experiment_id})}}
\\label{{tab:latency-{experiment_id.replace("_", "-")}}}
\\begin{{tabular}}{{lr}}
\\toprule
\\textbf{{Statistic}} & \\textbf{{Value (\\si{{\\micro\\second}})}} \\\\
\\midrule
"""
    for row in rows:
        tex += f"{row[0]} & {row[1]} \\\\\n"
    tex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return tex, md


def generate_throughput_table(stats: dict, experiment_id: str) -> tuple[str, str]:
    """Generate throughput summary table."""
    throughput = stats.get("throughput", {})
    
    rows = [
        ["Duration (s)", format_number(throughput.get("total_duration_sec"), 2)],
        ["Total Messages", format_number(throughput.get("total_messages"), 0)],
        ["Mean (msg/s)", format_number(throughput.get("mean_msgs_per_sec"), 1)],
        ["Max (msg/s)", format_number(throughput.get("max_msgs_per_sec"), 1)],
        ["Min (msg/s)", format_number(throughput.get("min_msgs_per_sec"), 1)],
        ["Std Dev", format_number(throughput.get("std_msgs_per_sec"), 1)],
    ]
    
    md = tabulate(rows, headers=["Metric", "Value"], tablefmt="pipe")
    
    tex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Throughput Statistics ({experiment_id})}}
\\label{{tab:throughput-{experiment_id.replace("_", "-")}}}
\\begin{{tabular}}{{lr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
"""
    for row in rows:
        tex += f"{row[0]} & {row[1]} \\\\\n"
    tex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return tex, md


def generate_queue_delay_table(stats: dict, experiment_id: str) -> tuple[str, str]:
    """Generate queue delay statistics table."""
    queue_delay = stats.get("queue_delay", {})
    
    if not queue_delay:
        return "% No queue delay data available", "*No queue delay data available*"
    
    rows = [
        ["Count", format_number(queue_delay.get("count"), 0)],
        ["Mean (μs)", format_number(queue_delay.get("mean"), 1)],
        ["Std Dev", format_number(queue_delay.get("std"), 1)],
        ["p50 (Median)", format_number(queue_delay.get("p50"), 1)],
        ["p90", format_number(queue_delay.get("p90"), 1)],
        ["p99", format_number(queue_delay.get("p99"), 1)],
    ]
    
    md = tabulate(rows, headers=["Statistic", "Value"], tablefmt="pipe")
    
    tex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Queue Delay Statistics ({experiment_id})}}
\\label{{tab:queue-delay-{experiment_id.replace("_", "-")}}}
\\begin{{tabular}}{{lr}}
\\toprule
\\textbf{{Statistic}} & \\textbf{{Value}} \\\\
\\midrule
"""
    for row in rows:
        tex += f"{row[0]} & {row[1]} \\\\\n"
    tex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return tex, md


def generate_adapter_comparison_table(stats: dict, experiment_id: str) -> tuple[str, str]:
    """Generate adapter/algorithm comparison table."""
    per_algorithm = stats.get("per_algorithm", {})
    
    if not per_algorithm:
        return "% No algorithm comparison data available", "*No algorithm comparison data available*"
    
    rows = []
    for algo, s in sorted(per_algorithm.items()):
        rows.append([
            algo,
            format_number(s.get("count"), 0),
            format_number(s.get("mean"), 1),
            format_number(s.get("std"), 1),
            format_number(s.get("p50"), 1),
            format_number(s.get("p99"), 1),
        ])
    
    md = tabulate(
        rows,
        headers=["Algorithm", "N", "Mean (μs)", "Std", "p50", "p99"],
        tablefmt="pipe"
    )
    
    tex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Algorithm Latency Comparison ({experiment_id})}}
\\label{{tab:algorithm-comparison-{experiment_id.replace("_", "-")}}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
\\textbf{{Algorithm}} & \\textbf{{N}} & \\textbf{{Mean (\\si{{\\micro\\second}})}} & \\textbf{{Std}} & \\textbf{{p50}} & \\textbf{{p99}} \\\\
\\midrule
"""
    for row in rows:
        tex += f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} \\\\\n"
    tex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return tex, md


def generate_effect_size_table(effect_sizes: list[dict]) -> tuple[str, str]:
    """Generate effect size comparison table."""
    if not effect_sizes:
        return "% No effect size data available", "*No effect size data available*"
    
    rows = []
    for es in effect_sizes:
        rows.append([
            es.get("comparison", "N/A"),
            format_number(es.get("cohens_d"), 3),
            es.get("interpretation", "N/A"),
            format_number(es.get("cliffs_delta"), 3),
            format_number(es.get("ks_statistic"), 3),
        ])
    
    md = tabulate(
        rows,
        headers=["Comparison", "Cohen's d", "Interpretation", "Cliff's δ", "KS Stat"],
        tablefmt="pipe"
    )
    
    tex = """\\begin{table}[htbp]
\\centering
\\caption{Effect Size Comparison}
\\label{tab:effect-sizes}
\\begin{tabular}{lrlrr}
\\toprule
\\textbf{Comparison} & \\textbf{Cohen's $d$} & \\textbf{Interpretation} & \\textbf{Cliff's $\\delta$} & \\textbf{KS Stat} \\\\
\\midrule
"""
    for row in rows:
        tex += f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} \\\\\n"
    tex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return tex, md


def generate_cluster_scaling_table(stats: dict, experiment_id: str) -> tuple[str, str]:
    """Generate cluster scaling behavior table."""
    worker_skew = stats.get("worker_skew", {})
    
    if not worker_skew or "events_per_worker" not in worker_skew:
        return "% No cluster scaling data available", "*No cluster scaling data available*"
    
    events_per_worker = worker_skew.get("events_per_worker", {})
    
    rows = []
    for worker_id, count in sorted(events_per_worker.items(), key=lambda x: int(x[0])):
        rows.append([f"Worker {worker_id}", format_number(count, 0)])
    
    # Add summary row
    rows.append(["---", "---"])
    rows.append(["Mean", format_number(worker_skew.get("mean_events"), 1)])
    rows.append(["Std Dev", format_number(worker_skew.get("std_events"), 1)])
    rows.append(["CV", format_number(worker_skew.get("coefficient_of_variation"), 3)])
    
    md = tabulate(rows, headers=["Worker", "Events"], tablefmt="pipe")
    
    tex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Worker Load Distribution ({experiment_id})}}
\\label{{tab:worker-scaling-{experiment_id.replace("_", "-")}}}
\\begin{{tabular}}{{lr}}
\\toprule
\\textbf{{Worker}} & \\textbf{{Events}} \\\\
\\midrule
"""
    for row in rows:
        if row[0] == "---":
            tex += "\\midrule\n"
        else:
            tex += f"{row[0]} & {row[1]} \\\\\n"
    tex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return tex, md


def generate_tables(
    experiment_id: str,
    stats_file: Path,
    output_dir: Path,
    effect_sizes_file: Optional[Path] = None,
) -> dict[str, tuple[str, str]]:
    """Generate all tables."""
    print(f"Generating tables for {experiment_id}")
    print(f"  Stats file: {stats_file}")
    print(f"  Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load statistics
    with open(stats_file) as f:
        stats = json.load(f)
    
    # Load effect sizes if available
    effect_sizes = []
    if effect_sizes_file and effect_sizes_file.exists():
        with open(effect_sizes_file) as f:
            es_data = json.load(f)
            if isinstance(es_data, list):
                effect_sizes = es_data
            else:
                effect_sizes = [es_data]
    
    tables = {}
    
    # Generate each table
    generators = [
        ("latency_quantiles", lambda: generate_latency_table(stats, experiment_id)),
        ("throughput_summary", lambda: generate_throughput_table(stats, experiment_id)),
        ("queue_delay_stats", lambda: generate_queue_delay_table(stats, experiment_id)),
        ("adapter_comparison", lambda: generate_adapter_comparison_table(stats, experiment_id)),
        ("effect_sizes", lambda: generate_effect_size_table(effect_sizes)),
        ("cluster_scaling", lambda: generate_cluster_scaling_table(stats, experiment_id)),
    ]
    
    for name, generator in generators:
        print(f"  Generating {name}...")
        tex, md = generator()
        tables[name] = (tex, md)
        
        # Save LaTeX
        tex_path = output_dir / f"{name}.tex"
        with open(tex_path, "w") as f:
            f.write(tex)
        
        # Save Markdown
        md_path = output_dir / f"{name}.md"
        with open(md_path, "w") as f:
            f.write(md)
    
    print(f"  Generated {len(tables)} tables")
    
    return tables


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX and Markdown tables")
    parser.add_argument("--exp-id", required=True, help="Experiment identifier")
    parser.add_argument("--stats-file", required=True, help="Path to summary.json")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--effect-sizes", help="Path to effect sizes JSON file")
    
    args = parser.parse_args()
    
    effect_sizes_file = Path(args.effect_sizes) if args.effect_sizes else None
    
    tables = generate_tables(
        experiment_id=args.exp_id,
        stats_file=Path(args.stats_file),
        output_dir=Path(args.out),
        effect_sizes_file=effect_sizes_file,
    )
    
    print(f"\nGenerated {len(tables)} tables!")
    for name in tables:
        print(f"  - {name}.tex / {name}.md")


if __name__ == "__main__":
    main()

