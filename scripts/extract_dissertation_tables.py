#!/usr/bin/env python3
"""
Extract tables for dissertation from analysis results.

Generates LaTeX/CSV tables for:
- Performance comparison tables
- Effect size tables
- Environment comparison tables
- Statistical test results

Usage:
    python scripts/extract_dissertation_tables.py \
        --aggregated final-results/aggregated_stats.json \
        --hypothesis final-results/hypothesis_tests.json \
        --output final-results/tables/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def extract_performance_table(aggregated_stats: dict, output_dir: Path):
    """Extract performance comparison table."""
    aggregated = aggregated_stats.get('aggregated', [])
    
    if not aggregated:
        print("No aggregated statistics available")
        return
    
    # Create DataFrame
    rows = []
    for stat in aggregated:
        rows.append({
            'Algorithm': stat['algorithm'],
            'Environment': stat['environment'],
            'Payload (B)': stat['payload_size'],
            'Rate (msg/s)': stat['rate'],
            'p50 (μs)': f"{stat['p50']['mean']:.2f}",
            'p95 (μs)': f"{stat['p95']['mean']:.2f}",
            'p99 (μs)': f"{stat['p99']['mean']:.2f}",
            'Throughput (ops/s)': f"{stat['throughput']['mean']:.0f}",
            'n_runs': stat['n_runs'],
        })
    
    df = pd.DataFrame(rows)
    
    # Save CSV
    csv_path = output_dir / 'performance_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"Written: {csv_path}")
    
    # Save LaTeX
    latex_path = output_dir / 'performance_table.tex'
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False, float_format="%.2f"))
    print(f"Written: {latex_path}")


def extract_effect_size_table(hypothesis_tests: dict, output_dir: Path):
    """Extract effect size table."""
    tests = hypothesis_tests.get('tests', [])
    
    if not tests:
        print("No hypothesis test results available")
        return
    
    rows = []
    for test in tests:
        effect = test.get('effect_size', {})
        rows.append({
            'Comparison': test.get('comparison_id', ''),
            'Group A': test.get('group_a_name', ''),
            'Group B': test.get('group_b_name', ''),
            "Cohen's d": f"{effect.get('cohens_d', 0):.3f}",
            'Interpretation': effect.get('interpretation', ''),
            'p-value': f"{test.get('welch_pvalue', 1.0):.4f}",
            'Significant': 'Yes' if test.get('welch_significant', False) else 'No',
        })
    
    df = pd.DataFrame(rows)
    
    # Save CSV
    csv_path = output_dir / 'effect_size_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"Written: {csv_path}")
    
    # Save LaTeX
    latex_path = output_dir / 'effect_size_table.tex'
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False, float_format="%.3f"))
    print(f"Written: {latex_path}")


def extract_environment_delta_table(aggregated_stats: dict, output_dir: Path):
    """Extract environment comparison table."""
    deltas = aggregated_stats.get('environment_deltas', [])
    
    if not deltas:
        print("No environment deltas available")
        return
    
    rows = []
    for delta in deltas:
        rows.append({
            'Algorithm': delta['algorithm'],
            'Payload (B)': delta['payload_size'],
            'Rate (msg/s)': delta['rate'],
            'Native p95 (μs)': f"{delta.get('native_p95_mean', 0):.2f}" if delta.get('native_p95_mean') else 'N/A',
            'Minikube p95 (μs)': f"{delta.get('minikube_p95_mean', 0):.2f}" if delta.get('minikube_p95_mean') else 'N/A',
            'GCP p95 (μs)': f"{delta.get('gcp_p95_mean', 0):.2f}" if delta.get('gcp_p95_mean') else 'N/A',
            'Native→Minikube (%)': f"{delta.get('native_to_minikube_pct', 0):.1f}" if delta.get('native_to_minikube_pct') is not None else 'N/A',
            'Native→GCP (%)': f"{delta.get('native_to_gcp_pct', 0):.1f}" if delta.get('native_to_gcp_pct') is not None else 'N/A',
        })
    
    df = pd.DataFrame(rows)
    
    # Save CSV
    csv_path = output_dir / 'environment_delta_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"Written: {csv_path}")
    
    # Save LaTeX
    latex_path = output_dir / 'environment_delta_table.tex'
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False, float_format="%.2f"))
    print(f"Written: {latex_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract dissertation tables")
    parser.add_argument('--aggregated', type=Path, help='Path to aggregated_stats.json')
    parser.add_argument('--hypothesis', type=Path, help='Path to hypothesis_tests.json')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Load aggregated stats
    aggregated_stats = {}
    if args.aggregated and args.aggregated.exists():
        with open(args.aggregated) as f:
            aggregated_stats = json.load(f)
        extract_performance_table(aggregated_stats, args.output)
        extract_environment_delta_table(aggregated_stats, args.output)
    else:
        print("Warning: aggregated_stats.json not found, skipping performance tables")
    
    # Load hypothesis tests
    hypothesis_tests = {}
    if args.hypothesis and args.hypothesis.exists():
        with open(args.hypothesis) as f:
            hypothesis_tests = json.load(f)
        extract_effect_size_table(hypothesis_tests, args.output)
    else:
        print("Warning: hypothesis_tests.json not found, skipping effect size tables")
    
    print("\nTable extraction complete!")


if __name__ == '__main__':
    main()
