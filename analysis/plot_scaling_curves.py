#!/usr/bin/env python3
"""
Generate scaling curve plots for throughput and latency.

Produces:
- Throughput vs payload size
- Latency vs message rate
- Cloud vs local scaling curves

Usage:
    python analysis/plot_scaling_curves.py \
        --index final-results/index.json \
        --output final-results/figures
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Don't suppress warnings - they're useful diagnostics for empty data
# Empty legend warnings indicate no data was loaded, which is a real issue to investigate

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color palette for algorithms
ALGO_COLORS = {
    'rsa2048': '#9b59b6',
    'ecdsa_p256': '#e67e22',
    'kyber512': '#1abc9c',
    'dilithium2': '#2980b9',
    'hybrid_kyber_dilithium': '#c0392b',
}

# Markers for environments
ENV_MARKERS = {
    'native': 'o',
    'minikube': 's',
    'gcp': '^',
}

ENV_COLORS = {
    'native': '#2ecc71',
    'minikube': '#3498db',
    'gcp': '#e74c3c',
}


def load_aggregated_stats(path: Path) -> dict:
    """Load aggregated statistics."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def plot_throughput_vs_payload(
    stats: dict,
    output_dir: Path,
):
    """Plot throughput scaling with payload size."""
    # Group by algorithm -> environment -> payload -> throughput
    data: dict = defaultdict(lambda: defaultdict(dict))
    
    entries_with_data = 0
    entries_missing_throughput = 0
    
    for entry in stats.get('aggregated', []):
        algo = entry['algorithm']
        env = entry['environment']
        payload = entry['payload_size']
        
        # Check if throughput data exists
        if 'throughput' not in entry:
            entries_missing_throughput += 1
            continue
        
        throughput = entry['throughput'].get('mean', 0)
        throughput_std = entry['throughput'].get('std', 0)
        
        if throughput > 0:  # Only include non-zero throughput
            data[algo][env][payload] = (throughput, throughput_std)
            entries_with_data += 1
    
    if entries_missing_throughput > 0:
        print(f"  Warning: {entries_missing_throughput} entries missing throughput data", file=sys.stderr)
    
    if len(data) == 0:
        print("  Warning: No throughput data available for plotting", file=sys.stderr)
        return
    
    # Plot per environment
    for env in ['native', 'minikube', 'gcp']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for algo in data:
            if env not in data[algo]:
                continue
            
            payloads = sorted(data[algo][env].keys())
            means = [data[algo][env][p][0] for p in payloads]
            stds = [data[algo][env][p][1] for p in payloads]
            
            ax.errorbar(payloads, means, yerr=stds,
                       color=ALGO_COLORS.get(algo, '#333'),
                       marker='o',
                       linewidth=2,
                       markersize=8,
                       capsize=5,
                       label=algo.replace('_', ' ').title())
        
        ax.set_xlabel('Payload Size (bytes)')
        ax.set_ylabel('Throughput (ops/sec)')
        ax.set_title(f'Throughput vs Payload Size ({env.capitalize()})')
        # Only add legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc='upper right')
        ax.set_xscale('log')
        
        plt.tight_layout()
        output_path = output_dir / f'throughput_vs_payload_{env}.png'
        plt.savefig(output_path)
        plt.close()
        print(f"  Saved: {output_path}")
    
    # Combined plot with environment comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for algo in data:
        for env in ['native', 'minikube', 'gcp']:
            if env not in data[algo]:
                continue
            
            payloads = sorted(data[algo][env].keys())
            means = [data[algo][env][p][0] for p in payloads]
            
            ax.plot(payloads, means,
                   color=ALGO_COLORS.get(algo, '#333'),
                   marker=ENV_MARKERS.get(env, 'o'),
                   linestyle='-' if env == 'native' else ('--' if env == 'minikube' else ':'),
                   linewidth=1.5,
                   markersize=6,
                   alpha=0.8,
                   label=f'{algo} ({env})')
    
    ax.set_xlabel('Payload Size (bytes)')
    ax.set_ylabel('Throughput (ops/sec)')
    ax.set_title('Throughput Scaling with Payload Size')
    # Only add legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.set_xscale('log')
    
    plt.tight_layout()
    output_path = output_dir / 'throughput_vs_payload_all.png'
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_latency_vs_rate(
    stats: dict,
    output_dir: Path,
):
    """Plot latency scaling with message rate."""
    # Group by algorithm -> environment -> rate -> latency
    data: dict = defaultdict(lambda: defaultdict(dict))
    
    entries_with_data = 0
    entries_missing_p95 = 0
    
    for entry in stats.get('aggregated', []):
        algo = entry['algorithm']
        env = entry['environment']
        rate = entry['rate']
        
        # Check if p95 data exists
        if 'p95' not in entry:
            entries_missing_p95 += 1
            continue
        
        p95_mean = entry['p95'].get('mean', 0)
        p95_std = entry['p95'].get('std', 0)
        
        if p95_mean > 0:  # Only include non-zero latency
            # Aggregate across payload sizes
            if rate not in data[algo][env]:
                data[algo][env][rate] = {'means': [], 'stds': []}
            data[algo][env][rate]['means'].append(p95_mean)
            data[algo][env][rate]['stds'].append(p95_std)
            entries_with_data += 1
    
    if entries_missing_p95 > 0:
        print(f"  Warning: {entries_missing_p95} entries missing p95 latency data", file=sys.stderr)
    
    if len(data) == 0:
        print("  Warning: No latency data available for plotting", file=sys.stderr)
        return
    
    # Average across payload sizes
    for algo in data:
        for env in data[algo]:
            for rate in data[algo][env]:
                means = data[algo][env][rate]['means']
                stds = data[algo][env][rate]['stds']
                data[algo][env][rate] = (np.mean(means), np.mean(stds))
    
    # Plot per environment
    for env in ['native', 'minikube', 'gcp']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for algo in data:
            if env not in data[algo]:
                continue
            
            rates = sorted(data[algo][env].keys())
            means = [data[algo][env][r][0] for r in rates]
            stds = [data[algo][env][r][1] for r in rates]
            
            ax.errorbar(rates, means, yerr=stds,
                       color=ALGO_COLORS.get(algo, '#333'),
                       marker='o',
                       linewidth=2,
                       markersize=8,
                       capsize=5,
                       label=algo.replace('_', ' ').title())
        
        ax.set_xlabel('Message Rate (msgs/sec)')
        ax.set_ylabel('p95 Latency (μs)')
        ax.set_title(f'Latency vs Message Rate ({env.capitalize()})')
        # Only add legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc='upper left')
        ax.set_xscale('log')
        
        plt.tight_layout()
        output_path = output_dir / f'latency_vs_rate_{env}.png'
        plt.savefig(output_path)
        plt.close()
        print(f"  Saved: {output_path}")


def plot_environment_scaling(
    stats: dict,
    output_dir: Path,
):
    """Plot environment overhead scaling."""
    # Group by algorithm -> rate -> env -> latency
    data: dict = defaultdict(lambda: defaultdict(dict))
    
    for entry in stats.get('aggregated', []):
        algo = entry['algorithm']
        env = entry['environment']
        rate = entry['rate']
        p95_mean = entry['p95']['mean']
        
        if rate not in data[algo]:
            data[algo][rate] = {}
        if env not in data[algo][rate]:
            data[algo][rate][env] = []
        data[algo][rate][env].append(p95_mean)
    
    # Average across payload sizes
    for algo in data:
        for rate in data[algo]:
            for env in data[algo][rate]:
                data[algo][rate][env] = np.mean(data[algo][rate][env])
    
    # Plot overhead ratio (minikube/native and gcp/native)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Minikube vs Native
    ax = axes[0]
    for algo in data:
        rates = sorted(data[algo].keys())
        ratios = []
        for r in rates:
            native = data[algo][r].get('native', 0)
            minikube = data[algo][r].get('minikube', 0)
            if native > 0 and minikube > 0:
                ratios.append(minikube / native)
            else:
                ratios.append(np.nan)
        
        ax.plot(rates, ratios,
               color=ALGO_COLORS.get(algo, '#333'),
               marker='o',
               linewidth=2,
               markersize=8,
               label=algo.replace('_', ' ').title())
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Message Rate (msgs/sec)')
    ax.set_ylabel('Latency Ratio (Minikube / Native)')
    ax.set_title('Container Overhead: Minikube vs Native')
    # Only add legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc='upper left')
    ax.set_xscale('log')
    
    # GCP vs Native
    ax = axes[1]
    for algo in data:
        rates = sorted(data[algo].keys())
        ratios = []
        for r in rates:
            native = data[algo][r].get('native', 0)
            gcp = data[algo][r].get('gcp', 0)
            if native > 0 and gcp > 0:
                ratios.append(gcp / native)
            else:
                ratios.append(np.nan)
        
        ax.plot(rates, ratios,
               color=ALGO_COLORS.get(algo, '#333'),
               marker='^',
               linewidth=2,
               markersize=8,
               label=algo.replace('_', ' ').title())
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Message Rate (msgs/sec)')
    ax.set_ylabel('Latency Ratio (GCP / Native)')
    ax.set_title('Cloud Overhead: GCP vs Native')
    # Only add legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc='upper left')
    ax.set_xscale('log')
    
    plt.suptitle('Environment Overhead Scaling', fontsize=14)
    plt.tight_layout()
    
    output_path = output_dir / 'scaling_curves.png'
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_classical_vs_pqc(
    stats: dict,
    output_dir: Path,
):
    """Plot classical vs PQC algorithm comparison."""
    classical = ['rsa2048', 'ecdsa_p256']
    pqc = ['kyber512', 'dilithium2', 'hybrid_kyber_dilithium']
    
    # Group by environment -> algorithm -> metrics
    data: dict = defaultdict(lambda: defaultdict(list))
    
    for entry in stats.get('aggregated', []):
        algo = entry['algorithm']
        env = entry['environment']
        p95_mean = entry['p95']['mean']
        data[env][algo].append(p95_mean)
    
    # Average across configurations
    for env in data:
        for algo in data[env]:
            data[env][algo] = np.mean(data[env][algo])
    
    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(classical) + len(pqc))
    width = 0.25
    
    for idx, env in enumerate(['native', 'minikube', 'gcp']):
        if env not in data:
            continue
        
        heights = []
        labels = []
        for algo in classical + pqc:
            heights.append(data[env].get(algo, 0))
            labels.append(algo.replace('_', '\n').title())
        
        ax.bar(x + idx * width, heights, width,
               label=env.capitalize(),
               color=ENV_COLORS.get(env, '#333'),
               alpha=0.8)
    
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('p95 Latency (μs)')
    ax.set_title('Classical vs Post-Quantum Cryptography Performance')
    ax.set_xticks(x + width)
    ax.set_xticklabels([a.replace('_', '\n') for a in classical + pqc], rotation=0)
    # Only add legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()
    
    # Add vertical line separating classical and PQC
    ax.axvline(x=len(classical) - 0.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(len(classical) / 2 - 0.5, ax.get_ylim()[1] * 0.95, 'Classical', ha='center', fontsize=10)
    ax.text(len(classical) + len(pqc) / 2 - 0.5, ax.get_ylim()[1] * 0.95, 'Post-Quantum', ha='center', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / 'classical_vs_pqc.png'
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate scaling curve plots")
    parser.add_argument('--index', '-i', type=Path, required=True, help='Path to index.json')
    parser.add_argument('--output', '-o', type=Path, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Load aggregated stats
    stats_path = args.index.parent / 'aggregated_stats.json'
    if not stats_path.exists():
        stats_path = args.index.parent / 'stats' / 'aggregated_stats.json'
    
    if not stats_path.exists():
        print(f"Error: Aggregated stats not found. Run aggregate_results.py first.", file=sys.stderr)
        sys.exit(1)
    
    stats = load_aggregated_stats(stats_path)
    aggregated_entries = stats.get('aggregated', [])
    print(f"Loaded aggregated stats with {len(aggregated_entries)} entries")
    
    if len(aggregated_entries) == 0:
        print("\n  ⚠️  WARNING: No aggregated statistics found!", file=sys.stderr)
        print("     This may indicate:", file=sys.stderr)
        print("     - Analysis pipeline failed for all experiments", file=sys.stderr)
        print("     - Summary files are missing or invalid", file=sys.stderr)
        print("     - Aggregation script failed to process data", file=sys.stderr)
        print("     - Data format issues preventing aggregation", file=sys.stderr)
        print(f"     Stats file: {stats_path}", file=sys.stderr)
    else:
        # Count entries by environment and algorithm for diagnostics
        by_env = defaultdict(int)
        by_algo = defaultdict(int)
        for entry in aggregated_entries:
            by_env[entry.get('environment', 'unknown')] += 1
            by_algo[entry.get('algorithm', 'unknown')] += 1
        print(f"  By environment: {dict(by_env)}")
        print(f"  By algorithm: {dict(by_algo)}")
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating throughput vs payload plots...")
    plot_throughput_vs_payload(stats, args.output)
    
    print("\nGenerating latency vs rate plots...")
    plot_latency_vs_rate(stats, args.output)
    
    print("\nGenerating environment scaling plots...")
    plot_environment_scaling(stats, args.output)
    
    print("\nGenerating classical vs PQC comparison...")
    plot_classical_vs_pqc(stats, args.output)
    
    print("\nScaling curve generation complete!")


if __name__ == "__main__":
    main()

