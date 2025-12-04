#!/usr/bin/env python3
"""
Plot replica scaling analysis for horizontal scaling experiments.

Produces:
- Throughput vs replicas
- Latency vs replicas
- Scaling efficiency metric (speedup / replicas)
- Interference effects visualization

Usage:
    python analysis/plot_replica_scaling.py \
        --index final-results/index.json \
        --output final-results/figures/scaling
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available", file=sys.stderr)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# Color scheme for algorithms
ALGO_COLORS = {
    'rsa2048': '#e74c3c',
    'ecdsa_p256': '#3498db',
    'kyber512': '#2ecc71',
    'dilithium2': '#9b59b6',
    'hybrid_kyber_dilithium': '#f39c12',
}

# Marker styles for environments
ENV_MARKERS = {
    'native': 'o',
    'minikube': 's',
    'gcp': '^',
}


def load_scaling_data(index_path: Path) -> dict:
    """Load scaling experiment data from index."""
    with open(index_path) as f:
        index = json.load(f)
    
    # Structure: {algorithm: {environment: {replicas: [summary_data, ...]}}}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for exp in index.get('experiments', []):
        if exp.get('status') not in ['success', 'cached']:
            continue
        
        algorithm = exp.get('algorithm', 'unknown')
        environment = exp.get('environment', 'unknown')
        replicas = exp.get('replicas', 1)
        output_dir = Path(exp.get('output_dir', ''))
        
        # Look for summary.json
        summary_path = output_dir / 'stats' / 'summary.json'
        if not summary_path.exists():
            summary_path = output_dir / 'aggregated_stats.json'
        
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
                
                # Extract key metrics
                latency = summary.get('latency', {})
                throughput_data = summary.get('throughput', {})
                
                metrics = {
                    'replicas': replicas,
                    'p50': latency.get('p50', latency.get('p50', {}).get('mean', None)),
                    'p95': latency.get('p95', latency.get('p95', {}).get('mean', None)),
                    'p99': latency.get('p99', latency.get('p99', {}).get('mean', None)),
                    'mean': latency.get('mean', latency.get('mean', {}).get('mean', None)),
                    'throughput': throughput_data.get('mean', throughput_data.get('ops_per_sec', None)),
                    'count': summary.get('count', 0),
                }
                
                # Handle nested structures
                if isinstance(metrics['p50'], dict):
                    metrics['p50'] = metrics['p50'].get('mean')
                if isinstance(metrics['p95'], dict):
                    metrics['p95'] = metrics['p95'].get('mean')
                if isinstance(metrics['p99'], dict):
                    metrics['p99'] = metrics['p99'].get('mean')
                if isinstance(metrics['throughput'], dict):
                    metrics['throughput'] = metrics['throughput'].get('mean')
                
                data[algorithm][environment][replicas].append(metrics)
                
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {summary_path}: {e}", file=sys.stderr)
    
    return data


def compute_scaling_metrics(data: dict) -> dict:
    """
    Compute scaling metrics from raw data.
    
    Returns structure with:
    - throughput scaling
    - latency scaling
    - efficiency (speedup / replicas)
    """
    results = defaultdict(lambda: defaultdict(dict))
    
    for algo in data:
        for env in data[algo]:
            replica_counts = sorted(data[algo][env].keys())
            
            if not replica_counts:
                continue
            
            # Get baseline (1 replica) metrics
            baseline_data = data[algo][env].get(1, [])
            if not baseline_data:
                baseline_data = data[algo][env].get(min(replica_counts), [])
            
            if not baseline_data:
                continue
            
            baseline_throughput = np.mean([d['throughput'] for d in baseline_data if d['throughput']])
            baseline_p95 = np.mean([d['p95'] for d in baseline_data if d['p95']])
            
            # Compute metrics for each replica count
            metrics = {
                'replica_counts': [],
                'throughput_mean': [],
                'throughput_std': [],
                'throughput_speedup': [],
                'efficiency': [],
                'p50_mean': [],
                'p50_std': [],
                'p95_mean': [],
                'p95_std': [],
                'p99_mean': [],
                'p99_std': [],
                'latency_ratio': [],  # vs baseline
                'interference_factor': [],  # how much latency increases
            }
            
            for r in replica_counts:
                run_data = data[algo][env][r]
                
                if not run_data:
                    continue
                
                throughputs = [d['throughput'] for d in run_data if d['throughput']]
                p50s = [d['p50'] for d in run_data if d['p50']]
                p95s = [d['p95'] for d in run_data if d['p95']]
                p99s = [d['p99'] for d in run_data if d['p99']]
                
                if not throughputs:
                    continue
                
                mean_throughput = np.mean(throughputs)
                mean_p50 = np.mean(p50s) if p50s else 0
                mean_p95 = np.mean(p95s) if p95s else 0
                mean_p99 = np.mean(p99s) if p99s else 0
                
                metrics['replica_counts'].append(r)
                metrics['throughput_mean'].append(mean_throughput)
                metrics['throughput_std'].append(np.std(throughputs) if len(throughputs) > 1 else 0)
                
                # Speedup = current throughput / baseline throughput
                speedup = mean_throughput / baseline_throughput if baseline_throughput else 1
                metrics['throughput_speedup'].append(speedup)
                
                # Efficiency = speedup / replicas (ideal = 1.0)
                efficiency = speedup / r if r > 0 else 0
                metrics['efficiency'].append(efficiency)
                
                metrics['p50_mean'].append(mean_p50)
                metrics['p50_std'].append(np.std(p50s) if len(p50s) > 1 else 0)
                metrics['p95_mean'].append(mean_p95)
                metrics['p95_std'].append(np.std(p95s) if len(p95s) > 1 else 0)
                metrics['p99_mean'].append(mean_p99)
                metrics['p99_std'].append(np.std(p99s) if len(p99s) > 1 else 0)
                
                # Latency ratio vs baseline
                latency_ratio = mean_p95 / baseline_p95 if baseline_p95 else 1
                metrics['latency_ratio'].append(latency_ratio)
                
                # Interference factor = (actual_latency - ideal_latency) / ideal_latency
                # Ideal latency = baseline (no contention)
                interference = (mean_p95 - baseline_p95) / baseline_p95 if baseline_p95 else 0
                metrics['interference_factor'].append(interference)
            
            results[algo][env] = metrics
    
    return results


def plot_throughput_scaling(metrics: dict, output_dir: Path) -> None:
    """Plot throughput vs replicas."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Throughput vs Replicas
    ax1 = axes[0]
    for algo in sorted(metrics.keys()):
        for env in sorted(metrics[algo].keys()):
            m = metrics[algo][env]
            if not m.get('replica_counts'):
                continue
            
            color = ALGO_COLORS.get(algo, '#333333')
            marker = ENV_MARKERS.get(env, 'o')
            label = f"{algo} ({env})"
            
            ax1.errorbar(
                m['replica_counts'],
                m['throughput_mean'],
                yerr=m['throughput_std'],
                label=label,
                color=color,
                marker=marker,
                linestyle='-',
                capsize=3,
                markersize=8,
            )
    
    ax1.set_xlabel('Number of Replicas', fontsize=12)
    ax1.set_ylabel('Throughput (ops/s)', fontsize=12)
    ax1.set_title('Throughput Scaling', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([1, 2, 4, 8])
    
    # Right: Speedup vs Replicas
    ax2 = axes[1]
    
    # Add ideal scaling line
    ax2.plot([1, 8], [1, 8], 'k--', label='Ideal (linear)', alpha=0.5)
    
    for algo in sorted(metrics.keys()):
        for env in sorted(metrics[algo].keys()):
            m = metrics[algo][env]
            if not m.get('replica_counts'):
                continue
            
            color = ALGO_COLORS.get(algo, '#333333')
            marker = ENV_MARKERS.get(env, 'o')
            label = f"{algo} ({env})"
            
            ax2.plot(
                m['replica_counts'],
                m['throughput_speedup'],
                label=label,
                color=color,
                marker=marker,
                linestyle='-',
                markersize=8,
            )
    
    ax2.set_xlabel('Number of Replicas', fontsize=12)
    ax2.set_ylabel('Speedup (×)', fontsize=12)
    ax2.set_title('Throughput Speedup', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([1, 2, 4, 8])
    ax2.set_yticks([1, 2, 4, 8])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ throughput_scaling.png")


def plot_latency_scaling(metrics: dict, output_dir: Path) -> None:
    """Plot latency vs replicas."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: P95 Latency vs Replicas
    ax1 = axes[0]
    for algo in sorted(metrics.keys()):
        for env in sorted(metrics[algo].keys()):
            m = metrics[algo][env]
            if not m.get('replica_counts'):
                continue
            
            color = ALGO_COLORS.get(algo, '#333333')
            marker = ENV_MARKERS.get(env, 'o')
            label = f"{algo} ({env})"
            
            ax1.errorbar(
                m['replica_counts'],
                m['p95_mean'],
                yerr=m['p95_std'],
                label=label,
                color=color,
                marker=marker,
                linestyle='-',
                capsize=3,
                markersize=8,
            )
    
    ax1.set_xlabel('Number of Replicas', fontsize=12)
    ax1.set_ylabel('P95 Latency (μs)', fontsize=12)
    ax1.set_title('P95 Latency vs Replicas', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([1, 2, 4, 8])
    
    # Right: Latency Ratio (normalized to 1 replica)
    ax2 = axes[1]
    
    # Add ideal line (no degradation)
    ax2.axhline(y=1.0, color='k', linestyle='--', label='Ideal (no degradation)', alpha=0.5)
    
    for algo in sorted(metrics.keys()):
        for env in sorted(metrics[algo].keys()):
            m = metrics[algo][env]
            if not m.get('replica_counts'):
                continue
            
            color = ALGO_COLORS.get(algo, '#333333')
            marker = ENV_MARKERS.get(env, 'o')
            label = f"{algo} ({env})"
            
            ax2.plot(
                m['replica_counts'],
                m['latency_ratio'],
                label=label,
                color=color,
                marker=marker,
                linestyle='-',
                markersize=8,
            )
    
    ax2.set_xlabel('Number of Replicas', fontsize=12)
    ax2.set_ylabel('Latency Ratio (vs 1 replica)', fontsize=12)
    ax2.set_title('Latency Degradation', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([1, 2, 4, 8])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ latency_scaling.png")


def plot_efficiency(metrics: dict, output_dir: Path) -> None:
    """Plot scaling efficiency (speedup / replicas)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add ideal efficiency line
    ax.axhline(y=1.0, color='k', linestyle='--', label='Ideal (100%)', alpha=0.5, linewidth=2)
    ax.axhline(y=0.5, color='r', linestyle=':', label='50% efficiency', alpha=0.3)
    
    bar_width = 0.15
    x_positions = np.array([1, 2, 4, 8])
    offset = 0
    
    all_labels = []
    
    for algo in sorted(metrics.keys()):
        for env in sorted(metrics[algo].keys()):
            m = metrics[algo][env]
            if not m.get('replica_counts'):
                continue
            
            color = ALGO_COLORS.get(algo, '#333333')
            label = f"{algo} ({env})"
            
            # Match replica counts to x positions
            eff_values = []
            for x in x_positions:
                if x in m['replica_counts']:
                    idx = m['replica_counts'].index(x)
                    eff_values.append(m['efficiency'][idx])
                else:
                    eff_values.append(0)
            
            ax.bar(
                x_positions + offset * bar_width,
                eff_values,
                bar_width,
                label=label,
                color=color,
                alpha=0.8,
            )
            
            offset += 1
            all_labels.append(label)
    
    ax.set_xlabel('Number of Replicas', fontsize=12)
    ax.set_ylabel('Scaling Efficiency (speedup / replicas)', fontsize=12)
    ax.set_title('Horizontal Scaling Efficiency', fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions + (offset - 1) * bar_width / 2)
    ax.set_xticklabels(['1', '2', '4', '8'])
    ax.set_ylim(0, 1.2)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ scaling_efficiency.png")


def plot_interference(metrics: dict, output_dir: Path) -> None:
    """Plot interference effects heatmap."""
    # Collect data for heatmap
    algos = sorted(metrics.keys())
    envs = set()
    for algo in algos:
        envs.update(metrics[algo].keys())
    envs = sorted(envs)
    
    replicas = [1, 2, 4, 8]
    
    fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 6), squeeze=False)
    
    for env_idx, env in enumerate(envs):
        ax = axes[0][env_idx]
        
        # Build matrix: algorithms x replicas
        matrix = np.zeros((len(algos), len(replicas)))
        matrix[:] = np.nan
        
        for algo_idx, algo in enumerate(algos):
            if env not in metrics[algo]:
                continue
            
            m = metrics[algo][env]
            for rep_idx, rep in enumerate(replicas):
                if rep in m['replica_counts']:
                    idx = m['replica_counts'].index(rep)
                    # Interference factor as percentage
                    matrix[algo_idx, rep_idx] = m['interference_factor'][idx] * 100
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=-10, vmax=50)
        
        ax.set_xticks(range(len(replicas)))
        ax.set_xticklabels(replicas)
        ax.set_yticks(range(len(algos)))
        ax.set_yticklabels(algos)
        
        ax.set_xlabel('Replicas')
        ax.set_ylabel('Algorithm')
        ax.set_title(f'{env.upper()} Interference', fontweight='bold')
        
        # Add text annotations
        for i in range(len(algos)):
            for j in range(len(replicas)):
                if not np.isnan(matrix[i, j]):
                    text = f'{matrix[i, j]:.0f}%'
                    color = 'white' if abs(matrix[i, j]) > 25 else 'black'
                    ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    cbar.set_label('Latency Increase (%)', fontsize=10)
    
    plt.suptitle('Interference Effects (Latency Increase vs Single Replica)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'interference_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ interference_heatmap.png")


def save_scaling_data(metrics: dict, output_dir: Path) -> None:
    """Save scaling metrics to JSON."""
    # Convert to serializable format
    output = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'algorithms': {},
    }
    
    for algo in metrics:
        output['algorithms'][algo] = {}
        for env in metrics[algo]:
            m = metrics[algo][env]
            output['algorithms'][algo][env] = {
                'replica_counts': m.get('replica_counts', []),
                'throughput': {
                    'mean': m.get('throughput_mean', []),
                    'std': m.get('throughput_std', []),
                    'speedup': m.get('throughput_speedup', []),
                },
                'latency_p95': {
                    'mean': m.get('p95_mean', []),
                    'std': m.get('p95_std', []),
                    'ratio': m.get('latency_ratio', []),
                },
                'efficiency': m.get('efficiency', []),
                'interference_factor': m.get('interference_factor', []),
            }
    
    output_path = output_dir / 'scaling_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  ✓ scaling_metrics.json")


def main():
    parser = argparse.ArgumentParser(
        description="Generate replica scaling analysis plots"
    )
    parser.add_argument(
        '--index', '-i', type=Path, required=True,
        help='Path to index.json'
    )
    parser.add_argument(
        '--output', '-o', type=Path, required=True,
        help='Output directory for figures'
    )
    
    args = parser.parse_args()
    
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for plotting", file=sys.stderr)
        sys.exit(1)
    
    if not args.index.exists():
        print(f"Error: Index file not found: {args.index}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading scaling data from: {args.index}")
    data = load_scaling_data(args.index)
    
    if not data:
        print("No scaling data found in index", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found data for {len(data)} algorithms")
    
    print("Computing scaling metrics...")
    metrics = compute_scaling_metrics(data)
    
    print(f"\nGenerating plots to: {args.output}")
    
    # Generate all plots
    plot_throughput_scaling(metrics, args.output)
    plot_latency_scaling(metrics, args.output)
    plot_efficiency(metrics, args.output)
    plot_interference(metrics, args.output)
    save_scaling_data(metrics, args.output)
    
    print("\n✓ Scaling analysis complete!")


if __name__ == "__main__":
    main()

