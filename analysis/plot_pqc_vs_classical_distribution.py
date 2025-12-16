#!/usr/bin/env python3
"""
Generate violin/boxplot comparison: PQC vs Classical distribution comparison.

Produces a combined visualization showing:
- Violin plots or box-and-whisker plots
- X-axis: Algorithm (grouped: Classical vs PQC)
- Y-axis: Latency (log-scale if needed)
- Overlay: median, p95, p99 markers

This makes three things instantly visible:
- Median shift
- Variance reduction
- Tail compression

Usage:
    python analysis/plot_pqc_vs_classical_distribution.py \
        --index final-results/index.json \
        --output final-results/figures
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

# Environment label mapping (for display)
ENV_DISPLAY_NAMES = {
    'native': 'Bare-metal',
    'minikube': 'Local-K8s',
    'gcp': 'Cloud-K8s'
}

# Algorithm groupings
CLASSICAL_ALGOS = ['rsa2048', 'ecdsa', 'ecdhe']
PQC_ALGOS = ['kyber512', 'dilithium2', 'hybrid_kyber_dilithium']

# Algorithm display names
ALGO_DISPLAY_NAMES = {
    'rsa2048': 'RSA-2048',
    'ecdsa': 'ECDSA P-256',
    'ecdhe': 'ECDHE P-256',
    'kyber512': 'Kyber-512',
    'dilithium2': 'Dilithium-2',
    'hybrid_kyber_dilithium': 'Hybrid',
}

# Colors: Classical (grays) vs PQC (blues)
ALGO_COLORS = {
    'rsa2048': '#666666',
    'ecdsa': '#888888',
    'ecdhe': '#aaaaaa',
    'kyber512': '#2980b9',
    'dilithium2': '#3498db',
    'hybrid_kyber_dilithium': '#5dade2',
}


def load_latencies_from_jsonl(path: Path) -> Optional[np.ndarray]:
    """Load latency values from merged JSONL or parquet file."""
    try:
        # Try parquet first
        parquet_path = path.parent / 'merged.parquet'
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                if 'latency_ns' in df.columns:
                    return (df['latency_ns'].values / 1000.0)  # Convert to microseconds
                elif 'latency_us' in df.columns:
                    return df['latency_us'].values
            except Exception as e:
                print(f"  Warning: Failed to read parquet {parquet_path}: {e}", file=sys.stderr)
        
        # Fall back to JSONL
        if not path.exists():
            return None
        
        latencies = []
        with open(path) as f:
            for line in f:
                try:
                    event = json.loads(line)
                    if 'latency_ns' in event:
                        latencies.append(event['latency_ns'] / 1000.0)
                    elif 'latency_us' in event:
                        latencies.append(event['latency_us'])
                except json.JSONDecodeError:
                    continue
        
        return np.array(latencies) if latencies else None
    except Exception as e:
        print(f"  Warning: Could not load {path}: {e}", file=sys.stderr)
        return None


def load_data_from_index(index_path: Path, environment: str = 'native', 
                         payload_size: int = 1024, rate: int = 2000):
    """Load latency data for all algorithms from index."""
    with open(index_path) as f:
        index = json.load(f)
    
    data = {}
    
    for exp in index.get('experiments', []):
        if (exp.get('environment') == environment and 
            exp.get('payload_size') == payload_size and
            exp.get('rate') == rate and
            exp.get('status') == 'success'):
            
            algo = exp.get('algorithm')
            output_dir_str = exp.get('output_dir', '')
            
            # Convert container paths (/workspace/...) to host paths
            if output_dir_str.startswith('/workspace/'):
                # Convert /workspace/results/... to results/... (relative to project root)
                output_dir = Path(output_dir_str.replace('/workspace/', ''))
            elif output_dir_str.startswith('/'):
                # Absolute path - use as-is but check if it exists
                output_dir = Path(output_dir_str)
            else:
                # Relative path - use as-is
                output_dir = Path(output_dir_str)
            
            # Try multiple possible locations
            possible_paths = [
                output_dir / 'merged' / 'merged.jsonl',
                output_dir / 'merged' / 'merged.parquet',
                Path('results') / environment / exp.get('scenario_id', '') / 'merged' / 'merged.jsonl',
                Path('results') / environment / exp.get('scenario_id', '') / 'merged' / 'merged.parquet',
            ]
            
            merged_jsonl = None
            merged_parquet = None
            for path in possible_paths:
                if path.exists():
                    if path.suffix == '.parquet':
                        merged_parquet = path
                    else:
                        merged_jsonl = path
                    break
            
            latencies = None
            if merged_parquet and merged_parquet.exists():
                latencies = load_latencies_from_jsonl(merged_parquet)
            elif merged_jsonl and merged_jsonl.exists():
                latencies = load_latencies_from_jsonl(merged_jsonl)
            
            if latencies is not None and len(latencies) > 0:
                if algo not in data:
                    data[algo] = []
                data[algo].append(latencies)
    
    # Combine all runs for each algorithm
    combined_data = {}
    for algo, runs in data.items():
        if runs:
            combined_data[algo] = np.concatenate(runs)
    
    return combined_data


def plot_pqc_vs_classical_distribution(data: dict, output_path: Path):
    """Create violin/boxplot comparison of PQC vs Classical distributions."""
    # Prepare data for plotting
    plot_data = []
    
    for algo_key, latencies in data.items():
        # Normalize algorithm key
        algo_normalized = algo_key.lower().replace('-', '').replace('_', '')
        
        # Determine category
        if any(c in algo_normalized for c in ['rsa', 'ecdsa', 'ecdhe']):
            category = 'Classical'
        elif any(p in algo_normalized for p in ['kyber', 'dilithium', 'hybrid']):
            category = 'PQC'
        else:
            continue
        
        # Get display name
        display_name = ALGO_DISPLAY_NAMES.get(algo_normalized, algo_key)
        
        # Sample if too large (for performance)
        if len(latencies) > 100000:
            latencies = np.random.choice(latencies, 100000, replace=False)
        
        # Calculate percentiles
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        plot_data.append({
            'algorithm': display_name,
            'category': category,
            'latency': latencies,
            'p50': p50,
            'p95': p95,
            'p99': p99,
            'algo_key': algo_normalized,
        })
    
    if not plot_data:
        print("Error: No data to plot", file=sys.stderr)
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare DataFrame for seaborn
    df_list = []
    for item in plot_data:
        for latency in item['latency']:
            df_list.append({
                'Algorithm': item['algorithm'],
                'Category': item['category'],
                'Latency (μs)': latency,
            })
    df = pd.DataFrame(df_list)
    
    # Sort algorithms: Classical first, then PQC
    classical_order = ['RSA-2048', 'ECDSA P-256', 'ECDHE P-256']
    pqc_order = ['Kyber-512', 'Dilithium-2', 'Hybrid']
    algo_order = [a for a in classical_order + pqc_order if a in df['Algorithm'].unique()]
    
    # Plot 1: Violin plot
    sns.violinplot(data=df, x='Algorithm', y='Latency (μs)', 
                   order=algo_order, ax=ax1, palette=[ALGO_COLORS.get(a.lower().replace('-', '').replace(' ', ''), '#333333') for a in algo_order])
    ax1.set_yscale('log')
    ax1.set_title('Latency Distribution: PQC vs Classical', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Algorithm', fontsize=11)
    ax1.set_ylabel('Latency (μs, log scale)', fontsize=11)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add percentile markers
    for item in plot_data:
        algo_idx = algo_order.index(item['algorithm']) if item['algorithm'] in algo_order else None
        if algo_idx is not None:
            # Add horizontal lines for percentiles
            ax1.axhline(y=item['p50'], xmin=(algo_idx-0.4)/len(algo_order), 
                       xmax=(algo_idx+0.4)/len(algo_order), 
                       color='red', linestyle='--', linewidth=1, alpha=0.7, label='p50' if algo_idx == 0 else '')
            ax1.axhline(y=item['p95'], xmin=(algo_idx-0.3)/len(algo_order), 
                       xmax=(algo_idx+0.3)/len(algo_order), 
                       color='orange', linestyle=':', linewidth=1, alpha=0.7, label='p95' if algo_idx == 0 else '')
            ax1.axhline(y=item['p99'], xmin=(algo_idx-0.2)/len(algo_order), 
                       xmax=(algo_idx+0.2)/len(algo_order), 
                       color='purple', linestyle='-.', linewidth=1, alpha=0.7, label='p99' if algo_idx == 0 else '')
    
    # Plot 2: Box plot with percentile markers
    box_plot = sns.boxplot(data=df, x='Algorithm', y='Latency (μs)', 
                          order=algo_order, ax=ax2,
                          palette=[ALGO_COLORS.get(a.lower().replace('-', '').replace(' ', ''), '#333333') for a in algo_order])
    ax2.set_yscale('log')
    ax2.set_title('Latency Distribution: Box Plot with Percentiles', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Algorithm', fontsize=11)
    ax2.set_ylabel('Latency (μs, log scale)', fontsize=11)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add percentile markers as points
    for item in plot_data:
        algo_idx = algo_order.index(item['algorithm']) if item['algorithm'] in algo_order else None
        if algo_idx is not None:
            ax2.scatter([algo_idx], [item['p50']], color='red', marker='s', s=50, zorder=10, label='p50' if algo_idx == 0 else '')
            ax2.scatter([algo_idx], [item['p95']], color='orange', marker='^', s=50, zorder=10, label='p95' if algo_idx == 0 else '')
            ax2.scatter([algo_idx], [item['p99']], color='purple', marker='D', s=50, zorder=10, label='p99' if algo_idx == 0 else '')
    
    # Add legend for percentiles
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='p50 (median)'),
        Line2D([0], [0], color='orange', linestyle=':', linewidth=1, label='p95'),
        Line2D([0], [0], color='purple', linestyle='-.', linewidth=1, label='p99'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate PQC vs Classical distribution comparison plots')
    parser.add_argument('--index', type=Path, required=True, help='Path to index.json')
    parser.add_argument('--output', type=Path, required=True, help='Output directory for figures')
    parser.add_argument('--environment', default='native', choices=['native', 'minikube', 'gcp'],
                       help='Environment to plot (default: native)')
    parser.add_argument('--payload-size', type=int, default=1024, help='Payload size in bytes (default: 1024)')
    parser.add_argument('--rate', type=int, default=2000, help='Workload rate in msg/s (default: 2000)')
    
    args = parser.parse_args()
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {args.index}...")
    data = load_data_from_index(args.index, args.environment, args.payload_size, args.rate)
    
    if not data:
        print("Error: No data loaded. Check index file and parameters.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded data for {len(data)} algorithms")
    
    # Map environment name to new terminology for filename
    env_map = {'native': 'bare-metal', 'minikube': 'local-k8s', 'gcp': 'cloud-k8s'}
    env_for_file = env_map.get(args.environment, args.environment)
    output_path = args.output / f'pqc_vs_classical_distribution_{env_for_file}.png'
    plot_pqc_vs_classical_distribution(data, output_path)
    
    print("Done!")


if __name__ == '__main__':
    main()
