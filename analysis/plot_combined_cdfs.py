#!/usr/bin/env python3
"""
Generate combined ECDF plots across algorithms and environments.

Produces:
- Combined ECDF overlays for each algorithm across all environments
- Per-payload-size panels
- Algorithm comparison plots

Usage:
    python analysis/plot_combined_cdfs.py \
        --index final-results/index.json \
        --output final-results/figures
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# Color palette for environments
ENV_COLORS = {
    'native': '#2ecc71',     # Green
    'minikube': '#3498db',   # Blue
    'gcp': '#e74c3c',        # Red
}

# Color palette for algorithms
ALGO_COLORS = {
    'rsa2048': '#9b59b6',          # Purple
    'ecdsa_p256': '#e67e22',       # Orange
    'kyber512': '#1abc9c',         # Teal
    'dilithium2': '#2980b9',       # Blue
    'hybrid_kyber_dilithium': '#c0392b',  # Dark red
}

# Line styles for environments
ENV_LINESTYLES = {
    'native': '-',
    'minikube': '--',
    'gcp': ':',
}


def load_latencies_from_jsonl(path: Path) -> Optional[np.ndarray]:
    """Load latency values from merged JSONL file."""
    try:
        # Try parquet first
        parquet_path = path.parent / 'merged.parquet'
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            if 'latency_us' in df.columns:
                return df['latency_us'].values
        
        # Fall back to JSONL
        if not path.exists():
            return None
        
        latencies = []
        with open(path) as f:
            for line in f:
                try:
                    event = json.loads(line)
                    if 'latency_us' in event:
                        latencies.append(event['latency_us'])
                except json.JSONDecodeError:
                    continue
        
        return np.array(latencies) if latencies else None
    except Exception as e:
        print(f"  Warning: Could not load {path}: {e}")
        return None


def compute_ecdf(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical CDF."""
    sorted_data = np.sort(data)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, ecdf


def plot_combined_ecdf_by_algorithm(
    data: dict[str, dict[str, list[np.ndarray]]],
    output_dir: Path,
):
    """Plot combined ECDF for each algorithm across environments."""
    for algorithm in data:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for env in ['native', 'minikube', 'gcp']:
            if env not in data[algorithm]:
                continue
            
            # Combine all runs for this algorithm/environment
            all_latencies = np.concatenate(data[algorithm][env]) if data[algorithm][env] else np.array([])
            
            if len(all_latencies) == 0:
                continue
            
            x, y = compute_ecdf(all_latencies)
            ax.plot(x, y, 
                   color=ENV_COLORS.get(env, '#333'),
                   linestyle=ENV_LINESTYLES.get(env, '-'),
                   linewidth=2,
                   label=f'{env.capitalize()}')
        
        ax.set_xlabel('Latency (μs)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'Latency ECDF: {algorithm}')
        ax.legend(loc='lower right')
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        
        # Add percentile markers
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='_p50')
        ax.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, label='_p95')
        ax.axhline(y=0.99, color='gray', linestyle=':', alpha=0.5, label='_p99')
        
        plt.tight_layout()
        output_path = output_dir / f'ecdf_{algorithm}.png'
        plt.savefig(output_path)
        plt.close()
        print(f"  Saved: {output_path}")


def plot_combined_ecdf_all_algorithms(
    data: dict[str, dict[str, list[np.ndarray]]],
    output_dir: Path,
    environment: str = 'native',
):
    """Plot combined ECDF for all algorithms in one environment."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for algorithm in data:
        if environment not in data[algorithm]:
            continue
        
        all_latencies = np.concatenate(data[algorithm][environment]) if data[algorithm][environment] else np.array([])
        
        if len(all_latencies) == 0:
            continue
        
        x, y = compute_ecdf(all_latencies)
        ax.plot(x, y,
               color=ALGO_COLORS.get(algorithm, '#333'),
               linewidth=2,
               label=algorithm.replace('_', ' ').title())
    
    ax.set_xlabel('Latency (μs)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'Latency ECDF - All Algorithms ({environment.capitalize()})')
    ax.legend(loc='lower right')
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    output_path = output_dir / f'combined_ecdf_{environment}.png'
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_environment_comparison_panel(
    data: dict[str, dict[str, list[np.ndarray]]],
    output_dir: Path,
):
    """Plot 3-panel comparison: native vs minikube vs gcp."""
    envs = ['native', 'minikube', 'gcp']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for idx, env in enumerate(envs):
        ax = axes[idx]
        
        for algorithm in data:
            if env not in data[algorithm]:
                continue
            
            all_latencies = np.concatenate(data[algorithm][env]) if data[algorithm][env] else np.array([])
            
            if len(all_latencies) == 0:
                continue
            
            x, y = compute_ecdf(all_latencies)
            ax.plot(x, y,
                   color=ALGO_COLORS.get(algorithm, '#333'),
                   linewidth=1.5,
                   label=algorithm.replace('_', ' ').title())
        
        ax.set_xlabel('Latency (μs)')
        if idx == 0:
            ax.set_ylabel('Cumulative Probability')
        ax.set_title(env.capitalize())
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        
        if idx == 2:
            ax.legend(loc='lower right', fontsize=8)
    
    plt.suptitle('Latency Distribution by Environment', fontsize=14)
    plt.tight_layout()
    
    output_path = output_dir / 'native_vs_minikube_vs_gcp.png'
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_payload_panels(
    data_by_payload: dict[int, dict[str, dict[str, list[np.ndarray]]]],
    output_dir: Path,
):
    """Plot per-payload-size panels."""
    payloads = sorted(data_by_payload.keys())
    
    if not payloads:
        return
    
    fig, axes = plt.subplots(1, len(payloads), figsize=(5 * len(payloads), 5), sharey=True)
    
    if len(payloads) == 1:
        axes = [axes]
    
    for idx, payload in enumerate(payloads):
        ax = axes[idx]
        payload_data = data_by_payload[payload]
        
        for algorithm in payload_data:
            for env in ['native', 'minikube', 'gcp']:
                if env not in payload_data[algorithm]:
                    continue
                
                all_latencies = np.concatenate(payload_data[algorithm][env]) if payload_data[algorithm][env] else np.array([])
                
                if len(all_latencies) == 0:
                    continue
                
                x, y = compute_ecdf(all_latencies)
                ax.plot(x, y,
                       color=ALGO_COLORS.get(algorithm, '#333'),
                       linestyle=ENV_LINESTYLES.get(env, '-'),
                       linewidth=1.5,
                       alpha=0.8)
        
        ax.set_xlabel('Latency (μs)')
        if idx == 0:
            ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'Payload: {payload} bytes')
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
    
    plt.suptitle('Latency Distribution by Payload Size', fontsize=14)
    plt.tight_layout()
    
    output_path = output_dir / 'ecdf_by_payload.png'
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate combined ECDF plots")
    parser.add_argument('--index', '-i', type=Path, required=True, help='Path to index.json')
    parser.add_argument('--output', '-o', type=Path, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Load index
    if not args.index.exists():
        print(f"Error: Index file not found: {args.index}", file=sys.stderr)
        sys.exit(1)
    
    with open(args.index) as f:
        index = json.load(f)
    
    print(f"Loaded index with {len(index.get('experiments', []))} experiments")
    
    # Organize data by algorithm -> environment -> list of latency arrays
    data: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    data_by_payload: dict[int, dict[str, dict[str, list[np.ndarray]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    
    for entry in index.get('experiments', []):
        if entry.get('status') not in ['success', 'cached']:
            continue
        
        output_dir = Path(entry['output_dir'])
        jsonl_path = output_dir / 'merged' / 'merged.jsonl'
        
        # Try alternative paths
        if not jsonl_path.exists():
            jsonl_path = output_dir / 'merged.jsonl'
        
        latencies = load_latencies_from_jsonl(jsonl_path)
        
        if latencies is not None and len(latencies) > 0:
            algorithm = entry['algorithm']
            environment = entry['environment']
            payload = entry['payload_size']
            
            data[algorithm][environment].append(latencies)
            data_by_payload[payload][algorithm][environment].append(latencies)
    
    print(f"Loaded latency data for {len(data)} algorithms")
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating ECDF plots by algorithm...")
    plot_combined_ecdf_by_algorithm(data, args.output)
    
    print("\nGenerating combined ECDF plots...")
    for env in ['native', 'minikube', 'gcp']:
        plot_combined_ecdf_all_algorithms(data, args.output, env)
    
    print("\nGenerating environment comparison panel...")
    plot_environment_comparison_panel(data, args.output)
    
    print("\nGenerating payload-size panels...")
    plot_payload_panels(data_by_payload, args.output)
    
    # Create combined ECDF (master plot)
    print("\nGenerating master combined ECDF...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for algorithm in data:
        for env in ['native', 'minikube', 'gcp']:
            if env not in data[algorithm]:
                continue
            
            all_latencies = np.concatenate(data[algorithm][env]) if data[algorithm][env] else np.array([])
            
            if len(all_latencies) == 0:
                continue
            
            x, y = compute_ecdf(all_latencies)
            ax.plot(x, y,
                   color=ALGO_COLORS.get(algorithm, '#333'),
                   linestyle=ENV_LINESTYLES.get(env, '-'),
                   linewidth=1.5,
                   alpha=0.8,
                   label=f'{algorithm} ({env})')
    
    ax.set_xlabel('Latency (μs)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Complete Latency ECDF - All Algorithms and Environments')
    ax.legend(loc='lower right', fontsize=7, ncol=2)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    output_path = args.output / 'combined_ecdf.png'
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")
    
    print("\nCDF plot generation complete!")


if __name__ == "__main__":
    main()

