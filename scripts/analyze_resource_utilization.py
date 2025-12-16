#!/usr/bin/env python3
"""
Analyze resource utilization (CPU and Memory) across algorithms and environments.
"""

import json
from pathlib import Path
from collections import defaultdict


def analyze_resource_utilization(results_dir: Path):
    """Analyze CPU and memory utilization from summary files."""
    
    results = {
        'cpu': defaultdict(list),
        'memory': defaultdict(list),
        'by_algorithm': defaultdict(lambda: {'cpu': [], 'memory': []}),
        'by_environment': defaultdict(lambda: {'cpu': [], 'memory': []})
    }
    
    algorithm_names = {
        'rsa2048': 'RSA-2048',
        'ecdsa': 'ECDSA P-256',
        'ecdhe': 'ECDHE P-256',
        'kyber512': 'Kyber-512',
        'dilithium2': 'Dilithium-2',
        'hybrid': 'Hybrid'
    }
    
    environment_names = {
        'native': 'Native',
        'minikube': 'Minikube',
        'gcp': 'GCP'
    }
    
    # Find all summary files
    summary_files = list(results_dir.glob('**/summary.json'))
    
    print(f"Found {len(summary_files)} summary files")
    print()
    
    processed = 0
    skipped_env = 0
    skipped_alg = 0
    for summary_file in summary_files:
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        # Extract environment from path (results/{env}/...)
        parts = summary_file.parts
        # Find 'results' in path and get the next element
        environment = None
        try:
            results_idx = parts.index('results')
            if results_idx + 1 < len(parts):
                environment = parts[results_idx + 1]
        except ValueError:
            pass
        
        if environment not in environment_names:
            skipped_env += 1
            continue
        
        # Extract algorithm from experiment_id
        exp_id = summary.get('experiment_id', '').lower()
        algorithm = None
        for alg_key in algorithm_names.keys():
            if alg_key in exp_id:
                algorithm = alg_key
                break
        
        if not algorithm:
            skipped_alg += 1
            continue
        
        # Extract CPU metrics
        cpu_data = summary.get('cpu', {})
        if cpu_data:
            mean_util = cpu_data.get('mean_utilization')
            
            if mean_util is not None:
                results['cpu'][algorithm].append(mean_util)
                results['by_algorithm'][algorithm]['cpu'].append(mean_util)
                results['by_environment'][environment]['cpu'].append(mean_util)
        
        # Extract memory metrics
        memory_data = summary.get('memory', {})
        if memory_data:
            # Try mean_rss_mb first (already converted), then mean_rss_bytes
            mean_rss_mb = memory_data.get('mean_rss_mb')
            if mean_rss_mb is None:
                mean_rss_bytes = memory_data.get('mean_rss_bytes')
                if mean_rss_bytes is not None:
                    mean_rss_mb = mean_rss_bytes / (1024 * 1024)
            
            if mean_rss_mb is not None:
                results['memory'][algorithm].append(mean_rss_mb)
                results['by_algorithm'][algorithm]['memory'].append(mean_rss_mb)
                results['by_environment'][environment]['memory'].append(mean_rss_mb)
        
        processed += 1
    
    print(f"Processed {processed} files successfully")
    print(f"Skipped {skipped_env} files (environment not recognized)")
    print(f"Skipped {skipped_alg} files (algorithm not recognized)")
    print(f"CPU data collected for: {list(results['cpu'].keys())}")
    print(f"Memory data collected for: {list(results['memory'].keys())}")
    print()
    
    return results, algorithm_names, environment_names


def calculate_statistics(values):
    """Calculate mean, std, min, max from a list of values."""
    if not values:
        return None
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = variance ** 0.5
    min_val = min(values)
    max_val = max(values)
    
    return {
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val,
        'count': len(values)
    }


def main():
    results_dir = Path('/home/ausmarton/scratchpad/quantum-resilient/results')
    
    print("=" * 80)
    print("Resource Utilization Analysis")
    print("=" * 80)
    print()
    
    results, algorithm_names, environment_names = analyze_resource_utilization(results_dir)
    
    # CPU Analysis by Algorithm
    print("CPU Utilization by Algorithm (%):")
    print("-" * 80)
    cpu_by_alg = {}
    for alg_key, alg_name in algorithm_names.items():
        if results['cpu'][alg_key]:  # Check if list has items
            stats = calculate_statistics(results['cpu'][alg_key])
            if stats:
                cpu_by_alg[alg_name] = stats
                print(f"{alg_name}:")
                print(f"  Mean: {stats['mean']:.2f}%")
                print(f"  Std: {stats['std']:.2f}%")
                print(f"  Range: {stats['min']:.2f}% - {stats['max']:.2f}%")
                print(f"  Configurations: {stats['count']}")
                print()
    
    if not cpu_by_alg:
        print("  ⚠️  No CPU data available (all values may be 0)")
        print()
    
    # Memory Analysis by Algorithm
    print("Memory Utilization by Algorithm (MB):")
    print("-" * 80)
    memory_by_alg = {}
    for alg_key, alg_name in algorithm_names.items():
        if results['memory'][alg_key]:  # Check if list has items
            stats = calculate_statistics(results['memory'][alg_key])
            if stats:
                memory_by_alg[alg_name] = stats
                print(f"{alg_name}:")
                print(f"  Mean: {stats['mean']:.2f} MB")
                print(f"  Std: {stats['std']:.2f} MB")
                print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f} MB")
                print(f"  Configurations: {stats['count']}")
                print()
    
    if not memory_by_alg:
        print("  ⚠️  No memory data available")
        print()
    
    # Environment Comparison
    print("Resource Utilization by Environment:")
    print("-" * 80)
    
    for env_key, env_name in environment_names.items():
        if env_key in results['by_environment']:
            env_data = results['by_environment'][env_key]
            
            print(f"{env_name}:")
            
            if env_data['cpu']:
                cpu_stats = calculate_statistics(env_data['cpu'])
                if cpu_stats:
                    print(f"  CPU (mean): {cpu_stats['mean']:.2f}%")
            
            if env_data['memory']:
                mem_stats = calculate_statistics(env_data['memory'])
                if mem_stats:
                    print(f"  Memory (mean): {mem_stats['mean']:.2f} MB")
            
            print()
    
    print("=" * 80)
    print("Analysis Complete")
    print("=" * 80)
    
    # Summary
    print()
    print("Summary:")
    print(f"  CPU data available: {len(cpu_by_alg) > 0}")
    print(f"  Memory data available: {len(memory_by_alg) > 0}")
    
    if len(cpu_by_alg) > 0 and len(memory_by_alg) > 0:
        print("  ✅ Resource utilization data is available for Section 4.2.8")
    else:
        print("  ⚠️  Limited resource utilization data - may need to note limitations")


if __name__ == '__main__':
    main()
