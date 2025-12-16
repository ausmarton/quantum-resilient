#!/usr/bin/env python3
"""
Analyze horizontal scaling behavior from scaling experiments.
"""

import json
import re
from pathlib import Path
from collections import defaultdict


def extract_replica_count(exp_id: str) -> int:
    """Extract replica count from experiment ID."""
    # Format: {algorithm}_p{payload}_r{rate}_scaling_{hash}_r{replicas}
    match = re.search(r'_r(\d+)$', exp_id)
    if match:
        return int(match.group(1))
    return 1  # Base case (no _r suffix)


def analyze_scaling(results_dir: Path):
    """Analyze scaling behavior from summary files."""
    
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    algorithm_names = {
        'rsa2048': 'RSA-2048',
        'ecdsa': 'ECDSA P-256',
        'ecdhe': 'ECDHE P-256',
        'kyber512': 'Kyber-512',
        'dilithium2': 'Dilithium-2',
        'hybrid': 'Hybrid'
    }
    
    # Find all summary files with "scaling" in path
    summary_files = list(results_dir.glob('**/summary.json'))
    
    for summary_file in summary_files:
        if 'scaling' not in str(summary_file):
            continue
        
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            # Extract environment
            parts = summary_file.parts
            try:
                results_idx = parts.index('results')
                if results_idx + 1 < len(parts):
                    environment = parts[results_idx + 1]
                else:
                    continue
            except ValueError:
                continue
            
            if environment not in ['native', 'minikube', 'gcp']:
                continue
            
            # Extract algorithm
            exp_id = summary.get('experiment_id', '').lower()
            algorithm = None
            for alg_key in algorithm_names.keys():
                if alg_key in exp_id:
                    algorithm = alg_key
                    break
            
            if not algorithm:
                continue
            
            # Extract replica count
            replica_count = extract_replica_count(exp_id)
            
            # Extract metrics
            latency_p95 = summary.get('latency', {}).get('p95')
            throughput_mean = summary.get('throughput', {}).get('mean_msgs_per_sec')
            
            if latency_p95 is not None and throughput_mean is not None:
                results[environment][algorithm][replica_count].append({
                    'latency_p95': latency_p95,
                    'throughput': throughput_mean
                })
        
        except Exception as e:
            continue
    
    return results, algorithm_names


def calculate_statistics(values, key):
    """Calculate mean from a list of dicts."""
    if not values:
        return None
    
    vals = [v[key] for v in values if key in v]
    if not vals:
        return None
    
    return {
        'mean': sum(vals) / len(vals),
        'count': len(vals)
    }


def main():
    results_dir = Path('/home/ausmarton/scratchpad/quantum-resilient/results')
    
    print("=" * 80)
    print("Horizontal Scaling Analysis")
    print("=" * 80)
    print()
    
    results, algorithm_names = analyze_scaling(results_dir)
    
    # Analyze by environment
    for env in ['minikube', 'gcp']:
        if env not in results:
            continue
        
        print(f"{env.upper()} Scaling Analysis:")
        print("-" * 80)
        
        for alg_key, alg_name in algorithm_names.items():
            if alg_key not in results[env]:
                continue
            
            replicas_data = results[env][alg_key]
            replica_counts = sorted(replicas_data.keys())
            
            if len(replica_counts) < 2:
                continue
            
            print(f"\n{alg_name}:")
            
            for replica_count in replica_counts:
                data = replicas_data[replica_count]
                latency_stats = calculate_statistics(data, 'latency_p95')
                throughput_stats = calculate_statistics(data, 'throughput')
                
                if latency_stats and throughput_stats:
                    print(f"  Replica {replica_count}:")
                    print(f"    Latency (p95): {latency_stats['mean']:.2f} μs")
                    print(f"    Throughput: {throughput_stats['mean']:.2f} msg/s")
                    
                    # Calculate speedup if we have replica 1
                    if replica_count > 1 and 1 in replicas_data:
                        base_throughput = calculate_statistics(replicas_data[1], 'throughput')
                        if base_throughput:
                            speedup = throughput_stats['mean'] / base_throughput['mean']
                            efficiency = speedup / replica_count * 100
                            print(f"    Speedup: {speedup:.2f}x (Efficiency: {efficiency:.1f}%)")
        
        print()
    
    print("=" * 80)
    print("Analysis Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
