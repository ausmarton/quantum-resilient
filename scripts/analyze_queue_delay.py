#!/usr/bin/env python3
"""
Analyze queue delay contributions to total latency.
"""

import json
from collections import defaultdict
from pathlib import Path


def load_summary(filepath: Path) -> dict:
    """Load a summary.json file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_queue_delay(results_dir: Path) -> dict:
    """Analyze queue delay contributions."""
    data = defaultdict(list)
    
    for summary_file in results_dir.rglob('summary.json'):
        summary = load_summary(summary_file)
        
        # Only analyze native environment
        parts = summary_file.parts
        if 'native' not in parts:
            continue
        
        # Extract algorithm
        exp_id = summary.get('experiment_id', '')
        algorithm = None
        for alg in ['rsa2048', 'ecdsa', 'ecdhe', 'kyber512', 'dilithium2', 'hybrid']:
            if alg in exp_id.lower():
                algorithm = alg
                break
        
        if not algorithm:
            continue
        
        # Filter to low-rate experiments only (r100, r500) to avoid saturation effects
        # High rates (r2000, r10000) show extreme queue delays that dominate the analysis
        if 'r2000' in exp_id or 'r10000' in exp_id:
            continue
        
        # Get latency and queue delay (both in microseconds)
        # Use mean values to avoid extreme outliers at high rates
        latency_mean = summary.get('latency', {}).get('mean')
        queue_delay_mean = summary.get('queue_delay', {}).get('mean')
        latency_p95 = summary.get('latency', {}).get('p95')
        
        if latency_mean and queue_delay_mean and latency_mean > 0:
            contribution_pct = (queue_delay_mean / latency_mean) * 100
            data[algorithm].append({
                'latency_mean': latency_mean,
                'latency_p95': latency_p95,
                'queue_delay_mean': queue_delay_mean,
                'contribution_pct': contribution_pct
            })
    
    # Calculate statistics
    results = {}
    for algorithm, values in data.items():
        if values:
            contributions = [v['contribution_pct'] for v in values]
            queue_delays = [v['queue_delay_mean'] for v in values]
            latencies_mean = [v['latency_mean'] for v in values]
            latencies_p95 = [v['latency_p95'] for v in values if v.get('latency_p95')]
            
            results[algorithm] = {
                'mean_contribution_pct': sum(contributions) / len(contributions),
                'min_contribution_pct': min(contributions),
                'max_contribution_pct': max(contributions),
                'mean_queue_delay_us': sum(queue_delays) / len(queue_delays),
                'mean_latency_us': sum([v['latency_mean'] for v in values]) / len(values),
                'mean_latency_p95_us': sum(latencies_p95) / len(latencies_p95) if latencies_p95 else None,
                'count': len(values)
            }
    
    return results


def main():
    results_dir = Path('/home/ausmarton/scratchpad/quantum-resilient/results')
    
    print("=" * 80)
    print("Queue Delay Analysis")
    print("=" * 80)
    print()
    
    results = analyze_queue_delay(results_dir)
    
    # Algorithm name mapping
    alg_names = {
        'rsa2048': 'RSA-2048',
        'ecdsa': 'ECDSA P-256',
        'ecdhe': 'ECDHE P-256',
        'kyber512': 'Kyber-512',
        'dilithium2': 'Dilithium-2',
        'hybrid': 'Hybrid'
    }
    
    for alg_key, alg_name in alg_names.items():
        if alg_key in results:
            stats = results[alg_key]
            print(f"{alg_name}:")
            print(f"  Mean queue delay contribution: {stats['mean_contribution_pct']:.1f}%")
            print(f"  Range: {stats['min_contribution_pct']:.1f}% - {stats['max_contribution_pct']:.1f}%")
            print(f"  Mean queue delay: {stats['mean_queue_delay_us']:.2f} μs")
            print(f"  Mean total latency: {stats['mean_latency_us']:.2f} μs")
            if stats.get('mean_latency_p95_us'):
                print(f"  Mean total latency (p95): {stats['mean_latency_p95_us']:.2f} μs")
            print(f"  Configurations: {stats['count']}")
            print()


if __name__ == '__main__':
    main()
