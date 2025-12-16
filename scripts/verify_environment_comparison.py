#!/usr/bin/env python3
"""
Verify environment comparison numbers from summary files.
"""

import json
from collections import defaultdict
from pathlib import Path


def load_summary(filepath: Path) -> dict:
    """Load a summary.json file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_environment_overhead(results_dir: Path) -> dict:
    """Calculate environment overhead percentages."""
    # Group summaries by algorithm and payload/rate combination
    experiments = defaultdict(lambda: {'native': None, 'minikube': None, 'gcp': None})
    
    # Load all summaries
    for summary_file in results_dir.rglob('summary.json'):
        summary = load_summary(summary_file)
        
        # Extract environment and algorithm from path
        parts = summary_file.parts
        env_idx = parts.index('results') if 'results' in parts else -1
        if env_idx < 0 or env_idx + 1 >= len(parts):
            continue
        
        env = parts[env_idx + 1]
        if env not in ['native', 'minikube', 'gcp']:
            continue
        
        # Extract algorithm from experiment_id
        exp_id = summary.get('experiment_id', '')
        algorithm = None
        for alg in ['rsa2048', 'ecdsa', 'ecdhe', 'kyber512', 'dilithium2', 'hybrid']:
            if alg in exp_id.lower():
                algorithm = alg
                break
        
        if not algorithm:
            continue
        
        # Extract payload and rate to create unique key
        # Try to extract from experiment_id
        payload = None
        rate = None
        for part in exp_id.split('_'):
            if part.startswith('p') and part[1:].isdigit():
                payload = part
            elif part.startswith('r') and part[1:].isdigit():
                rate = part
        
        if payload and rate:
            key = f"{algorithm}_{payload}_{rate}"
            p95_latency = summary.get('latency', {}).get('p95')
            if p95_latency:
                experiments[key][env] = p95_latency
    
    # Calculate overheads
    overheads = {
        'minikube': [],
        'gcp': []
    }
    
    for key, envs in experiments.items():
        native_latency = envs.get('native')
        minikube_latency = envs.get('minikube')
        gcp_latency = envs.get('gcp')
        
        if native_latency:
            if minikube_latency:
                overhead_pct = ((minikube_latency - native_latency) / native_latency) * 100
                overheads['minikube'].append(overhead_pct)
            
            if gcp_latency:
                overhead_pct = ((gcp_latency - native_latency) / native_latency) * 100
                overheads['gcp'].append(overhead_pct)
    
    # Calculate statistics
    results = {}
    for env, values in overheads.items():
        if values:
            results[env] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
    
    return results


def main():
    results_dir = Path('/home/ausmarton/scratchpad/quantum-resilient/results')
    
    print("=" * 80)
    print("Environment Overhead Analysis")
    print("=" * 80)
    print()
    
    overheads = calculate_environment_overhead(results_dir)
    
    for env, stats in overheads.items():
        print(f"{env.upper()} Overhead (vs Native):")
        print(f"  Mean: {stats['mean']:.1f}%")
        print(f"  Range: {stats['min']:.1f}% to {stats['max']:.1f}%")
        print(f"  Configurations: {stats['count']}")
        print()


if __name__ == '__main__':
    main()
