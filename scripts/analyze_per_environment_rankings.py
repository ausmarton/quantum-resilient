#!/usr/bin/env python3
"""
Analyze per-environment algorithm rankings and environment-normalized ratios.
"""

import json
from pathlib import Path
from collections import defaultdict


def analyze_per_environment_rankings(results_dir: Path):
    """Analyze algorithm performance per environment."""
    
    results = {
        'native': defaultdict(list),
        'minikube': defaultdict(list),
        'gcp': defaultdict(list)
    }
    
    algorithm_names = {
        'rsa2048': 'RSA-2048',
        'ecdsa': 'ECDSA P-256',
        'ecdhe': 'ECDHE P-256',
        'kyber512': 'Kyber-512',
        'dilithium2': 'Dilithium-2',
        'hybrid': 'Hybrid'
    }
    
    # Find all summary files
    summary_files = list(results_dir.glob('**/summary.json'))
    
    for summary_file in summary_files:
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
            
            if environment not in results:
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
            
            # Extract p95 latency
            latency_p95 = summary.get('latency', {}).get('p95')
            if latency_p95 is not None:
                results[environment][algorithm].append(latency_p95)
        
        except Exception:
            continue
    
    return results, algorithm_names


def calculate_mean(values):
    """Calculate mean from a list of values."""
    if not values:
        return None
    return sum(values) / len(values)


def main():
    results_dir = Path('/home/ausmarton/scratchpad/quantum-resilient/results')
    
    print("=" * 80)
    print("Per-Environment Algorithm Rankings")
    print("=" * 80)
    print()
    
    results, algorithm_names = analyze_per_environment_rankings(results_dir)
    
    # Calculate means per environment
    env_means = {}
    for env in ['native', 'minikube', 'gcp']:
        env_means[env] = {}
        for alg_key, alg_name in algorithm_names.items():
            if alg_key in results[env] and results[env][alg_key]:
                mean = calculate_mean(results[env][alg_key])
                if mean:
                    env_means[env][alg_key] = mean
    
    # Print per-environment rankings
    for env in ['native', 'minikube', 'gcp']:
        print(f"{env.upper()} Environment Rankings:")
        print("-" * 80)
        
        # Sort by mean latency
        sorted_algs = sorted(env_means[env].items(), key=lambda x: x[1])
        
        print("Rank | Algorithm | Mean p95 Latency (μs)")
        print("-" * 80)
        for rank, (alg_key, mean_latency) in enumerate(sorted_algs, 1):
            alg_name = algorithm_names[alg_key]
            print(f"{rank:4d} | {alg_name:20s} | {mean_latency:8.2f}")
        print()
    
    # Calculate environment-normalized ratios
    print("=" * 80)
    print("Environment-Normalized Ratios (GCP/Native and Minikube/Native)")
    print("=" * 80)
    print()
    
    if 'native' in env_means:
        print("Algorithm | Minikube/Native | GCP/Native")
        print("-" * 80)
        for alg_key, alg_name in algorithm_names.items():
            if alg_key in env_means['native']:
                native_lat = env_means['native'][alg_key]
                minikube_ratio = None
                gcp_ratio = None
                
                if alg_key in env_means.get('minikube', {}):
                    minikube_lat = env_means['minikube'][alg_key]
                    minikube_ratio = minikube_lat / native_lat
                
                if alg_key in env_means.get('gcp', {}):
                    gcp_lat = env_means['gcp'][alg_key]
                    gcp_ratio = gcp_lat / native_lat
                
                minikube_str = f"{minikube_ratio:.2f}" if minikube_ratio else "N/A"
                gcp_str = f"{gcp_ratio:.2f}" if gcp_ratio else "N/A"
                print(f"{alg_name:20s} | {minikube_str:15s} | {gcp_str}")
        print()


if __name__ == '__main__':
    main()
