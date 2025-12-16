#!/usr/bin/env python3
"""
Verify payload size and workload rate impact analysis from summary files.
"""

import json
from collections import defaultdict
from pathlib import Path


def load_summary(filepath: Path) -> dict:
    """Load a summary.json file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_payload_size_impact(results_dir: Path) -> dict:
    """Analyze payload size impact on latency."""
    # Group by algorithm and payload size
    data = defaultdict(lambda: defaultdict(list))
    
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
        
        # Extract payload size
        payload = None
        for part in exp_id.split('_'):
            if part.startswith('p') and part[1:].isdigit():
                payload = int(part[1:])
                break
        
        if not payload:
            continue
        
        # Get p95 latency
        p95_latency = summary.get('latency', {}).get('p95')
        if p95_latency:
            data[algorithm][payload].append(p95_latency)
    
    # Calculate statistics
    results = {}
    for algorithm, payloads in data.items():
        results[algorithm] = {}
        for payload, latencies in sorted(payloads.items()):
            if latencies:
                results[algorithm][payload] = {
                    'mean': sum(latencies) / len(latencies),
                    'min': min(latencies),
                    'max': max(latencies),
                    'count': len(latencies)
                }
    
    return results


def analyze_workload_rate_impact(results_dir: Path) -> dict:
    """Analyze workload rate impact on throughput."""
    # Group by algorithm and rate
    data = defaultdict(lambda: defaultdict(list))
    
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
        
        # Extract rate
        rate = None
        for part in exp_id.split('_'):
            if part.startswith('r') and part[1:].isdigit():
                rate = int(part[1:])
                break
        
        if not rate:
            continue
        
        # Get throughput
        throughput = summary.get('throughput', {}).get('mean_msgs_per_sec')
        if throughput:
            data[algorithm][rate].append(throughput)
    
    # Calculate statistics
    results = {}
    for algorithm, rates in data.items():
        results[algorithm] = {}
        for rate, throughputs in sorted(rates.items()):
            if throughputs:
                results[algorithm][rate] = {
                    'mean': sum(throughputs) / len(throughputs),
                    'min': min(throughputs),
                    'max': max(throughputs),
                    'count': len(throughputs)
                }
    
    return results


def main():
    results_dir = Path('/home/ausmarton/scratchpad/quantum-resilient/results')
    
    print("=" * 80)
    print("Payload Size Impact Analysis")
    print("=" * 80)
    print()
    
    payload_results = analyze_payload_size_impact(results_dir)
    
    # Focus on algorithms mentioned in dissertation
    key_algorithms = {
        'kyber512': 'Kyber-512',
        'dilithium2': 'Dilithium-2',
        'ecdsa': 'ECDSA P-256',
        'rsa2048': 'RSA-2048'
    }
    
    for alg_key, alg_name in key_algorithms.items():
        if alg_key in payload_results:
            print(f"{alg_name}:")
            payloads = sorted(payload_results[alg_key].items())
            if len(payloads) >= 2:
                first = payloads[0][1]['mean']
                last = payloads[-1][1]['mean']
                increase_pct = ((last - first) / first) * 100
                print(f"  {payloads[0][0]}B: {first:.1f}μs")
                print(f"  {payloads[-1][0]}B: {last:.1f}μs")
                print(f"  Increase: {increase_pct:.1f}%")
            print()
    
    print("=" * 80)
    print("Workload Rate Impact Analysis")
    print("=" * 80)
    print()
    
    workload_results = analyze_workload_rate_impact(results_dir)
    
    for alg_key, alg_name in key_algorithms.items():
        if alg_key in workload_results:
            print(f"{alg_name}:")
            rates = sorted(workload_results[alg_key].items())
            for rate, stats in rates:
                print(f"  {rate} msg/s: {stats['mean']:.0f} msg/s achieved (mean)")
            print()


if __name__ == '__main__':
    main()
