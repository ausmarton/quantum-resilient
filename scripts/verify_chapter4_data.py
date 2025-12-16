#!/usr/bin/env python3
"""
Verify Chapter 4 data counts and calculate metrics.

This script verifies all data claims in Chapter 4 and calculates
missing metrics like ECDHE P-256 performance.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# import pandas as pd  # Not needed for this script


def load_summary(filepath: Path) -> Optional[dict]:
    """Load a summary.json file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def verify_data_counts(results_dir: Path) -> dict:
    """Verify experiment, run, and event counts."""
    results = {
        'experiments': {'total': 0, 'by_env': defaultdict(int)},
        'runs': {'total': 0, 'by_env': defaultdict(int)},
        'events': {'total': 0, 'by_env': defaultdict(int)},
        'algorithms': set(),
        'environments': set(),
    }
    
    # Find all summary files
    for summary_file in results_dir.rglob('summary.json'):
        summary = load_summary(summary_file)
        if not summary:
            continue
        
        # Extract environment from path
        parts = summary_file.parts
        env_idx = parts.index('results') if 'results' in parts else -1
        if env_idx >= 0 and env_idx + 1 < len(parts):
            env = parts[env_idx + 1]
            results['environments'].add(env)
            results['experiments']['by_env'][env] += 1
            results['experiments']['total'] += 1
        
        # Count events
        total_events = summary.get('total_events', 0)
        results['events']['total'] += total_events
        if env_idx >= 0 and env_idx + 1 < len(parts):
            results['events']['by_env'][env] += total_events
        
        # Extract algorithm from experiment_id
        exp_id = summary.get('experiment_id', '')
        # Try to extract algorithm from experiment ID
        if 'rsa2048' in exp_id.lower():
            results['algorithms'].add('RSA-2048')
        elif 'ecdsa' in exp_id.lower() or 'p256' in exp_id.lower():
            if 'ecdh' in exp_id.lower() or 'ecdhe' in exp_id.lower():
                results['algorithms'].add('ECDHE P-256')
            else:
                results['algorithms'].add('ECDSA P-256')
        elif 'kyber' in exp_id.lower():
            if 'hybrid' in exp_id.lower():
                results['algorithms'].add('Hybrid')
            else:
                results['algorithms'].add('Kyber-512')
        elif 'dilithium' in exp_id.lower():
            results['algorithms'].add('Dilithium-2')
    
    # Count runs (run.jsonl files)
    for run_file in results_dir.rglob('run.jsonl'):
        parts = run_file.parts
        env_idx = parts.index('results') if 'results' in parts else -1
        if env_idx >= 0 and env_idx + 1 < len(parts):
            env = parts[env_idx + 1]
            results['runs']['by_env'][env] += 1
            results['runs']['total'] += 1
    
    return results


def calculate_ecdhe_metrics(results_dir: Path) -> Optional[dict]:
    """Calculate ECDHE P-256 performance metrics from summary files."""
    ecdhe_summaries = []
    
    # Find all summaries with ECDHE
    for summary_file in results_dir.rglob('summary.json'):
        summary = load_summary(summary_file)
        if not summary:
            continue
        
        exp_id = summary.get('experiment_id', '').lower()
        # Check if this is an ECDHE experiment
        if 'ecdhe' in exp_id or ('ecdh' in exp_id and 'p256' in exp_id):
            # Also check environment - we want native for Table 4.1
            parts = summary_file.parts
            if 'native' in parts:
                ecdhe_summaries.append(summary)
    
    if not ecdhe_summaries:
        print("No ECDHE P-256 summaries found in native environment")
        return None
    
    # Extract p95 latencies
    p95_latencies = []
    for summary in ecdhe_summaries:
        latency = summary.get('latency', {})
        p95 = latency.get('p95')
        if p95 is not None:
            p95_latencies.append(p95)
    
    if not p95_latencies:
        print("No p95 latency data found for ECDHE P-256")
        return None
    
    # Calculate statistics (using basic Python, no numpy needed)
    mean_p95 = sum(p95_latencies) / len(p95_latencies)
    variance = sum((x - mean_p95) ** 2 for x in p95_latencies) / len(p95_latencies)
    std_p95 = variance ** 0.5
    
    return {
        'mean_p95_latency_us': mean_p95,
        'std_p95_latency_us': std_p95,
        'min_p95_latency_us': min(p95_latencies),
        'max_p95_latency_us': max(p95_latencies),
        'configurations': len(ecdhe_summaries),
        'range_us': f"{min(p95_latencies):.2f} - {max(p95_latencies):.2f}",
    }


def main():
    results_dir = Path('/home/ausmarton/scratchpad/quantum-resilient/results')
    
    print("=" * 80)
    print("Chapter 4 Data Verification")
    print("=" * 80)
    print()
    
    # Verify data counts
    print("1. Verifying Data Counts...")
    print("-" * 80)
    counts = verify_data_counts(results_dir)
    
    print(f"Total Experiments: {counts['experiments']['total']}")
    print("  By Environment:")
    for env, count in sorted(counts['experiments']['by_env'].items()):
        print(f"    {env}: {count}")
    
    print(f"\nTotal Runs: {counts['runs']['total']}")
    print("  By Environment:")
    for env, count in sorted(counts['runs']['by_env'].items()):
        print(f"    {env}: {count}")
    
    print(f"\nTotal Events: {counts['events']['total']:,}")
    print("  By Environment:")
    for env, count in sorted(counts['events']['by_env'].items()):
        print(f"    {env}: {count:,}")
    
    print(f"\nAlgorithms Found: {sorted(counts['algorithms'])}")
    print(f"Environments Found: {sorted(counts['environments'])}")
    
    print()
    print("=" * 80)
    print("2. Calculating ECDHE P-256 Metrics...")
    print("-" * 80)
    
    ecdhe_metrics = calculate_ecdhe_metrics(results_dir)
    if ecdhe_metrics:
        print("ECDHE P-256 Performance (Native Environment):")
        print(f"  Mean p95 Latency: {ecdhe_metrics['mean_p95_latency_us']:.2f} μs")
        print(f"  Standard Deviation: {ecdhe_metrics['std_p95_latency_us']:.2f} μs")
        print(f"  Range: {ecdhe_metrics['range_us']} μs")
        print(f"  Configurations: {ecdhe_metrics['configurations']}")
    else:
        print("  Could not calculate ECDHE metrics")
    
    print()
    print("=" * 80)
    print("Verification Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
