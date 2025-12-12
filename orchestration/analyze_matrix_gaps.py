#!/usr/bin/env python3
"""
Analyze experiment matrix to identify missing combinations.

This script checks if we have complete coverage for all dissertation claims
by comparing what's defined vs what's needed.
"""

import yaml
from collections import defaultdict
from pathlib import Path

def load_matrix(matrix_path):
    """Load experiment matrix YAML."""
    with open(matrix_path) as f:
        return yaml.safe_load(f)

def analyze_coverage(matrix):
    """Analyze what combinations are covered."""
    experiments = matrix.get('experiments', [])
    
    # Track coverage by dimension
    coverage = {
        'algorithms': set(),
        'payloads': set(),
        'rates': set(),
        'patterns': set(),
        'durations': set(),
        'scaling': set(),
    }
    
    # Track combinations
    combinations = defaultdict(set)
    
    for exp in experiments:
        algo = exp['algorithm']
        coverage['algorithms'].add(algo)
        
        payloads = exp.get('payload_sizes', [])
        rates = exp.get('rates', [])
        pattern = exp.get('workload_pattern', 'constant')
        duration = exp.get('duration_sec', 30)
        is_scaling = exp.get('scaling_experiment', False)
        
        coverage['patterns'].add(pattern)
        coverage['durations'].add(duration)
        
        if is_scaling:
            coverage['scaling'].add(algo)
        
        for payload in payloads:
            coverage['payloads'].add(payload)
            for rate in rates:
                coverage['rates'].add(rate)
                key = (algo, payload, rate, pattern, duration, is_scaling)
                combinations[key] = True
    
    return coverage, combinations

def identify_gaps(matrix):
    """Identify missing combinations needed for dissertation claims."""
    experiments = matrix.get('experiments', [])
    
    # All algorithms
    all_algorithms = ['rsa2048', 'ecdsa_p256', 'kyber512', 'dilithium2', 'hybrid_kyber_dilithium']
    
    # Track what exists
    existing = {
        'baseline': set(),  # (algo, payload, rate)
        'burst': set(),     # (algo, payload, rate)
        '10k': set(),       # (algo, payload)
        'sustained': set(), # (algo, payload, rate)
        'scaling': set(),   # (algo,)
    }
    
    for exp in experiments:
        algo = exp['algorithm']
        payloads = exp.get('payload_sizes', [])
        rates = exp.get('rates', [])
        pattern = exp.get('workload_pattern', 'constant')
        duration = exp.get('duration_sec', 30)
        is_scaling = exp.get('scaling_experiment', False)
        
        if is_scaling:
            existing['scaling'].add(algo)
        elif pattern == 'burst':
            for payload in payloads:
                for rate in rates:
                    existing['burst'].add((algo, payload, rate))
        elif 10000 in rates:
            for payload in payloads:
                existing['10k'].add((algo, payload))
        elif duration == 300:
            for payload in payloads:
                for rate in rates:
                    existing['sustained'].add((algo, payload, rate))
        else:
            for payload in payloads:
                for rate in rates:
                    existing['baseline'].add((algo, payload, rate))
    
    # Identify gaps
    gaps = {
        'sustained_missing': [],
        'scaling_missing': [],
        'burst_incomplete': [],
    }
    
    # Check sustained load coverage
    sustained_payloads = [1024]
    sustained_rates = [2000]
    for algo in all_algorithms:
        for payload in sustained_payloads:
            for rate in sustained_rates:
                if (algo, payload, rate) not in existing['sustained']:
                    gaps['sustained_missing'].append((algo, payload, rate))
    
    # Check scaling coverage
    for algo in all_algorithms:
        if algo not in existing['scaling']:
            gaps['scaling_missing'].append(algo)
    
    # Check burst coverage completeness
    # Currently burst only has 2 payloads (1024, 4096) and 1 rate (2000)
    # This might be intentional, but let's note it
    burst_payloads = [1024, 4096]
    burst_rates = [2000]
    for algo in all_algorithms:
        for payload in burst_payloads:
            for rate in burst_rates:
                if (algo, payload, rate) not in existing['burst']:
                    gaps['burst_incomplete'].append((algo, payload, rate))
    
    return gaps, existing

def print_analysis(matrix_path):
    """Print comprehensive gap analysis."""
    matrix = load_matrix(matrix_path)
    coverage, combinations = analyze_coverage(matrix)
    gaps, existing = identify_gaps(matrix)
    
    print("=" * 80)
    print("EXPERIMENT MATRIX COVERAGE ANALYSIS")
    print("=" * 80)
    
    print("\n1. ALGORITHM COVERAGE")
    print("-" * 80)
    all_algorithms = ['rsa2048', 'ecdsa_p256', 'kyber512', 'dilithium2', 'hybrid_kyber_dilithium']
    print(f"All algorithms: {', '.join(all_algorithms)}")
    print(f"Covered algorithms: {', '.join(sorted(coverage['algorithms']))}")
    
    print("\n2. EXPERIMENT TYPE COVERAGE")
    print("-" * 80)
    print(f"Baseline (constant, 30s): {len(existing['baseline'])} combinations")
    print(f"Burst pattern: {len(existing['burst'])} combinations")
    print(f"10K msg/s rate: {len(existing['10k'])} combinations")
    print(f"Sustained load (5-min): {len(existing['sustained'])} combinations")
    print(f"Scaling experiments: {len(existing['scaling'])} algorithms")
    
    print("\n3. IDENTIFIED GAPS")
    print("-" * 80)
    
    if gaps['sustained_missing']:
        print(f"\n❌ MISSING: Sustained Load (5-minute) Experiments")
        print(f"   Missing {len(gaps['sustained_missing'])} combinations:")
        for algo, payload, rate in gaps['sustained_missing']:
            print(f"     - {algo}: {payload}B @ {rate} msg/s (5-min duration)")
        print(f"\n   Impact: Cannot compare sustained load behavior across all algorithms")
        print(f"   Required for claims:")
        print(f"     - 'Algorithm X handles sustained load better than baseline Z'")
        print(f"     - 'Sustained load performance comparison'")
    else:
        print("\n✅ Sustained load: Complete coverage")
    
    if gaps['scaling_missing']:
        print(f"\n⚠️  MISSING: Scaling Experiments")
        print(f"   Missing {len(gaps['scaling_missing'])} algorithms:")
        for algo in gaps['scaling_missing']:
            print(f"     - {algo}")
        print(f"\n   Note: Scaling is primarily for PQC deployment analysis")
        print(f"   Impact: Cannot compare scaling behavior for classical algorithms")
        print(f"   Priority: MEDIUM (less critical than sustained load)")
    else:
        print("\n✅ Scaling: Complete coverage")
    
    if gaps['burst_incomplete']:
        print(f"\n⚠️  INCOMPLETE: Burst Pattern Coverage")
        print(f"   Missing {len(gaps['burst_incomplete'])} combinations:")
        for algo, payload, rate in gaps['burst_incomplete']:
            print(f"     - {algo}: {payload}B @ {rate} msg/s (burst)")
        print(f"\n   Note: Current burst coverage may be intentional (2 payloads, 1 rate)")
        print(f"   Impact: Limited burst pattern analysis")
        print(f"   Priority: LOW (current coverage may be sufficient)")
    else:
        print("\n✅ Burst pattern: Complete coverage")
    
    print("\n4. RECOMMENDATIONS")
    print("-" * 80)
    print("CRITICAL (Must add):")
    print("  1. Add sustained load (5-minute) experiments for RSA-2048 and ECDSA P-256")
    print("     - This enables fair comparison across all algorithms")
    print("     - Required for dissertation claims about sustained load behavior")
    print()
    print("MEDIUM (Consider adding):")
    print("  2. Consider adding scaling experiments for RSA-2048 and ECDSA P-256")
    print("     - Less critical since scaling is primarily for PQC deployment")
    print("     - But would enable complete scaling comparison")
    print()
    print("LOW (Optional):")
    print("  3. Consider expanding burst pattern coverage")
    print("     - Current coverage (2 payloads, 1 rate) may be sufficient")
    print("     - Could add more payload sizes or rates if needed")
    
    print("\n" + "=" * 80)
    
    return gaps

if __name__ == "__main__":
    matrix_path = Path(__file__).parent / "experiment_matrix.yaml"
    gaps = print_analysis(matrix_path)
    
    # Exit with error code if critical gaps found
    if gaps['sustained_missing']:
        exit(1)
    exit(0)

