#!/usr/bin/env python3
"""
Analyze test type coverage (patterns, durations, rates) to identify gaps.

This script checks if we have complete coverage of test type combinations
needed for dissertation claims.
"""

import yaml
from collections import defaultdict
from pathlib import Path

def load_matrix(matrix_path):
    """Load experiment matrix YAML."""
    with open(matrix_path) as f:
        return yaml.safe_load(f)

def analyze_test_type_coverage(matrix):
    """Analyze what test type combinations are covered."""
    experiments = matrix.get('experiments', [])
    
    # Track coverage by test type dimensions
    coverage = {
        'patterns': defaultdict(set),  # pattern -> set of (algo, payload, rate, duration)
        'durations': defaultdict(set),  # duration -> set of (algo, pattern, payload, rate)
        'rates': defaultdict(set),       # rate -> set of (algo, pattern, payload, duration)
        'combinations': set(),           # (pattern, duration, rate) combinations
    }
    
    # Track what exists
    test_types = {
        'constant_30s': defaultdict(set),      # (algo, payload, rate)
        'constant_300s': defaultdict(set),    # (algo, payload, rate)
        'burst_30s': defaultdict(set),         # (algo, payload, rate)
        'burst_300s': defaultdict(set),        # (algo, payload, rate)
        '10k_constant': defaultdict(set),      # (algo, payload)
        '10k_burst': defaultdict(set),         # (algo, payload)
    }
    
    for exp in experiments:
        algo = exp['algorithm']
        payloads = exp.get('payload_sizes', [])
        rates = exp.get('rates', [])
        pattern = exp.get('workload_pattern', 'constant')
        duration = exp.get('duration_sec', 30)
        
        for payload in payloads:
            for rate in rates:
                key = (algo, payload, rate)
                combo_key = (pattern, duration, rate)
                
                coverage['patterns'][pattern].add((algo, payload, rate, duration))
                coverage['durations'][duration].add((algo, pattern, payload, rate))
                coverage['rates'][rate].add((algo, pattern, payload, duration))
                coverage['combinations'].add(combo_key)
                
                # Categorize by test type
                if rate == 10000:
                    if pattern == 'burst':
                        test_types['10k_burst'][algo].add(payload)
                    else:
                        test_types['10k_constant'][algo].add(payload)
                elif duration == 300:
                    if pattern == 'burst':
                        test_types['burst_300s'][algo].add((payload, rate))
                    else:
                        test_types['constant_300s'][algo].add((payload, rate))
                elif pattern == 'burst':
                    test_types['burst_30s'][algo].add((payload, rate))
                else:
                    test_types['constant_30s'][algo].add((payload, rate))
    
    return coverage, test_types

def identify_test_type_gaps(matrix):
    """Identify missing test type combinations."""
    experiments = matrix.get('experiments', [])
    all_algorithms = ['rsa2048', 'ecdsa_p256', 'kyber512', 'dilithium2', 'hybrid_kyber_dilithium']
    
    # Define what combinations should exist
    expected_combinations = {
        'constant_30s': {
            'payloads': [256, 1024, 4096, 16384],
            'rates': [100, 500, 2000],
        },
        'constant_300s': {
            'payloads': [1024],  # Sustained load
            'rates': [2000],
        },
        'burst_30s': {
            'payloads': [1024, 4096],
            'rates': [2000],
        },
        'burst_300s': {
            'payloads': [],  # Not currently tested
            'rates': [],
        },
        '10k_constant': {
            'payloads': [256, 1024, 4096, 16384],
            'rates': [10000],
        },
        '10k_burst': {
            'payloads': [],  # Not currently tested
            'rates': [],
        },
    }
    
    # Track what exists
    existing = {
        'constant_30s': defaultdict(set),
        'constant_300s': defaultdict(set),
        'burst_30s': defaultdict(set),
        'burst_300s': defaultdict(set),
        '10k_constant': defaultdict(set),
        '10k_burst': defaultdict(set),
    }
    
    for exp in experiments:
        algo = exp['algorithm']
        payloads = exp.get('payload_sizes', [])
        rates = exp.get('rates', [])
        pattern = exp.get('workload_pattern', 'constant')
        duration = exp.get('duration_sec', 30)
        
        for payload in payloads:
            for rate in rates:
                if rate == 10000:
                    if pattern == 'burst':
                        existing['10k_burst'][algo].add(payload)
                    else:
                        existing['10k_constant'][algo].add(payload)
                elif duration == 300:
                    if pattern == 'burst':
                        existing['burst_300s'][algo].add((payload, rate))
                    else:
                        existing['constant_300s'][algo].add((payload, rate))
                elif pattern == 'burst':
                    existing['burst_30s'][algo].add((payload, rate))
                else:
                    existing['constant_30s'][algo].add((payload, rate))
    
    # Identify gaps
    gaps = {
        'burst_300s': [],  # Burst pattern with 5-minute duration
        '10k_burst': [],   # 10K msg/s with burst pattern
        'burst_more_rates': [],  # Burst with other rates (100, 500)
        'burst_more_payloads': [],  # Burst with other payloads (256, 16384)
        'sustained_more_rates': [],  # Sustained load with other rates
        'sustained_more_payloads': [],  # Sustained load with other payloads
    }
    
    # Check for burst + 5-minute combination
    for algo in all_algorithms:
        if not existing['burst_300s'][algo]:
            gaps['burst_300s'].append(algo)
    
    # Check for 10K + burst combination
    for algo in all_algorithms:
        if algo in existing['10k_constant'] and algo not in existing['10k_burst']:
            gaps['10k_burst'].append(algo)
    
    # Check for burst with other rates (currently only 2000)
    for algo in all_algorithms:
        if algo in existing['burst_30s']:
            burst_combos = existing['burst_30s'][algo]
            rates_used = {rate for _, rate in burst_combos}
            if 100 not in rates_used:
                gaps['burst_more_rates'].append((algo, 100))
            if 500 not in rates_used:
                gaps['burst_more_rates'].append((algo, 500))
    
    # Check for burst with other payloads (currently only 1024, 4096)
    for algo in all_algorithms:
        if algo in existing['burst_30s']:
            burst_combos = existing['burst_30s'][algo]
            payloads_used = {payload for payload, _ in burst_combos}
            if 256 not in payloads_used:
                gaps['burst_more_payloads'].append((algo, 256))
            if 16384 not in payloads_used:
                gaps['burst_more_payloads'].append((algo, 16384))
    
    # Check for sustained load with other rates (currently only 2000)
    for algo in all_algorithms:
        if algo in existing['constant_300s']:
            sustained_combos = existing['constant_300s'][algo]
            rates_used = {rate for _, rate in sustained_combos}
            if 100 not in rates_used:
                gaps['sustained_more_rates'].append((algo, 100))
            if 500 not in rates_used:
                gaps['sustained_more_rates'].append((algo, 500))
            if 10000 not in rates_used:
                gaps['sustained_more_rates'].append((algo, 10000))
    
    # Check for sustained load with other payloads (currently only 1024)
    for algo in all_algorithms:
        if algo in existing['constant_300s']:
            sustained_combos = existing['constant_300s'][algo]
            payloads_used = {payload for payload, _ in sustained_combos}
            if 256 not in payloads_used:
                gaps['sustained_more_payloads'].append((algo, 256))
            if 4096 not in payloads_used:
                gaps['sustained_more_payloads'].append((algo, 4096))
            if 16384 not in payloads_used:
                gaps['sustained_more_payloads'].append((algo, 16384))
    
    return gaps, existing

def print_test_type_analysis(matrix_path):
    """Print comprehensive test type coverage analysis."""
    matrix = load_matrix(matrix_path)
    coverage, test_types = analyze_test_type_coverage(matrix)
    gaps, existing = identify_test_type_gaps(matrix)
    
    print("=" * 80)
    print("TEST TYPE COVERAGE ANALYSIS")
    print("=" * 80)
    
    print("\n1. CURRENT TEST TYPE COVERAGE")
    print("-" * 80)
    
    print("\n✅ Constant Pattern, 30s Duration:")
    print(f"   Algorithms: {len(test_types['constant_30s'])}")
    for algo, combos in sorted(test_types['constant_30s'].items()):
        payloads = sorted(set(p for _, p, _ in [c + (None,) for c in combos] if p))
        rates = sorted(set(r for _, _, r in [c + (None,) for c in combos] if r))
        print(f"     {algo}: {len(combos)} combinations")
    
    print("\n✅ Constant Pattern, 300s Duration (Sustained Load):")
    print(f"   Algorithms: {len(test_types['constant_300s'])}")
    for algo, combos in sorted(test_types['constant_300s'].items()):
        payloads = sorted(set(p for p, _ in combos))
        rates = sorted(set(r for _, r in combos))
        print(f"     {algo}: payloads={payloads}, rates={rates}")
    
    print("\n✅ Burst Pattern, 30s Duration:")
    print(f"   Algorithms: {len(test_types['burst_30s'])}")
    for algo, combos in sorted(test_types['burst_30s'].items()):
        payloads = sorted(set(p for p, _ in combos))
        rates = sorted(set(r for _, r in combos))
        print(f"     {algo}: payloads={payloads}, rates={rates}")
    
    print("\n✅ 10K msg/s Rate, Constant Pattern:")
    print(f"   Algorithms: {len(test_types['10k_constant'])}")
    for algo, payloads in sorted(test_types['10k_constant'].items()):
        print(f"     {algo}: payloads={sorted(payloads)}")
    
    print("\n2. IDENTIFIED GAPS")
    print("-" * 80)
    
    if gaps['burst_300s']:
        print(f"\n❌ MISSING: Burst Pattern + 5-minute Duration")
        print(f"   Missing for {len(gaps['burst_300s'])} algorithms:")
        for algo in gaps['burst_300s']:
            print(f"     - {algo}: burst pattern @ 5-minute duration")
        print(f"\n   Impact: Cannot test burst pattern behavior under sustained load")
        print(f"   Priority: MEDIUM (burst + sustained load combination)")
        print(f"   Rationale: Enterprise workloads may have sustained burst periods")
    else:
        print("\n✅ Burst + 5-minute: Not currently tested (may be intentional)")
    
    if gaps['10k_burst']:
        print(f"\n❌ MISSING: 10K msg/s Rate + Burst Pattern")
        print(f"   Missing for {len(gaps['10k_burst'])} algorithms:")
        for algo in gaps['10k_burst']:
            print(f"     - {algo}: 10K msg/s @ burst pattern")
        print(f"\n   Impact: Cannot test high-rate burst behavior")
        print(f"   Priority: MEDIUM (high-rate burst is realistic enterprise scenario)")
        print(f"   Rationale: Enterprise systems may have bursts at high rates")
    else:
        print("\n✅ 10K + Burst: Not currently tested (may be intentional)")
    
    if gaps['burst_more_rates']:
        print(f"\n⚠️  INCOMPLETE: Burst Pattern Coverage (Rates)")
        print(f"   Burst currently only tested at 2000 msg/s")
        print(f"   Missing {len(gaps['burst_more_rates'])} combinations:")
        for algo, rate in gaps['burst_more_rates'][:10]:  # Limit output
            print(f"     - {algo}: burst @ {rate} msg/s")
        if len(gaps['burst_more_rates']) > 10:
            print(f"     ... and {len(gaps['burst_more_rates']) - 10} more")
        print(f"\n   Impact: Limited burst pattern analysis at different rates")
        print(f"   Priority: LOW (current coverage may be sufficient)")
        print(f"   Rationale: Burst at high rate (2000) is most representative")
    else:
        print("\n✅ Burst rates: Complete coverage")
    
    if gaps['burst_more_payloads']:
        print(f"\n⚠️  INCOMPLETE: Burst Pattern Coverage (Payloads)")
        print(f"   Burst currently only tested at 1024B and 4096B")
        print(f"   Missing {len(gaps['burst_more_payloads'])} combinations:")
        for algo, payload in gaps['burst_more_payloads'][:10]:
            print(f"     - {algo}: burst @ {payload}B")
        if len(gaps['burst_more_payloads']) > 10:
            print(f"     ... and {len(gaps['burst_more_payloads']) - 10} more")
        print(f"\n   Impact: Limited burst pattern analysis at different payloads")
        print(f"   Priority: LOW (current coverage may be sufficient)")
        print(f"   Rationale: Burst at medium payloads (1KB, 4KB) is most representative")
    else:
        print("\n✅ Burst payloads: Complete coverage")
    
    if gaps['sustained_more_rates']:
        print(f"\n⚠️  INCOMPLETE: Sustained Load Coverage (Rates)")
        print(f"   Sustained load currently only tested at 2000 msg/s")
        print(f"   Missing {len(gaps['sustained_more_rates'])} combinations:")
        for algo, rate in gaps['sustained_more_rates'][:10]:
            print(f"     - {algo}: sustained load @ {rate} msg/s")
        if len(gaps['sustained_more_rates']) > 10:
            print(f"     ... and {len(gaps['sustained_more_rates']) - 10} more")
        print(f"\n   Impact: Limited sustained load analysis at different rates")
        print(f"   Priority: LOW (current coverage may be sufficient)")
        print(f"   Rationale: Sustained load at high rate (2000) is most representative")
    else:
        print("\n✅ Sustained rates: Complete coverage")
    
    if gaps['sustained_more_payloads']:
        print(f"\n⚠️  INCOMPLETE: Sustained Load Coverage (Payloads)")
        print(f"   Sustained load currently only tested at 1024B")
        print(f"   Missing {len(gaps['sustained_more_payloads'])} combinations:")
        for algo, payload in gaps['sustained_more_payloads'][:10]:
            print(f"     - {algo}: sustained load @ {payload}B")
        if len(gaps['sustained_more_payloads']) > 10:
            print(f"     ... and {len(gaps['sustained_more_payloads']) - 10} more")
        print(f"\n   Impact: Limited sustained load analysis at different payloads")
        print(f"   Priority: LOW (current coverage may be sufficient)")
        print(f"   Rationale: Sustained load at medium payload (1KB) is most representative")
    else:
        print("\n✅ Sustained payloads: Complete coverage")
    
    print("\n3. TEST TYPE COMBINATION MATRIX")
    print("-" * 80)
    print("\nCurrent coverage:")
    print("  Pattern × Duration × Rate combinations:")
    for combo in sorted(coverage['combinations']):
        pattern, duration, rate = combo
        print(f"    {pattern:8s} × {duration:4d}s × {rate:5d} msg/s")
    
    print("\n4. RECOMMENDATIONS")
    print("-" * 80)
    print("MEDIUM Priority (Consider adding):")
    print("  1. Burst + 5-minute duration")
    print("     - Tests burst pattern under sustained load")
    print("     - Realistic for enterprise workloads with sustained burst periods")
    print("     - Would add: 5 algorithms × 2 payloads × 1 rate × 3 runs = 30 scenarios")
    print()
    print("  2. 10K msg/s + Burst pattern")
    print("     - Tests high-rate burst behavior")
    print("     - Realistic for enterprise systems with high-rate bursts")
    print("     - Would add: 5 algorithms × 2 payloads × 1 rate × 5 runs = 50 scenarios")
    print()
    print("LOW Priority (Optional):")
    print("  3. Burst at other rates (100, 500 msg/s)")
    print("     - Current coverage (2000 msg/s) may be sufficient")
    print("     - Would add: 5 algorithms × 2 payloads × 2 rates × 5 runs = 100 scenarios")
    print()
    print("  4. Burst at other payloads (256B, 16KB)")
    print("     - Current coverage (1KB, 4KB) may be sufficient")
    print("     - Would add: 5 algorithms × 2 payloads × 1 rate × 5 runs = 50 scenarios")
    print()
    print("  5. Sustained load at other rates/payloads")
    print("     - Current coverage (1KB @ 2000 msg/s) may be sufficient")
    print("     - Would significantly increase experiment time (5 minutes per run)")
    
    print("\n" + "=" * 80)
    
    return gaps

if __name__ == "__main__":
    matrix_path = Path(__file__).parent / "experiment_matrix.yaml"
    gaps = print_test_type_analysis(matrix_path)
    
    # Exit with error code if medium/high priority gaps found
    if gaps['burst_300s'] or gaps['10k_burst']:
        exit(1)
    exit(0)

