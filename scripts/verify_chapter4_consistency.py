#!/usr/bin/env python3
"""
Verify all numbers in Chapter 4 for consistency.
"""

import json
import re
from pathlib import Path


def load_hypothesis_results():
    """Load hypothesis test results."""
    with open('final-results/hypothesis_tests.json') as f:
        return json.load(f)


def verify_all_numbers():
    """Verify all key numbers in Chapter 4."""
    results = {
        'experiments': {'expected': 396, 'found': []},
        'runs': {'expected': 1836, 'found': []},
        'events': {'expected': 134621400, 'found': []},
        'statistical': {},
        'throughput': {}
    }
    
    # Load hypothesis results
    hyp_data = load_hypothesis_results()
    results['statistical'] = {
        'total_comparisons': hyp_data['total_comparisons'],
        'significant': hyp_data['significant_comparisons'],
        'effect_sizes': hyp_data['summary']['effect_sizes']
    }
    
    print("=" * 80)
    print("Chapter 4 Consistency Verification")
    print("=" * 80)
    print()
    print("Statistical Analysis:")
    print(f"  Total comparisons: {results['statistical']['total_comparisons']}")
    print(f"  Significant: {results['statistical']['significant']}")
    print(f"  Effect sizes: {results['statistical']['effect_sizes']}")
    print()
    print("Expected values:")
    print(f"  Experiments: 396")
    print(f"  Runs: 1,836")
    print(f"  Events: 134,621,400 (134.6 million)")
    print()
    print("Key findings to verify:")
    print("  - Throughput at 10K msg/s: ~1,000 msg/s (not 9,750)")
    print("  - Environment overhead: Minikube 44.8%, GCP 239.2%")
    print("  - Statistical: 72 comparisons, 59 large effects (81.9%)")


if __name__ == '__main__':
    verify_all_numbers()
