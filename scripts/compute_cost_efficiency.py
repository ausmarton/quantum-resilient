#!/usr/bin/env python3
"""
Compute cost efficiency metrics for GCP deployments (FR13).

Calculates:
- Operations per dollar (ops/$)
- Latency per dollar (μs/$)
- Cost efficiency comparison across algorithms

Usage:
    python scripts/compute_cost_efficiency.py \
        --stats final-results/aggregated_stats.json \
        --output final-results/cost_efficiency.json
"""

import argparse
import json
from pathlib import Path
from typing import Optional

# GCP pricing estimates (as of 2025, approximate)
# These should be updated with actual GCP pricing if available
GCP_PRICING = {
    'compute_hourly': 0.10,  # $0.10 per vCPU-hour (approximate)
    'storage_gb_monthly': 0.02,  # $0.02 per GB-month
    'network_gb': 0.12,  # $0.12 per GB egress
}


def estimate_gcp_cost(experiment_duration_seconds: float, vcpus: int = 1) -> float:
    """Estimate GCP cost for an experiment.
    
    Args:
        experiment_duration_seconds: Duration of experiment
        vcpus: Number of vCPUs used
    
    Returns:
        Estimated cost in USD
    """
    hours = experiment_duration_seconds / 3600.0
    compute_cost = hours * vcpus * GCP_PRICING['compute_hourly']
    # Add small overhead for storage/network (10% of compute)
    total_cost = compute_cost * 1.1
    return total_cost


def compute_cost_efficiency_metrics(stats: dict) -> list[dict]:
    """Compute cost efficiency metrics for GCP experiments."""
    gcp_experiments = [a for a in stats.get('aggregated', []) 
                      if a.get('environment') == 'gcp']
    
    metrics = []
    
    for exp in gcp_experiments:
        algorithm = exp.get('algorithm')
        payload_size = exp.get('payload_size', 0)
        rate = exp.get('rate', 0)
        
        # Get throughput and latency
        throughput = exp.get('throughput', {}).get('mean', 0)  # ops/sec
        p95_latency = exp.get('p95', {}).get('mean', 0)  # microseconds
        
        # Estimate experiment duration (assume 5 minutes = 300 seconds for baseline)
        # This is approximate - actual duration should come from experiment metadata
        duration_seconds = 300.0
        
        # Estimate cost
        estimated_cost = estimate_gcp_cost(duration_seconds, vcpus=1)
        
        # Calculate metrics
        total_operations = throughput * duration_seconds
        ops_per_dollar = total_operations / estimated_cost if estimated_cost > 0 else 0
        latency_per_dollar = p95_latency / estimated_cost if estimated_cost > 0 else 0
        
        metrics.append({
            'algorithm': algorithm,
            'payload_size': payload_size,
            'rate': rate,
            'throughput_ops_per_sec': throughput,
            'p95_latency_us': p95_latency,
            'estimated_cost_usd': round(estimated_cost, 4),
            'total_operations': int(total_operations),
            'ops_per_dollar': round(ops_per_dollar, 2),
            'latency_per_dollar_us': round(latency_per_dollar, 2),
            'cost_per_million_ops': round(estimated_cost / (total_operations / 1_000_000), 4) if total_operations > 0 else 0,
        })
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compute cost efficiency metrics")
    parser.add_argument('--stats', type=Path, required=True,
                       help='Path to aggregated_stats.json')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output path for cost efficiency JSON')
    
    args = parser.parse_args()
    
    # Load stats
    with open(args.stats) as f:
        stats = json.load(f)
    
    # Compute metrics
    metrics = compute_cost_efficiency_metrics(stats)
    
    # Write output
    output_data = {
        'generated_at': stats.get('generated_at', ''),
        'pricing_notes': 'GCP pricing estimates - update with actual pricing if available',
        'metrics': metrics,
        'summary': {
            'total_configurations': len(metrics),
            'algorithms': list(set(m['algorithm'] for m in metrics)),
        }
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Cost efficiency metrics computed: {len(metrics)} configurations")
    print(f"Written: {args.output}")


if __name__ == '__main__':
    main()
