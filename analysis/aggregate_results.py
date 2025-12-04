#!/usr/bin/env python3
"""
Aggregate results across all experiments and compute summary statistics.

Computes:
- Mean/std/CI for p50/p95/p99 across runs
- Effect sizes (PQC vs classical, PQC vs PQC)
- Environment deltas (native vs minikube vs gcp)

Usage:
    python analysis/aggregate_results.py \
        --index final-results/index.json \
        --output final-results
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""
    scenario_id: str
    algorithm: str
    payload_size: int
    rate: int
    environment: str
    run_index: int = 0
    
    # Latency metrics (microseconds)
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    p999: float = 0.0
    mean_latency: float = 0.0
    std_latency: float = 0.0
    
    # Throughput
    mean_throughput: float = 0.0
    max_throughput: float = 0.0
    
    # Counts
    total_events: int = 0


@dataclass
class AggregatedStats:
    """Aggregated statistics across multiple runs."""
    algorithm: str
    payload_size: int
    rate: int
    environment: str
    n_runs: int = 0
    
    # Aggregated latency
    p50_mean: float = 0.0
    p50_std: float = 0.0
    p50_ci_low: float = 0.0
    p50_ci_high: float = 0.0
    
    p95_mean: float = 0.0
    p95_std: float = 0.0
    p95_ci_low: float = 0.0
    p95_ci_high: float = 0.0
    
    p99_mean: float = 0.0
    p99_std: float = 0.0
    p99_ci_low: float = 0.0
    p99_ci_high: float = 0.0
    
    # Aggregated throughput
    throughput_mean: float = 0.0
    throughput_std: float = 0.0
    throughput_ci_low: float = 0.0
    throughput_ci_high: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'algorithm': self.algorithm,
            'payload_size': self.payload_size,
            'rate': self.rate,
            'environment': self.environment,
            'n_runs': self.n_runs,
            'p50': {'mean': self.p50_mean, 'std': self.p50_std, 'ci_low': self.p50_ci_low, 'ci_high': self.p50_ci_high},
            'p95': {'mean': self.p95_mean, 'std': self.p95_std, 'ci_low': self.p95_ci_low, 'ci_high': self.p95_ci_high},
            'p99': {'mean': self.p99_mean, 'std': self.p99_std, 'ci_low': self.p99_ci_low, 'ci_high': self.p99_ci_high},
            'throughput': {'mean': self.throughput_mean, 'std': self.throughput_std, 'ci_low': self.throughput_ci_low, 'ci_high': self.throughput_ci_high},
        }


def load_summary(path: Path) -> Optional[dict]:
    """Load summary.json file."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def extract_result(entry: dict, summary: dict) -> ExperimentResult:
    """Extract experiment result from index entry and summary."""
    result = ExperimentResult(
        scenario_id=entry['scenario_id'],
        algorithm=entry['algorithm'],
        payload_size=entry['payload_size'],
        rate=entry['rate'],
        environment=entry['environment'],
    )
    
    # Extract run index from scenario ID
    if '_run' in entry['scenario_id']:
        try:
            result.run_index = int(entry['scenario_id'].split('_run')[1].split('_')[0])
        except (ValueError, IndexError):
            result.run_index = 0
    
    # Extract latency metrics
    if 'latency' in summary:
        lat = summary['latency']
        result.p50 = lat.get('p50', 0)
        result.p90 = lat.get('p90', 0)
        result.p95 = lat.get('p95', 0)
        result.p99 = lat.get('p99', 0)
        result.p999 = lat.get('p999', 0)
        result.mean_latency = lat.get('mean', 0)
        result.std_latency = lat.get('std', 0)
    
    # Extract throughput
    if 'throughput' in summary:
        tput = summary['throughput']
        result.mean_throughput = tput.get('mean_msgs_per_sec', 0)
        result.max_throughput = tput.get('max_msgs_per_sec', 0)
    
    result.total_events = summary.get('total_events', 0)
    
    return result


def compute_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Compute confidence interval using t-distribution."""
    if len(values) < 2:
        return (values[0] if values else 0, values[0] if values else 0)
    
    n = len(values)
    mean = np.mean(values)
    std_err = stats.sem(values)
    
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (mean - h, mean + h)


def aggregate_results(results: list[ExperimentResult]) -> AggregatedStats:
    """Aggregate results from multiple runs."""
    if not results:
        return AggregatedStats(algorithm='', payload_size=0, rate=0, environment='')
    
    first = results[0]
    agg = AggregatedStats(
        algorithm=first.algorithm,
        payload_size=first.payload_size,
        rate=first.rate,
        environment=first.environment,
        n_runs=len(results),
    )
    
    # Collect values
    p50_vals = [r.p50 for r in results if r.p50 > 0]
    p95_vals = [r.p95 for r in results if r.p95 > 0]
    p99_vals = [r.p99 for r in results if r.p99 > 0]
    tput_vals = [r.mean_throughput for r in results if r.mean_throughput > 0]
    
    # Compute aggregates for p50
    if p50_vals:
        agg.p50_mean = np.mean(p50_vals)
        agg.p50_std = np.std(p50_vals, ddof=1) if len(p50_vals) > 1 else 0
        agg.p50_ci_low, agg.p50_ci_high = compute_ci(p50_vals)
    
    # Compute aggregates for p95
    if p95_vals:
        agg.p95_mean = np.mean(p95_vals)
        agg.p95_std = np.std(p95_vals, ddof=1) if len(p95_vals) > 1 else 0
        agg.p95_ci_low, agg.p95_ci_high = compute_ci(p95_vals)
    
    # Compute aggregates for p99
    if p99_vals:
        agg.p99_mean = np.mean(p99_vals)
        agg.p99_std = np.std(p99_vals, ddof=1) if len(p99_vals) > 1 else 0
        agg.p99_ci_low, agg.p99_ci_high = compute_ci(p99_vals)
    
    # Compute aggregates for throughput
    if tput_vals:
        agg.throughput_mean = np.mean(tput_vals)
        agg.throughput_std = np.std(tput_vals, ddof=1) if len(tput_vals) > 1 else 0
        agg.throughput_ci_low, agg.throughput_ci_high = compute_ci(tput_vals)
    
    return agg


def compute_effect_size(group_a: list[float], group_b: list[float]) -> dict:
    """Compute Cohen's d effect size."""
    if not group_a or not group_b:
        return {'cohens_d': 0, 'interpretation': 'insufficient_data'}
    
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)
    
    # Pooled standard deviation
    n_a, n_b = len(group_a), len(group_b)
    var_a = np.var(group_a, ddof=1) if n_a > 1 else 0
    var_b = np.var(group_b, ddof=1) if n_b > 1 else 0
    
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    
    if pooled_std == 0:
        return {'cohens_d': 0, 'interpretation': 'no_variance'}
    
    d = (mean_b - mean_a) / pooled_std
    
    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = 'negligible'
    elif abs_d < 0.5:
        interpretation = 'small'
    elif abs_d < 0.8:
        interpretation = 'medium'
    else:
        interpretation = 'large'
    
    return {
        'cohens_d': round(d, 4),
        'interpretation': interpretation,
        'mean_a': round(mean_a, 2),
        'mean_b': round(mean_b, 2),
        'n_a': n_a,
        'n_b': n_b,
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument('--index', '-i', type=Path, required=True, help='Path to index.json')
    parser.add_argument('--output', '-o', type=Path, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Load index
    if not args.index.exists():
        print(f"Error: Index file not found: {args.index}", file=sys.stderr)
        sys.exit(1)
    
    with open(args.index) as f:
        index = json.load(f)
    
    print(f"Loaded index with {len(index.get('experiments', []))} experiments")
    
    # Load all results
    results: list[ExperimentResult] = []
    
    for entry in index.get('experiments', []):
        if entry.get('status') not in ['success', 'cached']:
            continue
        
        output_dir = Path(entry['output_dir'])
        summary_path = output_dir / 'stats' / 'summary.json'
        
        # Try alternative paths
        if not summary_path.exists():
            summary_path = output_dir / 'summary.json'
        
        summary = load_summary(summary_path)
        if summary:
            result = extract_result(entry, summary)
            results.append(result)
    
    print(f"Loaded {len(results)} valid results")
    
    if not results:
        print("Warning: No results found. Creating empty aggregation.")
        args.output.mkdir(parents=True, exist_ok=True)
        with open(args.output / 'aggregated_stats.json', 'w') as f:
            json.dump({'aggregated': [], 'effect_sizes': [], 'environment_deltas': []}, f, indent=2)
        return
    
    # Group results by (algorithm, payload, rate, environment)
    groups: dict[tuple, list[ExperimentResult]] = defaultdict(list)
    for r in results:
        key = (r.algorithm, r.payload_size, r.rate, r.environment)
        groups[key].append(r)
    
    # Aggregate each group
    aggregated: list[AggregatedStats] = []
    for key, group_results in groups.items():
        agg = aggregate_results(group_results)
        aggregated.append(agg)
    
    print(f"Computed {len(aggregated)} aggregated statistics")
    
    # Compute effect sizes for comparisons
    effect_sizes = []
    
    # Group by (algorithm, payload, rate) to compare environments
    by_config: dict[tuple, dict[str, list[ExperimentResult]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        config_key = (r.algorithm, r.payload_size, r.rate)
        by_config[config_key][r.environment].append(r)
    
    # Native vs Minikube vs GCP
    for config_key, env_results in by_config.items():
        algorithm, payload, rate = config_key
        
        native_p95 = [r.p95 for r in env_results.get('native', [])]
        minikube_p95 = [r.p95 for r in env_results.get('minikube', [])]
        gcp_p95 = [r.p95 for r in env_results.get('gcp', [])]
        
        if native_p95 and minikube_p95:
            effect = compute_effect_size(native_p95, minikube_p95)
            effect['comparison'] = 'native_vs_minikube'
            effect['algorithm'] = algorithm
            effect['payload_size'] = payload
            effect['rate'] = rate
            effect['metric'] = 'p95_latency_us'
            effect_sizes.append(effect)
        
        if native_p95 and gcp_p95:
            effect = compute_effect_size(native_p95, gcp_p95)
            effect['comparison'] = 'native_vs_gcp'
            effect['algorithm'] = algorithm
            effect['payload_size'] = payload
            effect['rate'] = rate
            effect['metric'] = 'p95_latency_us'
            effect_sizes.append(effect)
        
        if minikube_p95 and gcp_p95:
            effect = compute_effect_size(minikube_p95, gcp_p95)
            effect['comparison'] = 'minikube_vs_gcp'
            effect['algorithm'] = algorithm
            effect['payload_size'] = payload
            effect['rate'] = rate
            effect['metric'] = 'p95_latency_us'
            effect_sizes.append(effect)
    
    # PQC vs Classical comparisons
    classical_algos = ['rsa2048', 'ecdsa_p256']
    pqc_algos = ['kyber512', 'dilithium2', 'hybrid_kyber_dilithium']
    
    for env in ['native', 'minikube', 'gcp']:
        for payload in [256, 1024, 4096]:
            for rate in [100, 500, 2000]:
                # Collect classical results
                classical_p95 = []
                for algo in classical_algos:
                    key = (algo, payload, rate)
                    if key in by_config:
                        classical_p95.extend([r.p95 for r in by_config[key].get(env, [])])
                
                # Compare each PQC algo to classical
                for pqc_algo in pqc_algos:
                    key = (pqc_algo, payload, rate)
                    if key in by_config:
                        pqc_p95 = [r.p95 for r in by_config[key].get(env, [])]
                        if classical_p95 and pqc_p95:
                            effect = compute_effect_size(classical_p95, pqc_p95)
                            effect['comparison'] = f'classical_vs_{pqc_algo}'
                            effect['algorithm'] = pqc_algo
                            effect['payload_size'] = payload
                            effect['rate'] = rate
                            effect['environment'] = env
                            effect['metric'] = 'p95_latency_us'
                            effect_sizes.append(effect)
    
    print(f"Computed {len(effect_sizes)} effect sizes")
    
    # Compute environment deltas
    env_deltas = []
    for config_key, env_results in by_config.items():
        algorithm, payload, rate = config_key
        
        native_mean = np.mean([r.p95 for r in env_results.get('native', [])]) if env_results.get('native') else None
        minikube_mean = np.mean([r.p95 for r in env_results.get('minikube', [])]) if env_results.get('minikube') else None
        gcp_mean = np.mean([r.p95 for r in env_results.get('gcp', [])]) if env_results.get('gcp') else None
        
        delta = {
            'algorithm': algorithm,
            'payload_size': payload,
            'rate': rate,
            'native_p95_mean': round(native_mean, 2) if native_mean else None,
            'minikube_p95_mean': round(minikube_mean, 2) if minikube_mean else None,
            'gcp_p95_mean': round(gcp_mean, 2) if gcp_mean else None,
        }
        
        if native_mean and minikube_mean:
            delta['native_to_minikube_pct'] = round((minikube_mean - native_mean) / native_mean * 100, 2)
        if native_mean and gcp_mean:
            delta['native_to_gcp_pct'] = round((gcp_mean - native_mean) / native_mean * 100, 2)
        if minikube_mean and gcp_mean:
            delta['minikube_to_gcp_pct'] = round((gcp_mean - minikube_mean) / minikube_mean * 100, 2)
        
        env_deltas.append(delta)
    
    # Write outputs
    args.output.mkdir(parents=True, exist_ok=True)
    
    # JSON output
    output_json = {
        'generated_at': index.get('generated_at', ''),
        'total_experiments': len(results),
        'aggregated': [a.to_dict() for a in aggregated],
        'effect_sizes': effect_sizes,
        'environment_deltas': env_deltas,
    }
    
    json_path = args.output / 'aggregated_stats.json'
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"Written: {json_path}")
    
    # CSV output
    csv_path = args.output / 'aggregated_stats.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'algorithm', 'payload_size', 'rate', 'environment', 'n_runs',
            'p50_mean', 'p50_std', 'p50_ci_low', 'p50_ci_high',
            'p95_mean', 'p95_std', 'p95_ci_low', 'p95_ci_high',
            'p99_mean', 'p99_std', 'p99_ci_low', 'p99_ci_high',
            'throughput_mean', 'throughput_std',
        ])
        for a in aggregated:
            writer.writerow([
                a.algorithm, a.payload_size, a.rate, a.environment, a.n_runs,
                round(a.p50_mean, 2), round(a.p50_std, 2), round(a.p50_ci_low, 2), round(a.p50_ci_high, 2),
                round(a.p95_mean, 2), round(a.p95_std, 2), round(a.p95_ci_low, 2), round(a.p95_ci_high, 2),
                round(a.p99_mean, 2), round(a.p99_std, 2), round(a.p99_ci_low, 2), round(a.p99_ci_high, 2),
                round(a.throughput_mean, 2), round(a.throughput_std, 2),
            ])
    print(f"Written: {csv_path}")
    
    # Also write to stats subdirectory
    stats_dir = args.output / 'stats'
    stats_dir.mkdir(exist_ok=True)
    
    with open(stats_dir / 'aggregated_stats.json', 'w') as f:
        json.dump(output_json, f, indent=2)
    
    with open(stats_dir / 'effect_sizes.json', 'w') as f:
        json.dump(effect_sizes, f, indent=2)
    
    with open(stats_dir / 'environment_deltas.json', 'w') as f:
        json.dump(env_deltas, f, indent=2)
    
    print(f"\nAggregation complete!")
    print(f"  - {len(aggregated)} aggregated statistics")
    print(f"  - {len(effect_sizes)} effect sizes")
    print(f"  - {len(env_deltas)} environment deltas")


if __name__ == "__main__":
    main()

