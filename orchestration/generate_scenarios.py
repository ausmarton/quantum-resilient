#!/usr/bin/env python3
"""
Generate individual scenario YAML files from experiment matrix.

Creates a directory structure:
  generated-scenarios/<algorithm>/<payload>/<rate>/run_<N>/scenario.yaml

Each scenario has a deterministic RNG seed derived from its parameters.

Usage:
    python orchestration/generate_scenarios.py [--matrix experiment_matrix.yaml] [--output generated-scenarios]
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def compute_rng_seed(algorithm: str, payload: int, rate: int, run: int, base: int = 42) -> int:
    """Compute deterministic RNG seed from experiment parameters."""
    seed_str = f"{algorithm}:{payload}:{rate}:{run}:{base}"
    hash_bytes = hashlib.sha256(seed_str.encode()).digest()
    # Use first 8 bytes as seed
    return int.from_bytes(hash_bytes[:8], byteorder='big') % (2**63)


def generate_experiment_id(algorithm: str, payload: int, rate: int, run: int) -> str:
    """Generate unique experiment ID."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{algorithm}_p{payload}_r{rate}_run{run}_{timestamp}"


def generate_scenario_yaml(
    experiment: dict,
    payload_size: int,
    rate: int,
    run_index: int,
    defaults: dict,
) -> dict:
    """Generate a single scenario YAML configuration."""
    algorithm = experiment['algorithm']
    rng_seed_base = defaults.get('rng_seed_base', 42)
    
    rng_seed = compute_rng_seed(algorithm, payload_size, rate, run_index, rng_seed_base)
    exp_id = generate_experiment_id(algorithm, payload_size, rate, run_index)
    
    scenario = {
        'id': exp_id,
        'description': f"{experiment.get('description', algorithm)} - {payload_size}B @ {rate} msg/s (run {run_index})",
        'rng_seed': rng_seed,
        'workload': {
            'msgs_per_sec': rate,
            'msg_size_bytes': payload_size,
            'duration_sec': defaults.get('duration_sec', 30),
            'pattern': 'constant',
        },
        'algorithm': {
            'adapter': experiment.get('adapter', algorithm),
            'operation': experiment.get('operation', 'sign'),
        },
        'execution': defaults.get('execution', {
            'mode': 'fixed_pool',
            'workers': 4,
            'queue_capacity': 4000,
        }),
        'metrics': {
            'jsonl_out': './results/raw/run.jsonl',
        },
        'metadata': {
            'algorithm': algorithm,
            'payload_size_bytes': payload_size,
            'rate_msgs_per_sec': rate,
            'run_index': run_index,
            'category': experiment.get('category', 'unknown'),
        },
    }
    
    # Add KEM configuration if present
    if 'kem' in experiment:
        scenario['algorithm']['kem'] = experiment['kem']
    
    # Add signature configuration if present
    if 'sig' in experiment:
        scenario['algorithm']['sig'] = experiment['sig']
    
    return scenario


def generate_all_scenarios(matrix: dict, output_dir: Path) -> list[dict]:
    """Generate all scenarios from experiment matrix."""
    defaults = matrix.get('defaults', {})
    experiments = matrix.get('experiments', [])
    
    generated = []
    
    for experiment in experiments:
        algorithm = experiment['algorithm']
        payload_sizes = experiment.get('payload_sizes', [1024])
        rates = experiment.get('rates', [500])
        runs = experiment.get('runs', defaults.get('runs', 5))
        
        for payload_size in payload_sizes:
            for rate in rates:
                for run_index in range(1, runs + 1):
                    # Generate scenario
                    scenario = generate_scenario_yaml(
                        experiment, payload_size, rate, run_index, defaults
                    )
                    
                    # Create directory structure
                    scenario_dir = output_dir / algorithm / f"p{payload_size}" / f"r{rate}" / f"run_{run_index}"
                    scenario_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Write scenario file
                    scenario_path = scenario_dir / "scenario.yaml"
                    with open(scenario_path, 'w') as f:
                        yaml.dump(scenario, f, default_flow_style=False, sort_keys=False)
                    
                    # Track generated scenario
                    generated.append({
                        'id': scenario['id'],
                        'algorithm': algorithm,
                        'payload_size': payload_size,
                        'rate': rate,
                        'run_index': run_index,
                        'path': str(scenario_path),
                        'category': experiment.get('category', 'unknown'),
                    })
    
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate scenario files from experiment matrix")
    parser.add_argument(
        '--matrix', '-m',
        type=Path,
        default=Path('orchestration/experiment_matrix.yaml'),
        help='Path to experiment matrix YAML'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('generated-scenarios'),
        help='Output directory for generated scenarios'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be generated without writing files'
    )
    
    args = parser.parse_args()
    
    # Load matrix
    if not args.matrix.exists():
        print(f"Error: Matrix file not found: {args.matrix}", file=sys.stderr)
        sys.exit(1)
    
    with open(args.matrix) as f:
        matrix = yaml.safe_load(f)
    
    print(f"Loaded experiment matrix: {args.matrix}")
    print(f"Output directory: {args.output}")
    
    # Count expected scenarios
    total = 0
    for exp in matrix.get('experiments', []):
        payloads = len(exp.get('payload_sizes', [1024]))
        rates = len(exp.get('rates', [500]))
        runs = exp.get('runs', matrix.get('defaults', {}).get('runs', 5))
        count = payloads * rates * runs
        total += count
        print(f"  {exp['algorithm']}: {payloads} payloads × {rates} rates × {runs} runs = {count}")
    
    print(f"\nTotal scenarios to generate: {total}")
    
    if args.dry_run:
        print("\nDry run - no files written")
        return
    
    # Generate scenarios
    generated = generate_all_scenarios(matrix, args.output)
    
    # Write manifest
    manifest_path = args.output / "manifest.json"
    manifest = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'matrix_file': str(args.matrix),
        'total_scenarios': len(generated),
        'scenarios': generated,
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nGenerated {len(generated)} scenarios")
    print(f"Manifest: {manifest_path}")
    
    # Summary by algorithm
    from collections import Counter
    by_algo = Counter(s['algorithm'] for s in generated)
    print("\nBy algorithm:")
    for algo, count in sorted(by_algo.items()):
        print(f"  {algo}: {count}")


if __name__ == "__main__":
    main()

