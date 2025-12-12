#!/usr/bin/env python3
"""
Generate individual scenario YAML files from experiment matrix.

Creates thousands of scenarios combining:
  algorithms × payload sizes × message rates × durations × runs

Directory structure:
  generated-scenarios/<algorithm>/<payload>/<rate>/run-<N>/scenario.yaml

Each scenario has:
  - Deterministic RNG seed = hash(algorithm + payload + rate + run_index)
  - Globally unique scenario ID
  - Full metadata section

Usage:
    python orchestration/generate_scenarios.py [--matrix experiment_matrix.yaml] [--output generated-scenarios]
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml


def compute_rng_seed(algorithm: str, payload: int, rate: int, run: int) -> int:
    """
    Compute deterministic RNG seed from experiment parameters.
    
    Uses SHA-256 hash of the concatenated parameters to produce a
    reproducible seed value.
    """
    seed_str = f"{algorithm}:{payload}:{rate}:{run}"
    hash_bytes = hashlib.sha256(seed_str.encode()).digest()
    # Use first 8 bytes as seed (64-bit integer)
    return int.from_bytes(hash_bytes[:8], byteorder='big') % (2**63)


def compute_scenario_hash(algorithm: str, payload: int, rate: int, run: int, pattern: str = "constant", duration: int = None, is_scaling: bool = False) -> str:
    """Compute short hash for globally unique scenario ID."""
    # Always include pattern, duration, and scaling flag in hash for uniqueness
    seed_parts = [algorithm, str(payload), str(rate), str(run), pattern]
    if duration and duration != 30:
        seed_parts.append(str(duration))
    if is_scaling:
        seed_parts.append("scaling")
    seed_str = ":".join(seed_parts)
    return hashlib.sha256(seed_str.encode()).hexdigest()[:8]


def generate_scenario_id(algorithm: str, payload: int, rate: int, run: int, pattern: str = "constant", duration: int = None, is_scaling: bool = False) -> str:
    """
    Generate globally unique scenario ID.
    
    Format: <algorithm>_p<payload>_r<rate>_run<N>_<hash>
    With pattern: <algorithm>_p<payload>_r<rate>_<pattern>_run<N>_<hash>
    With duration: <algorithm>_p<payload>_r<rate>_<duration>m_run<N>_<hash>
    
    Note: Scenario IDs are the same regardless of smoke-test vs full-scale mode.
    The distinction comes from the experiment matrix filtering and parameters, not the ID.
    """
    hash_suffix = compute_scenario_hash(algorithm, payload, rate, run, pattern, duration, is_scaling)
    
    # Build ID with optional pattern, duration, and scaling suffixes
    # Same format for both smoke-test and full-scale
    parts = [algorithm, f"p{payload}", f"r{rate}"]
    
    if pattern and pattern != "constant":
        parts.append(pattern)
    
    if duration and duration != 30:
        if duration == 300:
            parts.append("5m")
        else:
            parts.append(f"{duration}s")
    
    if is_scaling:
        parts.append("scaling")  # Add scaling suffix to make IDs unique
    
    parts.append(f"run{run}")
    parts.append(hash_suffix)
    
    return "_".join(parts)


def generate_scenario_yaml(
    experiment: dict,
    payload_size: int,
    rate: int,
    run_index: int,
    defaults: dict,
    generation_timestamp: str,
    smoke_test: bool = False,
) -> dict:
    """
    Generate a single scenario YAML configuration.
    
    Includes full metadata section for traceability.
    """
    algorithm = experiment['algorithm']
    
    # Get workload pattern (default: constant)
    workload_pattern = experiment.get('workload_pattern', 'constant')
    
    # Get duration override (if specified, otherwise use defaults)
    duration_sec = experiment.get('duration_sec', defaults.get('duration_sec', 30))
    if smoke_test:
        duration_sec = 5
    
    # Check if this is a scaling experiment
    is_scaling = experiment.get('scaling_experiment', False)
    
    # Compute deterministic seed (include pattern and duration for uniqueness)
    rng_seed = compute_rng_seed(algorithm, payload_size, rate, run_index)
    
    # Generate globally unique ID (include pattern, duration, and scaling flag)
    # Note: ID is the same regardless of smoke_test mode - distinction comes from matrix filtering
    scenario_id = generate_scenario_id(algorithm, payload_size, rate, run_index, workload_pattern, duration_sec, is_scaling)
    
    # Map adapter name for hybrid operations
    # Hybrid operations use 'kyber' adapter with special operations
    adapter_name = experiment.get('adapter', algorithm)
    operation = experiment.get('operation', 'sign')
    
    # For hybrid_kyber_dilithium, use 'kyber' adapter
    # The operation (kem_aead_sign) handles both KEM and signature
    if adapter_name == 'hybrid_kyber_dilithium':
        adapter_name = 'kyber'
    
    # Build workload configuration
    workload_config = {
        'msgs_per_sec': rate,
        'msg_size_bytes': payload_size,
        'duration_sec': duration_sec,
        'pattern': workload_pattern,
    }
    
    # Add burst configuration if pattern is burst
    if workload_pattern == 'burst' and 'burst_config' in experiment:
        workload_config['burst'] = experiment['burst_config'].copy()
    
    # Build scenario
    scenario = {
        'id': scenario_id,
        'description': f"{experiment.get('description', algorithm)} - {payload_size}B @ {rate} msg/s (run {run_index})",
        'rng_seed': rng_seed,
        
        # Workload configuration
        'workload': workload_config,
        
        # Algorithm configuration
        'algorithm': {
            'adapter': adapter_name,
            'operation': operation,
        },
        
        # Execution configuration
        'execution': defaults.get('execution', {
            'mode': 'fixed_pool',
            'workers': 4,
            'queue_capacity': 4000,
        }),
        
        # Metrics configuration
        # Use absolute path for containerized environments (GCP, Minikube)
        # Relative paths fail with readOnlyRootFilesystem: true
        'metrics': {
            'jsonl_out': '/results/raw/run.jsonl',
        },
        
        # Full metadata section
        'metadata': {
            'scenario_id': scenario_id,
            'algorithm': algorithm,
            'adapter': adapter_name,  # Use mapped adapter name
            'operation': operation,
            'category': experiment.get('category', 'unknown'),
            'payload_size_bytes': payload_size,
            'msgs_per_sec': rate,
            'duration_sec': duration_sec,
            'workload_pattern': workload_pattern,
            'run_index': run_index,
            'total_runs': 1 if smoke_test else experiment.get('runs', defaults.get('runs', 5)),
            'scaling_experiment': is_scaling,  # Include scaling flag in metadata
            'seed': rng_seed,
            'seed_hash': compute_scenario_hash(algorithm, payload_size, rate, run_index),
            'generated_at': generation_timestamp,
            'generator_version': '2.0.0',
            # Note: No 'mode' field - scenarios are the same regardless of smoke-test flag
            # The distinction comes from experiment matrix filtering and parameters
        },
    }
    
    # Add KEM configuration if present
    if 'kem' in experiment:
        scenario['algorithm']['kem'] = experiment['kem'].copy()
        scenario['metadata']['kem_paramset'] = experiment['kem'].get('paramset', 'unknown')
    
    # Add signature configuration if present
    if 'sig' in experiment:
        scenario['algorithm']['sig'] = experiment['sig'].copy()
        scenario['metadata']['sig_paramset'] = experiment['sig'].get('paramset', 'unknown')
    
    return scenario


def validate_scenario(scenario: dict) -> tuple[bool, Optional[str]]:
    """
    Validate a generated scenario configuration.
    
    Returns (is_valid, error_message).
    """
    required_fields = ['id', 'workload', 'algorithm', 'metadata']
    
    for field in required_fields:
        if field not in scenario:
            return False, f"Missing required field: {field}"
    
    # Validate workload
    workload = scenario.get('workload', {})
    if workload.get('msgs_per_sec', 0) <= 0:
        return False, "msgs_per_sec must be positive"
    if workload.get('msg_size_bytes', 0) <= 0:
        return False, "msg_size_bytes must be positive"
    if workload.get('duration_sec', 0) <= 0:
        return False, "duration_sec must be positive"
    
    # Validate algorithm
    algorithm = scenario.get('algorithm', {})
    if not algorithm.get('adapter'):
        return False, "algorithm.adapter is required"
    if not algorithm.get('operation'):
        return False, "algorithm.operation is required"
    
    # Validate metadata
    metadata = scenario.get('metadata', {})
    if not metadata.get('scenario_id'):
        return False, "metadata.scenario_id is required"
    if metadata.get('seed') is None:
        return False, "metadata.seed is required"
    
    return True, None


def generate_all_scenarios(matrix: dict, output_dir: Path, smoke_test: bool = False, mini_smoke_test: bool = False) -> tuple[list[dict], list[str]]:
    """
    Generate all scenarios from experiment matrix.
    
    Returns (generated_scenarios, validation_errors).
    """
    defaults = matrix.get('defaults', {})
    experiments = matrix.get('experiments', [])
    generation_timestamp = datetime.now(timezone.utc).isoformat()
    
    generated = []
    errors = []
    seen_ids = set()
    
    # Smoke-test mode: restrict to subset of algorithms and experiment types
    smoke_test_algorithms = ['rsa2048', 'kyber512', 'dilithium2', 'hybrid_kyber_dilithium']
    
    # Mini smoke-test mode: only 2 algorithms (1 classical, 1 PQC)
    if mini_smoke_test:
        smoke_test_algorithms = ['rsa2048', 'kyber512']
        smoke_test = True  # Enable smoke test mode
    
    for experiment in experiments:
        algorithm = experiment['algorithm']
        
        # In smoke-test mode, only generate scenarios for specific algorithms
        if smoke_test and algorithm not in smoke_test_algorithms:
            continue
        
        # In smoke-test mode, filter experiments to avoid duplicates and reduce scope
        # Include: baseline constant experiments, burst experiments, scaling experiments
        if smoke_test:
            workload_pattern = experiment.get('workload_pattern', 'constant')
            is_scaling_exp = experiment.get('scaling_experiment', False)
            rates_orig = experiment.get('rates', [])
            
            # Skip 10K msg/s experiments (too high for smoke test)
            if 10000 in rates_orig:
                continue
            
            # Skip 5-minute duration experiments (too long for smoke test)
            if experiment.get('duration_sec') == 300:
                continue
            
            # Mini smoke test: only constant pattern experiments (no burst, no scaling)
            if mini_smoke_test:
                if workload_pattern != 'constant' or is_scaling_exp:
                    continue
            
            # Include all remaining experiments:
            # - Baseline constant experiments (will use smoke test parameters)
            # - Burst experiments (will use smoke test parameters but keep burst pattern)
            # - Scaling experiments (will use smoke test parameters but keep scaling flag)
            # They'll be distinguished by pattern and scaling flag in scenario ID
        
        # In smoke-test mode, use reduced parameters
        # CRITICAL: Hardware (machine_type, CPU, memory, disk) MUST stay identical
        if mini_smoke_test:
            # Mini smoke test: absolute minimum - 1 payload, 1 rate, 1 run
            payload_sizes = [256]  # Minimal size only
            rates = [100]  # Low rate only
            runs = 1
        elif smoke_test:
            # Enhanced smoke test: use 2 payload sizes and 2 rates for better coverage
            payload_sizes = [256, 1024]  # Minimal + common size
            rates = [100, 500]  # Low + medium rate
            runs = 1
        else:
            payload_sizes = experiment.get('payload_sizes', [1024])
            rates = experiment.get('rates', [500])
            runs = experiment.get('runs', defaults.get('runs', 5))
        
        for payload_size in payload_sizes:
            for rate in rates:
                for run_index in range(1, runs + 1):
                    # Generate scenario
                    scenario = generate_scenario_yaml(
                        experiment, payload_size, rate, run_index, 
                        defaults, generation_timestamp, smoke_test
                    )
                    
                    # Validate scenario
                    is_valid, error = validate_scenario(scenario)
                    if not is_valid:
                        errors.append(f"{scenario['id']}: {error}")
                        continue
                    
                    # Check for duplicate IDs
                    if scenario['id'] in seen_ids:
                        errors.append(f"Duplicate scenario ID: {scenario['id']}")
                        continue
                    seen_ids.add(scenario['id'])
                    
                    # Create directory structure: <algo>/<payload>/<rate>/run-<N>/
                    # Include pattern and scaling in directory path to avoid overwrites
                    # Format: <algo>/<pattern>[-scaling]/<payload>/<rate>/run-<N>/
                    workload_pattern = experiment.get('workload_pattern', 'constant')
                    is_scaling_exp = experiment.get('scaling_experiment', False)
                    
                    # Build directory path components
                    dir_parts = [algorithm]
                    
                    # Add pattern subdirectory if not constant (to distinguish burst)
                    if workload_pattern != 'constant':
                        dir_parts.append(workload_pattern)
                    elif is_scaling_exp:
                        # Add 'scaling' subdirectory for scaling experiments
                        dir_parts.append('scaling')
                    # Otherwise, constant non-scaling goes in algorithm root
                    
                    dir_parts.extend([
                        f"p{payload_size}",
                        f"r{rate}",
                        f"run-{run_index}"
                    ])
                    
                    scenario_dir = output_dir / Path(*dir_parts)
                    scenario_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Write scenario file
                    scenario_path = scenario_dir / "scenario.yaml"
                    with open(scenario_path, 'w') as f:
                        # Use custom representer for cleaner YAML output
                        yaml.dump(
                            scenario, f, 
                            default_flow_style=False, 
                            sort_keys=False,
                            allow_unicode=True,
                            width=120,
                        )
                    
                    # Track generated scenario
                    generated.append({
                        'id': scenario['id'],
                        'algorithm': algorithm,
                        'payload_size': payload_size,
                        'rate': rate,
                        'run_index': run_index,
                        'total_runs': scenario['metadata']['total_runs'],  # Include total_runs for run grouping
                        'seed': scenario['rng_seed'],
                        'path': str(scenario_path),
                        'relative_path': str(scenario_path.relative_to(output_dir)),
                        'category': experiment.get('category', 'unknown'),
                        'scaling_experiment': experiment.get('scaling_experiment', False),  # Include scaling flag
                        'workload_pattern': scenario['metadata'].get('workload_pattern', 'constant'),  # Include workload pattern for counting
                        'duration_sec': scenario['metadata'].get('duration_sec', defaults.get('duration_sec', 30)),  # Include duration for counting
                    })
    
    return generated, errors


def print_summary(generated: list[dict], matrix: dict) -> None:
    """Print generation summary."""
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    
    # By algorithm
    by_algo = Counter(s['algorithm'] for s in generated)
    print("\nScenarios by algorithm:")
    for algo, count in sorted(by_algo.items()):
        print(f"  {algo}: {count}")
    
    # By payload
    by_payload = Counter(s['payload_size'] for s in generated)
    print("\nScenarios by payload size:")
    for payload, count in sorted(by_payload.items()):
        print(f"  {payload}B: {count}")
    
    # By rate
    by_rate = Counter(s['rate'] for s in generated)
    print("\nScenarios by message rate:")
    for rate, count in sorted(by_rate.items()):
        print(f"  {rate} msg/s: {count}")
    
    # Combinations
    algorithms = len(by_algo)
    payloads = len(by_payload)
    rates = len(by_rate)
    runs = matrix.get('defaults', {}).get('runs', 5)
    
    print(f"\nMatrix dimensions:")
    print(f"  Algorithms: {algorithms}")
    print(f"  Payload sizes: {payloads}")
    print(f"  Message rates: {rates}")
    print(f"  Runs per config: {runs}")
    print(f"  Total combinations: {algorithms} × {payloads} × {rates} × {runs} = {len(generated)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate scenario files from experiment matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all scenarios
  python orchestration/generate_scenarios.py

  # Dry run to see what would be generated
  python orchestration/generate_scenarios.py --dry-run

  # Custom matrix and output
  python orchestration/generate_scenarios.py -m custom_matrix.yaml -o my-scenarios

Output structure:
  generated-scenarios/
  ├── rsa2048/
  │   ├── p256/
  │   │   ├── r100/
  │   │   │   ├── run-1/scenario.yaml
  │   │   │   ├── run-2/scenario.yaml
  │   │   │   └── ...
  │   │   ├── r500/
  │   │   └── r2000/
  │   ├── p1024/
  │   └── p4096/
  ├── kyber512/
  └── ...
        """
    )
    parser.add_argument(
        '--matrix', '-m',
        type=Path,
        default=Path('orchestration/experiment_matrix.yaml'),
        help='Path to experiment matrix YAML (default: orchestration/experiment_matrix.yaml)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('generated-scenarios'),
        help='Output directory for generated scenarios (default: generated-scenarios)'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Print what would be generated without writing files'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Validate existing scenarios without generating new ones'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )
    parser.add_argument(
        '--smoke-test',
        action='store_true',
        help='Generate smoke-test scenarios (reduced scale, minimal cost)'
    )
    parser.add_argument(
        '--mini-smoke-test',
        action='store_true',
        help='Generate minimal smoke-test scenarios (2 experiments: 1 classical, 1 PQC)'
    )
    
    args = parser.parse_args()
    
    # Load matrix
    if not args.matrix.exists():
        print(f"Error: Matrix file not found: {args.matrix}", file=sys.stderr)
        sys.exit(1)
    
    with open(args.matrix) as f:
        matrix = yaml.safe_load(f)
    
    if not args.quiet:
        print(f"Loaded experiment matrix: {args.matrix}")
        print(f"Output directory: {args.output}")
    
    # Count expected scenarios
    total = 0
    defaults = matrix.get('defaults', {})
    
    if not args.quiet:
        print("\nExperiment configuration:")
    
    for exp in matrix.get('experiments', []):
        payloads = len(exp.get('payload_sizes', [1024]))
        rates = len(exp.get('rates', [500]))
        runs = exp.get('runs', defaults.get('runs', 5))
        count = payloads * rates * runs
        total += count
        
        if not args.quiet:
            print(f"  {exp['algorithm']}: {payloads} payloads × {rates} rates × {runs} runs = {count}")
    
    if not args.quiet:
        print(f"\nTotal scenarios to generate: {total}")
    
    if args.dry_run:
        print("\n[DRY RUN] No files will be written")
        
        # Show sample scenarios
        print("\nSample scenario IDs that would be generated:")
        sample_count = 0
        for exp in matrix.get('experiments', []):
            algorithm = exp['algorithm']
            for payload in exp.get('payload_sizes', [1024])[:1]:
                for rate in exp.get('rates', [500])[:1]:
                    for run in range(1, min(3, exp.get('runs', 5) + 1)):
                        scenario_id = generate_scenario_id(algorithm, payload, rate, run)
                        seed = compute_rng_seed(algorithm, payload, rate, run)
                        print(f"  {scenario_id} (seed: {seed})")
                        sample_count += 1
                        if sample_count >= 10:
                            break
                    if sample_count >= 10:
                        break
                if sample_count >= 10:
                    break
            if sample_count >= 10:
                break
        
        if total > 10:
            print(f"  ... and {total - 10} more")
        
        return
    
    # Generate scenarios
    if not args.quiet:
        print("\nGenerating scenarios...")
        if args.smoke_test:
            print("Smoke-test mode: reduced scale, minimal cost")
    
    generated, errors = generate_all_scenarios(matrix, args.output, args.smoke_test, args.mini_smoke_test)
    
    # Report errors
    if errors:
        print(f"\n⚠ {len(errors)} validation errors:", file=sys.stderr)
        for err in errors[:10]:
            print(f"  - {err}", file=sys.stderr)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more", file=sys.stderr)
    
    # Write manifest
    manifest_path = args.output / "manifest.json"
    manifest = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'generator_version': '2.0.0',
        'matrix_file': str(args.matrix),
        'total_scenarios': len(generated),
        'validation_errors': len(errors),
        'algorithms': list(set(s['algorithm'] for s in generated)),
        'payload_sizes': sorted(set(s['payload_size'] for s in generated)),
        'rates': sorted(set(s['rate'] for s in generated)),
        'scenarios': generated,
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    if not args.quiet:
        print(f"\n✓ Generated {len(generated)} scenarios")
        print(f"✓ Manifest: {manifest_path}")
        print_summary(generated, matrix)
    else:
        print(f"Generated {len(generated)} scenarios in {args.output}")
    
    # Exit with error if there were validation issues
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
