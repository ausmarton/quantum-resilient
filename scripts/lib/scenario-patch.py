#!/usr/bin/env python3
# =============================================================================
# scripts/lib/scenario-patch.py - Unified scenario YAML patching
#
# Patches scenario YAML files for containerized execution:
# - Sets metrics.jsonl_out to /results/raw/run.jsonl
# - Handles smoke-test mode (reduces duration)
# - Sets RNG seed if provided
#
# Usage:
#   python3 scripts/lib/scenario-patch.py \
#     --input scenario.yaml \
#     --output patched-scenario.yaml \
#     [--smoke-test] \
#     [--seed 12345] \
#     [--jsonl-out /results/raw/run.jsonl]
# =============================================================================

import argparse
import sys
import yaml

def patch_scenario(
    input_path: str,
    output_path: str,
    smoke_test: bool = False,
    seed: int = None,
    jsonl_out: str = "/results/raw/run.jsonl",
    duration: int = None,
) -> None:
    """Patch scenario YAML for containerized execution."""
    try:
        with open(input_path, 'r') as f:
            scenario = yaml.safe_load(f) or {}
        
        # Ensure metrics section exists
        if 'metrics' not in scenario:
            scenario['metrics'] = {}
        
        # Set JSONL output path
        scenario['metrics']['jsonl_out'] = jsonl_out
        
        # Handle smoke-test mode or explicit duration override
        if smoke_test and 'workload' in scenario:
            scenario['workload']['duration_sec'] = 5
        elif duration is not None and 'workload' in scenario:
            scenario['workload']['duration_sec'] = duration
        
        # Set RNG seed if provided
        if seed is not None:
            if 'rng_seed' in scenario:
                scenario['rng_seed'] = seed
            else:
                # Insert after 'id' field if it exists
                if 'id' in scenario:
                    # Create ordered dict-like structure
                    new_scenario = {}
                    for key, value in scenario.items():
                        new_scenario[key] = value
                        if key == 'id':
                            new_scenario['rng_seed'] = seed
                    scenario = new_scenario
                else:
                    scenario['rng_seed'] = seed
        
        # Write patched scenario
        with open(output_path, 'w') as f:
            yaml.dump(scenario, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
    except Exception as e:
        print(f"ERROR: Failed to patch scenario: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Patch scenario YAML for containerized execution')
    parser.add_argument('--input', required=True, help='Input scenario YAML file')
    parser.add_argument('--output', required=True, help='Output scenario YAML file')
    parser.add_argument('--smoke-test', action='store_true', help='Enable smoke-test mode (reduce duration)')
    parser.add_argument('--seed', type=int, help='RNG seed value')
    parser.add_argument('--jsonl-out', default='/results/raw/run.jsonl', help='JSONL output path')
    parser.add_argument('--duration', type=int, help='Override duration in seconds')
    
    args = parser.parse_args()
    
    patch_scenario(
        input_path=args.input,
        output_path=args.output,
        smoke_test=args.smoke_test,
        seed=args.seed,
        jsonl_out=args.jsonl_out,
        duration=args.duration,
    )

if __name__ == '__main__':
    main()

