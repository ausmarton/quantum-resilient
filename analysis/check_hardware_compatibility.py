#!/usr/bin/env python3
"""
Check hardware compatibility for cross-environment comparisons.

Validates that hardware differences are documented and warns about
comparisons that may be confounded by hardware differences.

Usage:
    python3 analysis/check_hardware_compatibility.py \
        --native results/native/*/hardware_metadata.json \
        --minikube results/minikube/*/container_metadata.json \
        --gcp results/gcp/*/cloud_metadata.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def load_metadata(path: Path) -> Optional[dict]:
    """Load metadata from JSON file."""
    if not path.exists():
        return None
    
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}", file=sys.stderr)
        return None


def extract_hardware_specs(metadata: dict) -> dict:
    """Extract hardware specifications from metadata."""
    return {
        'cpu_model': metadata.get('cpu_model', 'unknown'),
        'cpu_count': metadata.get('cpu_count', metadata.get('cpu_count', 'unknown')),
        'memory_kb': metadata.get('memory_total_kb', metadata.get('memory_total_kb', 'unknown')),
        'arch': metadata.get('arch', 'unknown'),
        'kernel': metadata.get('kernel_version', 'unknown'),
    }


def check_compatibility(specs_a: dict, specs_b: dict, env_a: str, env_b: str) -> list[str]:
    """Check hardware compatibility and return warnings."""
    warnings = []
    
    # CPU model
    if specs_a.get('cpu_model') != specs_b.get('cpu_model'):
        warnings.append(
            f"Different CPU models: {specs_a.get('cpu_model')} vs {specs_b.get('cpu_model')}"
        )
    
    # CPU count
    cpu_a = specs_a.get('cpu_count')
    cpu_b = specs_b.get('cpu_count')
    if cpu_a != 'unknown' and cpu_b != 'unknown' and cpu_a != cpu_b:
        warnings.append(
            f"Different CPU counts: {cpu_a} vs {cpu_b} (normalize throughput by CPU count)"
        )
    
    # Memory
    mem_a = specs_a.get('memory_kb')
    mem_b = specs_b.get('memory_kb')
    if mem_a != 'unknown' and mem_b != 'unknown':
        mem_a_val = int(mem_a) if isinstance(mem_a, (int, str)) and str(mem_a).isdigit() else 0
        mem_b_val = int(mem_b) if isinstance(mem_b, (int, str)) and str(mem_b).isdigit() else 0
        if mem_a_val > 0 and mem_b_val > 0:
            diff_pct = abs(mem_a_val - mem_b_val) / mem_a_val * 100
            if diff_pct > 10:  # More than 10% difference
                warnings.append(
                    f"Significant memory difference: {mem_a_val/1024/1024:.1f} GB vs "
                    f"{mem_b_val/1024/1024:.1f} GB ({diff_pct:.1f}% difference)"
                )
    
    # Architecture
    if specs_a.get('arch') != specs_b.get('arch'):
        warnings.append(
            f"Different architectures: {specs_a.get('arch')} vs {specs_b.get('arch')}"
        )
    
    return warnings


def main():
    parser = argparse.ArgumentParser(
        description="Check hardware compatibility for cross-environment comparisons"
    )
    parser.add_argument(
        "--native",
        type=Path,
        help="Path to native hardware_metadata.json (glob pattern supported)"
    )
    parser.add_argument(
        "--minikube",
        type=Path,
        help="Path to minikube container_metadata.json (glob pattern supported)"
    )
    parser.add_argument(
        "--gcp",
        type=Path,
        help="Path to GCP cloud_metadata.json (glob pattern supported)"
    )
    
    args = parser.parse_args()
    
    # Load metadata
    native_metadata = None
    minikube_metadata = None
    gcp_metadata = None
    
    if args.native:
        native_path = Path(args.native)
        if native_path.exists():
            native_metadata = load_metadata(native_path)
        else:
            # Try glob pattern
            native_paths = list(Path().glob(str(args.native)))
            if native_paths:
                native_metadata = load_metadata(native_paths[0])
    
    if args.minikube:
        minikube_path = Path(args.minikube)
        if minikube_path.exists():
            minikube_metadata = load_metadata(minikube_path)
        else:
            # Try glob pattern
            minikube_paths = list(Path().glob(str(args.minikube)))
            if minikube_paths:
                minikube_metadata = load_metadata(minikube_paths[0])
    
    if args.gcp:
        gcp_path = Path(args.gcp)
        if gcp_path.exists():
            gcp_metadata = load_metadata(gcp_path)
        else:
            # Try glob pattern
            gcp_paths = list(Path().glob(str(args.gcp)))
            if gcp_paths:
                gcp_metadata = load_metadata(gcp_paths[0])
    
    # Extract specs
    specs = {}
    if native_metadata:
        specs['native'] = extract_hardware_specs(native_metadata)
    if minikube_metadata:
        specs['minikube'] = extract_hardware_specs(minikube_metadata)
    if gcp_metadata:
        specs['gcp'] = extract_hardware_specs(gcp_metadata)
    
    # Print hardware specs
    if RICH_AVAILABLE:
        console = Console()
        table = Table(title="Hardware Specifications")
        table.add_column("Environment", style="cyan")
        table.add_column("CPU Model", style="yellow")
        table.add_column("CPU Count", justify="right")
        table.add_column("Memory (GB)", justify="right")
        table.add_column("Architecture")
        
        for env, spec in specs.items():
            mem_gb = "unknown"
            if spec.get('memory_kb') != 'unknown':
                try:
                    mem_kb = int(spec['memory_kb'])
                    mem_gb = f"{mem_kb / 1024 / 1024:.1f}"
                except:
                    pass
            
            table.add_row(
                env,
                spec.get('cpu_model', 'unknown'),
                str(spec.get('cpu_count', 'unknown')),
                mem_gb,
                spec.get('arch', 'unknown')
            )
        
        console.print(table)
        console.print()
    else:
        print("=== Hardware Specifications ===")
        for env, spec in specs.items():
            print(f"\n{env.upper()}:")
            print(f"  CPU Model: {spec.get('cpu_model', 'unknown')}")
            print(f"  CPU Count: {spec.get('cpu_count', 'unknown')}")
            print(f"  Memory: {spec.get('memory_kb', 'unknown')} KB")
            print(f"  Architecture: {spec.get('arch', 'unknown')}")
        print()
    
    # Check compatibility
    print("=== Compatibility Checks ===")
    
    all_warnings = []
    
    if 'native' in specs and 'minikube' in specs:
        warnings = check_compatibility(specs['native'], specs['minikube'], 'native', 'minikube')
        if warnings:
            print("\n⚠️  Native vs Minikube:")
            for warning in warnings:
                print(f"  - {warning}")
                all_warnings.append(('native', 'minikube', warning))
        else:
            print("\n✅ Native vs Minikube: Compatible (same hardware)")
    
    if 'native' in specs and 'gcp' in specs:
        warnings = check_compatibility(specs['native'], specs['gcp'], 'native', 'gcp')
        if warnings:
            print("\n⚠️  Native vs GCP:")
            for warning in warnings:
                print(f"  - {warning}")
                all_warnings.append(('native', 'gcp', warning))
        else:
            print("\n✅ Native vs GCP: Compatible")
    
    if 'minikube' in specs and 'gcp' in specs:
        warnings = check_compatibility(specs['minikube'], specs['gcp'], 'minikube', 'gcp')
        if warnings:
            print("\n⚠️  Minikube vs GCP:")
            for warning in warnings:
                print(f"  - {warning}")
                all_warnings.append(('minikube', 'gcp', warning))
        else:
            print("\n✅ Minikube vs GCP: Compatible")
    
    # Summary
    print("\n=== Summary ===")
    if all_warnings:
        print(f"⚠️  Found {len(all_warnings)} hardware compatibility issue(s)")
        print("\nRecommendations:")
        print("  - Frame comparisons as 'deployment context' (hardware + environment)")
        print("  - Use normalization for throughput comparisons (per CPU)")
        print("  - Focus on relative patterns, not absolute values")
        print("  - Document hardware differences in dissertation")
        return 1
    else:
        print("✅ All environments use compatible hardware")
        print("  - Direct comparisons are valid")
        return 0


if __name__ == "__main__":
    sys.exit(main())

