#!/usr/bin/env python3
"""
Generate/update interpretation documents with latest analysis data.

Extracts key findings from aggregated statistics and hypothesis tests,
then populates interpretation documents with actual data.

Usage:
    python scripts/generate_interpretation_docs.py \
        --stats final-results/aggregated_stats.json \
        --hypothesis final-results/hypothesis_tests.json \
        --output docs/analysis
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def extract_payload_impact(stats: dict) -> dict:
    """Extract payload size impact data."""
    native = [a for a in stats.get('aggregated', []) if a.get('environment') == 'native']
    by_algo_payload = defaultdict(lambda: defaultdict(list))
    
    for agg in native:
        algo = agg.get('algorithm')
        payload = agg.get('payload_size', 0)
        p95 = agg.get('p95', {}).get('mean', 0)
        if p95 > 0 and payload > 0:
            by_algo_payload[algo][payload].append(p95)
    
    return dict(by_algo_payload)


def extract_environment_overhead(stats: dict) -> dict:
    """Extract environment overhead data."""
    deltas = stats.get('environment_deltas', [])
    
    native_to_minikube = [d.get('native_to_minikube_pct') for d in deltas 
                         if d.get('native_to_minikube_pct') is not None]
    native_to_gcp = [d.get('native_to_gcp_pct') for d in deltas 
                    if d.get('native_to_gcp_pct') is not None]
    
    return {
        'native_to_minikube': {
            'avg': sum(native_to_minikube) / len(native_to_minikube) if native_to_minikube else 0,
            'min': min(native_to_minikube) if native_to_minikube else 0,
            'max': max(native_to_minikube) if native_to_minikube else 0,
            'count': len(native_to_minikube)
        },
        'native_to_gcp': {
            'avg': sum(native_to_gcp) / len(native_to_gcp) if native_to_gcp else 0,
            'min': min(native_to_gcp) if native_to_gcp else 0,
            'max': max(native_to_gcp) if native_to_gcp else 0,
            'count': len(native_to_gcp)
        }
    }


def extract_effect_sizes(stats: dict) -> dict:
    """Extract effect size distribution."""
    effects = stats.get('effect_sizes', [])
    
    large = [e for e in effects if abs(e.get('cohens_d', 0)) >= 0.8]
    medium = [e for e in effects if 0.5 <= abs(e.get('cohens_d', 0)) < 0.8]
    small = [e for e in effects if 0.2 <= abs(e.get('cohens_d', 0)) < 0.5]
    negligible = [e for e in effects if abs(e.get('cohens_d', 0)) < 0.2]
    
    return {
        'total': len(effects),
        'large': len(large),
        'medium': len(medium),
        'small': len(small),
        'negligible': len(negligible)
    }


def update_payload_impact_doc(output_dir: Path, payload_data: dict):
    """Update payload-size-impact.md with extracted data."""
    doc_path = output_dir / 'payload-size-impact.md'
    if not doc_path.exists():
        print(f"Warning: {doc_path} does not exist, skipping update")
        return
    
    # Read existing content
    with open(doc_path) as f:
        content = f.read()
    
    # Generate payload impact section
    lines = []
    lines.append("## Key Findings")
    lines.append("")
    lines.append("### Native Environment")
    lines.append("")
    lines.append("**Algorithm Performance by Payload Size (p95 latency in microseconds)**:")
    lines.append("")
    
    for algo in sorted(payload_data.keys()):
        payloads = sorted(payload_data[algo].keys())
        if len(payloads) >= 2:
            lines.append(f"**{algo}**:")
            prev_latency = None
            prev_payload = None
            for payload in payloads:
                values = payload_data[algo][payload]
                avg = sum(values) / len(values)
                lines.append(f"- {payload} bytes: {avg:.2f}μs")
                if prev_latency and prev_latency > 0 and prev_payload:
                    increase = ((avg - prev_latency) / prev_latency) * 100
                    size_diff_kb = (payload - prev_payload) / 1024
                    if size_diff_kb > 0:
                        per_kb = increase / size_diff_kb
                        lines.append(f"  → +{increase:+.1f}% ({per_kb:+.2f}% per KB)")
                prev_latency = avg
                prev_payload = payload
            lines.append("")
    
    # Replace the "Key Findings" section
    import re
    pattern = r'## Key Findings.*?(?=## |\Z)'
    replacement = '\n'.join(lines) + '\n'
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Write back
    with open(doc_path, 'w') as f:
        f.write(content)
    
    print(f"Updated: {doc_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate interpretation documents")
    parser.add_argument('--stats', type=Path, required=True,
                       help='Path to aggregated_stats.json')
    parser.add_argument('--hypothesis', type=Path,
                       help='Path to hypothesis_tests.json')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for interpretation docs')
    
    args = parser.parse_args()
    
    # Load data
    with open(args.stats) as f:
        stats = json.load(f)
    
    # Extract data
    payload_data = extract_payload_impact(stats)
    env_overhead = extract_environment_overhead(stats)
    effect_sizes = extract_effect_sizes(stats)
    
    # Update documents
    args.output.mkdir(parents=True, exist_ok=True)
    update_payload_impact_doc(args.output, payload_data)
    
    print("Interpretation documents updated successfully")


if __name__ == '__main__':
    main()
