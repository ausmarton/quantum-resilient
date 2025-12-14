#!/usr/bin/env python3
"""
Validate that summary.json files accurately represent the raw data.

This script:
1. Loads raw data from run-*/raw/run.jsonl files
2. Computes statistics from raw data
3. Compares with existing summary.json
4. Reports any discrepancies

Usage:
    python scripts/validate_summaries_against_raw.py [--fix]
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np


def load_raw_data(exp_dir: Path) -> pd.DataFrame:
    """Load all raw data files for an experiment."""
    raw_files = list(exp_dir.rglob("run-*/raw/run.jsonl"))
    
    if not raw_files:
        return None
    
    chunks = []
    for raw_file in sorted(raw_files):
        file_size = raw_file.stat().st_size
        use_chunks = file_size > 100 * 1024 * 1024  # 100MB threshold
        chunk_size = 100000
        
        if use_chunks:
            for chunk in pd.read_json(raw_file, lines=True, chunksize=chunk_size):
                chunks.append(chunk)
        else:
            chunks.append(pd.read_json(raw_file, lines=True))
    
    if not chunks:
        return None
    
    return pd.concat(chunks, ignore_index=True)


def compute_statistics_from_raw(df: pd.DataFrame) -> dict:
    """Compute statistics from raw data (matching compute_statistics.py logic)."""
    if df is None or len(df) == 0:
        return None
    
    # Convert timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
    
    # Latency statistics (in nanoseconds)
    latency_col = None
    for col in ['latency_ns', 'latency', 'duration_ns', 'duration']:
        if col in df.columns:
            latency_col = col
            break
    
    if latency_col is None:
        return None
    
    latency_data = df[latency_col].dropna()
    
    if len(latency_data) == 0:
        return None
    
    # Compute percentiles
    stats = {
        'total_events': len(df),
        'latency': {
            'mean': float(latency_data.mean()),
            'std': float(latency_data.std()),
            'p50': float(latency_data.quantile(0.50)),
            'p90': float(latency_data.quantile(0.90)),
            'p95': float(latency_data.quantile(0.95)),
            'p99': float(latency_data.quantile(0.99)),
            'p999': float(latency_data.quantile(0.999)),
            'min': float(latency_data.min()),
            'max': float(latency_data.max()),
        }
    }
    
    # Throughput (if timestamp available)
    if 'timestamp' in df.columns and len(df) > 1:
        df_sorted = df.sort_values('timestamp')
        time_span = (df_sorted['timestamp'].iloc[-1] - df_sorted['timestamp'].iloc[0]).total_seconds()
        if time_span > 0:
            stats['throughput'] = {
                'mean_msgs_per_sec': float(len(df) / time_span),
                'total_events': len(df),
                'duration_seconds': float(time_span),
            }
    
    return stats


def compare_summaries(raw_stats: dict, summary_stats: dict, tolerance: float = 0.01) -> dict:
    """Compare raw-computed stats with summary stats."""
    issues = []
    warnings = []
    
    if raw_stats is None:
        issues.append("Could not compute stats from raw data")
        return {'valid': False, 'issues': issues, 'warnings': warnings}
    
    # Check total events
    raw_events = raw_stats.get('total_events', 0)
    summary_events = summary_stats.get('total_events', 0)
    
    if abs(raw_events - summary_events) > 0:
        issues.append(f"Event count mismatch: raw={raw_events}, summary={summary_events}")
    
    # Check latency statistics
    raw_latency = raw_stats.get('latency', {})
    summary_latency = summary_stats.get('latency', {})
    
    if not raw_latency or not summary_latency:
        issues.append("Missing latency data in one or both")
        return {'valid': False, 'issues': issues, 'warnings': warnings}
    
    # Detect unit mismatch: if summary values are ~1000x smaller, they're in microseconds
    # Raw data is in nanoseconds, summary might be in microseconds
    sample_raw = raw_latency.get('p50', 0)
    sample_summary = summary_latency.get('p50', 0)
    
    if sample_raw > 0 and sample_summary > 0:
        ratio = sample_raw / sample_summary
        if ratio > 900 and ratio < 1100:  # ~1000x difference indicates unit mismatch
            # Summary is in microseconds, convert to nanoseconds for comparison
            unit_conversion = 1000
            warnings.append("Summary appears to be in microseconds, converting for comparison")
        else:
            unit_conversion = 1
    else:
        unit_conversion = 1
    
    # Compare percentiles
    for percentile in ['p50', 'p95', 'p99', 'mean']:
        raw_val = raw_latency.get(percentile)
        summary_val = summary_latency.get(percentile)
        
        if raw_val is None or summary_val is None:
            warnings.append(f"Missing {percentile} in one or both")
            continue
        
        # Convert summary to nanoseconds if needed
        summary_val_ns = summary_val * unit_conversion
        
        # Calculate relative difference
        if raw_val > 0:
            rel_diff = abs(raw_val - summary_val_ns) / raw_val
            if rel_diff > tolerance:
                issues.append(
                    f"Latency {percentile} mismatch: raw={raw_val:.2f}ns, "
                    f"summary={summary_val_ns:.2f}ns (from {summary_val:.2f}), diff={rel_diff*100:.2f}%"
                )
            elif rel_diff > tolerance / 10:
                warnings.append(
                    f"Latency {percentile} small difference: {rel_diff*100:.2f}%"
                )
    
    # Check throughput if available
    raw_tput = raw_stats.get('throughput', {})
    summary_tput = summary_stats.get('throughput', {})
    
    if raw_tput and summary_tput:
        raw_rate = raw_tput.get('mean_msgs_per_sec', 0)
        summary_rate = summary_tput.get('mean_msgs_per_sec', 0)
        
        if raw_rate > 0:
            rel_diff = abs(raw_rate - summary_rate) / raw_rate
            if rel_diff > tolerance:
                issues.append(
                    f"Throughput mismatch: raw={raw_rate:.2f}, "
                    f"summary={summary_rate:.2f}, diff={rel_diff*100:.2f}%"
                )
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'raw_events': raw_events,
        'summary_events': summary_events,
    }


def validate_experiment(exp_dir: Path) -> dict:
    """Validate a single experiment's summary."""
    summary_file = exp_dir / "stats" / "summary.json"
    
    if not summary_file.exists():
        return {
            'exp_name': exp_dir.name,
            'valid': False,
            'error': 'No summary.json file'
        }
    
    # Load summary
    try:
        with open(summary_file) as f:
            summary_stats = json.load(f)
    except Exception as e:
        return {
            'exp_name': exp_dir.name,
            'valid': False,
            'error': f'Error loading summary: {e}'
        }
    
    # Load and compute from raw data
    try:
        raw_df = load_raw_data(exp_dir)
        if raw_df is None or len(raw_df) == 0:
            return {
                'exp_name': exp_dir.name,
                'valid': False,
                'error': 'No raw data found'
            }
        
        raw_stats = compute_statistics_from_raw(raw_df)
        if raw_stats is None:
            return {
                'exp_name': exp_dir.name,
                'valid': False,
                'error': 'Could not compute stats from raw data'
            }
        
        # Compare
        comparison = compare_summaries(raw_stats, summary_stats)
        
        return {
            'exp_name': exp_dir.name,
            'valid': comparison['valid'],
            'issues': comparison.get('issues', []),
            'warnings': comparison.get('warnings', []),
            'raw_events': comparison.get('raw_events', 0),
            'summary_events': comparison.get('summary_events', 0),
        }
    
    except Exception as e:
        return {
            'exp_name': exp_dir.name,
            'valid': False,
            'error': f'Error processing: {e}'
        }


def main():
    parser = argparse.ArgumentParser(description="Validate summaries against raw data")
    parser.add_argument('--fix', action='store_true', help='Regenerate invalid summaries')
    parser.add_argument('--tolerance', type=float, default=0.01, help='Tolerance for comparisons (default: 0.01 = 1%%)')
    parser.add_argument('--output', type=Path, help='Output validation report')
    
    args = parser.parse_args()
    
    # Find all experiments
    results_dir = Path('results')
    experiments = []
    
    for env_dir in ['native', 'minikube', 'gcp']:
        env_path = results_dir / env_dir
        if env_path.exists():
            for exp_dir in env_path.iterdir():
                if exp_dir.is_dir():
                    summary_file = exp_dir / "stats" / "summary.json"
                    if summary_file.exists():
                        experiments.append(exp_dir)
    
    print(f"Validating {len(experiments)} experiments...")
    print()
    
    results = []
    for exp_dir in experiments:
        result = validate_experiment(exp_dir)
        results.append(result)
        
        if not result['valid']:
            status = "❌"
            if 'error' in result:
                print(f"{status} {result['exp_name']}: {result['error']}")
            else:
                print(f"{status} {result['exp_name']}: {len(result.get('issues', []))} issues")
        elif result.get('warnings'):
            print(f"⚠️  {result['exp_name']}: {len(result['warnings'])} warnings")
    
    # Summary
    valid_count = sum(1 for r in results if r['valid'])
    invalid_count = len(results) - valid_count
    warning_count = sum(1 for r in results if r.get('warnings'))
    
    print()
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total experiments: {len(results)}")
    print(f"Valid: {valid_count} ({valid_count/len(results)*100:.1f}%)")
    print(f"Invalid: {invalid_count} ({invalid_count/len(results)*100:.1f}%)")
    print(f"With warnings: {warning_count}")
    print()
    
    if invalid_count > 0:
        print("Invalid experiments:")
        for r in results:
            if not r['valid']:
                print(f"  - {r['exp_name']}")
                if 'error' in r:
                    print(f"    Error: {r['error']}")
                for issue in r.get('issues', []):
                    print(f"    Issue: {issue}")
        print()
    
    # Save report if requested
    if args.output:
        report = {
            'total': len(results),
            'valid': valid_count,
            'invalid': invalid_count,
            'results': results
        }
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {args.output}")
    
    # Exit code
    sys.exit(0 if invalid_count == 0 else 1)


if __name__ == '__main__':
    main()
