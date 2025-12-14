#!/usr/bin/env python3
"""
Verify that all generated artifacts comply with REQUIREMENTS_SPECIFICATION.md.

Checks:
- All required outputs generated
- All required figures exist
- All required tables exist
- All required statistics computed
- All dissertation claims supported

Usage:
    python scripts/verify_requirements_compliance.py [--output report.json]
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


def check_file_exists(path: Path, description: str) -> dict:
    """Check if a file exists."""
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    return {
        'description': description,
        'path': str(path),
        'exists': exists,
        'size_bytes': size,
        'valid': exists and size > 0
    }


def check_directory_exists(path: Path, description: str) -> dict:
    """Check if a directory exists and has files."""
    exists = path.exists() and path.is_dir()
    file_count = len(list(path.glob('*'))) if exists else 0
    return {
        'description': description,
        'path': str(path),
        'exists': exists,
        'file_count': file_count,
        'valid': exists and file_count > 0
    }


def verify_analysis_outputs(base_dir: Path) -> dict:
    """Verify all analysis outputs exist."""
    results = {
        'aggregated_statistics': check_file_exists(
            base_dir / 'aggregated_stats.json',
            'Aggregated statistics JSON'
        ),
        'aggregated_statistics_csv': check_file_exists(
            base_dir / 'aggregated_stats.csv',
            'Aggregated statistics CSV'
        ),
        'hypothesis_tests': check_file_exists(
            base_dir / 'hypothesis_tests.json',
            'Hypothesis test results JSON'
        ),
        'hypothesis_table': check_file_exists(
            base_dir / 'hypothesis_table.csv',
            'Hypothesis test table CSV'
        ),
        'figures_directory': check_directory_exists(
            base_dir / 'figures',
            'Figures directory'
        ),
        'tables_directory': check_directory_exists(
            base_dir / 'tables',
            'Tables directory'
        ),
    }
    
    # Check specific figures
    figures_dir = base_dir / 'figures'
    if figures_dir.exists():
        required_figures = [
            'combined_ecdf.png',
            'combined_ecdf_native.png',
            'combined_ecdf_minikube.png',
            'combined_ecdf_gcp.png',
            'native_vs_minikube_vs_gcp.png',
            'scaling_curves.png',
            'classical_vs_pqc.png',
        ]
        for fig in required_figures:
            results[f'figure_{fig}'] = check_file_exists(
                figures_dir / fig,
                f'Figure: {fig}'
            )
    
    # Check specific tables
    tables_dir = base_dir / 'tables'
    if tables_dir.exists():
        required_tables = [
            'performance_table.csv',
            'performance_table.tex',
            'environment_delta_table.csv',
            'environment_delta_table.tex',
        ]
        for table in required_tables:
            results[f'table_{table}'] = check_file_exists(
                tables_dir / table,
                f'Table: {table}'
            )
    
    return results


def verify_data_completeness(base_dir: Path) -> dict:
    """Verify data completeness requirements."""
    results = {}
    
    # Check summaries
    summaries = list(Path('results').rglob('**/stats/summary.json'))
    results['summaries'] = {
        'description': 'Experiment summaries',
        'count': len(summaries),
        'expected': 330,
        'valid': len(summaries) >= 330
    }
    
    # Check index
    index_file = base_dir / 'index.json'
    if index_file.exists():
        with open(index_file) as f:
            index_data = json.load(f)
        results['index'] = {
            'description': 'Experiment index',
            'count': len(index_data.get('experiments', [])),
            'expected': 330,
            'valid': len(index_data.get('experiments', [])) >= 330
        }
    
    return results


def verify_statistical_requirements(base_dir: Path) -> dict:
    """Verify statistical analysis requirements."""
    results = {}
    
    # Load aggregated stats
    stats_file = base_dir / 'aggregated_stats.json'
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)
        
        results['aggregated_configs'] = {
            'description': 'Aggregated configurations',
            'count': len(stats.get('aggregated', [])),
            'valid': len(stats.get('aggregated', [])) > 0
        }
        
        results['effect_sizes'] = {
            'description': 'Effect size calculations',
            'count': len(stats.get('effect_sizes', [])),
            'valid': len(stats.get('effect_sizes', [])) > 0
        }
        
        results['environment_deltas'] = {
            'description': 'Environment delta calculations',
            'count': len(stats.get('environment_deltas', [])),
            'valid': len(stats.get('environment_deltas', [])) > 0
        }
    
    # Load hypothesis tests
    hyp_file = base_dir / 'hypothesis_tests.json'
    if hyp_file.exists():
        with open(hyp_file) as f:
            hyp = json.load(f)
        
        # Check both 'tests' and 'results' keys (different script versions)
        tests = hyp.get('tests', [])
        results_list = hyp.get('results', [])
        
        # Use whichever is available
        if results_list:
            test_count = len(results_list)
            significant = sum(1 for t in results_list if t.get('any_significant', False) or t.get('welch_sig', False) == 'Yes')
        elif tests:
            test_count = len(tests)
            significant = sum(1 for t in tests if t.get('welch_significant', False) or t.get('any_significant', False))
        else:
            # Fall back to summary counts
            test_count = hyp.get('total_comparisons', 0)
            significant = hyp.get('significant_comparisons', 0)
        
        results['hypothesis_tests'] = {
            'description': 'Hypothesis test comparisons',
            'count': test_count,
            'valid': test_count > 0
        }
        
        results['significant_tests'] = {
            'description': 'Statistically significant results',
            'count': significant,
            'total': test_count,
            'valid': test_count > 0
        }
    
    return results


def verify_dissertation_claims(base_dir: Path) -> dict:
    """Verify that data supports dissertation claims."""
    results = {}
    
    # Load data
    stats_file = base_dir / 'aggregated_stats.json'
    if not stats_file.exists():
        return {'error': 'aggregated_stats.json not found'}
    
    with open(stats_file) as f:
        stats = json.load(f)
    
    # Claim: PQC algorithms have data
    native = [a for a in stats.get('aggregated', []) if a.get('environment') == 'native']
    algorithms = set(a.get('algorithm') for a in native)
    required_algorithms = {'rsa2048', 'ecdsa_p256', 'kyber512', 'dilithium2', 'hybrid_kyber_dilithium'}
    
    # Normalize algorithm names for comparison (handle variations)
    normalized_found = set()
    for algo in algorithms:
        normalized = algo.lower().replace('_', '').replace('-', '')
        normalized_found.add(normalized)
    
    normalized_required = {a.lower().replace('_', '').replace('-', '') for a in required_algorithms}
    
    # Check matches
    missing = []
    for req in required_algorithms:
        req_norm = req.lower().replace('_', '').replace('-', '')
        if req_norm not in normalized_found:
            # Try to find close match
            close_match = [a for a in algorithms if req_norm in a.lower().replace('_', '').replace('-', '') or a.lower().replace('_', '').replace('-', '') in req_norm]
            if not close_match:
                missing.append(req)
    
    results['algorithms_covered'] = {
        'description': 'Required algorithms have data',
        'found': list(algorithms),
        'required': list(required_algorithms),
        'missing': missing,
        'valid': len(missing) == 0
    }
    
    # Claim: All environments have data
    environments = set(a.get('environment') for a in stats.get('aggregated', []))
    required_envs = {'native', 'minikube', 'gcp'}
    
    results['environments_covered'] = {
        'description': 'Required environments have data',
        'found': list(environments),
        'required': list(required_envs),
        'missing': list(required_envs - environments),
        'valid': required_envs.issubset(environments)
    }
    
    # Claim: Effect sizes computed
    effects = stats.get('effect_sizes', [])
    large_effects = [e for e in effects if abs(e.get('cohens_d', 0)) >= 0.8]
    
    results['large_effect_sizes'] = {
        'description': 'Large effect sizes (|d| ≥ 0.8) found',
        'count': len(large_effects),
        'total': len(effects),
        'valid': len(large_effects) > 0
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Verify requirements compliance")
    parser.add_argument('--output', type=Path, help='Output JSON report')
    parser.add_argument('--base-dir', type=Path, default=Path('final-results'),
                       help='Base directory for analysis outputs')
    
    args = parser.parse_args()
    
    base_dir = args.base_dir
    
    print("Verifying requirements compliance...")
    print("=" * 80)
    print()
    
    # Run all checks
    all_results = {
        'analysis_outputs': verify_analysis_outputs(base_dir),
        'data_completeness': verify_data_completeness(base_dir),
        'statistical_requirements': verify_statistical_requirements(base_dir),
        'dissertation_claims': verify_dissertation_claims(base_dir),
    }
    
    # Summary
    total_checks = 0
    passed_checks = 0
    
    print("## Analysis Outputs")
    print("-" * 80)
    for key, result in all_results['analysis_outputs'].items():
        total_checks += 1
        status = "✅" if result.get('valid', False) else "❌"
        print(f"{status} {result.get('description', key)}")
        if result.get('valid', False):
            passed_checks += 1
        elif 'size_bytes' in result:
            print(f"   Size: {result.get('size_bytes', 0)} bytes")
        elif 'file_count' in result:
            print(f"   Files: {result.get('file_count', 0)}")
    print()
    
    print("## Data Completeness")
    print("-" * 80)
    for key, result in all_results['data_completeness'].items():
        total_checks += 1
        status = "✅" if result.get('valid', False) else "❌"
        count = result.get('count', 0)
        expected = result.get('expected', 0)
        print(f"{status} {result.get('description', key)}: {count}/{expected}")
        if result.get('valid', False):
            passed_checks += 1
    print()
    
    print("## Statistical Requirements")
    print("-" * 80)
    for key, result in all_results['statistical_requirements'].items():
        total_checks += 1
        status = "✅" if result.get('valid', False) else "❌"
        count = result.get('count', 0)
        print(f"{status} {result.get('description', key)}: {count}")
        if result.get('valid', False):
            passed_checks += 1
    print()
    
    print("## Dissertation Claims Support")
    print("-" * 80)
    for key, result in all_results['dissertation_claims'].items():
        total_checks += 1
        status = "✅" if result.get('valid', False) else "❌"
        print(f"{status} {result.get('description', key)}")
        if 'found' in result:
            print(f"   Found: {result.get('found', [])}")
        if 'missing' in result and result.get('missing'):
            print(f"   Missing: {result.get('missing', [])}")
        if 'count' in result:
            print(f"   Count: {result.get('count', 0)}")
        if result.get('valid', False):
            passed_checks += 1
    print()
    
    # Overall summary
    print("=" * 80)
    print(f"SUMMARY: {passed_checks}/{total_checks} checks passed ({passed_checks/total_checks*100:.1f}%)")
    print("=" * 80)
    
    if args.output:
        report = {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'compliance_percentage': passed_checks / total_checks * 100 if total_checks > 0 else 0,
            'results': all_results
        }
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output}")
    
    sys.exit(0 if passed_checks == total_checks else 1)


if __name__ == '__main__':
    main()
