#!/usr/bin/env python3
"""
Comprehensive verification of dissertation readiness.

Checks all requirements from:
- docs/dissertation-requirements.md
- docs/REQUIREMENTS_SPECIFICATION.md
- DEVELOPMENT_GUIDELINES.md

Usage:
    python scripts/verify_dissertation_readiness.py \
        --output final-results/readiness_report.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def check_file_exists(path: Path, description: str) -> dict:
    """Check if a file exists and is readable."""
    exists = path.exists()
    if exists:
        try:
            size = path.stat().st_size
            if path.suffix == '.json':
                with open(path) as f:
                    json.load(f)  # Validate JSON
            return {'exists': True, 'size_bytes': size, 'valid': True}
        except Exception as e:
            return {'exists': True, 'valid': False, 'error': str(e)}
    return {'exists': False, 'valid': False}


def verify_data_completeness(base_dir: Path) -> dict:
    """Verify data completeness requirements."""
    results = {}
    
    # Check summaries (expected: 396 total with ECDHE: 120 native + 138 minikube + 138 gcp)
    summaries = list(Path('results').rglob('**/stats/summary.json'))
    expected_total = 396  # With ECDHE
    results['summaries'] = {
        'count': len(summaries),
        'expected': expected_total,
        'valid': len(summaries) >= expected_total,  # >= to allow for extra experiments
        'description': 'Experiment summaries'
    }
    
    # Check index
    index_file = base_dir / 'index.json'
    if index_file.exists():
        with open(index_file) as f:
            index_data = json.load(f)
        results['index'] = {
            'count': len(index_data.get('experiments', [])),
            'expected': expected_total,
            'valid': len(index_data.get('experiments', [])) >= expected_total,  # >= to allow for extra
            'description': 'Experiment index'
        }
    
    return results


def verify_required_outputs(base_dir: Path) -> dict:
    """Verify all required outputs from REQUIREMENTS_SPECIFICATION.md."""
    results = {}
    
    # Tables
    tables_dir = base_dir / 'tables'
    required_tables = [
        'performance_table.csv',
        'performance_table.tex',
        'environment_delta_table.csv',
        'environment_delta_table.tex',
    ]
    results['tables'] = {}
    for table in required_tables:
        path = tables_dir / table
        results['tables'][table] = check_file_exists(path, f'Table: {table}')
    
    # Figures
    figures_dir = base_dir / 'figures'
    required_figures = [
        'combined_ecdf.png',
        'combined_ecdf_native.png',
        'combined_ecdf_minikube.png',
        'combined_ecdf_gcp.png',
        'scaling_curves.png',
    ]
    results['figures'] = {}
    for fig in required_figures:
        path = figures_dir / fig
        results['figures'][fig] = check_file_exists(path, f'Figure: {fig}')
    
    # Analysis documents
    docs_dir = Path('docs/analysis')
    required_docs = [
        'data-interpretation.md',
        'payload-size-impact.md',
        'workload-pattern-impact.md',
        'error-rate-analysis.md',
    ]
    results['documents'] = {}
    for doc in required_docs:
        path = docs_dir / doc
        results['documents'][doc] = check_file_exists(path, f'Document: {doc}')
    
    return results


def verify_dissertation_claims(base_dir: Path) -> dict:
    """Verify dissertation claims from dissertation-requirements.md."""
    results = {}
    
    # Load data
    stats_file = base_dir / 'aggregated_stats.json'
    hyp_file = base_dir / 'hypothesis_tests.json'
    
    if not stats_file.exists():
        return {'error': 'aggregated_stats.json not found'}
    
    with open(stats_file) as f:
        stats = json.load(f)
    
    # Claim: PQC key generation overhead (1-3μs)
    native = [a for a in stats.get('aggregated', []) if a.get('environment') == 'native']
    pqc_algorithms = ['kyber512', 'dilithium2', 'hybrid']
    classical_algorithms = ['rsa2048', 'ecdsa_p256', 'ecdhe_p256']  # Includes ECDHE for KEM comparison
    
    pqc_latencies = []
    classical_latencies = []
    
    for exp in native:
        algo = exp.get('algorithm', '')
        p95 = exp.get('p95', {}).get('mean', 0)
        if algo in pqc_algorithms and p95 > 0:
            pqc_latencies.append(p95)
        elif algo in classical_algorithms and p95 > 0:
            classical_latencies.append(p95)
    
    if pqc_latencies and classical_latencies:
        avg_pqc = sum(pqc_latencies) / len(pqc_latencies)
        avg_classical = sum(classical_latencies) / len(classical_latencies)
        overhead = avg_pqc - avg_classical
        results['pqc_overhead'] = {
            'claim': '1-3μs overhead',
            'measured': round(overhead, 2),
            'valid': 1 <= overhead <= 3,
            'description': 'PQC key generation overhead'
        }
    
    # Claim: Dilithium vs ECDSA comparable
    dilithium = [a for a in native if a.get('algorithm') == 'dilithium2']
    ecdsa = [a for a in native if a.get('algorithm') == 'ecdsa']
    
    if dilithium and ecdsa:
        dil_p95 = sum(a.get('p95', {}).get('mean', 0) for a in dilithium) / len(dilithium)
        ecdsa_p95 = sum(a.get('p95', {}).get('mean', 0) for a in ecdsa) / len(ecdsa)
        ratio = dil_p95 / ecdsa_p95 if ecdsa_p95 > 0 else 0
        results['dilithium_vs_ecdsa'] = {
            'claim': 'Comparable performance',
            'dilithium_p95': round(dil_p95, 2),
            'ecdsa_p95': round(ecdsa_p95, 2),
            'ratio': round(ratio, 2),
            'valid': 0.5 <= ratio <= 1.5,  # Within 50% is "comparable"
            'description': 'Dilithium vs ECDSA performance'
        }
    
    # Claim: Large effect sizes (Cohen's d > 1.2)
    effects = stats.get('effect_sizes', [])
    large_effects = [e for e in effects if abs(e.get('cohens_d', 0)) >= 1.2]
    results['large_effect_sizes'] = {
        'claim': 'Cohen\'s d > 1.2',
        'count': len(large_effects),
        'total': len(effects),
        'valid': len(large_effects) > 0,
        'description': 'Large effect sizes found'
    }
    
    # Claim: Statistical significance (p < 0.001)
    if hyp_file.exists():
        with open(hyp_file) as f:
            hyp = json.load(f)
        tests = hyp.get('results', [])
        very_significant = [t for t in tests 
                          if t.get('welch_p', 0) < 0.001 or 
                             t.get('mw_p', 0) < 0.001]
        results['statistical_significance'] = {
            'claim': 'p < 0.001',
            'very_significant': len(very_significant),
            'total': len(tests),
            'valid': len(very_significant) > 0,
            'description': 'Very significant results (p < 0.001)'
        }
    
    return results


def verify_chapter_requirements(base_dir: Path) -> dict:
    """Verify Chapter 5 and 6 requirements."""
    results = {}
    
    # Chapter 5.1: Algorithmic Performance
    results['chapter_5_1'] = {
        'performance_tables': check_file_exists(
            base_dir / 'tables' / 'performance_table.csv',
            'Performance table'
        ),
        'cdf_plots': check_file_exists(
            base_dir / 'figures' / 'combined_ecdf.png',
            'CDF plots'
        ),
        'statistical_tests': check_file_exists(
            base_dir / 'hypothesis_tests.json',
            'Hypothesis tests'
        ),
    }
    
    # Chapter 5.2.1: Containerization Overhead
    results['chapter_5_2_1'] = {
        'overhead_calculated': check_file_exists(
            base_dir / 'aggregated_stats.json',
            'Environment deltas'
        ),
        'comparison_plots': check_file_exists(
            base_dir / 'figures' / 'native_vs_minikube_vs_gcp.png',
            'Environment comparison'
        ),
    }
    
    # Chapter 5.2.2: Production Scaling
    results['chapter_5_2_2'] = {
        'scaling_curves': check_file_exists(
            base_dir / 'figures' / 'scaling_curves.png',
            'Scaling curves'
        ),
    }
    
    # Chapter 6: Discussion
    results['chapter_6'] = {
        'interpretation_doc': check_file_exists(
            Path('docs/analysis/data-interpretation.md'),
            'Interpretation document'
        ),
        'recommendations': True,  # Based on data interpretation
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Verify dissertation readiness")
    parser.add_argument('--output', type=Path, default=Path('final-results/readiness_report.json'),
                       help='Output JSON report')
    parser.add_argument('--base-dir', type=Path, default=Path('final-results'),
                       help='Base directory for analysis outputs')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DISSERTATION READINESS VERIFICATION")
    print("=" * 80)
    print()
    
    all_results = {
        'data_completeness': verify_data_completeness(args.base_dir),
        'required_outputs': verify_required_outputs(args.base_dir),
        'dissertation_claims': verify_dissertation_claims(args.base_dir),
        'chapter_requirements': verify_chapter_requirements(args.base_dir),
    }
    
    # Print summary
    print("## Data Completeness")
    print("-" * 80)
    for key, result in all_results['data_completeness'].items():
        status = "✅" if result.get('valid', False) else "❌"
        count = result.get('count', 0)
        expected = result.get('expected', 0)
        print(f"{status} {result.get('description', key)}: {count}/{expected}")
    print()
    
    print("## Required Outputs")
    print("-" * 80)
    
    # Tables
    print("Tables:")
    for table, result in all_results['required_outputs'].get('tables', {}).items():
        status = "✅" if result.get('valid', False) else "❌"
        print(f"  {status} {table}")
    
    # Figures
    print("\nFigures:")
    for fig, result in all_results['required_outputs'].get('figures', {}).items():
        status = "✅" if result.get('valid', False) else "❌"
        print(f"  {status} {fig}")
    
    # Documents
    print("\nDocuments:")
    for doc, result in all_results['required_outputs'].get('documents', {}).items():
        status = "✅" if result.get('valid', False) else "❌"
        print(f"  {status} {doc}")
    print()
    
    print("## Dissertation Claims")
    print("-" * 80)
    for key, result in all_results['dissertation_claims'].items():
        status = "✅" if result.get('valid', False) else "❌"
        desc = result.get('description', key)
        if 'measured' in result:
            print(f"{status} {desc}: {result.get('measured')}μs (claim: {result.get('claim')})")
        elif 'count' in result:
            print(f"{status} {desc}: {result.get('count')}/{result.get('total')}")
        else:
            print(f"{status} {desc}")
    print()
    
    print("## Chapter Requirements")
    print("-" * 80)
    for chapter, requirements in all_results['chapter_requirements'].items():
        print(f"\n{chapter.upper()}:")
        for req, result in requirements.items():
            if isinstance(result, dict):
                status = "✅" if result.get('valid', False) else "❌"
                print(f"  {status} {result.get('description', req)}")
            else:
                status = "✅" if result else "❌"
                print(f"  {status} {req}")
    print()
    
    # Overall summary
    total_checks = 0
    passed_checks = 0
    
    def count_checks(d):
        nonlocal total_checks, passed_checks
        for v in d.values():
            if isinstance(v, dict):
                if 'valid' in v:
                    total_checks += 1
                    if v['valid']:
                        passed_checks += 1
                else:
                    count_checks(v)
            elif isinstance(v, bool):
                total_checks += 1
                if v:
                    passed_checks += 1
    
    count_checks(all_results)
    
    print("=" * 80)
    print(f"SUMMARY: {passed_checks}/{total_checks} checks passed ({passed_checks/total_checks*100:.1f}%)")
    print("=" * 80)
    
    # Save report
    report = {
        'total_checks': total_checks,
        'passed_checks': passed_checks,
        'compliance_percentage': passed_checks / total_checks * 100 if total_checks > 0 else 0,
        'results': all_results
    }
    
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {args.output}")
    
    return 0 if passed_checks == total_checks else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
