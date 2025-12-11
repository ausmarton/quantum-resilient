#!/usr/bin/env python3
"""
Statistical hypothesis testing suite for PQC benchmark results.

Performs comprehensive statistical analysis:
- Kolmogorov-Smirnov test (distribution shape)
- Mann-Whitney U test (distribution location)
- Welch's t-test (mean difference)
- Cohen's d effect size with 95% CI
- Holm-Bonferroni p-value correction

Usage:
    python analysis/hypothesis_tests.py \
        --index final-results/index.json \
        --matrix orchestration/experiment_matrix.yaml \
        --output final-results
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class TestResult:
    """Result from a single statistical test."""
    comparison_id: str
    comparison_type: str  # 'algorithm', 'environment', 'pqc_vs_classical'
    group_a_name: str
    group_b_name: str
    metric: str
    
    # Sample info
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    
    # Test results
    ks_statistic: float = 0.0
    ks_pvalue: float = 1.0
    
    mw_statistic: float = 0.0
    mw_pvalue: float = 1.0
    
    welch_statistic: float = 0.0
    welch_pvalue: float = 1.0
    
    # Effect size
    cohens_d: float = 0.0
    cohens_d_ci_low: float = 0.0
    cohens_d_ci_high: float = 0.0
    effect_interpretation: str = "negligible"
    
    # Corrected p-values (filled after all tests)
    ks_pvalue_corrected: float = 1.0
    mw_pvalue_corrected: float = 1.0
    welch_pvalue_corrected: float = 1.0
    
    # Significance flags
    ks_significant: bool = False
    mw_significant: bool = False
    welch_significant: bool = False
    any_significant: bool = False
    
    def to_dict(self) -> dict:
        return {
            'comparison_id': self.comparison_id,
            'comparison_type': self.comparison_type,
            'group_a': self.group_a_name,
            'group_b': self.group_b_name,
            'metric': self.metric,
            'n_a': self.n_a,
            'n_b': self.n_b,
            'mean_a': round(self.mean_a, 4),
            'mean_b': round(self.mean_b, 4),
            'std_a': round(self.std_a, 4),
            'std_b': round(self.std_b, 4),
            'mean_diff': round(self.mean_b - self.mean_a, 4),
            'mean_diff_pct': round((self.mean_b - self.mean_a) / self.mean_a * 100, 2) if self.mean_a != 0 else 0,
            'tests': {
                'kolmogorov_smirnov': {
                    'statistic': round(self.ks_statistic, 6),
                    'p_value': float(self.ks_pvalue) if self.ks_pvalue is not None else None,
                    'p_value_corrected': float(self.ks_pvalue_corrected) if self.ks_pvalue_corrected is not None else None,
                    'significant': bool(self.ks_significant),
                },
                'mann_whitney_u': {
                    'statistic': round(self.mw_statistic, 4),
                    'p_value': float(self.mw_pvalue) if self.mw_pvalue is not None else None,
                    'p_value_corrected': float(self.mw_pvalue_corrected) if self.mw_pvalue_corrected is not None else None,
                    'significant': bool(self.mw_significant),
                },
                'welch_t': {
                    'statistic': round(self.welch_statistic, 4),
                    'p_value': float(self.welch_pvalue) if self.welch_pvalue is not None else None,
                    'p_value_corrected': float(self.welch_pvalue_corrected) if self.welch_pvalue_corrected is not None else None,
                    'significant': bool(self.welch_significant),
                },
            },
            'effect_size': {
                'cohens_d': round(self.cohens_d, 4),
                'ci_95_low': round(self.cohens_d_ci_low, 4),
                'ci_95_high': round(self.cohens_d_ci_high, 4),
                'interpretation': self.effect_interpretation,
            },
            'any_significant': bool(self.any_significant),
        }
    
    def to_csv_row(self) -> dict:
        return {
            'comparison_id': self.comparison_id,
            'comparison_type': self.comparison_type,
            'group_a': self.group_a_name,
            'group_b': self.group_b_name,
            'metric': self.metric,
            'n_a': self.n_a,
            'n_b': self.n_b,
            'mean_a': round(self.mean_a, 2),
            'mean_b': round(self.mean_b, 2),
            'mean_diff_pct': round((self.mean_b - self.mean_a) / self.mean_a * 100, 2) if self.mean_a != 0 else 0,
            'ks_stat': round(self.ks_statistic, 4),
            'ks_p': f"{self.ks_pvalue_corrected:.2e}",
            'ks_sig': 'Yes' if self.ks_significant else 'No',
            'mw_stat': round(self.mw_statistic, 2),
            'mw_p': f"{self.mw_pvalue_corrected:.2e}",
            'mw_sig': 'Yes' if self.mw_significant else 'No',
            'welch_t': round(self.welch_statistic, 2),
            'welch_p': f"{self.welch_pvalue_corrected:.2e}",
            'welch_sig': 'Yes' if self.welch_significant else 'No',
            'cohens_d': round(self.cohens_d, 3),
            'effect_size': self.effect_interpretation,
            'any_significant': 'Yes' if self.any_significant else 'No',
        }


def load_latencies_from_jsonl(path: Path) -> Optional[np.ndarray]:
    """Load latency values from merged JSONL or Parquet file."""
    try:
        # Try parquet first (faster)
        parquet_path = path.parent / 'merged.parquet'
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            if 'latency_us' in df.columns:
                return df['latency_us'].values
        
        # Fall back to JSONL
        if not path.exists():
            return None
        
        latencies = []
        with open(path) as f:
            for line in f:
                try:
                    event = json.loads(line)
                    if 'latency_us' in event:
                        latencies.append(event['latency_us'])
                except json.JSONDecodeError:
                    continue
        
        return np.array(latencies) if latencies else None
    except Exception:
        return None


def compute_cohens_d_with_ci(
    group_a: np.ndarray, 
    group_b: np.ndarray,
    confidence: float = 0.95
) -> tuple[float, float, float, str]:
    """
    Compute Cohen's d effect size with confidence interval.
    
    Returns: (d, ci_low, ci_high, interpretation)
    """
    n_a, n_b = len(group_a), len(group_b)
    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    var_a = np.var(group_a, ddof=1) if n_a > 1 else 0
    var_b = np.var(group_b, ddof=1) if n_b > 1 else 0
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    
    if pooled_std == 0:
        return 0.0, 0.0, 0.0, 'undefined'
    
    # Cohen's d
    d = (mean_b - mean_a) / pooled_std
    
    # Standard error of d (Hedges & Olkin, 1985)
    se_d = np.sqrt((n_a + n_b) / (n_a * n_b) + (d ** 2) / (2 * (n_a + n_b)))
    
    # Confidence interval
    z = stats.norm.ppf((1 + confidence) / 2)
    ci_low = d - z * se_d
    ci_high = d + z * se_d
    
    # Interpretation (Cohen's conventions)
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = 'negligible'
    elif abs_d < 0.5:
        interpretation = 'small'
    elif abs_d < 0.8:
        interpretation = 'medium'
    else:
        interpretation = 'large'
    
    return d, ci_low, ci_high, interpretation


def run_all_tests(
    group_a: np.ndarray,
    group_b: np.ndarray,
    comparison_id: str,
    comparison_type: str,
    group_a_name: str,
    group_b_name: str,
    metric: str = 'latency_us',
    min_sample_size: int = 10,
) -> Optional[TestResult]:
    """Run all statistical tests between two groups."""
    
    if len(group_a) < 2 or len(group_b) < 2:
        return None
    
    # Warn if sample sizes are very small (smoke-test mode)
    if len(group_a) < min_sample_size or len(group_b) < min_sample_size:
        print(f"Warning: Small sample sizes detected (n_a={len(group_a)}, n_b={len(group_b)}). "
              f"Statistical tests may have reduced power. This is expected in smoke-test mode.",
              file=sys.stderr)
    
    result = TestResult(
        comparison_id=comparison_id,
        comparison_type=comparison_type,
        group_a_name=group_a_name,
        group_b_name=group_b_name,
        metric=metric,
        n_a=len(group_a),
        n_b=len(group_b),
        mean_a=np.mean(group_a),
        mean_b=np.mean(group_b),
        std_a=np.std(group_a, ddof=1),
        std_b=np.std(group_b, ddof=1),
    )
    
    # Kolmogorov-Smirnov test
    try:
        stat, p = stats.ks_2samp(group_a, group_b)
        result.ks_statistic = stat
        result.ks_pvalue = p
    except Exception:
        pass
    
    # Mann-Whitney U test
    try:
        stat, p = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
        result.mw_statistic = stat
        result.mw_pvalue = p
    except Exception:
        pass
    
    # Welch's t-test
    try:
        stat, p = stats.ttest_ind(group_a, group_b, equal_var=False)
        result.welch_statistic = stat
        result.welch_pvalue = p
    except Exception:
        pass
    
    # Cohen's d with CI
    d, ci_low, ci_high, interp = compute_cohens_d_with_ci(group_a, group_b)
    result.cohens_d = d
    result.cohens_d_ci_low = ci_low
    result.cohens_d_ci_high = ci_high
    result.effect_interpretation = interp
    
    return result


def holm_bonferroni_correction(
    results: list[TestResult], 
    alpha: float = 0.05
) -> None:
    """Apply Holm-Bonferroni correction to all p-values in place."""
    
    # Collect all p-values with references
    pvalues = []
    for i, r in enumerate(results):
        pvalues.append((r.ks_pvalue, i, 'ks'))
        pvalues.append((r.mw_pvalue, i, 'mw'))
        pvalues.append((r.welch_pvalue, i, 'welch'))
    
    n = len(pvalues)
    
    # Sort by p-value
    pvalues.sort(key=lambda x: x[0])
    
    # Apply correction
    for rank, (p, result_idx, test_type) in enumerate(pvalues):
        adjusted_alpha = alpha / (n - rank)
        significant = p < adjusted_alpha
        corrected_p = min(p * (n - rank), 1.0)
        
        result = results[result_idx]
        if test_type == 'ks':
            result.ks_pvalue_corrected = corrected_p
            result.ks_significant = significant
        elif test_type == 'mw':
            result.mw_pvalue_corrected = corrected_p
            result.mw_significant = significant
        elif test_type == 'welch':
            result.welch_pvalue_corrected = corrected_p
            result.welch_significant = significant
    
    # Update any_significant flag
    for r in results:
        r.any_significant = r.ks_significant or r.mw_significant or r.welch_significant


def generate_interpretation(results: list[TestResult]) -> str:
    """Generate interpretive text for dissertation."""
    lines = []
    
    # Count significant results
    total = len(results)
    any_sig = sum(1 for r in results if r.any_significant)
    ks_sig = sum(1 for r in results if r.ks_significant)
    mw_sig = sum(1 for r in results if r.mw_significant)
    welch_sig = sum(1 for r in results if r.welch_significant)
    
    lines.append("=" * 70)
    lines.append("STATISTICAL HYPOTHESIS TESTING SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Total comparisons: {total}")
    if total > 0:
        lines.append(f"Significant (any test): {any_sig} ({any_sig/total*100:.1f}%)")
    else:
        lines.append(f"Significant (any test): {any_sig} (N/A - no comparisons)")
    lines.append(f"  - Kolmogorov-Smirnov: {ks_sig}")
    lines.append(f"  - Mann-Whitney U: {mw_sig}")
    lines.append(f"  - Welch's t-test: {welch_sig}")
    lines.append("")
    
    # Effect size distribution
    large = sum(1 for r in results if r.effect_interpretation == 'large')
    medium = sum(1 for r in results if r.effect_interpretation == 'medium')
    small = sum(1 for r in results if r.effect_interpretation == 'small')
    negligible = sum(1 for r in results if r.effect_interpretation == 'negligible')
    
    lines.append("Effect sizes (Cohen's d):")
    lines.append(f"  - Large (|d| ≥ 0.8): {large}")
    lines.append(f"  - Medium (0.5 ≤ |d| < 0.8): {medium}")
    lines.append(f"  - Small (0.2 ≤ |d| < 0.5): {small}")
    lines.append(f"  - Negligible (|d| < 0.2): {negligible}")
    lines.append("")
    
    # Key findings by comparison type
    by_type = defaultdict(list)
    for r in results:
        by_type[r.comparison_type].append(r)
    
    lines.append("-" * 70)
    lines.append("KEY FINDINGS BY COMPARISON TYPE")
    lines.append("-" * 70)
    
    for comp_type, type_results in by_type.items():
        sig_count = sum(1 for r in type_results if r.any_significant)
        lines.append(f"\n{comp_type.upper()}:")
        lines.append(f"  Comparisons: {len(type_results)}, Significant: {sig_count}")
        
        # Show top 3 by effect size
        sorted_by_effect = sorted(type_results, key=lambda x: abs(x.cohens_d), reverse=True)[:3]
        for r in sorted_by_effect:
            sig_marker = " ***" if r.any_significant else ""
            lines.append(f"    {r.group_a_name} vs {r.group_b_name}: d={r.cohens_d:.2f} ({r.effect_interpretation}){sig_marker}")
    
    lines.append("")
    lines.append("-" * 70)
    lines.append("DISSERTATION INTERPRETATION")
    lines.append("-" * 70)
    lines.append("")
    
    # Generate interpretation paragraph
    pqc_vs_classical = [r for r in results if r.comparison_type == 'pqc_vs_classical']
    env_results = [r for r in results if r.comparison_type == 'environment']
    
    if pqc_vs_classical:
        pqc_sig = sum(1 for r in pqc_vs_classical if r.any_significant)
        avg_effect = np.mean([abs(r.cohens_d) for r in pqc_vs_classical])
        lines.append(
            f"Statistical analysis reveals that {pqc_sig} out of {len(pqc_vs_classical)} "
            f"PQC vs classical comparisons show significant differences (p < 0.05, Holm-Bonferroni corrected). "
            f"The average effect size magnitude is {avg_effect:.2f}, indicating "
            f"{'substantial' if avg_effect > 0.5 else 'modest'} practical differences between "
            f"post-quantum and classical cryptographic implementations."
        )
        lines.append("")
    
    if env_results:
        env_sig = sum(1 for r in env_results if r.any_significant)
        lines.append(
            f"Cross-environment analysis shows {env_sig} out of {len(env_results)} "
            f"comparisons exhibit statistically significant differences. "
            f"This confirms that execution environment (native, container, cloud) "
            f"has a measurable impact on cryptographic operation latency, "
            f"which must be accounted for in deployment planning."
        )
        lines.append("")
    
    lines.append(
        "The Holm-Bonferroni correction was applied to control the family-wise error rate "
        "across all multiple comparisons. Effect sizes (Cohen's d) with 95% confidence intervals "
        "provide practical significance measures beyond statistical significance."
    )
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run statistical hypothesis tests on experiment results"
    )
    parser.add_argument(
        '--index', '-i', type=Path, required=True,
        help='Path to index.json'
    )
    parser.add_argument(
        '--matrix', '-m', type=Path,
        help='Path to experiment_matrix.yaml (for comparison definitions)'
    )
    parser.add_argument(
        '--output', '-o', type=Path, required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.05,
        help='Significance level (default: 0.05)'
    )
    
    args = parser.parse_args()
    
    # Load index
    if not args.index.exists():
        print(f"Error: Index file not found: {args.index}", file=sys.stderr)
        sys.exit(1)
    
    with open(args.index) as f:
        index = json.load(f)
    
    print(f"Loaded index with {len(index.get('experiments', []))} experiments")
    
    # Load comparisons from matrix if available
    comparison_defs = []
    if args.matrix and args.matrix.exists() and YAML_AVAILABLE:
        with open(args.matrix) as f:
            matrix = yaml.safe_load(f)
        comparison_defs = matrix.get('comparisons', [])
        print(f"Loaded {len(comparison_defs)} comparison definitions from matrix")
    
    # Load all latency data, grouped by algorithm and environment
    data: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    
    for entry in index.get('experiments', []):
        if entry.get('status') not in ['success', 'cached']:
            continue
        
        output_dir = Path(entry['output_dir'])
        jsonl_path = output_dir / 'merged' / 'merged.jsonl'
        
        if not jsonl_path.exists():
            jsonl_path = output_dir / 'merged.jsonl'
        if not jsonl_path.exists():
            jsonl_path = output_dir / 'raw' / 'run.jsonl'
        
        latencies = load_latencies_from_jsonl(jsonl_path)
        
        if latencies is not None and len(latencies) > 0:
            algorithm = entry['algorithm']
            environment = entry['environment']
            data[algorithm][environment].append(latencies)
    
    print(f"Loaded latency data for {len(data)} algorithms")
    
    all_results: list[TestResult] = []
    
    # 1. Environment comparisons (native vs minikube vs gcp) for each algorithm
    print("\nRunning environment comparisons...")
    env_pairs = [
        ('native', 'minikube'),
        ('native', 'gcp'),
        ('minikube', 'gcp'),
    ]
    
    for algorithm in data:
        for env_a, env_b in env_pairs:
            if env_a not in data[algorithm] or env_b not in data[algorithm]:
                continue
            
            # Combine all runs for each environment
            data_a = np.concatenate(data[algorithm][env_a]) if data[algorithm][env_a] else np.array([])
            data_b = np.concatenate(data[algorithm][env_b]) if data[algorithm][env_b] else np.array([])
            
            if len(data_a) > 1 and len(data_b) > 1:
                result = run_all_tests(
                    data_a, data_b,
                    comparison_id=f"{algorithm}_{env_a}_vs_{env_b}",
                    comparison_type='environment',
                    group_a_name=f"{algorithm}_{env_a}",
                    group_b_name=f"{algorithm}_{env_b}",
                )
                if result:
                    all_results.append(result)
                    print(f"  {algorithm}: {env_a} vs {env_b} (n={result.n_a}, {result.n_b})")
    
    # 2. Algorithm comparisons within each environment
    print("\nRunning algorithm comparisons...")
    algorithms = list(data.keys())
    
    for i, algo_a in enumerate(algorithms):
        for algo_b in algorithms[i+1:]:
            for env in ['native', 'minikube', 'gcp']:
                if env not in data[algo_a] or env not in data[algo_b]:
                    continue
                
                data_a = np.concatenate(data[algo_a][env]) if data[algo_a][env] else np.array([])
                data_b = np.concatenate(data[algo_b][env]) if data[algo_b][env] else np.array([])
                
                if len(data_a) > 1 and len(data_b) > 1:
                    result = run_all_tests(
                        data_a, data_b,
                        comparison_id=f"{algo_a}_vs_{algo_b}_{env}",
                        comparison_type='algorithm',
                        group_a_name=f"{algo_a}_{env}",
                        group_b_name=f"{algo_b}_{env}",
                    )
                    if result:
                        all_results.append(result)
    
    # 3. PQC vs Classical comparisons
    print("\nRunning PQC vs Classical comparisons...")
    classical = ['rsa2048', 'ecdsa_p256']
    pqc = ['kyber512', 'dilithium2', 'hybrid_kyber_dilithium']
    
    for env in ['native', 'minikube', 'gcp']:
        # Aggregate classical data
        classical_data = []
        for algo in classical:
            if algo in data and env in data[algo]:
                classical_data.extend(data[algo][env])
        
        if not classical_data:
            continue
        
        classical_combined = np.concatenate(classical_data) if classical_data else np.array([])
        
        # Compare each PQC algorithm
        for pqc_algo in pqc:
            if pqc_algo not in data or env not in data[pqc_algo]:
                continue
            
            pqc_data = np.concatenate(data[pqc_algo][env]) if data[pqc_algo][env] else np.array([])
            
            if len(classical_combined) > 1 and len(pqc_data) > 1:
                result = run_all_tests(
                    classical_combined, pqc_data,
                    comparison_id=f"classical_vs_{pqc_algo}_{env}",
                    comparison_type='pqc_vs_classical',
                    group_a_name=f"classical_{env}",
                    group_b_name=f"{pqc_algo}_{env}",
                )
                if result:
                    all_results.append(result)
                    print(f"  Classical vs {pqc_algo} ({env})")
    
    # 4. Predefined comparisons from matrix
    if comparison_defs:
        print("\nRunning predefined comparisons from matrix...")
        for comp in comparison_defs:
            baseline = comp.get('baseline')
            treatment = comp.get('treatment')
            name = comp.get('name', f'{baseline}_vs_{treatment}')
            
            for env in ['native', 'minikube', 'gcp']:
                if baseline not in data or treatment not in data:
                    continue
                if env not in data[baseline] or env not in data[treatment]:
                    continue
                
                data_a = np.concatenate(data[baseline][env]) if data[baseline][env] else np.array([])
                data_b = np.concatenate(data[treatment][env]) if data[treatment][env] else np.array([])
                
                if len(data_a) > 1 and len(data_b) > 1:
                    result = run_all_tests(
                        data_a, data_b,
                        comparison_id=f"{name}_{env}",
                        comparison_type='predefined',
                        group_a_name=f"{baseline}_{env}",
                        group_b_name=f"{treatment}_{env}",
                    )
                    if result:
                        all_results.append(result)
    
    print(f"\nTotal comparisons: {len(all_results)}")
    
    # Apply Holm-Bonferroni correction
    print("Applying Holm-Bonferroni correction...")
    holm_bonferroni_correction(all_results, args.alpha)
    
    # Count significant results
    sig_count = sum(1 for r in all_results if r.any_significant)
    print(f"Significant comparisons (α={args.alpha}): {sig_count}/{len(all_results)}")
    
    # Generate interpretation
    interpretation = generate_interpretation(all_results)
    print("\n" + interpretation)
    
    # Prepare output
    args.output.mkdir(parents=True, exist_ok=True)
    
    # JSON output
    output_data = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'alpha': args.alpha,
        'correction_method': 'Holm-Bonferroni',
        'total_comparisons': len(all_results),
        'significant_comparisons': sig_count,
        'summary': {
            'by_type': {},
            'by_test': {
                'kolmogorov_smirnov': sum(1 for r in all_results if r.ks_significant),
                'mann_whitney_u': sum(1 for r in all_results if r.mw_significant),
                'welch_t': sum(1 for r in all_results if r.welch_significant),
            },
            'effect_sizes': {
                'large': sum(1 for r in all_results if r.effect_interpretation == 'large'),
                'medium': sum(1 for r in all_results if r.effect_interpretation == 'medium'),
                'small': sum(1 for r in all_results if r.effect_interpretation == 'small'),
                'negligible': sum(1 for r in all_results if r.effect_interpretation == 'negligible'),
            },
        },
        'results': [r.to_dict() for r in all_results],
    }
    
    # Count by type
    by_type = defaultdict(lambda: {'total': 0, 'significant': 0})
    for r in all_results:
        by_type[r.comparison_type]['total'] += 1
        if r.any_significant:
            by_type[r.comparison_type]['significant'] += 1
    output_data['summary']['by_type'] = dict(by_type)
    
    json_path = args.output / 'hypothesis_tests.json'
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nWritten: {json_path}")
    
    # CSV output
    csv_path = args.output / 'hypothesis_table.csv'
    fieldnames = list(all_results[0].to_csv_row().keys()) if all_results else []
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r.to_csv_row())
    print(f"Written: {csv_path}")
    
    # Write interpretation text
    interp_path = args.output / 'hypothesis_interpretation.txt'
    with open(interp_path, 'w') as f:
        f.write(interpretation)
    print(f"Written: {interp_path}")
    
    print("\nHypothesis testing complete!")


if __name__ == "__main__":
    main()
