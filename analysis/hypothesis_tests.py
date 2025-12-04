#!/usr/bin/env python3
"""
Perform statistical hypothesis testing on experiment results.

Runs:
- Mann-Whitney U test (distribution difference)
- Kolmogorov-Smirnov test (shape difference)
- Welch's t-test (mean difference)

Reports:
- p-values (with Holm-Bonferroni correction)
- Effect sizes (Cohen's d)
- Confidence intervals

Usage:
    python analysis/hypothesis_tests.py \
        --index final-results/index.json \
        --matrix orchestration/experiment_matrix.yaml \
        --output final-results/hypothesis_tests.json
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class HypothesisTestResult:
    """Result of a hypothesis test."""
    comparison: str
    group_a: str
    group_b: str
    metric: str
    test_name: str
    statistic: float
    p_value: float
    p_value_corrected: float
    significant: bool
    effect_size: float
    effect_interpretation: str
    ci_low: float
    ci_high: float
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    
    def to_dict(self) -> dict:
        return {
            'comparison': self.comparison,
            'group_a': self.group_a,
            'group_b': self.group_b,
            'metric': self.metric,
            'test_name': self.test_name,
            'statistic': round(self.statistic, 4),
            'p_value': self.p_value,
            'p_value_corrected': self.p_value_corrected,
            'significant': self.significant,
            'effect_size': round(self.effect_size, 4),
            'effect_interpretation': self.effect_interpretation,
            'ci_low': round(self.ci_low, 2),
            'ci_high': round(self.ci_high, 2),
            'n_a': self.n_a,
            'n_b': self.n_b,
            'mean_a': round(self.mean_a, 2),
            'mean_b': round(self.mean_b, 2),
        }


def load_latencies_from_jsonl(path: Path) -> Optional[np.ndarray]:
    """Load latency values from merged JSONL file."""
    try:
        parquet_path = path.parent / 'merged.parquet'
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            if 'latency_us' in df.columns:
                return df['latency_us'].values
        
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


def compute_cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> tuple[float, str]:
    """Compute Cohen's d effect size."""
    n_a, n_b = len(group_a), len(group_b)
    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)
    
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    
    if pooled_std == 0:
        return 0.0, 'undefined'
    
    d = (mean_b - mean_a) / pooled_std
    
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = 'negligible'
    elif abs_d < 0.5:
        interpretation = 'small'
    elif abs_d < 0.8:
        interpretation = 'medium'
    else:
        interpretation = 'large'
    
    return d, interpretation


def compute_ci_difference(group_a: np.ndarray, group_b: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Compute confidence interval for difference in means."""
    mean_diff = np.mean(group_b) - np.mean(group_a)
    
    se_a = stats.sem(group_a)
    se_b = stats.sem(group_b)
    se_diff = np.sqrt(se_a**2 + se_b**2)
    
    # Use Welch-Satterthwaite degrees of freedom
    df = (se_a**2 + se_b**2)**2 / (
        (se_a**4 / (len(group_a) - 1)) + (se_b**4 / (len(group_b) - 1))
    ) if (se_a > 0 or se_b > 0) else 1
    
    t_crit = stats.t.ppf((1 + confidence) / 2, df)
    margin = t_crit * se_diff
    
    return mean_diff - margin, mean_diff + margin


def holm_bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[tuple[float, bool]]:
    """Apply Holm-Bonferroni correction to p-values."""
    n = len(p_values)
    indexed = [(p, i) for i, p in enumerate(p_values)]
    indexed.sort()
    
    corrected = [None] * n
    
    for rank, (p, original_idx) in enumerate(indexed):
        adjusted_alpha = alpha / (n - rank)
        significant = p < adjusted_alpha
        corrected_p = min(p * (n - rank), 1.0)
        corrected[original_idx] = (corrected_p, significant)
    
    return corrected


def run_hypothesis_tests(
    group_a: np.ndarray,
    group_b: np.ndarray,
    comparison_name: str,
    group_a_name: str,
    group_b_name: str,
    metric: str = 'latency_us',
) -> list[HypothesisTestResult]:
    """Run all hypothesis tests between two groups."""
    results = []
    
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)
    effect_size, effect_interp = compute_cohens_d(group_a, group_b)
    ci_low, ci_high = compute_ci_difference(group_a, group_b)
    
    # Mann-Whitney U test
    try:
        stat, p = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
        results.append(HypothesisTestResult(
            comparison=comparison_name,
            group_a=group_a_name,
            group_b=group_b_name,
            metric=metric,
            test_name='Mann-Whitney U',
            statistic=stat,
            p_value=p,
            p_value_corrected=p,  # Will be corrected later
            significant=p < 0.05,
            effect_size=effect_size,
            effect_interpretation=effect_interp,
            ci_low=ci_low,
            ci_high=ci_high,
            n_a=len(group_a),
            n_b=len(group_b),
            mean_a=mean_a,
            mean_b=mean_b,
        ))
    except Exception as e:
        print(f"  Warning: Mann-Whitney U failed: {e}")
    
    # Kolmogorov-Smirnov test
    try:
        stat, p = stats.ks_2samp(group_a, group_b)
        results.append(HypothesisTestResult(
            comparison=comparison_name,
            group_a=group_a_name,
            group_b=group_b_name,
            metric=metric,
            test_name='Kolmogorov-Smirnov',
            statistic=stat,
            p_value=p,
            p_value_corrected=p,
            significant=p < 0.05,
            effect_size=effect_size,
            effect_interpretation=effect_interp,
            ci_low=ci_low,
            ci_high=ci_high,
            n_a=len(group_a),
            n_b=len(group_b),
            mean_a=mean_a,
            mean_b=mean_b,
        ))
    except Exception as e:
        print(f"  Warning: KS test failed: {e}")
    
    # Welch's t-test
    try:
        stat, p = stats.ttest_ind(group_a, group_b, equal_var=False)
        results.append(HypothesisTestResult(
            comparison=comparison_name,
            group_a=group_a_name,
            group_b=group_b_name,
            metric=metric,
            test_name="Welch's t-test",
            statistic=stat,
            p_value=p,
            p_value_corrected=p,
            significant=p < 0.05,
            effect_size=effect_size,
            effect_interpretation=effect_interp,
            ci_low=ci_low,
            ci_high=ci_high,
            n_a=len(group_a),
            n_b=len(group_b),
            mean_a=mean_a,
            mean_b=mean_b,
        ))
    except Exception as e:
        print(f"  Warning: Welch's t-test failed: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run hypothesis tests on experiment results")
    parser.add_argument('--index', '-i', type=Path, required=True, help='Path to index.json')
    parser.add_argument('--matrix', '-m', type=Path, help='Path to experiment_matrix.yaml')
    parser.add_argument('--output', '-o', type=Path, required=True, help='Output JSON file')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    
    args = parser.parse_args()
    
    # Load index
    if not args.index.exists():
        print(f"Error: Index file not found: {args.index}", file=sys.stderr)
        sys.exit(1)
    
    with open(args.index) as f:
        index = json.load(f)
    
    print(f"Loaded index with {len(index.get('experiments', []))} experiments")
    
    # Load comparisons from matrix if provided
    comparisons_config = []
    if args.matrix and args.matrix.exists() and yaml:
        with open(args.matrix) as f:
            matrix = yaml.safe_load(f)
        comparisons_config = matrix.get('comparisons', [])
        print(f"Loaded {len(comparisons_config)} comparison definitions")
    
    # Load all latency data
    data: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    
    for entry in index.get('experiments', []):
        if entry.get('status') not in ['success', 'cached']:
            continue
        
        output_dir = Path(entry['output_dir'])
        jsonl_path = output_dir / 'merged' / 'merged.jsonl'
        
        if not jsonl_path.exists():
            jsonl_path = output_dir / 'merged.jsonl'
        
        latencies = load_latencies_from_jsonl(jsonl_path)
        
        if latencies is not None and len(latencies) > 0:
            key = f"{entry['algorithm']}_{entry['environment']}"
            data[entry['algorithm']][entry['environment']].append(latencies)
    
    print(f"Loaded latency data for {len(data)} algorithms")
    
    all_results: list[HypothesisTestResult] = []
    
    # Run predefined comparisons
    print("\nRunning predefined comparisons...")
    for comp in comparisons_config:
        baseline_algo = comp.get('baseline')
        treatment_algo = comp.get('treatment')
        comp_name = comp.get('name', f'{baseline_algo}_vs_{treatment_algo}')
        
        for env in ['native', 'minikube', 'gcp']:
            if baseline_algo not in data or treatment_algo not in data:
                continue
            if env not in data[baseline_algo] or env not in data[treatment_algo]:
                continue
            
            baseline_data = np.concatenate(data[baseline_algo][env])
            treatment_data = np.concatenate(data[treatment_algo][env])
            
            if len(baseline_data) > 0 and len(treatment_data) > 0:
                print(f"  {comp_name} ({env})")
                results = run_hypothesis_tests(
                    baseline_data, treatment_data,
                    f"{comp_name}_{env}",
                    baseline_algo, treatment_algo,
                )
                all_results.extend(results)
    
    # Run environment comparisons
    print("\nRunning environment comparisons...")
    for algorithm in data:
        env_pairs = [
            ('native', 'minikube'),
            ('native', 'gcp'),
            ('minikube', 'gcp'),
        ]
        
        for env_a, env_b in env_pairs:
            if env_a not in data[algorithm] or env_b not in data[algorithm]:
                continue
            
            data_a = np.concatenate(data[algorithm][env_a])
            data_b = np.concatenate(data[algorithm][env_b])
            
            if len(data_a) > 0 and len(data_b) > 0:
                print(f"  {algorithm}: {env_a} vs {env_b}")
                results = run_hypothesis_tests(
                    data_a, data_b,
                    f"{algorithm}_{env_a}_vs_{env_b}",
                    f"{algorithm}_{env_a}",
                    f"{algorithm}_{env_b}",
                )
                all_results.extend(results)
    
    # Run PQC vs Classical comparisons
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
        
        classical_combined = np.concatenate(classical_data)
        
        # Compare each PQC algo
        for pqc_algo in pqc:
            if pqc_algo not in data or env not in data[pqc_algo]:
                continue
            
            pqc_data = np.concatenate(data[pqc_algo][env])
            
            if len(classical_combined) > 0 and len(pqc_data) > 0:
                print(f"  Classical vs {pqc_algo} ({env})")
                results = run_hypothesis_tests(
                    classical_combined, pqc_data,
                    f"classical_vs_{pqc_algo}_{env}",
                    f"classical_{env}",
                    f"{pqc_algo}_{env}",
                )
                all_results.extend(results)
    
    # Apply Holm-Bonferroni correction
    print("\nApplying Holm-Bonferroni correction...")
    p_values = [r.p_value for r in all_results]
    corrected = holm_bonferroni_correction(p_values, args.alpha)
    
    for i, (corrected_p, significant) in enumerate(corrected):
        all_results[i].p_value_corrected = corrected_p
        all_results[i].significant = significant
    
    # Count significant results
    significant_count = sum(1 for r in all_results if r.significant)
    print(f"\nSignificant results: {significant_count}/{len(all_results)} (α={args.alpha})")
    
    # Write output
    output = {
        'generated_at': index.get('generated_at', ''),
        'alpha': args.alpha,
        'total_tests': len(all_results),
        'significant_tests': significant_count,
        'correction_method': 'Holm-Bonferroni',
        'tests': [r.to_dict() for r in all_results],
        'summary': {
            'by_comparison': {},
            'by_test': {},
        },
    }
    
    # Summarize by comparison type
    by_comparison: dict = defaultdict(list)
    for r in all_results:
        by_comparison[r.comparison].append(r.to_dict())
    output['summary']['by_comparison'] = dict(by_comparison)
    
    # Summarize by test type
    by_test: dict = defaultdict(lambda: {'total': 0, 'significant': 0})
    for r in all_results:
        by_test[r.test_name]['total'] += 1
        if r.significant:
            by_test[r.test_name]['significant'] += 1
    output['summary']['by_test'] = dict(by_test)
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults written to: {args.output}")
    
    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    
    # Find most significant PQC vs Classical comparisons
    pqc_vs_classical = [r for r in all_results if 'classical_vs' in r.comparison and r.test_name == "Welch's t-test"]
    for r in sorted(pqc_vs_classical, key=lambda x: x.p_value)[:5]:
        sig_marker = "***" if r.significant else ""
        print(f"{r.comparison}: d={r.effect_size:.2f} ({r.effect_interpretation}), p={r.p_value_corrected:.4f} {sig_marker}")
    
    print("\nHypothesis testing complete!")


if __name__ == "__main__":
    main()

