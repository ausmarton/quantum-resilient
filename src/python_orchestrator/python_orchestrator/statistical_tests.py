"""Statistical tests for comparing PQC vs classical algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ComparisonResult:
	"""Results of statistical comparison between two algorithms."""
	algorithm_a: str
	algorithm_b: str
	operation: str
	n_a: int
	n_b: int
	mean_a: float
	mean_b: float
	median_a: float
	median_b: float
	std_a: float
	std_b: float
	
	# Parametric tests
	ttest_statistic: Optional[float] = None
	ttest_pvalue: Optional[float] = None
	paired_ttest_statistic: Optional[float] = None
	paired_ttest_pvalue: Optional[float] = None
	
	# Non-parametric tests
	mannwhitneyu_statistic: Optional[float] = None
	mannwhitneyu_pvalue: Optional[float] = None
	wilcoxon_statistic: Optional[float] = None
	wilcoxon_pvalue: Optional[float] = None
	
	# Effect sizes
	cohens_d: Optional[float] = None
	cohens_dz: Optional[float] = None  # For paired data
	rank_biserial: Optional[float] = None
	
	# Difference metrics
	mean_difference: Optional[float] = None
	percent_difference: Optional[float] = None
	faster_algorithm: Optional[str] = None


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
	"""
	Calculate Cohen's d for independent samples.
	"""
	n1, n2 = len(group1), len(group2)
	var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
	pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
	
	if pooled_std == 0:
		return 0.0
	
	return (np.mean(group1) - np.mean(group2)) / pooled_std


def cohens_dz_paired(diff: np.ndarray) -> float:
	"""
	Calculate Cohen's dz for paired samples.
	dz = mean(differences) / std(differences)
	"""
	if len(diff) == 0 or np.std(diff, ddof=1) == 0:
		return 0.0
	
	return np.mean(diff) / np.std(diff, ddof=1)


def rank_biserial_r(group1: np.ndarray, group2: np.ndarray) -> float:
	"""
	Calculate rank-biserial correlation (effect size for Mann-Whitney U).
	r = 1 - (2U)/(n1*n2)
	"""
	try:
		u_stat, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')
		n1, n2 = len(group1), len(group2)
		r = 1 - (2 * u_stat) / (n1 * n2)
		return r
	except Exception:
		return 0.0


def compare_algorithms(
	df: pd.DataFrame,
	algorithm_a: str,
	algorithm_b: str,
	operation: str,
	metric_column: str = 'latency_micros',
	paired: bool = False,
) -> Optional[ComparisonResult]:
	"""
	Compare two algorithms for a specific operation.
	
	Args:
		df: DataFrame with metrics
		algorithm_a: Name of first algorithm (e.g., "Kyber512")
		algorithm_b: Name of second algorithm (e.g., "RSA-2048")
		operation: Operation to compare (e.g., "Keygen", "Encapsulate")
		metric_column: Column to compare (default: latency_micros)
		paired: Whether to use paired tests (default: False)
	
	Returns:
		ComparisonResult with test statistics and effect sizes
	"""
	# Filter data
	data_a = df[(df['algorithm'] == algorithm_a) & (df['operation'] == operation)][metric_column].dropna()
	data_b = df[(df['algorithm'] == algorithm_b) & (df['operation'] == operation)][metric_column].dropna()
	
	if len(data_a) == 0 or len(data_b) == 0:
		return None
	
	result = ComparisonResult(
		algorithm_a=algorithm_a,
		algorithm_b=algorithm_b,
		operation=operation,
		n_a=len(data_a),
		n_b=len(data_b),
		mean_a=float(np.mean(data_a)),
		mean_b=float(np.mean(data_b)),
		median_a=float(np.median(data_a)),
		median_b=float(np.median(data_b)),
		std_a=float(np.std(data_a, ddof=1)),
		std_b=float(np.std(data_b, ddof=1)),
	)
	
	# Mean difference
	result.mean_difference = result.mean_a - result.mean_b
	if result.mean_b != 0:
		result.percent_difference = (result.mean_difference / result.mean_b) * 100
	result.faster_algorithm = algorithm_a if result.mean_a < result.mean_b else algorithm_b
	
	# Parametric tests
	if paired and len(data_a) == len(data_b):
		# Paired t-test
		try:
			stat, pval = stats.ttest_rel(data_a, data_b)
			result.paired_ttest_statistic = float(stat)
			result.paired_ttest_pvalue = float(pval)
			
			# Cohen's dz for paired
			diff = np.array(data_a) - np.array(data_b)
			result.cohens_dz = cohens_dz_paired(diff)
		except Exception as e:
			print(f"Warning: Paired t-test failed for {algorithm_a} vs {algorithm_b}: {e}")
	else:
		# Independent t-test
		try:
			stat, pval = stats.ttest_ind(data_a, data_b)
			result.ttest_statistic = float(stat)
			result.ttest_pvalue = float(pval)
			
			# Cohen's d for independent samples
			result.cohens_d = cohens_d(np.array(data_a), np.array(data_b))
		except Exception as e:
			print(f"Warning: Independent t-test failed for {algorithm_a} vs {algorithm_b}: {e}")
	
	# Non-parametric tests
	if paired and len(data_a) == len(data_b):
		# Wilcoxon signed-rank test
		try:
			stat, pval = stats.wilcoxon(data_a, data_b)
			result.wilcoxon_statistic = float(stat)
			result.wilcoxon_pvalue = float(pval)
		except Exception as e:
			print(f"Warning: Wilcoxon test failed for {algorithm_a} vs {algorithm_b}: {e}")
	else:
		# Mann-Whitney U test
		try:
			stat, pval = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
			result.mannwhitneyu_statistic = float(stat)
			result.mannwhitneyu_pvalue = float(pval)
			
			# Rank-biserial correlation
			result.rank_biserial = rank_biserial_r(np.array(data_a), np.array(data_b))
		except Exception as e:
			print(f"Warning: Mann-Whitney U test failed for {algorithm_a} vs {algorithm_b}: {e}")
	
	return result


def compare_pqc_vs_classical(df: pd.DataFrame, metric_column: str = 'latency_micros') -> pd.DataFrame:
	"""
	Compare all PQC algorithms against their classical counterparts.
	
	Returns DataFrame with comparison results.
	"""
	comparisons = []
	
	# KEM comparisons
	kem_pairs = [
		("Kyber512", "RSA-2048", "Keygen"),
		("Kyber512", "RSA-2048", "Encapsulate"),
		("Kyber768", "RSA-2048", "Keygen"),
		("Kyber768", "RSA-2048", "Encapsulate"),
		("Kyber512", "ECDHE-P256", "Keygen"),
		("Kyber768", "ECDHE-P256", "Keygen"),
	]
	
	# Signature comparisons
	sig_pairs = [
		("Dilithium2", "ECDSA-P256", "Sign"),
		("Dilithium2", "ECDSA-P256", "Verify"),
		("Dilithium3", "ECDSA-P256", "Sign"),
		("Dilithium3", "ECDSA-P256", "Verify"),
		("Dilithium2", "RSA-2048", "Sign"),
		("Dilithium3", "RSA-2048", "Sign"),
	]
	
	all_pairs = kem_pairs + sig_pairs
	
	for pqc_alg, classical_alg, operation in all_pairs:
		result = compare_algorithms(df, pqc_alg, classical_alg, operation, metric_column)
		if result:
			comparisons.append(result)
	
	if not comparisons:
		return pd.DataFrame()
	
	# Convert to DataFrame
	return pd.DataFrame([vars(c) for c in comparisons])


def interpret_pvalue(pvalue: float) -> str:
	"""Interpret p-value with standard thresholds."""
	if pvalue < 0.001:
		return "***  (p < 0.001, highly significant)"
	elif pvalue < 0.01:
		return "**   (p < 0.01, very significant)"
	elif pvalue < 0.05:
		return "*    (p < 0.05, significant)"
	elif pvalue < 0.10:
		return ".    (p < 0.10, marginally significant)"
	else:
		return "ns   (not significant)"


def interpret_effect_size(cohens_d: float) -> str:
	"""Interpret Cohen's d effect size."""
	abs_d = abs(cohens_d)
	if abs_d < 0.2:
		return "negligible"
	elif abs_d < 0.5:
		return "small"
	elif abs_d < 0.8:
		return "medium"
	else:
		return "large"

