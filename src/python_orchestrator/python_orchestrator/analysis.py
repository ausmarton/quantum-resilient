from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SummaryRow:
	algorithm: str
	op: str
	n: int
	p50_ms: float
	p95_ms: float
	p99_ms: float
	mean_ms: float
	std_ms: float
	ci95_lower_ms: float
	ci95_upper_ms: float
	throughput_ops_per_s_mean: float
	cpu_user_s_mean: float
	cpu_system_s_mean: float
	max_rss_mb_mean: float


def _ci95(mean: float, std: float, n: int) -> Tuple[float, float]:
	if n <= 1 or std <= 0:
		return (mean, mean)
	se = std / np.sqrt(n)
	t_crit = stats.t.ppf(0.975, df=n - 1)
	return (mean - t_crit * se, mean + t_crit * se)


def load_metrics_df(results_dir: str) -> pd.DataFrame:
	jsonl = Path(results_dir) / "metrics.jsonl"
	csv = Path(results_dir) / "metrics.csv"
	if jsonl.exists():
		records = []
		with jsonl.open("r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				try:
					records.append(json.loads(line))
				except Exception:
					continue
		df = pd.DataFrame.from_records(records)
	elif csv.exists():
		df = pd.read_csv(csv)
	else:
		return pd.DataFrame()
	# Normalizations
	if "latency_micros" in df.columns:
		df["latency_ms"] = df["latency_micros"].astype(float) / 1000.0
		df["latency_s"] = df["latency_micros"].astype(float) / 1_000_000.0
	else:
		df["latency_ms"] = np.nan
		df["latency_s"] = np.nan
	df["throughput_ops_per_s"] = np.where(df["latency_s"] > 0, 1.0 / df["latency_s"], np.nan)
	if "cpu_user_micros" in df.columns:
		df["cpu_user_s"] = df["cpu_user_micros"].astype(float) / 1_000_000.0
	else:
		df["cpu_user_s"] = np.nan
	if "cpu_system_micros" in df.columns:
		df["cpu_system_s"] = df["cpu_system_micros"].astype(float) / 1_000_000.0
	else:
		df["cpu_system_s"] = np.nan
	if "max_rss_bytes" in df.columns:
		df["max_rss_mb"] = df["max_rss_bytes"].astype(float) / (1024.0 * 1024.0)
	else:
		df["max_rss_mb"] = np.nan
	# Expect algorithm/op labels if present
	if "algorithm" not in df.columns:
		df["algorithm"] = "unknown"
	if "operation" in df.columns:
		df["op"] = df["operation"]
	else:
		df["op"] = "unknown"
	return df


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
	if df.empty:
		return df
	group_cols = ["algorithm", "op"]
	summaries: List[SummaryRow] = []
	for (alg, op), g in df.groupby(group_cols):
		lat_ms = g["latency_ms"].dropna().to_numpy()
		thr = g["throughput_ops_per_s"].dropna().to_numpy()
		cpu_u = g["cpu_user_s"].dropna().to_numpy()
		cpu_s = g["cpu_system_s"].dropna().to_numpy()
		rss = g["max_rss_mb"].dropna().to_numpy()
		n = int(lat_ms.size)
		if n == 0:
			continue
		p50 = float(np.percentile(lat_ms, 50))
		p95 = float(np.percentile(lat_ms, 95))
		p99 = float(np.percentile(lat_ms, 99))
		mean = float(np.mean(lat_ms))
		std = float(np.std(lat_ms, ddof=1)) if n > 1 else 0.0
		ci_lo, ci_hi = _ci95(mean, std, n)
		row = SummaryRow(
			algorithm=str(alg),
			op=str(op),
			n=n,
			p50_ms=p50,
			p95_ms=p95,
			p99_ms=p99,
			mean_ms=mean,
			std_ms=std,
			ci95_lower_ms=float(ci_lo),
			ci95_upper_ms=float(ci_hi),
			throughput_ops_per_s_mean=float(np.mean(thr)) if thr.size else float("nan"),
			cpu_user_s_mean=float(np.mean(cpu_u)) if cpu_u.size else float("nan"),
			cpu_system_s_mean=float(np.mean(cpu_s)) if cpu_s.size else float("nan"),
			max_rss_mb_mean=float(np.mean(rss)) if rss.size else float("nan"),
		)
		summaries.append(row)
	return pd.DataFrame([asdict(r) for r in summaries])


def _paired_index_join(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
	if "pair_id" in a.columns and "pair_id" in b.columns:
		merged = pd.merge(a[["pair_id", "latency_ms"]], b[["pair_id", "latency_ms"]], on="pair_id", how="inner", suffixes=("_a", "_b"))
		return merged["latency_ms_a"].to_numpy(), merged["latency_ms_b"].to_numpy()
	# fallback: align by index length
	n = min(len(a), len(b))
	return a["latency_ms"].to_numpy()[:n], b["latency_ms"].to_numpy()[:n]


def paired_comparisons(df: pd.DataFrame, op: str, group_a: str, group_b: str) -> Dict[str, float]:
	ga = df[(df["op"] == op) & (df["algorithm"] == group_a)]
	gb = df[(df["op"] == op) & (df["algorithm"] == group_b)]
	if ga.empty or gb.empty:
		return {}
	x, y = _paired_index_join(ga, gb)
	if x.size == 0 or y.size == 0:
		return {}
	diff = x - y
	# Paired t-test
	tt = stats.ttest_rel(x, y, nan_policy="omit")
	# Wilcoxon (requires non-zero diffs)
	try:
		wil = stats.wilcoxon(x, y)
		wil_p = float(wil.pvalue)
	except Exception:
		wil_p = float("nan")
	# Effect sizes
	cohen_dz = float(np.mean(diff) / np.std(diff, ddof=1)) if diff.size > 1 and np.std(diff, ddof=1) > 0 else float("nan")
	# rank-biserial correlation approximation from Wilcoxon z if available
	try:
		# scipy wilcoxon returns statistic, not z; approximate r via t result as fallback
		r_effect = float(tt.statistic / np.sqrt(x.size)) if x.size > 0 and np.isfinite(tt.statistic) else float("nan")
	except Exception:
		r_effect = float("nan")
	return {
		"paired_t_pvalue": float(tt.pvalue) if np.isfinite(tt.pvalue) else float("nan"),
		"wilcoxon_pvalue": wil_p,
		"cohen_dz": cohen_dz,
		"rank_biserial_r_approx": r_effect,
	}


