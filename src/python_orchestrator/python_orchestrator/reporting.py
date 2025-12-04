from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import nbformat

from .analysis import compute_summary, load_metrics_df
from .analysis import paired_comparisons
try:
	from .statistical_tests import compare_pqc_vs_classical, interpret_pvalue, interpret_effect_size
	HAVE_STATISTICAL_TESTS = True
except ImportError:
	HAVE_STATISTICAL_TESTS = False


def generate_charts(df: pd.DataFrame, out_dir: Path) -> List[Path]:
	out_dir.mkdir(parents=True, exist_ok=True)
	files: List[Path] = []
	if df.empty:
		return files
	sns.set_theme(style="whitegrid")
	# Latency CDF
	for op, gop in df.groupby("op"):
		plt.figure(figsize=(8, 5))
		for alg, g in gop.groupby("algorithm"):
			x = np.sort(g["latency_ms"].dropna().to_numpy())
			if x.size == 0:
				continue
			y = np.arange(1, x.size + 1) / x.size
			plt.plot(x, y, label=str(alg))
		plt.xlabel("Latency (ms)")
		plt.ylabel("CDF")
		plt.title(f"Latency CDF - {op}")
		plt.legend()
		path = out_dir / f"latency_cdf_{op}.png"
		plt.tight_layout()
		plt.savefig(path, dpi=150)
		plt.close()
		files.append(path)
	# Throughput
	plt.figure(figsize=(8, 5))
	sns.boxplot(data=df, x="algorithm", y="throughput_ops_per_s", hue="op")
	plt.ylabel("Throughput (ops/s)")
	plt.title("Throughput by Algorithm and Operation")
	plt.xticks(rotation=30, ha="right")
	plt.tight_layout()
	path = out_dir / "throughput_boxplot.png"
	plt.savefig(path, dpi=150)
	plt.close()
	files.append(path)
	# CPU/memory
	if "cpu_user_s" in df.columns or "max_rss_mb" in df.columns:
		plt.figure(figsize=(8, 5))
		sns.barplot(data=df.groupby("algorithm", as_index=False)["cpu_user_s"].mean(), x="algorithm", y="cpu_user_s")
		plt.ylabel("CPU user (s, mean)")
		plt.title("CPU user time")
		plt.xticks(rotation=30, ha="right")
		plt.tight_layout()
		path = out_dir / "cpu_user_mean.png"
		plt.savefig(path, dpi=150)
		plt.close()
		files.append(path)
		plt.figure(figsize=(8, 5))
		sns.barplot(data=df.groupby("algorithm", as_index=False)["cpu_system_s"].mean(), x="algorithm", y="cpu_system_s")
		plt.ylabel("CPU system (s, mean)")
		plt.title("CPU system time")
		plt.xticks(rotation=30, ha="right")
		plt.tight_layout()
		path = out_dir / "cpu_system_mean.png"
		plt.savefig(path, dpi=150)
		plt.close()
		files.append(path)
		plt.figure(figsize=(8, 5))
		sns.barplot(data=df.groupby("algorithm", as_index=False)["max_rss_mb"].mean(), x="algorithm", y="max_rss_mb")
		plt.ylabel("Max RSS (MB, mean)")
		plt.title("Memory usage")
		plt.xticks(rotation=30, ha="right")
		plt.tight_layout()
		path = out_dir / "memory_rss_mean.png"
		plt.savefig(path, dpi=150)
		plt.close()
		files.append(path)
	return files


def write_markdown_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
	if summary_df.empty:
		out_path.write_text("# Summary\n\nNo metrics available.\n", encoding="utf-8")
		return
	lines = ["# Benchmark Summary", ""]
	lines.append(summary_df.to_markdown(index=False))
	out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json_summary(summary_df: pd.DataFrame, out_path: Path, env_snapshot: dict | None = None) -> None:
	records = []
	for _, r in summary_df.iterrows():
		records.append({
			"algorithm": r.get("algorithm"),
			"parameter_set": r.get("op"),
			"operations": {
				"keygen_time_ms": None,
				"encapsulate_time_ms": None,
				"decrypt_time_ms": None,
				"encrypt_time_ms": None,
				"sign_time_ms": None,
				"verify_time_ms": None,
			},
			"sizes": {
				"public_key_bytes": None,
				"secret_key_bytes": None,
				"signature_bytes": None,
				"ciphertext_bytes": None,
				"storage_overhead_pct": None,
			},
			"performance": {
				"throughput_ops_per_sec": r.get("throughput_ops_per_s_mean"),
				"latency_p50": r.get("p50_ms"),
				"latency_p95": r.get("p95_ms"),
				"latency_p99": r.get("p99_ms"),
				"stddev": r.get("std_ms"),
				"CI_95": {
					"lower_ms": r.get("ci95_lower_ms"),
					"upper_ms": r.get("ci95_upper_ms"),
				},
				"p_value": None,
			},
			"resources": {
				"avg_cpu_percent": None,
				"avg_memory_mb": r.get("max_rss_mb_mean"),
				"disk_io_bytes": None,
				"net_tx_bytes": None,
				"net_rx_bytes": None,
			},
			"context": {
				"env_snapshot": env_snapshot,
			},
		})
	out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def write_notebook(results_dir: Path, summary_csv: Path, charts_dir: Path, out_path: Path) -> None:
	nb = nbformat.v4.new_notebook()
	nb.cells = [
		nbformat.v4.new_markdown_cell("# PQC Benchmark Analysis"),
		nbformat.v4.new_code_cell(
			"import pandas as pd\n"
			"import seaborn as sns\n"
			"import matplotlib.pyplot as plt\n"
			"summary = pd.read_csv(r'%s')\n"
			"display(summary.head())\n"
			"sns.set_theme(style='whitegrid')\n"
			"plt.figure(figsize=(8,4));\n"
			"sns.barplot(data=summary, x='algorithm', y='p50_ms', hue='op');\n"
			"plt.xticks(rotation=30, ha='right');\n"
			"plt.title('p50 latency by algorithm');\n"
			"plt.show();\n" % str(summary_csv)
		),
	]
	nbformat.write(nb, str(out_path))


def package_report(artifacts: List[Path], out_zip: Path) -> None:
	with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
		for p in artifacts:
			if p.exists():
				zf.write(p, arcname=p.name)


def run_analysis_and_report(results_dir: str) -> None:
	out_dir = Path(results_dir)
	df = load_metrics_df(results_dir)
	summary_df = compute_summary(df)
	# Write CSV/JSON summary
	summary_csv = out_dir / "summary.csv"
	summary_json = out_dir / "summary.json"
	summary_df.to_csv(summary_csv, index=False)
	# Read environment snapshot if present
	env_snapshot = None
	env_path = out_dir / "environment.json"
	if env_path.exists():
		try:
			env_snapshot = json.loads(env_path.read_text(encoding="utf-8"))
		except Exception:
			env_snapshot = None
	write_json_summary(summary_df, summary_json, env_snapshot)
	# Write aggregated_metrics.json alias
	aggregated_json = out_dir / "aggregated_metrics.json"
	try:
		aggregated_json.write_text(summary_json.read_text(encoding="utf-8"), encoding="utf-8")
	except Exception:
		pass
	# Write env.json alias
	if env_path.exists():
		try:
			(out_dir / "env.json").write_text(env_path.read_text(encoding="utf-8"), encoding="utf-8")
		except Exception:
			pass
	# Optional comparisons: if at least two algorithms per op, compute basic paired tests for the top two by count
	try:
		comparisons = []
		if not df.empty:
			for op, gop in df.groupby("op"):
				alg_counts = gop["algorithm"].value_counts()
				if len(alg_counts.index) >= 2:
					a, b = alg_counts.index[:2]
					stats = paired_comparisons(df, op=str(op), group_a=str(a), group_b=str(b))
					if stats:
						comparisons.append({
							"op": str(op),
							"group_a": str(a),
							"group_b": str(b),
							**stats,
						})
		if comparisons:
			(out_dir / "comparisons.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
	except Exception:
		pass
	# Charts
	charts_dir = out_dir / "charts"
	chart_files = generate_charts(df, charts_dir)
	
	# Statistical comparisons (PQC vs Classical)
	if HAVE_STATISTICAL_TESTS and not df.empty:
		try:
			stat_comparisons = compare_pqc_vs_classical(df, metric_column='latency_micros')
			if not stat_comparisons.empty:
				# Save detailed statistical comparisons
				stat_csv = out_dir / "statistical_comparisons.csv"
				stat_comparisons.to_csv(stat_csv, index=False)
				
				# Create human-readable report
				stat_report = out_dir / "statistical_report.md"
				with open(stat_report, 'w') as f:
					f.write("# Statistical Comparison Report\n")
					f.write("## PQC vs Classical Algorithms\n\n")
					
					for _, row in stat_comparisons.iterrows():
						f.write(f"### {row['algorithm_a']} vs {row['algorithm_b']} ({row['operation']})\n\n")
						f.write(f"**Sample sizes:** n_A={row['n_a']}, n_B={row['n_b']}\n\n")
						f.write(f"**Means:** {row['mean_a']:.6f} vs {row['mean_b']:.6f} µs\n\n")
						f.write(f"**Medians:** {row['median_a']:.6f} vs {row['median_b']:.6f} µs\n\n")
						f.write(f"**Faster algorithm:** {row.get('faster_algorithm', 'N/A')}\n\n")
						mean_diff = row.get('mean_difference', 0)
						pct_diff = row.get('percent_difference', 0) or 0
						f.write(f"**Mean difference:** {mean_diff:.6f} µs ({pct_diff:.2f}%)\n\n")
						
						f.write("**Statistical Tests:**\n")
						if pd.notna(row.get('ttest_pvalue')):
							f.write(f"- Independent t-test: t={row['ttest_statistic']:.4f}, p={row['ttest_pvalue']:.6f} {interpret_pvalue(row['ttest_pvalue'])}\n")
						if pd.notna(row.get('mannwhitneyu_pvalue')):
							f.write(f"- Mann-Whitney U: U={row['mannwhitneyu_statistic']:.4f}, p={row['mannwhitneyu_pvalue']:.6f} {interpret_pvalue(row['mannwhitneyu_pvalue'])}\n")
						
						f.write("\n**Effect Sizes:**\n")
						if pd.notna(row.get('cohens_d')):
							f.write(f"- Cohen's d: {row['cohens_d']:.4f} ({interpret_effect_size(row['cohens_d'])})\n")
						if pd.notna(row.get('rank_biserial')):
							f.write(f"- Rank-biserial r: {row['rank_biserial']:.4f}\n")
						
						f.write("\n---\n\n")
				
				chart_files.append(stat_csv)
				chart_files.append(stat_report)
		except Exception as e:
			print(f"Warning: Statistical comparisons failed: {e}")
	
	# Markdown
	md_path = out_dir / "summary.md"
	write_markdown_summary(summary_df, md_path)
	# Notebook
	nb_path = out_dir / "analysis.ipynb"
	write_notebook(out_dir, summary_csv, charts_dir, nb_path)
	# Package zip
	artifacts = [summary_csv, summary_json, md_path, nb_path] + chart_files
	report_zip = out_dir / "report.zip"
	package_report(artifacts, report_zip)


