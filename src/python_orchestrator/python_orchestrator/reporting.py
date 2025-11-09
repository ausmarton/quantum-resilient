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
	# Charts
	charts_dir = out_dir / "charts"
	chart_files = generate_charts(df, charts_dir)
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


