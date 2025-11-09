#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
import csv
import yaml


def load_jsonl(path: Path):
	events = []
	if not path.exists():
		return events
	for line in path.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line:
			continue
		try:
			events.append(json.loads(line))
		except Exception:
			continue
	return events


def validate_jsonl(events, schema):
	errors = []
	for i, ev in enumerate(events):
		for req in ["operation", "latency_micros"]:
			if req not in ev:
				errors.append({"index": i, "error": f"missing key: {req}"})
		if "latency_micros" in ev and not isinstance(ev["latency_micros"], (int, float)):
			errors.append({"index": i, "error": "latency_micros must be number"})
	return errors


def validate_csv_headers(csv_path: Path, required_headers):
	if not csv_path.exists():
		return [f"{csv_path} missing"]
	with csv_path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.reader(f)
		headers = next(reader, [])
		missing = [h for h in required_headers if h not in headers]
		return missing


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--results", default="results", help="Path to results directory")
	args = parser.parse_args()
	results = Path(args.results)
	jsonl = results / "metrics.jsonl"
	csv_path = results / "metrics.csv"
	agg = results / "aggregated_metrics.json"
	schema_file = Path("configs/metrics_schema.yaml")
	ok = True

	# JSONL structure
	events = load_jsonl(jsonl)
	schema = yaml.safe_load(schema_file.read_text(encoding="utf-8")) if schema_file.exists() else {}
	jsonl_errors = validate_jsonl(events, schema)
	if jsonl_errors:
		ok = False
		print("[FAIL] JSONL schema errors:", json.dumps(jsonl_errors[:5], indent=2))
	else:
		print("[OK] JSONL basic structure validated.")

	# CSV headers check (core set)
	required_headers = ["operation", "latency_micros"]
	missing = validate_csv_headers(csv_path, required_headers)
	if missing:
		ok = False
		print("[FAIL] CSV missing headers:", missing)
	else:
		print("[OK] CSV headers validated.")

	# Aggregated metrics presence and fields
	if agg.exists():
		try:
			data = json.loads(agg.read_text(encoding="utf-8"))
			def keys(d): return set(d[0].keys()) if isinstance(d, list) and d else set(d.keys())
			k = keys(data)
			for field in ["p50_ms", "p95_ms", "p99_ms", "mean_ms", "std_ms", "ci95_lower_ms", "ci95_upper_ms"]:
				if field not in k:
					ok = False
					print(f"[FAIL] aggregated_metrics.json missing: {field}")
			print("[OK] Aggregated metrics fields present.")
		except Exception as e:
			ok = False
			print("[FAIL] aggregated_metrics.json parse error:", e)
	else:
		ok = False
		print("[FAIL] aggregated_metrics.json missing.")

	# Basic numeric ranges (latency >= 0, throughput >= 0)
	if events:
		bad = [e for e in events if (e.get("latency_micros", 0) < 0) or (("throughput_ops_per_sec" in e) and (e["throughput_ops_per_sec"] is not None) and (e["throughput_ops_per_sec"] < 0))]
		if bad:
			ok = False
			print("[FAIL] Negative latency/throughput in events.")
		else:
			print("[OK] Basic numeric range checks passed.")

	# Disk/Network presence (best effort)
	if events and not any("disk_io_bytes" in e for e in events):
		print("[WARN] disk_io_bytes not present in metrics.jsonl")
	if events and not any("net_tx_bytes" in e for e in events):
		print("[WARN] net_tx_bytes not present in metrics.jsonl")

	if not ok:
		sys.exit(1)
	print("VALIDATION PASSED")


if __name__ == "__main__":
	main()


