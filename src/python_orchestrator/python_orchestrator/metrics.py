from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any


def aggregate_jsonl_to_csv(jsonl_path: str, csv_path: str) -> None:
	in_path = Path(jsonl_path)
	out_path = Path(csv_path)
	rows: List[Dict[str, Any]] = []
	if not in_path.exists():
		return
	with in_path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				evt = json.loads(line)
			except Exception:
				continue
			rows.append(evt)
	if not rows:
		return
	# Flatten and write
	keys = sorted({k for row in rows for k in row.keys()})
	with out_path.open("w", newline="", encoding="utf-8") as f:
		w = csv.DictWriter(f, fieldnames=keys)
		w.writeheader()
		for r in rows:
			w.writerow(r)


def write_raw_events_csv(results_dir: str) -> None:
	"""Emit raw_events.csv from metrics.jsonl, preserving per-op timestamps and metrics."""
	in_path = Path(results_dir) / "metrics.jsonl"
	out_path = Path(results_dir) / "raw_events.csv"
	if not in_path.exists():
		return
	rows: List[Dict[str, Any]] = []
	with in_path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				evt = json.loads(line)
			except Exception:
				continue
			rows.append(evt)
	if not rows:
		return
	keys = sorted({k for row in rows for k in row.keys()})
	with out_path.open("w", newline="", encoding="utf-8") as f:
		w = csv.DictWriter(f, fieldnames=keys)
		w.writeheader()
		for r in rows:
			w.writerow(r)


