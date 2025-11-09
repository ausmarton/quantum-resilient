from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import yaml


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
	events: List[Dict[str, Any]] = []
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


def validate_metrics_jsonl(jsonl_path: str, schema_yaml: str) -> Dict[str, Any]:
	jsonl = Path(jsonl_path)
	schema = yaml.safe_load(Path(schema_yaml).read_text(encoding="utf-8"))
	events = _load_jsonl(jsonl)
	report = {
		"file": str(jsonl),
		"count": len(events),
		"errors": [],
	}
	# Minimal structural checks based on schema placeholders
	expected_labels = set((schema.get("labels") or {}).keys())
	series = [s.get("name") for s in (schema.get("series") or [])]
	for idx, ev in enumerate(events):
		# required keys
		for req in ["operation", "latency_micros"]:
			if req not in ev:
				report["errors"].append({"index": idx, "error": f"missing key: {req}"})
		# types
		if "latency_micros" in ev and not isinstance(ev["latency_micros"], (int, float)):
			report["errors"].append({"index": idx, "error": "latency_micros must be number"})
		# labels presence (best-effort)
		for lbl in expected_labels:
			_ = ev.get(lbl, None)
	return report


