from __future__ import annotations

import json
from python_orchestrator.schema_validate import validate_metrics_jsonl


def test_validate_metrics_jsonl_empty(tmp_path):
	jsonl = tmp_path / "metrics.jsonl"
	jsonl.write_text("", encoding="utf-8")
	schema = tmp_path / "schema.yaml"
	schema.write_text("schema: metrics\nlabels: {algorithm: {type: string}}\nseries: []\n", encoding="utf-8")
	rep = validate_metrics_jsonl(str(jsonl), str(schema))
	assert rep["count"] == 0
	assert rep["errors"] == []


