from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class Workload:
	type: str
	concurrency: int
	chunk_size_bytes: int


@dataclass
class ExperimentConfig:
	comparison_mode: str
	repetitions: int
	warmup_seconds: int
	workload: Workload
	algorithms: Dict[str, Any]
	network_emulation: Dict[str, Any]
	storage_backend: Dict[str, Any]
	environment: Dict[str, Any]
	output: Dict[str, Any]


def load_experiment_config(path: str) -> ExperimentConfig:
	with open(path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f)
	bench = data.get("benchmark", {})
	workload = bench.get("workload", {})
	cfg = ExperimentConfig(
		comparison_mode=bench.get("comparison_mode", "pqc_vs_classical"),
		repetitions=int(bench.get("repetitions", 1)),
		warmup_seconds=int(bench.get("warmup_seconds", 0)),
		workload=Workload(
			type=str(workload.get("type", "kem_encapsulate")),
			concurrency=int(workload.get("concurrency", 1)),
			chunk_size_bytes=int(workload.get("chunk_size_bytes", 16384)),
		),
		algorithms=bench.get("algorithms", {}),
		network_emulation=bench.get("network_emulation", {}),
		storage_backend=data.get("storage_backend", {}),
		environment=data.get("environment", {}),
		output=data.get("output", {"directory": "./results", "format": "csv"}),
	)
	return cfg


