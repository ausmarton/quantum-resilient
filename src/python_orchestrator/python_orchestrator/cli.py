from __future__ import annotations

import argparse
import sys
from .runner import run_experiment


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="PQC Benchmark Orchestrator")
	parser.add_argument("-c", "--config", required=True, help="Path to experiment YAML")
	args = parser.parse_args(argv)
	run_experiment(args.config)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())


