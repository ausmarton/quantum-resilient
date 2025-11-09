from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class EnvSnapshot:
	timestamp_utc: str
	host: str
	os: str
	os_release: str
	arch: str
	python: str
	cpu_count: int
	cpu_model: str
	liboqs_version: str
	git_hash: str


def _cpu_model_linux() -> str:
	try:
		with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
			for line in f:
				if line.lower().startswith("model name"):
					return line.split(":", 1)[1].strip()
	except Exception:
		pass
	return platform.processor() or "unknown"


def _git_hash() -> str:
	try:
		return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
	except Exception:
		return "unknown"


def _liboqs_version() -> str:
	ver = os.environ.get("LIBOQS_VERSION")
	if ver:
		return ver
	try:
		import oqs  # type: ignore
		return getattr(oqs, "__version__", "unknown") or "unknown"
	except Exception:
		return "unknown"


def snapshot_env() -> EnvSnapshot:
	return EnvSnapshot(
		timestamp_utc=datetime.now(timezone.utc).isoformat(),
		host=socket.gethostname(),
		os=platform.system(),
		os_release=platform.release(),
		arch=platform.machine(),
		python=platform.python_version(),
		cpu_count=os.cpu_count() or 0,
		cpu_model=_cpu_model_linux(),
		liboqs_version=_liboqs_version(),
		git_hash=_git_hash(),
	)


def write_snapshot_json(path: str) -> None:
	s = snapshot_env()
	with open(path, "w", encoding="utf-8") as f:
		json.dump(asdict(s), f, indent=2)


