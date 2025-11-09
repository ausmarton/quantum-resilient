#!/usr/bin/env bash
set -euo pipefail

usage() {
	echo "Usage: $0 --config <experiment.yaml> [--mode local|gcp|both]" >&2
	exit 1
}

CONFIG=""
MODE="both"
while [[ $# -gt 0 ]]; do
	case "$1" in
		-c|--config) CONFIG="${2:-}"; shift 2;;
		-m|--mode) MODE="${2:-}"; shift 2;;
		-h|--help) usage;;
		*) echo "Unknown arg: $1" >&2; usage;;
	esac
done
[[ -z "${CONFIG}" ]] && usage
if [[ ! -f "${CONFIG}" ]]; then
	echo "Config not found: ${CONFIG}" >&2
	exit 2
fi

# Determinism toggles (local and in-container when propagated)
export TZ=UTC
export LC_ALL=C
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

timestamp() { date -u +"%Y%m%dT%H%M%SZ"; }
sha256() { sha256sum | awk '{print $1}'; }

canon_json_sha() {
	python3 - <<'PY'
import io, json, sys, hashlib
path = sys.stdin.read().strip()
with open(path, 'r', encoding='utf-8') as f:
	obj = json.load(f)
data = json.dumps(obj, sort_keys=True, separators=(',', ':')).encode('utf-8')
print(hashlib.sha256(data).hexdigest())
PY
}

prepare_config_with_output() {
	local in_cfg="$1"
	local out_dir="$2"
	local tmp_cfg="$3"
	python3 - "$in_cfg" "$out_dir" "$tmp_cfg" <<'PY'
import sys, yaml, os
src, out_dir, dst = sys.argv[1:]
with open(src, 'r', encoding='utf-8') as f:
	cfg = yaml.safe_load(f)
cfg = cfg or {}
cfg.setdefault('output', {})
cfg['output']['directory'] = out_dir
os.makedirs(out_dir, exist_ok=True)
with open(dst, 'w', encoding='utf-8') as f:
	yaml.safe_dump(cfg, f, sort_keys=False)
print(dst)
PY
}

run_local_once() {
	local cfg="$1"
	if command -v pqc-orchestrator >/dev/null 2>&1; then
		pqc-orchestrator --config "$cfg"
	else
		python3 -m python_orchestrator.cli --config "$cfg"
	fi
}

compare_runs() {
	local dir_a="$1"
	local dir_b="$2"
	local fail=0
	echo "Comparing environment snapshots..."
	if [[ -f "${dir_a}/environment.json" && -f "${dir_b}/environment.json" ]]; then
		local sha_a sha_b
		sha_a="$(printf "%s" "${dir_a}/environment.json" | canon_json_sha)"
		sha_b="$(printf "%s" "${dir_b}/environment.json" | canon_json_sha)"
		if [[ "${sha_a}" != "${sha_b}" ]]; then
			echo "Environment mismatch between ${dir_a} and ${dir_b}" >&2
			fail=1
		fi
	else
		echo "Warning: environment.json missing; skipping environment comparison." >&2
	fi
	echo "Comparing metrics and reports..."
	for f in metrics.jsonl metrics.csv summary.csv summary.json summary.md analysis.ipynb; do
		if [[ -f "${dir_a}/${f}" && -f "${dir_b}/${f}" ]]; then
			local s1 s2
			s1="$(sha256sum "${dir_a}/${f}" | awk '{print $1}')"
			s2="$(sha256sum "${dir_b}/${f}" | awk '{print $1}')"
			if [[ "${s1}" != "${s2}" ]]; then
				echo "Mismatch: ${f}" >&2
				fail=1
			fi
		fi
	done
	# Compare charts by checksum if both exist
	if [[ -d "${dir_a}/charts" && -d "${dir_b}/charts" ]]; then
		for img in $(ls "${dir_a}/charts" 2>/dev/null || true); do
			if [[ -f "${dir_b}/charts/${img}" ]]; then
				local s1 s2
				s1="$(sha256sum "${dir_a}/charts/${img}" | awk '{print $1}')"
				s2="$(sha256sum "${dir_b}/charts/${img}" | awk '{print $1}')"
				if [[ "${s1}" != "${s2}" ]]; then
					echo "Mismatch chart: ${img}" >&2
					fail=1
				fi
			fi
		done
	fi
	return "${fail}"
}

do_local() {
	echo "=== Reproducing locally ==="
	local base="results/repro/local"
	local t1="$(timestamp)"
	local t2="$(timestamp)_b"
	local out1="${base}/${t1}"
	local out2="${base}/${t2}"
	mkdir -p "${out1}" "${out2}"
	local cfg1="${out1}/experiment.yaml"
	local cfg2="${out2}/experiment.yaml"
	prepare_config_with_output "${CONFIG}" "${out1}" "${cfg1}" >/dev/null
	prepare_config_with_output "${CONFIG}" "${out2}" "${cfg2}" >/dev/null
	run_local_once "${cfg1}"
	run_local_once "${cfg2}"
	compare_runs "${out1}" "${out2}"
}

do_gcp() {
	echo "=== Reproducing on GCP ==="
	local base="results/repro/gcp"
	local t1="$(timestamp)"
	local t2="$(timestamp)_b"
	local out1="${base}/${t1}"
	local out2="${base}/${t2}"
	mkdir -p "${out1}" "${out2}"
	local cfg1="${out1}/experiment.yaml"
	local cfg2="${out2}/experiment.yaml"
	prepare_config_with_output "${CONFIG}" "${out1}" "${cfg1}" >/dev/null
	prepare_config_with_output "${CONFIG}" "${out2}" "${cfg2}" >/dev/null
	# Delegate to project script (should run same YAML remotely and fetch results back into outX)
	if [[ -x "./scripts/run_gcp.sh" ]]; then
		./scripts/run_gcp.sh -c "${cfg1}" || true
		./scripts/run_gcp.sh -c "${cfg2}" || true
	else
		echo "scripts/run_gcp.sh not found or not executable; skipping GCP run." >&2
	fi
	compare_runs "${out1}" "${out2}"
}

case "${MODE}" in
	local) do_local;;
	gcp) do_gcp;;
	both) do_local; do_gcp;;
	*) echo "Invalid mode: ${MODE}" >&2; exit 3;;
esac

echo "Reproduction checks completed."


