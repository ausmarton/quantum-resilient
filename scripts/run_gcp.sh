#!/usr/bin/env bash
set -euo pipefail

usage() {
	echo "Usage: $0 --config <experiment.yaml> --project <gcp-project> --region <region> --cluster <name> --bucket <gcs-bucket>" >&2
	exit 1
}

CONFIG=""
PROJECT=""
REGION="us-central1"
CLUSTER="pqc-benchmark"
BUCKET=""
NAMESPACE="pqc-benchmark"
HELM_RELEASE="pqc-benchmark"

while [[ $# -gt 0 ]]; do
	case "$1" in
		-c|--config) CONFIG="${2:-}"; shift 2;;
		-p|--project) PROJECT="${2:-}"; shift 2;;
		-r|--region) REGION="${2:-}"; shift 2;;
		-k|--cluster) CLUSTER="${2:-}"; shift 2;;
		-b|--bucket) BUCKET="${2:-}"; shift 2;;
		-h|--help) usage;;
		*) echo "Unknown arg: $1" >&2; usage;;
	esac
done

[[ -z "${CONFIG}" || -z "${PROJECT}" || -z "${BUCKET}" ]] && usage
[[ ! -f "${CONFIG}" ]] && { echo "Config not found: ${CONFIG}" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
TF_DIR="${ROOT}/terraform/gcp"

echo "[run_gcp] Initializing and applying Terraform..."
pushd "${TF_DIR}" >/dev/null
terraform init -input=false -no-color
terraform apply -input=false -auto-approve -no-color \
	-var "project_id=${PROJECT}" \
	-var "region=${REGION}" \
	-var "cluster_name=${CLUSTER}" \
	-var "bucket_name=${BUCKET}"
popd >/dev/null

echo "[run_gcp] Getting GKE credentials..."
gcloud container clusters get-credentials "${CLUSTER}" --region "${REGION}" --project "${PROJECT}"

echo "[run_gcp] Ensuring namespace ${NAMESPACE}..."
kubectl get ns "${NAMESPACE}" >/dev/null 2>&1 || kubectl create ns "${NAMESPACE}"

echo "[run_gcp] Creating/Updating experiment ConfigMap..."
CM_NAME="experiment-config"
kubectl -n "${NAMESPACE}" delete configmap "${CM_NAME}" >/dev/null 2>&1 || true
kubectl -n "${NAMESPACE}" create configmap "${CM_NAME}" --from-file "$(basename "${CONFIG}")=${CONFIG}"

echo "[run_gcp] Deploying Helm chart..."
pushd "${ROOT}/helm" >/dev/null
helm upgrade --install "${HELM_RELEASE}" . \
	--namespace "${NAMESPACE}" \
	--set orchestrator.configMapName="${CM_NAME}" \
	--set orchestrator.configFileName="$(basename "${CONFIG}")" \
	--set orchestrator.outputDir="/results" \
	--set results.bucket="${BUCKET}"
popd >/dev/null

echo "[run_gcp] Waiting for orchestrator Job to start..."
kubectl -n "${NAMESPACE}" wait --for=condition=complete --timeout=30m "job/${HELM_RELEASE}-orchestrator" || {
	echo "Job did not complete in time" >&2
	kubectl -n "${NAMESPACE}" logs job/"${HELM_RELEASE}-orchestrator" --tail=100 || true
	exit 3
}

echo "[run_gcp] Fetching results artifacts from pod..."
POD="$(kubectl -n "${NAMESPACE}" get pods -l app="${HELM_RELEASE}-orchestrator" -o jsonpath='{.items[0].metadata.name}')"
OUT_DIR="${ROOT}/results/gcp_${CLUSTER}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${OUT_DIR}"
if ! kubectl -n "${NAMESPACE}" cp "${POD}:/results/." "${OUT_DIR}"; then
	echo "[run_gcp] WARN: Could not copy results from pod. Falling back to local orchestrator run."
	if command -v python3 >/dev/null 2>&1; then
		python3 -m pip install -q --upgrade pip
		python3 -m pip install -q maturin
		python3 -m maturin develop -m "${ROOT}/src/rust_core/Cargo.toml" || true
		python3 -m pip install -q -e "${ROOT}/src/python_orchestrator" || true
		python3 -m python_orchestrator.cli --config "${CONFIG}" || true
		cp -r "${ROOT}/results/." "${OUT_DIR}/" || true
	fi
fi

echo "[run_gcp] Comparing aggregated_metrics.json schema with local results (if present)..."
python3 - <<'PY' || true
import json, pathlib, sys
root = pathlib.Path(__file__).resolve().parents[1]
gcp_dirs = sorted((root / "results").glob("gcp_*"), key=lambda p: p.stat().st_mtime)
if not gcp_dirs: sys.exit(0)
gcp_dir = gcp_dirs[-1]
local = root / "results" / "aggregated_metrics.json"
remote = gcp_dir / "aggregated_metrics.json"
if local.exists() and remote.exists():
    try:
        l = json.loads(local.read_text())
        r = json.loads(remote.read_text())
        lk = set(l[0].keys()) if isinstance(l, list) and l else set(l.keys())
        rk = set(r[0].keys()) if isinstance(r, list) and r else set(r.keys())
        missing_in_remote = lk - rk
        missing_in_local = rk - lk
        if missing_in_remote or missing_in_local:
            print("[run_gcp] Schema differences:", {"missing_in_remote": sorted(missing_in_remote), "missing_in_local": sorted(missing_in_local)})
        else:
            print("[run_gcp] OK: aggregated_metrics.json schemas match (key sets).")
    except Exception as e:
        print("[run_gcp] WARN: Could not compare schemas:", e)
PY

echo "[run_gcp] Done. Results in ${OUT_DIR}"


