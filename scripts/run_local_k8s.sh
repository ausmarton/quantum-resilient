#!/usr/bin/env bash
set -euo pipefail

usage() {
	echo "Usage: $0 --config <experiment.yaml> [--namespace <ns>] [--release <name>]" >&2
	exit 1
}

CONFIG=""
NAMESPACE="pqc-benchmark"
HELM_RELEASE="pqc-benchmark"

while [[ $# -gt 0 ]]; do
	case "$1" in
		-c|--config) CONFIG="${2:-}"; shift 2;;
		-n|--namespace) NAMESPACE="${2:-}"; shift 2;;
		-r|--release) HELM_RELEASE="${2:-}"; shift 2;;
		-h|--help) usage;;
		*) echo "Unknown arg: $1" >&2; usage;;
	esac
done

[[ -z "${CONFIG}" ]] && usage
[[ ! -f "${CONFIG}" ]] && { echo "Config not found: ${CONFIG}" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

echo "[run_local_k8s] Using current kubectl context: $(kubectl config current-context)"

# Clean up ALL resources in pqc-benchmark namespace and reset network
echo "[run_local_k8s] Cleaning up all resources in pqc-benchmark namespace..."
kubectl -n pqc-benchmark delete deployment,replicaset,job,pod --all --force --grace-period=0 >/dev/null 2>&1 || true
sleep 2

# Always reset CNI network to avoid IP exhaustion
echo "[run_local_k8s] Resetting CNI network..."
minikube ssh -- "sudo rm -rf /var/lib/cni/networks/ptp/* 2>/dev/null; sudo systemctl restart containerd; sudo systemctl restart kubelet" >/dev/null 2>&1 || true
sleep 8

echo "[run_local_k8s] Building image with podman..."
BUILD_TAG="pqc-benchmarking-skeleton:dev-$(date -u +%Y%m%dT%H%M%SZ)"

podman build --no-cache -t "${BUILD_TAG}" -f "${ROOT}/docker/Dockerfile" "${ROOT}"

echo "[run_local_k8s] Saving and loading image into minikube..."
TAR_PATH="/tmp/pqc-image-$(date +%s).tar"
podman save "${BUILD_TAG}" -o "${TAR_PATH}"

# Load via minikube's cache
minikube image load "${TAR_PATH}" && rm -f "${TAR_PATH}"

# Verify it's available
echo "[run_local_k8s] Images in minikube:"
minikube image ls | grep pqc-benchmarking-skeleton || echo "WARNING: Image not found"

echo "[run_local_k8s] Ensuring namespace ${NAMESPACE}..."
kubectl get ns "${NAMESPACE}" >/dev/null 2>&1 || kubectl create ns "${NAMESPACE}"

echo "[run_local_k8s] Creating/Updating experiment ConfigMap..."
CM_NAME="experiment-config"
kubectl -n "${NAMESPACE}" delete configmap "${CM_NAME}" >/dev/null 2>&1 || true
kubectl -n "${NAMESPACE}" create configmap "${CM_NAME}" --from-file "experiment.yaml=${CONFIG}"

if command -v helm >/dev/null 2>&1; then
	echo "[run_local_k8s] Deploying Helm chart..."
	pushd "${ROOT}/helm" >/dev/null
	helm upgrade --install "${HELM_RELEASE}" . \
		--namespace "${NAMESPACE}" \
		--set image.repository="$(echo ${BUILD_TAG} | cut -d: -f1)" \
		--set image.tag="$(echo ${BUILD_TAG} | cut -d: -f2)" \
		--set image.pullPolicy=IfNotPresent \
		--set orchestrator.configMapName="${CM_NAME}" \
		--set orchestrator.configFileName="experiment.yaml" \
		--set orchestrator.outputDir="/results"
	popd >/dev/null
else
	echo "[run_local_k8s] helm not found; applying plain k8s manifests..."
	# Apply rust-core Deployment/Service
	kubectl apply -f "${ROOT}/k8s/deployment.yaml"
	kubectl apply -f "${ROOT}/k8s/service.yaml"
	
	# Create Job with the correct image
	cat "${ROOT}/k8s/orchestrator-job.yaml" | \
		sed "s|image: .*pqc-benchmarking-skeleton.*|image: ${BUILD_TAG}|" | \
		sed "s|imagePullPolicy: .*|imagePullPolicy: IfNotPresent|" | \
		kubectl apply -f -
	
	ACTUAL_IMAGE=$(kubectl -n "${NAMESPACE}" get job pqc-benchmark-orchestrator -o jsonpath='{.spec.template.spec.containers[0].image}')
	echo "[run_local_k8s] Job image -> ${ACTUAL_IMAGE}"
	if [[ "${ACTUAL_IMAGE}" != "${BUILD_TAG}" ]]; then
		echo "[run_local_k8s] ERROR: Job image mismatch! Expected ${BUILD_TAG}, got ${ACTUAL_IMAGE}"
		exit 2
	fi
fi

echo "[run_local_k8s] Waiting for orchestrator Job to complete (up to 5m)..."
TIMEOUT="5m"
JOB_NAME="${HELM_RELEASE}-orchestrator"

# Fail-fast: check pod status every 10s for first minute
for i in {1..6}; do
	sleep 10
	POD_STATUS=$(kubectl -n "${NAMESPACE}" get pods -l app="${JOB_NAME}" -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "")
	POD_NAME=$(kubectl -n "${NAMESPACE}" get pods -l app="${JOB_NAME}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
	echo "[run_local_k8s] Pod ${POD_NAME} status: ${POD_STATUS}"
	
	if [[ "${POD_STATUS}" == "Failed" || "${POD_STATUS}" == "Error" ]]; then
		echo "[run_local_k8s] Pod failed early; printing logs immediately..."
		kubectl -n "${NAMESPACE}" logs "${POD_NAME}" --tail=100 || true
		kubectl -n "${NAMESPACE}" describe pod "${POD_NAME}" | tail -50 || true
		echo "[run_local_k8s] Job failed within 60s; exiting."
		exit 3
	elif [[ "${POD_STATUS}" == "Succeeded" ]]; then
		echo "[run_local_k8s] Pod succeeded early!"
		break
	elif [[ "${POD_STATUS}" == "Pending" ]]; then
		# Check for ImagePullBackOff or other issues
		POD_REASON=$(kubectl -n "${NAMESPACE}" get pods -l app="${JOB_NAME}" -o jsonpath='{.items[0].status.containerStatuses[0].state.waiting.reason}' 2>/dev/null || echo "")
		if [[ "${POD_REASON}" == "ImagePullBackOff" || "${POD_REASON}" == "ErrImagePull" ]]; then
			echo "[run_local_k8s] Image pull failed: ${POD_REASON}"
			kubectl -n "${NAMESPACE}" describe pod "${POD_NAME}" | grep -A 10 "Events:" || true
			exit 3
		fi
	fi
done

kubectl -n "${NAMESPACE}" wait --for=condition=complete --timeout="${TIMEOUT}" "job/${JOB_NAME}" || {
	echo "Job did not complete in time" >&2
	# Attempt to copy /results even on failure
	echo "--- Attempting to copy /results from latest pod ---"
	LASTPOD="$(kubectl -n "${NAMESPACE}" get pods -l app="${JOB_NAME}" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null || true)"
	if [ -n "${LASTPOD}" ]; then
		OUT_FAIL="${ROOT}/results/local_k8s_fail_$(date -u +%Y%m%dT%H%M%SZ)"
		mkdir -p "${OUT_FAIL}"
		kubectl -n "${NAMESPACE}" cp "${LASTPOD}:/results/." "${OUT_FAIL}" 2>/dev/null || true
		echo "[run_local_k8s] (failure) results copied to: ${OUT_FAIL}"
	fi
	echo "--- Job describe ---"
	kubectl -n "${NAMESPACE}" describe job/"${JOB_NAME}" || true
	echo "--- Pods list ---"
	kubectl -n "${NAMESPACE}" get pods -l app="${JOB_NAME}" -o wide || true
	echo "--- Pod describe (first) ---"
	FIRSTPOD="$(kubectl -n "${NAMESPACE}" get pods -l app="${JOB_NAME}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
	if [ -n "${FIRSTPOD}" ]; then
		kubectl -n "${NAMESPACE}" describe pod "${FIRSTPOD}" || true
		echo "--- Pod logs ---"
		kubectl -n "${NAMESPACE}" logs "${FIRSTPOD}" --tail=200 || true
	fi
	echo "--- Recent events ---"
	kubectl -n "${NAMESPACE}" get events --sort-by=.lastTimestamp | tail -n 50 || true
	exit 3
}

echo "[run_local_k8s] Fetching results artifacts from pod..."
POD="$(kubectl -n "${NAMESPACE}" get pods -l app="${HELM_RELEASE}-orchestrator" -o jsonpath='{.items[0].metadata.name}')"
OUT_DIR="${ROOT}/results/local_k8s_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${OUT_DIR}"
kubectl -n "${NAMESPACE}" cp "${POD}:/results/." "${OUT_DIR}"
echo "[run_local_k8s] Results in ${OUT_DIR}"

echo "[run_local_k8s] Optional: to view rust-core metrics locally, run:"
echo "kubectl -n ${NAMESPACE} port-forward svc/${HELM_RELEASE}-rust-core 9100:9100"
echo "Then open http://localhost:9100/metrics"
