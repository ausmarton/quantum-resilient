## Deployment Guide

This document explains how to build and run the benchmarking framework locally, on a local Kubernetes cluster (kind/minikube), and on Google Cloud (GKE).

### Prerequisites
- Docker 20+
- Python 3.11+
- kubectl 1.25+
- One of:
  - kind v0.20+ (recommended for local)
  - or minikube v1.30+
- For GCP:
  - gcloud CLI
  - A GCP project with GKE and Artifact Registry enabled

---

## Local (Docker)

```bash
# Build and run locally with Docker
make docker-build
make docker-run

# Results will appear in ./results
ls -la results/
```

Dockerfile: `docker/Dockerfile`
Compose: `docker/docker-compose.yml` (optional)

---

## Local Kubernetes (kind)

### 1) Create a kind cluster
```bash
kind create cluster --name pqc-bench --image kindest/node:v1.27.3
kubectl cluster-info --context kind-pqc-bench
```

### 2) Build image and load into kind
```bash
# Build image locally
make docker-build

# Load into kind
kind load docker-image pqc-benchmark:latest --name pqc-bench
```

### 3) Deploy manifests
```bash
# Apply namespace (if specified in k8s/manifests)
kubectl apply -f k8s/

# Check resources
kubectl get all -A | grep pqc

# Tail logs (replace with your workload name)
kubectl logs -f deployment/quantumresilient -n quantumresilient
```

### 4) Collect results
If the workload writes to a mounted volume, ensure a PersistentVolume/PersistentVolumeClaim is configured. The provided manifests may default to writing inside the container.

For quick collection:
```bash
# Exec into the pod and copy results
POD=$(kubectl get pods -n quantumresilient -o name | head -n1)
kubectl exec -n quantumresilient "$POD" -- ls -la /app/results || true
kubectl cp quantumresilient/${POD#pod/}:/app/results ./results
```

---

## Local Kubernetes (minikube)

### 1) Start cluster
```bash
minikube start --kubernetes-version=v1.27.3
```

### 2) Build image in minikube’s Docker daemon
```bash
eval $(minikube docker-env)
make docker-build
```

### 3) Deploy and view logs
```bash
kubectl apply -f k8s/
kubectl get pods -n quantumresilient
kubectl logs -f deployment/quantumresilient -n quantumresilient
```

### 4) Copy results
```bash
POD=$(kubectl get pods -n quantumresilient -o name | head -n1)
kubectl cp quantumresilient/${POD#pod/}:/app/results ./results
```

---

## Google Cloud (GKE)

### 1) Configure project and auth
```bash
# Set project
PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable services
gcloud services enable container.googleapis.com artifactregistry.googleapis.com

# Create Artifact Registry (if not exists)
REGION=us-central1
REPO=pqc
gcloud artifacts repositories create $REPO \
  --repository-format=docker \
  --location=$REGION \
  --description="PQC Benchmark images" || true
```

### 2) Build and push image to Artifact Registry
```bash
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/pqc-benchmark:latest"

gcloud auth configure-docker $REGION-docker.pkg.dev

docker build -f docker/Dockerfile -t $IMAGE .
docker push $IMAGE
```

### 3) Create GKE cluster and connect
```bash
CLUSTER=pqc-research
ZONE=us-central1-a

gcloud container clusters create $CLUSTER --zone $ZONE --num-nodes 2

gcloud container clusters get-credentials $CLUSTER --zone $ZONE --project $PROJECT_ID
```

### 4) Deploy manifests referencing the pushed image
Update your manifest image fields to use `$IMAGE` or patch after applying:
```bash
kubectl apply -f k8s/
# Example patch (replace deployment name and container if needed)
kubectl set image deployment/quantumresilient pqc-benchmark-container=$IMAGE -n quantumresilient
```

### 5) Monitor and collect results
```bash
kubectl get pods -n quantumresilient
kubectl logs -f deployment/quantumresilient -n quantumresilient

POD=$(kubectl get pods -n quantumresilient -o name | head -n1)
kubectl cp quantumresilient/${POD#pod/}:/app/results ./results
```

---

## Configuration
- Global configuration: `config.yaml`
- Output directory: `results/`
- CLI (preferred):
  - `python src/python_orchestrator/main.py benchmark` (Objective 3)
  - `python src/python_orchestrator/main.py compare` (Objective 4)
  - `python src/python_orchestrator/main.py run` (both)
- One-step runner:
  - `python run_benchmarks.py`

---

## Tips
- For batch workloads in K8s, consider using a `Job`/`CronJob` rather than a long-running `Deployment`.
- For persistent results in K8s, mount a PVC to `/app/results` and copy from the node or use an object store (e.g., GCS) uploader.
- Set logging via `framework.log_level` in `config.yaml` or `--log-level` flag.
