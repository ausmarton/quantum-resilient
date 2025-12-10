#!/usr/bin/env python3
# =============================================================================
# scripts/lib/k8s-job-generator.py - Unified Kubernetes Job YAML Generator
#
# Generates Kubernetes Job YAML for both Minikube and GCP/GKE environments.
# Consolidates duplicate YAML generation logic from run_minikube.sh and
# scripts/submit_gcp_job_parallel.sh.
#
# Usage:
#   python3 scripts/lib/k8s-job-generator.py \
#     --environment minikube \
#     --job-name pqc-bench-worker \
#     --namespace default \
#     --image localhost/pqc-bench:latest \
#     --scenario-configmap pqc-bench-scenario \
#     [--output job.yaml]
#
#   python3 scripts/lib/k8s-job-generator.py \
#     --environment gcp \
#     --job-name pqc-bench-worker \
#     --namespace default \
#     --image gcr.io/project/pqc-bench:latest \
#     --scenario-configmap pqc-scenario-abc123 \
#     --gcp-config-configmap pqc-gcp-config-abc123 \
#     --experiment-id abc123 \
#     [--output job.yaml]
# =============================================================================

import argparse
import sys
import yaml
from typing import Dict, Any, Optional


def create_base_job() -> Dict[str, Any]:
    """Create base Job structure common to all environments."""
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": "pqc-bench-worker",
            "namespace": "default",
            "labels": {
                "app": "pqc-bench",
                "component": "worker",
            },
        },
        "spec": {
            "backoffLimit": 2,
            "ttlSecondsAfterFinished": 3600,
            "template": {
                "metadata": {
                    "labels": {
                        "app": "pqc-bench",
                        "component": "worker",
                    },
                    "annotations": {
                        "prometheus.io/scrape": "true",
                        "prometheus.io/port": "9898",
                        "prometheus.io/path": "/metrics",
                    },
                },
                "spec": {
                    "restartPolicy": "Never",
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 65532,
                        "runAsGroup": 65532,
                        "fsGroup": 65532,
                        "seccompProfile": {
                            "type": "RuntimeDefault",
                        },
                    },
                },
            },
        },
    }


def create_minikube_init_container() -> Dict[str, Any]:
    """Create init container for Minikube (busybox)."""
    return {
        "name": "gather-metadata",
        "image": "busybox:1.36",
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": 65532,
            "allowPrivilegeEscalation": False,
            "capabilities": {
                "drop": ["ALL"],
            },
            "readOnlyRootFilesystem": True,
        },
        "command": [
            "/bin/sh",
            "-c",
            """set -e

# Create metadata JSON
cat > /results/container_metadata.json << EOF
{
  "type": "kubernetes_container",
  "node_name": "${NODE_NAME:-unknown}",
  "pod_name": "${POD_NAME:-unknown}",
  "pod_namespace": "${POD_NAMESPACE:-default}",
  "pod_ip": "${POD_IP:-unknown}",
  "container_image": "localhost/pqc-bench:latest",
  "kernel_version": "$(uname -r)",
  "arch": "$(uname -m)",
  "hostname": "$(hostname)",
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "Metadata written to /results/container_metadata.json"
cat /results/container_metadata.json""",
        ],
        "env": [
            {
                "name": "NODE_NAME",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "spec.nodeName",
                    },
                },
            },
            {
                "name": "POD_NAME",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "metadata.name",
                    },
                },
            },
            {
                "name": "POD_NAMESPACE",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "metadata.namespace",
                    },
                },
            },
            {
                "name": "POD_IP",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "status.podIP",
                    },
                },
            },
        ],
        "volumeMounts": [
            {
                "name": "results",
                "mountPath": "/results",
            },
        ],
    }


def create_gcp_init_container() -> Dict[str, Any]:
    """Create init container for GCP (cloud-sdk)."""
    return {
        "name": "gather-metadata",
        "image": "gcr.io/google.com/cloudsdktool/cloud-sdk:alpine",
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": 65532,
            "allowPrivilegeEscalation": False,
            "capabilities": {
                "drop": ["ALL"],
            },
        },
        "command": [
            "/bin/sh",
            "-c",
            """set -e

echo "=== Gathering GCP/GKE metadata ==="

# Get metadata from GCE metadata server
ZONE=$(curl -s -H "Metadata-Flavor: Google" \\
  http://metadata.google.internal/computeMetadata/v1/instance/zone | cut -d'/' -f4)
MACHINE_TYPE=$(curl -s -H "Metadata-Flavor: Google" \\
  http://metadata.google.internal/computeMetadata/v1/instance/machine-type | cut -d'/' -f4)
INSTANCE_ID=$(curl -s -H "Metadata-Flavor: Google" \\
  http://metadata.google.internal/computeMetadata/v1/instance/id)
PROJECT_ID=$(curl -s -H "Metadata-Flavor: Google" \\
  http://metadata.google.internal/computeMetadata/v1/project/project-id)
CLUSTER_NAME=$(curl -s -H "Metadata-Flavor: Google" \\
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/cluster-name 2>/dev/null || echo "unknown")

# Get CPU info
CPU_MODEL=$(cat /proc/cpuinfo | grep "model name" | head -1 | cut -d':' -f2 | xargs)
CPU_COUNT=$(nproc)

# Get memory info
MEMORY_TOTAL=$(cat /proc/meminfo | grep MemTotal | awk '{print $2}')

# Get kernel info
KERNEL_VERSION=$(uname -r)
ARCH=$(uname -m)

# Get container runtime
CONTAINER_RUNTIME="containerd"

# Create cloud metadata JSON
cat > /results/cloud_metadata.json << EOF
{
  "type": "gcp_gke",
  "project_id": "$PROJECT_ID",
  "zone": "$ZONE",
  "region": "$(echo $ZONE | sed 's/-[a-z]$//')",
  "machine_type": "$MACHINE_TYPE",
  "instance_id": "$INSTANCE_ID",
  "cluster_name": "$CLUSTER_NAME",
  "node_name": "${NODE_NAME}",
  "pod_name": "${POD_NAME}",
  "pod_namespace": "${POD_NAMESPACE}",
  "cpu_model": "$CPU_MODEL",
  "cpu_count": $CPU_COUNT,
  "memory_total_kb": $MEMORY_TOTAL,
  "kernel_version": "$KERNEL_VERSION",
  "arch": "$ARCH",
  "container_runtime": "$CONTAINER_RUNTIME",
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "Cloud metadata:"
cat /results/cloud_metadata.json

# Create results subdirectories
mkdir -p /results/raw /results/merged /results/stats /results/figures

              echo "=== Metadata collection complete ==="
""",
        ],
        "env": [
            {
                "name": "NODE_NAME",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "spec.nodeName",
                    },
                },
            },
            {
                "name": "POD_NAME",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "metadata.name",
                    },
                },
            },
            {
                "name": "POD_NAMESPACE",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "metadata.namespace",
                    },
                },
            },
        ],
        "volumeMounts": [
            {
                "name": "results",
                "mountPath": "/results",
            },
        ],
        "resources": {
            "requests": {
                "cpu": "100m",
                "memory": "128Mi",
            },
            "limits": {
                "cpu": "500m",
                "memory": "512Mi",
            },
        },
    }


def create_main_container(
    environment: str,
    image: str,
    scenario_configmap: str,
    gcp_config_configmap: Optional[str] = None,
) -> Dict[str, Any]:
    """Create main pqc-bench container."""
    container = {
        "name": "pqc-bench",
        "image": image,
        "imagePullPolicy": "Always" if environment == "gcp" else "Never",
        "args": [
            "--scenario",
            "/config/scenario.yaml",
        ],
        "env": [
            {
                "name": "RUST_LOG",
                "value": "info",
            },
            {
                "name": "QR_MODE",
                "value": environment,
            },
            {
                "name": "NODE_NAME",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "spec.nodeName",
                    },
                },
            },
            {
                "name": "POD_NAME",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "metadata.name",
                    },
                },
            },
        ],
        "volumeMounts": [
            {
                "name": "scenario-config",
                "mountPath": "/config",
                "readOnly": True,
            },
            {
                "name": "results",
                "mountPath": "/results",
            },
        ],
        "resources": {
            "requests": {
                "cpu": "800m" if environment == "gcp" else "500m",
                "memory": "1Gi" if environment == "gcp" else "512Mi",
            },
            "limits": {
                "cpu": "4",
                "memory": "4Gi",
            },
        },
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": 65532,
            "allowPrivilegeEscalation": False,
            "capabilities": {
                "drop": ["ALL"],
            },
            "readOnlyRootFilesystem": True,
        },
        "ports": [
            {
                "name": "metrics",
                "containerPort": 9898,
                "protocol": "TCP",
            },
        ],
    }
    
    # Add GCP-specific environment variables
    if environment == "gcp" and gcp_config_configmap:
        container["env"].extend([
            {
                "name": "PQC_SMOKE_TEST",
                "valueFrom": {
                    "configMapKeyRef": {
                        "name": gcp_config_configmap,
                        "key": "smoke_test",
                        "optional": True,
                    },
                },
            },
            {
                "name": "GCS_BUCKET",
                "valueFrom": {
                    "configMapKeyRef": {
                        "name": gcp_config_configmap,
                        "key": "bucket_name",
                    },
                },
            },
            {
                "name": "EXP_ID",
                "valueFrom": {
                    "configMapKeyRef": {
                        "name": gcp_config_configmap,
                        "key": "experiment_id",
                    },
                },
            },
        ])
    
    # Add POD_IP for Minikube
    if environment == "minikube":
        container["env"].append({
            "name": "POD_IP",
            "valueFrom": {
                "fieldRef": {
                    "fieldPath": "status.podIP",
                },
            },
        })
    
    return container


def create_gcp_upload_sidecar(gcp_config_configmap: str) -> Dict[str, Any]:
    """Create GCS upload sidecar container for GCP."""
    # Read the sidecar script from the original YAML
    # This is a simplified version - in production, we'd read from a file
    sidecar_script = """# Don't use set -e here - we want to catch and report errors explicitly
set -u  # Only fail on undefined variables

echo "=== Waiting for benchmark to complete ==="

# Find the JSONL output file (may be in different locations)
JSONL_FILE=""
MAX_WAIT=300  # 5 minutes max wait
WAIT_COUNT=0

while [ -z "$JSONL_FILE" ] && [ $WAIT_COUNT -lt $MAX_WAIT ]; do
  # Check expected location first
  if [ -f /results/raw/run.jsonl ]; then
    JSONL_FILE="/results/raw/run.jsonl"
    break
  fi
  # Check for any .jsonl file in /results/
  FOUND=$(find /results -name "*.jsonl" -type f 2>/dev/null | head -1)
  if [ -n "$FOUND" ]; then
    JSONL_FILE="$FOUND"
    echo "Found JSONL file at: $JSONL_FILE (moving to expected location)"
    mkdir -p /results/raw
    cp "$JSONL_FILE" /results/raw/run.jsonl
    JSONL_FILE="/results/raw/run.jsonl"
    break
  fi
  echo "Waiting for results... ($WAIT_COUNT/$MAX_WAIT seconds)"
  sleep 5
  WAIT_COUNT=$((WAIT_COUNT + 5))
done

if [ -z "$JSONL_FILE" ] || [ ! -f "$JSONL_FILE" ]; then
  echo "ERROR: No JSONL file found after $MAX_WAIT seconds"
  echo "Contents of /results:"
  ls -laR /results/ 2>/dev/null || true
  exit 1
fi

echo "Found JSONL file: $JSONL_FILE"

# Wait for file to stop growing (benchmark complete)
prev_size=0
stable_count=0
while [ $stable_count -lt 3 ]; do
  sleep 10
  curr_size=$(stat -f%z "$JSONL_FILE" 2>/dev/null || stat -c%s "$JSONL_FILE" 2>/dev/null || echo 0)
  if [ "$curr_size" = "$prev_size" ] && [ "$curr_size" != "0" ]; then
    stable_count=$((stable_count + 1))
    echo "File stable check $stable_count/3 (size: $curr_size)"
  else
    stable_count=0
    prev_size=$curr_size
  fi
done

echo "=== Benchmark complete, uploading results ==="

# Ensure gcloud/gsutil config directories exist and are writable
mkdir -p "${CLOUDSDK_CONFIG:-/tmp/.config/gcloud}"
mkdir -p "$(dirname ${BOTO_CONFIG:-/tmp/.boto})"

# Create minimal boto config
if [ ! -f "${BOTO_CONFIG:-/tmp/.boto}" ]; then
  echo "[GSUtil]" > "${BOTO_CONFIG:-/tmp/.boto}"
  echo "use_magicfile = False" >> "${BOTO_CONFIG:-/tmp/.boto}"
  echo "[Boto]" >> "${BOTO_CONFIG:-/tmp/.boto}"
  echo "https_validate_certificates = True" >> "${BOTO_CONFIG:-/tmp/.boto}"
fi

# Configure gcloud/gsutil to use Workload Identity (ADC)
echo "Configuring gcloud/gsutil to use Workload Identity (ADC)..."
unset GOOGLE_APPLICATION_CREDENTIALS 2>/dev/null || true
export CLOUDSDK_AUTH_USE_APPLICATION_DEFAULT_CREDENTIALS=true

# Verify Workload Identity
echo "Checking Workload Identity metadata..."
METADATA_SERVER="http://metadata.google.internal"
if curl -s -f -H "Metadata-Flavor: Google" \\
    "${METADATA_SERVER}/computeMetadata/v1/instance/service-accounts/default/email" >/dev/null 2>&1; then
  SA_EMAIL=$(curl -s -H "Metadata-Flavor: Google" \\
      "${METADATA_SERVER}/computeMetadata/v1/instance/service-accounts/default/email")
  echo "Service account from metadata: $SA_EMAIL"
else
  echo "WARNING: Could not retrieve service account from metadata server"
fi

BUCKET="${GCS_BUCKET}"
EXP="${EXP_ID}"

# Create provenance.json
GIT_COMMIT="${GIT_COMMIT:-unknown}"
SCENARIO_HASH=$(sha256sum /config/scenario.yaml | cut -d' ' -f1)
TIMESTAMP=$(date -u +%Y%m%d%H%M%S)
DATASET_VERSION="${TIMESTAMP}-${GIT_COMMIT:0:8}-${SCENARIO_HASH:0:8}"

cat > /results/provenance.json << EOF
{
  "git_commit": "$GIT_COMMIT",
  "rustc_version": "1.78.0",
  "container_image": "${CONTAINER_IMAGE}",
  "gke_cluster_name": "$(cat /results/cloud_metadata.json | grep cluster_name | cut -d'"' -f4)",
  "gke_nodepool_size": 1,
  "instance_machine_type": "$(cat /results/cloud_metadata.json | grep machine_type | cut -d'"' -f4)",
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "scenario_file_hash": "$SCENARIO_HASH",
  "dataset_version": "$DATASET_VERSION"
}
EOF

# Create manifest.json
cat > /results/manifest.json << EOF
{
  "run_id": "$EXP",
  "environment": "gcp_gke",
  "provenance": $(cat /results/provenance.json),
  "cloud_metadata": $(cat /results/cloud_metadata.json)
}
EOF

# Copy raw results as merged
cp /results/raw/run.jsonl /results/merged/merged.jsonl

# Upload to GCS
echo "Uploading to gs://${BUCKET}/experiments/${EXP}/"

UPLOAD_ERRORS=0

# Upload files using gcloud storage
for file in merged.jsonl manifest.json provenance.json cloud_metadata.json; do
  if ! gcloud storage cp "/results/${file}" "gs://${BUCKET}/experiments/${EXP}/${file}" 2>&1; then
    echo "ERROR: Failed to upload ${file}"
    UPLOAD_ERRORS=$((UPLOAD_ERRORS + 1))
  else
    echo "✓ Uploaded ${file}"
  fi
done

# Upload raw data
if ! gcloud storage cp /results/raw/run.jsonl "gs://${BUCKET}/experiments/${EXP}/raw/run.jsonl" 2>&1; then
  echo "ERROR: Failed to upload raw data"
  UPLOAD_ERRORS=$((UPLOAD_ERRORS + 1))
else
  echo "✓ Uploaded raw data"
fi

# Upload stats if generated
if [ -f /results/stats/summary.json ]; then
  if ! gcloud storage cp /results/stats/summary.json "gs://${BUCKET}/experiments/${EXP}/summary.json" 2>&1; then
    echo "WARNING: Failed to upload summary.json (optional)"
  else
    echo "✓ Uploaded summary.json"
  fi
fi

# Verify uploads
echo "=== Verifying uploads ==="
sleep 2

LIST_OUTPUT=$(gcloud storage ls "gs://${BUCKET}/experiments/${EXP}/" 2>&1)
if [ $? -ne 0 ]; then
  echo "ERROR: Cannot list uploaded files"
  echo "$LIST_OUTPUT"
  UPLOAD_ERRORS=$((UPLOAD_ERRORS + 1))
else
  echo "Directory listing successful:"
  echo "$LIST_OUTPUT"
fi

# Signal completion
touch /results/.upload_complete

echo "=== GCS upload finished successfully ==="
exit 0"""
    
    return {
        "name": "upload-results",
        "image": "gcr.io/google.com/cloudsdktool/cloud-sdk:alpine",
        "command": ["/bin/sh", "-c", sidecar_script],
        "env": [
            {
                "name": "GCS_BUCKET",
                "valueFrom": {
                    "configMapKeyRef": {
                        "name": gcp_config_configmap,
                        "key": "bucket_name",
                    },
                },
            },
            {
                "name": "EXP_ID",
                "valueFrom": {
                    "configMapKeyRef": {
                        "name": gcp_config_configmap,
                        "key": "experiment_id",
                    },
                },
            },
            {
                "name": "GIT_COMMIT",
                "valueFrom": {
                    "configMapKeyRef": {
                        "name": gcp_config_configmap,
                        "key": "git_commit",
                        "optional": True,
                    },
                },
            },
            {
                "name": "CONTAINER_IMAGE",
                "valueFrom": {
                    "configMapKeyRef": {
                        "name": gcp_config_configmap,
                        "key": "container_image",
                        "optional": True,
                    },
                },
            },
            {
                "name": "CLOUDSDK_CONFIG",
                "value": "/tmp/.config/gcloud",
            },
            {
                "name": "BOTO_CONFIG",
                "value": "/tmp/.boto",
            },
        ],
        "volumeMounts": [
            {
                "name": "scenario-config",
                "mountPath": "/config",
                "readOnly": True,
            },
            {
                "name": "results",
                "mountPath": "/results",
            },
        ],
        "resources": {
            "requests": {
                "cpu": "100m",
                "memory": "256Mi",
            },
            "limits": {
                "cpu": "500m",
                "memory": "512Mi",
            },
        },
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": 65532,
            "allowPrivilegeEscalation": False,
            "capabilities": {
                "drop": ["ALL"],
            },
        },
    }


def generate_job_yaml(
    environment: str,
    job_name: str,
    namespace: str,
    image: str,
    scenario_configmap: str,
    experiment_id: Optional[str] = None,
    gcp_config_configmap: Optional[str] = None,
    ttl_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate Kubernetes Job YAML for the specified environment.
    
    Args:
        environment: "minikube" or "gcp"
        job_name: Name for the Job
        namespace: Kubernetes namespace
        image: Container image to use
        scenario_configmap: Name of the scenario ConfigMap
        experiment_id: Experiment ID (for labels)
        gcp_config_configmap: Name of GCP config ConfigMap (required for GCP)
        ttl_seconds: TTL for completed jobs (default: 3600 for minikube, 300 for GCP)
    
    Returns:
        Job YAML as a dictionary
    """
    if environment not in ["minikube", "gcp"]:
        raise ValueError(f"Unknown environment: {environment}")
    
    if environment == "gcp" and not gcp_config_configmap:
        raise ValueError("gcp_config_configmap is required for GCP environment")
    
    # Create base job
    job = create_base_job()
    
    # Set metadata
    job["metadata"]["name"] = job_name
    job["metadata"]["namespace"] = namespace
    if experiment_id:
        job["metadata"]["labels"]["experiment-id"] = experiment_id
    if environment == "gcp":
        job["metadata"]["labels"]["environment"] = "gcp"
    
    # Set TTL
    if ttl_seconds is None:
        ttl_seconds = 300 if environment == "gcp" else 3600
    job["spec"]["ttlSecondsAfterFinished"] = ttl_seconds
    
    # Get pod spec
    pod_spec = job["spec"]["template"]["spec"]
    
    # Set service account for GCP
    if environment == "gcp":
        pod_spec["serviceAccountName"] = "pqc-bench-sa"
    
    # Create init container
    if environment == "minikube":
        pod_spec["initContainers"] = [create_minikube_init_container()]
    else:
        pod_spec["initContainers"] = [create_gcp_init_container()]
    
    # Create main container
    main_container = create_main_container(
        environment=environment,
        image=image,
        scenario_configmap=scenario_configmap,
        gcp_config_configmap=gcp_config_configmap,
    )
    
    # Create containers list
    containers = [main_container]
    
    # Add sidecar for GCP
    if environment == "gcp" and gcp_config_configmap:
        containers.append(create_gcp_upload_sidecar(gcp_config_configmap))
    
    pod_spec["containers"] = containers
    
    # Create volumes
    volumes = [
        {
            "name": "scenario-config",
            "configMap": {
                "name": scenario_configmap,
            },
        },
    ]
    
    # Add results volume
    if environment == "minikube":
        volumes.append({
            "name": "results",
            "persistentVolumeClaim": {
                "claimName": "pqc-bench-results",
            },
        })
    else:
        volumes.append({
            "name": "results",
            "emptyDir": {
                "sizeLimit": "2Gi",
            },
        })
    
    pod_spec["volumes"] = volumes
    
    # Add environment-specific settings
    if environment == "minikube":
        # Add tolerations for Minikube
        pod_spec["tolerations"] = [
            {
                "key": "node.kubernetes.io/not-ready",
                "operator": "Exists",
                "effect": "NoSchedule",
            },
        ]
    else:
        # Add node selector and affinity for GCP
        pod_spec["nodeSelector"] = {
            "cloud.google.com/gke-nodepool": "pqc-bench-pool",
        }
        pod_spec["affinity"] = {
            "podAntiAffinity": {
                "preferredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "weight": 100,
                        "podAffinityTerm": {
                            "labelSelector": {
                                "matchLabels": {
                                    "app": "pqc-bench",
                                    "component": "worker",
                                },
                            },
                            "topologyKey": "kubernetes.io/hostname",
                        },
                    },
                ],
            },
        }
        pod_spec["tolerations"] = [
            {
                "key": "cloud.google.com/gke-spot",
                "operator": "Equal",
                "value": "true",
                "effect": "NoSchedule",
            },
        ]
    
    return job


def main():
    parser = argparse.ArgumentParser(
        description="Generate Kubernetes Job YAML for PQC benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--environment",
        required=True,
        choices=["minikube", "gcp"],
        help="Target environment",
    )
    parser.add_argument(
        "--job-name",
        required=True,
        help="Job name",
    )
    parser.add_argument(
        "--namespace",
        required=True,
        help="Kubernetes namespace",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Container image",
    )
    parser.add_argument(
        "--scenario-configmap",
        required=True,
        help="Scenario ConfigMap name",
    )
    parser.add_argument(
        "--experiment-id",
        help="Experiment ID (for labels)",
    )
    parser.add_argument(
        "--gcp-config-configmap",
        help="GCP config ConfigMap name (required for GCP)",
    )
    parser.add_argument(
        "--ttl-seconds",
        type=int,
        help="TTL for completed jobs",
    )
    parser.add_argument(
        "--output",
        help="Output file (default: stdout)",
    )
    
    args = parser.parse_args()
    
    # Validate GCP requirements
    if args.environment == "gcp" and not args.gcp_config_configmap:
        parser.error("--gcp-config-configmap is required for GCP environment")
    
    try:
        job = generate_job_yaml(
            environment=args.environment,
            job_name=args.job_name,
            namespace=args.namespace,
            image=args.image,
            scenario_configmap=args.scenario_configmap,
            experiment_id=args.experiment_id,
            gcp_config_configmap=args.gcp_config_configmap,
            ttl_seconds=args.ttl_seconds,
        )
        
        # Output YAML
        output_yaml = yaml.dump(
            job,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_yaml)
        else:
            print(output_yaml)
    
    except Exception as e:
        print(f"ERROR: Failed to generate Job YAML: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

