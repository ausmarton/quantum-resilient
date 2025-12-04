//! Kubernetes Client
//!
//! Provides Kubernetes API operations for managing experiment resources.

use k8s_openapi::api::batch::v1::Job;
use k8s_openapi::api::core::v1::{ConfigMap, Pod};
use kube::api::{Api, DeleteParams, ListParams, PostParams};
use kube::Client;
use std::collections::BTreeMap;
use std::path::Path;
use thiserror::Error;
use tokio::io::AsyncWriteExt;
use tracing::{info, warn};

#[derive(Error, Debug)]
pub enum K8sClientError {
    #[error("Kubernetes API error: {0}")]
    KubeError(#[from] kube::Error),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Pod not found: {0}")]
    PodNotFound(String),
    #[error("Exec error: {0}")]
    ExecError(String),
}

/// Kubernetes client wrapper for experiment management
pub struct K8sClient {
    client: Client,
    namespace: String,
}

impl K8sClient {
    /// Create a new Kubernetes client
    pub async fn new(namespace: &str) -> Result<Self, K8sClientError> {
        let client = Client::try_default().await?;
        Ok(Self {
            client,
            namespace: namespace.to_string(),
        })
    }

    /// Create a ConfigMap with scenario configuration
    pub async fn create_scenario_configmap(
        &self,
        name: &str,
        scenario_yaml: &str,
    ) -> Result<(), K8sClientError> {
        let configmaps: Api<ConfigMap> = Api::namespaced(self.client.clone(), &self.namespace);

        let mut data = BTreeMap::new();
        data.insert("active_scenario.yaml".to_string(), scenario_yaml.to_string());

        let cm = ConfigMap {
            metadata: kube::api::ObjectMeta {
                name: Some(name.to_string()),
                namespace: Some(self.namespace.clone()),
                labels: Some(BTreeMap::from([
                    ("app".to_string(), "quantum-resilient".to_string()),
                    ("component".to_string(), "experiment-config".to_string()),
                ])),
                ..Default::default()
            },
            data: Some(data),
            ..Default::default()
        };

        configmaps.create(&PostParams::default(), &cm).await?;
        Ok(())
    }

    /// Delete a ConfigMap
    pub async fn delete_configmap(&self, name: &str) -> Result<(), K8sClientError> {
        let configmaps: Api<ConfigMap> = Api::namespaced(self.client.clone(), &self.namespace);
        configmaps.delete(name, &DeleteParams::default()).await?;
        Ok(())
    }

    /// Create a Job to run worker pods
    pub async fn create_worker_job(
        &self,
        experiment_id: &str,
        replicas: u32,
        image: &str,
        scenario_configmap: &str,
    ) -> Result<(), K8sClientError> {
        let jobs: Api<Job> = Api::namespaced(self.client.clone(), &self.namespace);

        let job_name = format!("qr-experiment-{}", experiment_id);

        let job: Job = serde_json::from_value(serde_json::json!({
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "namespace": self.namespace,
                "labels": {
                    "app": "quantum-resilient",
                    "component": "worker",
                    "experimentId": experiment_id
                }
            },
            "spec": {
                "parallelism": replicas,
                "completions": replicas,
                "backoffLimit": 0,
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "quantum-resilient",
                            "component": "worker",
                            "experimentId": experiment_id
                        },
                        "annotations": {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": "9898"
                        }
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "terminationGracePeriodSeconds": 10,
                        "containers": [{
                            "name": "pqc-bench",
                            "image": image,
                            "imagePullPolicy": "IfNotPresent",
                            "ports": [
                                {"name": "control", "containerPort": 6060},
                                {"name": "prom", "containerPort": 9898}
                            ],
                            "env": [
                                {"name": "QR_SCENARIO_PATH", "value": "/app/scenarios/active_scenario.yaml"},
                                {"name": "QR_RESULTS_DIR", "value": "/app/results"},
                                {"name": "QR_ORCHESTRATOR_ADDRESS", "value": format!("http://qr-orchestrator.{}.svc.cluster.local:7070", self.namespace)},
                                {"name": "QR_EXPERIMENT_ID", "value": experiment_id},
                                {"name": "QR_ENFORCE_TIMESYNC", "value": "true"},
                                {"name": "RUST_LOG", "value": "info"},
                                {
                                    "name": "POD_NAME",
                                    "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}
                                },
                                {
                                    "name": "POD_IP",
                                    "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}
                                }
                            ],
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "256Mi"},
                                "limits": {"cpu": "2", "memory": "2Gi"}
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/healthz", "port": 6060},
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/readyz", "port": 6060},
                                "initialDelaySeconds": 3,
                                "periodSeconds": 5
                            },
                            "volumeMounts": [
                                {
                                    "name": "scenario-config",
                                    "mountPath": "/app/scenarios/active_scenario.yaml",
                                    "subPath": "active_scenario.yaml",
                                    "readOnly": true
                                },
                                {
                                    "name": "results",
                                    "mountPath": "/app/results"
                                }
                            ]
                        }],
                        "volumes": [
                            {
                                "name": "scenario-config",
                                "configMap": {"name": scenario_configmap}
                            },
                            {
                                "name": "results",
                                "emptyDir": {}
                            }
                        ]
                    }
                }
            }
        }))?;

        jobs.create(&PostParams::default(), &job).await?;
        Ok(())
    }

    /// Delete a Job and its pods
    pub async fn delete_job(&self, experiment_id: &str) -> Result<(), K8sClientError> {
        let jobs: Api<Job> = Api::namespaced(self.client.clone(), &self.namespace);
        let job_name = format!("qr-experiment-{}", experiment_id);

        let dp = DeleteParams {
            propagation_policy: Some(kube::api::PropagationPolicy::Background),
            ..Default::default()
        };

        match jobs.delete(&job_name, &dp).await {
            Ok(_) => Ok(()),
            Err(kube::Error::Api(e)) if e.code == 404 => Ok(()), // Already deleted
            Err(e) => Err(e.into()),
        }
    }

    /// List pods for an experiment
    pub async fn list_experiment_pods(&self, experiment_id: &str) -> Result<Vec<Pod>, K8sClientError> {
        let pods: Api<Pod> = Api::namespaced(self.client.clone(), &self.namespace);
        let lp = ListParams::default().labels(&format!("experimentId={}", experiment_id));
        let pod_list = pods.list(&lp).await?;
        Ok(pod_list.items)
    }

    /// Copy results file from a pod (simplified - uses kubectl exec simulation)
    pub async fn copy_results_from_pod(
        &self,
        pod_name: &str,
        remote_path: &str,
        local_path: &Path,
    ) -> Result<(), K8sClientError> {
        // In a real implementation, this would use the Kubernetes exec API
        // For now, we'll use reqwest to fetch from the worker's API
        
        // Get pod IP
        let pods: Api<Pod> = Api::namespaced(self.client.clone(), &self.namespace);
        let pod = pods.get(pod_name).await?;
        
        let pod_ip = pod
            .status
            .as_ref()
            .and_then(|s| s.pod_ip.as_ref())
            .ok_or_else(|| K8sClientError::PodNotFound(pod_name.to_string()))?;

        // Fetch results via HTTP (workers expose /results endpoint in a real implementation)
        // For now, we'll create a placeholder
        info!(
            "Would copy {} from pod {} ({}) to {}",
            remote_path,
            pod_name,
            pod_ip,
            local_path.display()
        );

        // Create parent directory
        if let Some(parent) = local_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // For testing, create an empty file
        // In production, this would actually copy the file
        let mut file = tokio::fs::File::create(local_path).await?;
        file.write_all(b"").await?;

        Ok(())
    }

    /// Get pod details
    pub async fn get_pod(&self, name: &str) -> Result<Pod, K8sClientError> {
        let pods: Api<Pod> = Api::namespaced(self.client.clone(), &self.namespace);
        let pod = pods.get(name).await?;
        Ok(pod)
    }

    /// Patch a ConfigMap with new scenario data
    pub async fn patch_configmap(
        &self,
        name: &str,
        scenario_yaml: &str,
    ) -> Result<(), K8sClientError> {
        let configmaps: Api<ConfigMap> = Api::namespaced(self.client.clone(), &self.namespace);

        let patch = serde_json::json!({
            "data": {
                "active_scenario.yaml": scenario_yaml
            }
        });

        configmaps
            .patch(
                name,
                &kube::api::PatchParams::default(),
                &kube::api::Patch::Merge(patch),
            )
            .await?;

        Ok(())
    }
}

// Implement From for kube::Error to allow ? operator
impl From<serde_json::Error> for K8sClientError {
    fn from(err: serde_json::Error) -> Self {
        K8sClientError::ExecError(err.to_string())
    }
}

