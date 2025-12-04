//! Experiment Controller
//!
//! Manages the lifecycle of distributed benchmark experiments.

use crate::aggregator::ResultAggregator;
use crate::coordinator::{ExperimentCoordinator, WorkerInfo};
use crate::k8s_client::K8sClient;
use crate::storage;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use tracing::{error, info, warn};

#[derive(Error, Debug)]
pub enum ControllerError {
    #[error("Experiment not found: {0}")]
    ExperimentNotFound(String),
    #[error("Invalid experiment state: expected {expected}, got {actual}")]
    InvalidState { expected: String, actual: String },
    #[error("Kubernetes error: {0}")]
    K8sError(#[from] kube::Error),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Worker error: {0}")]
    WorkerError(String),
    #[error("Storage error: {0}")]
    StorageError(String),
    #[error("K8s client error: {0}")]
    K8sClientError(String),
}

impl From<crate::k8s_client::K8sClientError> for ControllerError {
    fn from(err: crate::k8s_client::K8sClientError) -> Self {
        ControllerError::K8sClientError(err.to_string())
    }
}

/// Experiment phase in its lifecycle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExperimentPhase {
    /// Experiment created, workers being spawned
    Pending,
    /// Workers spawned, waiting for all to register
    Waiting,
    /// Experiment running
    Running,
    /// Stop signal sent, waiting for workers to finish
    Stopping,
    /// Collecting results from workers
    Collecting,
    /// Experiment completed successfully
    Completed,
    /// Experiment failed
    Failed,
}

impl std::fmt::Display for ExperimentPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExperimentPhase::Pending => write!(f, "pending"),
            ExperimentPhase::Waiting => write!(f, "waiting"),
            ExperimentPhase::Running => write!(f, "running"),
            ExperimentPhase::Stopping => write!(f, "stopping"),
            ExperimentPhase::Collecting => write!(f, "collecting"),
            ExperimentPhase::Completed => write!(f, "completed"),
            ExperimentPhase::Failed => write!(f, "failed"),
        }
    }
}

/// Request to create a new experiment
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateExperimentRequest {
    pub scenario_config: String,
    pub replicas: u32,
    #[serde(default = "default_start_delay")]
    pub start_delay_ms: u64,
    pub experiment_id: String,
}

fn default_start_delay() -> u64 {
    5000
}

/// Experiment metadata and state
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Experiment {
    pub id: String,
    pub scenario_config: String,
    pub replicas: u32,
    pub start_delay_ms: u64,
    pub phase: ExperimentPhase,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    #[serde(skip)]
    pub coordinator: Arc<ExperimentCoordinator>,
}

/// Experiment status response
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExperimentStatus {
    pub id: String,
    pub replicas: u32,
    pub ready: u32,
    pub completed: u32,
    pub phase: ExperimentPhase,
    pub workers: Vec<WorkerStatus>,
}

/// Worker status in an experiment
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerStatus {
    pub worker_id: u32,
    pub pod_name: String,
    pub ready: bool,
    pub completed: bool,
}

/// Result of collecting experiment artifacts
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CollectResult {
    pub artifact_uri: String,
    pub events: u64,
    pub duration_sec: f64,
    pub summary: AggregationSummary,
}

/// Summary statistics from aggregation
#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct AggregationSummary {
    pub total_events: u64,
    pub total_duration_sec: f64,
    pub throughput_ops_sec: f64,
    pub latency_p50_us: f64,
    pub latency_p90_us: f64,
    pub latency_p95_us: f64,
    pub latency_p99_us: f64,
    pub queue_delay_p50_us: f64,
    pub queue_delay_p99_us: f64,
}

/// The main experiment controller
pub struct ExperimentController {
    k8s_client: K8sClient,
    worker_image: String,
    namespace: String,
    local_results_dir: String,
    storage_uri: Option<String>,
    max_time_drift_ns: u64,
    experiments: RwLock<HashMap<String, Experiment>>,
}

impl ExperimentController {
    pub fn new(
        k8s_client: K8sClient,
        worker_image: String,
        namespace: String,
        local_results_dir: String,
        storage_uri: Option<String>,
        max_time_drift_ns: u64,
    ) -> Self {
        Self {
            k8s_client,
            worker_image,
            namespace,
            local_results_dir,
            storage_uri,
            max_time_drift_ns,
            experiments: RwLock::new(HashMap::new()),
        }
    }

    /// Create a new experiment
    pub async fn create_experiment(
        &self,
        request: CreateExperimentRequest,
    ) -> Result<Experiment, ControllerError> {
        let experiment_id = request.experiment_id.clone();
        info!("Creating experiment: {}", experiment_id);

        // Create coordinator for this experiment
        let coordinator = Arc::new(ExperimentCoordinator::new(
            experiment_id.clone(),
            request.replicas,
            self.max_time_drift_ns,
        ));

        // Create the experiment object
        let experiment = Experiment {
            id: experiment_id.clone(),
            scenario_config: request.scenario_config.clone(),
            replicas: request.replicas,
            start_delay_ms: request.start_delay_ms,
            phase: ExperimentPhase::Pending,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            coordinator: coordinator.clone(),
        };

        // Store experiment
        {
            let mut experiments = self.experiments.write();
            experiments.insert(experiment_id.clone(), experiment.clone());
        }

        // Create ConfigMap with scenario
        let configmap_name = format!("qr-experiment-{}-scenario", experiment_id);
        self.k8s_client
            .create_scenario_configmap(&configmap_name, &request.scenario_config)
            .await?;
        info!("Created ConfigMap: {}", configmap_name);

        // Create worker Job
        self.k8s_client
            .create_worker_job(
                &experiment_id,
                request.replicas,
                &self.worker_image,
                &configmap_name,
            )
            .await?;
        info!(
            "Created Job with {} replicas for experiment {}",
            request.replicas, experiment_id
        );

        // Update phase to waiting
        self.update_phase(&experiment_id, ExperimentPhase::Waiting)?;

        let experiments = self.experiments.read();
        Ok(experiments.get(&experiment_id).unwrap().clone())
    }

    /// Register a worker with an experiment
    pub fn register_worker(
        &self,
        experiment_id: &str,
        pod_name: &str,
        pod_ip: &str,
    ) -> Result<(u32, u64), ControllerError> {
        let experiments = self.experiments.read();
        let experiment = experiments
            .get(experiment_id)
            .ok_or_else(|| ControllerError::ExperimentNotFound(experiment_id.to_string()))?;

        let (worker_id, start_time) = experiment.coordinator.register_worker(pod_name, pod_ip);
        info!(
            "Registered worker {} (pod: {}) for experiment {}",
            worker_id, pod_name, experiment_id
        );

        Ok((worker_id, start_time))
    }

    /// Start an experiment
    pub async fn start_experiment(&self, experiment_id: &str) -> Result<(), ControllerError> {
        info!("Starting experiment: {}", experiment_id);

        let coordinator = {
            let experiments = self.experiments.read();
            let experiment = experiments
                .get(experiment_id)
                .ok_or_else(|| ControllerError::ExperimentNotFound(experiment_id.to_string()))?;

            if experiment.phase != ExperimentPhase::Waiting {
                return Err(ControllerError::InvalidState {
                    expected: "waiting".to_string(),
                    actual: experiment.phase.to_string(),
                });
            }

            experiment.coordinator.clone()
        };

        // Calculate global start time
        let start_delay_ms = {
            let experiments = self.experiments.read();
            experiments.get(experiment_id).unwrap().start_delay_ms
        };
        let global_start = coordinator.calculate_global_start(start_delay_ms);

        // Signal all workers to start
        coordinator.signal_start(global_start).await;

        // Update experiment state
        self.update_phase(experiment_id, ExperimentPhase::Running)?;
        {
            let mut experiments = self.experiments.write();
            if let Some(exp) = experiments.get_mut(experiment_id) {
                exp.started_at = Some(Utc::now());
            }
        }

        info!(
            "Experiment {} started with global start time: {}",
            experiment_id, global_start
        );

        Ok(())
    }

    /// Stop an experiment gracefully
    pub async fn stop_experiment(&self, experiment_id: &str) -> Result<(), ControllerError> {
        info!("Stopping experiment: {}", experiment_id);

        let coordinator = {
            let experiments = self.experiments.read();
            let experiment = experiments
                .get(experiment_id)
                .ok_or_else(|| ControllerError::ExperimentNotFound(experiment_id.to_string()))?;

            experiment.coordinator.clone()
        };

        self.update_phase(experiment_id, ExperimentPhase::Stopping)?;

        // Signal all workers to stop
        coordinator.signal_stop().await;

        info!("Stop signal sent to all workers in experiment {}", experiment_id);

        Ok(())
    }

    /// Get experiment status
    pub fn get_status(&self, experiment_id: &str) -> Result<ExperimentStatus, ControllerError> {
        let experiments = self.experiments.read();
        let experiment = experiments
            .get(experiment_id)
            .ok_or_else(|| ControllerError::ExperimentNotFound(experiment_id.to_string()))?;

        let workers_info = experiment.coordinator.get_workers();
        let ready_count = workers_info.iter().filter(|w| w.ready).count() as u32;
        let completed_count = workers_info.iter().filter(|w| w.completed).count() as u32;

        let workers: Vec<WorkerStatus> = workers_info
            .iter()
            .map(|w| WorkerStatus {
                worker_id: w.id,
                pod_name: w.pod_name.clone(),
                ready: w.ready,
                completed: w.completed,
            })
            .collect();

        Ok(ExperimentStatus {
            id: experiment_id.to_string(),
            replicas: experiment.replicas,
            ready: ready_count,
            completed: completed_count,
            phase: experiment.phase,
            workers,
        })
    }

    /// Collect and aggregate results from all workers
    pub async fn collect_results(
        &self,
        experiment_id: &str,
    ) -> Result<CollectResult, ControllerError> {
        info!("Collecting results for experiment: {}", experiment_id);

        self.update_phase(experiment_id, ExperimentPhase::Collecting)?;

        let (coordinator, scenario_config) = {
            let experiments = self.experiments.read();
            let experiment = experiments
                .get(experiment_id)
                .ok_or_else(|| ControllerError::ExperimentNotFound(experiment_id.to_string()))?;
            (experiment.coordinator.clone(), experiment.scenario_config.clone())
        };

        // Create local directory for this experiment's results
        let exp_results_dir = PathBuf::from(&self.local_results_dir).join(experiment_id);
        tokio::fs::create_dir_all(&exp_results_dir).await?;

        // Get all worker info
        let workers = coordinator.get_workers();

        // Collect JSONL files from each worker pod
        let mut jsonl_files = Vec::new();
        for worker in &workers {
            let local_path = exp_results_dir.join(format!("worker_{}.jsonl", worker.id));
            match self
                .k8s_client
                .copy_results_from_pod(&worker.pod_name, "/app/results/results.jsonl", &local_path)
                .await
            {
                Ok(_) => {
                    info!("Collected results from worker {}", worker.id);
                    jsonl_files.push(local_path);
                }
                Err(e) => {
                    warn!("Failed to collect from worker {}: {}", worker.id, e);
                }
            }
        }

        // Aggregate results
        let aggregator = ResultAggregator::new();
        let merged_path = exp_results_dir.join("merged_results.jsonl");
        let summary = aggregator
            .aggregate(&jsonl_files, &merged_path)
            .await
            .map_err(|e| ControllerError::StorageError(e.to_string()))?;

        // Upload to storage if configured
        let artifact_uri = if let Some(ref storage_base) = self.storage_uri {
            let uri = format!("{}/{}/results.jsonl", storage_base.trim_end_matches('/'), experiment_id);
            storage::upload(&merged_path, &uri)
                .await
                .map_err(|e| ControllerError::StorageError(e.to_string()))?;
            uri
        } else {
            format!("file://{}", merged_path.display())
        };

        // Update experiment state
        self.update_phase(experiment_id, ExperimentPhase::Completed)?;
        {
            let mut experiments = self.experiments.write();
            if let Some(exp) = experiments.get_mut(experiment_id) {
                exp.completed_at = Some(Utc::now());
            }
        }

        info!(
            "Experiment {} completed. Results: {}",
            experiment_id, artifact_uri
        );

        Ok(CollectResult {
            artifact_uri,
            events: summary.total_events,
            duration_sec: summary.total_duration_sec,
            summary,
        })
    }

    /// Delete an experiment and its resources
    pub async fn delete_experiment(&self, experiment_id: &str) -> Result<(), ControllerError> {
        info!("Deleting experiment: {}", experiment_id);

        // Delete Kubernetes resources
        self.k8s_client.delete_job(experiment_id).await?;
        
        let configmap_name = format!("qr-experiment-{}-scenario", experiment_id);
        let _ = self.k8s_client.delete_configmap(&configmap_name).await;

        // Remove from memory
        let mut experiments = self.experiments.write();
        experiments.remove(experiment_id);

        info!("Experiment {} deleted", experiment_id);

        Ok(())
    }

    /// Get list of all experiments
    pub fn list_experiments(&self) -> Vec<Experiment> {
        let experiments = self.experiments.read();
        experiments.values().cloned().collect()
    }

    /// Get orchestrator's current timestamp (for time sync)
    pub fn current_timestamp_ns(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }

    fn update_phase(
        &self,
        experiment_id: &str,
        phase: ExperimentPhase,
    ) -> Result<(), ControllerError> {
        let mut experiments = self.experiments.write();
        let experiment = experiments
            .get_mut(experiment_id)
            .ok_or_else(|| ControllerError::ExperimentNotFound(experiment_id.to_string()))?;
        experiment.phase = phase;
        Ok(())
    }
}

