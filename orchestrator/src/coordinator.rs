//! Experiment Coordinator
//!
//! Manages worker registration, synchronization barriers, and signaling.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

/// Information about a registered worker
#[derive(Debug, Clone)]
pub struct WorkerInfo {
    pub id: u32,
    pub pod_name: String,
    pub pod_ip: String,
    pub registered_at: u64,
    pub ready: bool,
    pub completed: bool,
    pub time_drift_ns: i64,
}

/// Coordinates workers in a distributed experiment
#[derive(Debug)]
pub struct ExperimentCoordinator {
    experiment_id: String,
    expected_replicas: u32,
    max_time_drift_ns: u64,
    next_worker_id: AtomicU32,
    global_start_ns: AtomicU64,
    workers: RwLock<HashMap<u32, WorkerInfo>>,
    started: RwLock<bool>,
}

impl ExperimentCoordinator {
    pub fn new(experiment_id: String, expected_replicas: u32, max_time_drift_ns: u64) -> Self {
        Self {
            experiment_id,
            expected_replicas,
            max_time_drift_ns,
            next_worker_id: AtomicU32::new(0),
            global_start_ns: AtomicU64::new(0),
            workers: RwLock::new(HashMap::new()),
            started: RwLock::new(false),
        }
    }

    /// Register a worker and return (worker_id, global_start_timestamp)
    pub fn register_worker(&self, pod_name: &str, pod_ip: &str) -> (u32, u64) {
        let worker_id = self.next_worker_id.fetch_add(1, Ordering::SeqCst);
        let now_ns = current_time_ns();

        let worker = WorkerInfo {
            id: worker_id,
            pod_name: pod_name.to_string(),
            pod_ip: pod_ip.to_string(),
            registered_at: now_ns,
            ready: false,
            completed: false,
            time_drift_ns: 0,
        };

        {
            let mut workers = self.workers.write();
            workers.insert(worker_id, worker);
        }

        info!(
            "Worker {} registered for experiment {} (pod: {})",
            worker_id, self.experiment_id, pod_name
        );

        let global_start = self.global_start_ns.load(Ordering::SeqCst);
        (worker_id, global_start)
    }

    /// Mark a worker as ready
    pub fn mark_worker_ready(&self, worker_id: u32, worker_time_ns: u64) {
        let orchestrator_time = current_time_ns();
        let drift = worker_time_ns as i64 - orchestrator_time as i64;

        let mut workers = self.workers.write();
        if let Some(worker) = workers.get_mut(&worker_id) {
            worker.ready = true;
            worker.time_drift_ns = drift;

            if drift.unsigned_abs() > self.max_time_drift_ns {
                warn!(
                    "Worker {} has time drift of {} ns (max allowed: {} ns)",
                    worker_id,
                    drift,
                    self.max_time_drift_ns
                );
            }
        }
    }

    /// Mark a worker as completed
    pub fn mark_worker_completed(&self, worker_id: u32) {
        let mut workers = self.workers.write();
        if let Some(worker) = workers.get_mut(&worker_id) {
            worker.completed = true;
        }
    }

    /// Calculate global start time based on delay
    pub fn calculate_global_start(&self, delay_ms: u64) -> u64 {
        let now = current_time_ns();
        let start_time = now + (delay_ms * 1_000_000);
        self.global_start_ns.store(start_time, Ordering::SeqCst);
        start_time
    }

    /// Get the global start timestamp
    pub fn get_global_start(&self) -> u64 {
        self.global_start_ns.load(Ordering::SeqCst)
    }

    /// Signal all workers to start
    pub async fn signal_start(&self, global_start_ns: u64) {
        self.global_start_ns.store(global_start_ns, Ordering::SeqCst);
        *self.started.write() = true;

        // Clone worker info to release the lock before async operations
        let workers: Vec<WorkerInfo> = {
            let workers_guard = self.workers.read();
            workers_guard.values().cloned().collect()
        };

        let client = reqwest::Client::new();

        for worker in &workers {
            let url = format!("http://{}:6060/start_signal", worker.pod_ip);
            let body = serde_json::json!({
                "globalStartUnixNs": global_start_ns
            });

            match client.post(&url).json(&body).send().await {
                Ok(resp) if resp.status().is_success() => {
                    info!("Sent start signal to worker {} ({})", worker.id, worker.pod_name);
                }
                Ok(resp) => {
                    warn!(
                        "Worker {} returned error on start signal: {}",
                        worker.id,
                        resp.status()
                    );
                }
                Err(e) => {
                    warn!("Failed to send start signal to worker {}: {}", worker.id, e);
                }
            }
        }
    }

    /// Signal all workers to stop
    pub async fn signal_stop(&self) {
        // Clone worker info to release the lock before async operations
        let workers: Vec<WorkerInfo> = {
            let workers_guard = self.workers.read();
            workers_guard.values().cloned().collect()
        };

        let client = reqwest::Client::new();

        for worker in &workers {
            let url = format!("http://{}:6060/shutdown", worker.pod_ip);

            match client.post(&url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    info!("Sent stop signal to worker {} ({})", worker.id, worker.pod_name);
                }
                Ok(resp) => {
                    warn!(
                        "Worker {} returned error on stop signal: {}",
                        worker.id,
                        resp.status()
                    );
                }
                Err(e) => {
                    warn!("Failed to send stop signal to worker {}: {}", worker.id, e);
                }
            }
        }
    }

    /// Check if all expected workers are ready
    pub fn all_workers_ready(&self) -> bool {
        let workers = self.workers.read();
        if workers.len() < self.expected_replicas as usize {
            return false;
        }
        workers.values().all(|w| w.ready)
    }

    /// Check if all workers have completed
    pub fn all_workers_completed(&self) -> bool {
        let workers = self.workers.read();
        if workers.is_empty() {
            return false;
        }
        workers.values().all(|w| w.completed)
    }

    /// Get number of ready workers
    pub fn ready_count(&self) -> u32 {
        let workers = self.workers.read();
        workers.values().filter(|w| w.ready).count() as u32
    }

    /// Get all worker info
    pub fn get_workers(&self) -> Vec<WorkerInfo> {
        let workers = self.workers.read();
        workers.values().cloned().collect()
    }

    /// Check if experiment has started
    pub fn is_started(&self) -> bool {
        *self.started.read()
    }
}

/// Get current time in nanoseconds since UNIX epoch
fn current_time_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

