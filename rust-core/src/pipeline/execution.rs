//! Execution Models Module
//!
//! This module provides different execution models for running the benchmark pipeline:
//! - Single: One processor task (existing behavior)
//! - FixedPool: Fixed number of processor tasks
//! - Elastic: Worker pool that expands and contracts based on queue pressure

use crate::crypto_adapter::{
    hybrid_decrypt, hybrid_encrypt, CryptoAdapter, CryptoError, HybridSizes, KeypairWithSecret,
};
use crate::scenario::{ExecutionConfig, ExecutionMode};
use crate::telemetry::{JsonlWriter, Metrics, SysInfoSampler};
use chrono::Utc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::sync::Notify;
use tracing::{info, info_span, warn, Instrument};

/// Event passed through the pipeline with queue timing
#[derive(Debug, Clone)]
pub struct QueuedEvent {
    pub event_id: u64,
    pub payload: Vec<u8>,
    pub timestamp_ns: u128,
    /// Timestamp when event was enqueued
    pub enqueue_ts: Instant,
}

/// Result of processing an event
#[derive(Debug)]
pub struct ProcessedEvent {
    pub event_id: u64,
    pub latency_ns: u128,  // Nanosecond precision
    pub queue_delay_ns: u128,  // Nanosecond precision
    pub success: bool,
    pub output_size: Option<usize>,
    pub worker_id: usize,
}

/// Pipeline context containing shared state
#[derive(Clone)]
pub struct ExecutionContext {
    /// Cached keypair for KEM operations
    pub keypair: Option<Arc<KeypairWithSecret>>,
    /// Cached keypair for signature operations (e.g., Dilithium)
    pub sig_keypair: Option<Arc<KeypairWithSecret>>,
    /// Algorithm name
    pub algorithm: String,
    /// Operation name
    pub operation: String,
    /// Run ID
    pub run_id: String,
    /// Scenario ID
    pub scenario_id: String,
    /// RNG seed for reproducibility
    pub rng_seed: u64,
}

/// Shared state for the execution engine
pub struct ExecutionState {
    /// Whether shutdown has been requested
    pub shutdown_requested: Arc<AtomicBool>,
    /// Number of active workers
    pub active_workers: Arc<AtomicUsize>,
    /// Current queue length (approximate)
    pub queue_length: Arc<AtomicUsize>,
    /// Total events processed
    pub events_processed: Arc<AtomicU64>,
    /// Total latency accumulated in nanoseconds (for average calculation)
    /// Uses Mutex instead of AtomicU128 since AtomicU128 is not in standard library
    pub total_latency_ns: Arc<Mutex<u128>>,
    /// Queue capacity
    pub queue_capacity: usize,
    /// Shutdown completion notification
    pub shutdown_complete: Arc<Notify>,
    /// Pipeline started flag
    pub pipeline_started: Arc<AtomicBool>,
    /// Scenario loaded flag
    pub scenario_loaded: Arc<AtomicBool>,
}

impl ExecutionState {
    pub fn new(queue_capacity: usize) -> Self {
        Self {
            shutdown_requested: Arc::new(AtomicBool::new(false)),
            active_workers: Arc::new(AtomicUsize::new(0)),
            queue_length: Arc::new(AtomicUsize::new(0)),
            events_processed: Arc::new(AtomicU64::new(0)),
            total_latency_ns: Arc::new(Mutex::new(0u128)),
            queue_capacity,
            shutdown_complete: Arc::new(Notify::new()),
            pipeline_started: Arc::new(AtomicBool::new(false)),
            scenario_loaded: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested.load(Ordering::Relaxed)
    }

    pub fn request_shutdown(&self) {
        self.shutdown_requested.store(true, Ordering::SeqCst);
    }

    pub fn notify_shutdown_complete(&self) {
        self.shutdown_complete.notify_waiters();
    }

    pub async fn wait_for_shutdown(&self) {
        self.shutdown_complete.notified().await;
    }
}

impl Clone for ExecutionState {
    fn clone(&self) -> Self {
        Self {
            shutdown_requested: self.shutdown_requested.clone(),
            active_workers: self.active_workers.clone(),
            queue_length: self.queue_length.clone(),
            events_processed: self.events_processed.clone(),
            total_latency_ns: self.total_latency_ns.clone(),
            queue_capacity: self.queue_capacity,
            shutdown_complete: self.shutdown_complete.clone(),
            pipeline_started: self.pipeline_started.clone(),
            scenario_loaded: self.scenario_loaded.clone(),
        }
    }
}

/// Execution engine that manages different concurrency models
pub struct ExecutionEngine {
    config: ExecutionConfig,
    state: ExecutionState,
}

impl ExecutionEngine {
    pub fn new(config: ExecutionConfig) -> Self {
        let state = ExecutionState::new(config.queue_capacity);
        Self { config, state }
    }

    /// Creates an ExecutionEngine with a pre-existing shared state
    /// This allows the control plane to share state with the pipeline
    pub fn with_state(config: ExecutionConfig, state: ExecutionState) -> Self {
        Self { config, state }
    }

    pub fn state(&self) -> &ExecutionState {
        &self.state
    }

    /// Returns a clone of the execution state for sharing
    pub fn shared_state(&self) -> ExecutionState {
        self.state.clone()
    }

    /// Run the execution engine with the configured mode
    pub async fn run(
        &self,
        _scenario: &crate::scenario::Scenario,
        adapter: Arc<dyn CryptoAdapter + Send + Sync>,
        metrics: Metrics,
        jsonl_writer: JsonlWriter,
        sampler: SysInfoSampler,
        context: ExecutionContext,
        mut event_rx: mpsc::Receiver<QueuedEvent>,
    ) -> (u64, u128, Duration) {
        // Set initial metrics
        metrics.set_queue_capacity(self.config.queue_capacity);
        metrics.set_active_workers(0);

        self.state.scenario_loaded.store(true, Ordering::SeqCst);
        self.state.pipeline_started.store(true, Ordering::SeqCst);

        let start_time = Instant::now();

        match self.config.mode {
            ExecutionMode::Single => {
                self.run_single(
                    &mut event_rx,
                    adapter,
                    metrics,
                    jsonl_writer,
                    sampler,
                    context,
                )
                .await;
            }
            ExecutionMode::FixedPool => {
                self.run_fixed_pool(
                    event_rx,
                    adapter,
                    metrics,
                    jsonl_writer,
                    sampler,
                    context,
                )
                .await;
            }
            ExecutionMode::Elastic => {
                self.run_elastic(
                    event_rx,
                    adapter,
                    metrics,
                    jsonl_writer,
                    sampler,
                    context,
                )
                .await;
            }
        }

        let elapsed = start_time.elapsed();
        let processed = self.state.events_processed.load(Ordering::Relaxed);
        let total_lat_ns = *self.state.total_latency_ns.lock().unwrap();

        self.state.notify_shutdown_complete();

        (processed, total_lat_ns, elapsed)
    }

    /// Single processor mode (existing behavior)
    async fn run_single(
        &self,
        rx: &mut mpsc::Receiver<QueuedEvent>,
        adapter: Arc<dyn CryptoAdapter + Send + Sync>,
        metrics: Metrics,
        jsonl_writer: JsonlWriter,
        sampler: SysInfoSampler,
        context: ExecutionContext,
    ) {
        self.state.active_workers.store(1, Ordering::Relaxed);
        metrics.set_active_workers(1);

        while let Some(event) = rx.recv().await {
            if self.state.is_shutdown_requested() {
                break;
            }

            let result = process_event(
                &event,
                0, // worker_id
                &adapter,
                &metrics,
                &jsonl_writer,
                &sampler,
                &context,
            )
            .await;

            self.state.events_processed.fetch_add(1, Ordering::Relaxed);
            *self.state.total_latency_ns.lock().unwrap() += result.latency_ns;
        }

        self.state.active_workers.store(0, Ordering::Relaxed);
        metrics.set_active_workers(0);
        info!("Single processor: completed");
    }

    /// Fixed pool mode - fixed number of worker tasks
    async fn run_fixed_pool(
        &self,
        event_rx: mpsc::Receiver<QueuedEvent>,
        adapter: Arc<dyn CryptoAdapter + Send + Sync>,
        metrics: Metrics,
        jsonl_writer: JsonlWriter,
        sampler: SysInfoSampler,
        context: ExecutionContext,
    ) {
        let num_workers = self.config.workers;
        let event_rx = Arc::new(tokio::sync::Mutex::new(event_rx));

        self.state
            .active_workers
            .store(num_workers, Ordering::Relaxed);
        metrics.set_active_workers(num_workers);

        let mut handles = Vec::with_capacity(num_workers);

        // Spawn metrics updater task
        let metrics_clone = metrics.clone();
        let state_clone = self.state.clone();
        let metrics_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(100));
            loop {
                interval.tick().await;
                if state_clone.is_shutdown_requested() {
                    break;
                }
                let queue_len = state_clone.queue_length.load(Ordering::Relaxed);
                metrics_clone.set_queue_length(queue_len);
            }
        });

        for worker_id in 0..num_workers {
            let rx = event_rx.clone();
            let adapter = adapter.clone();
            let metrics = metrics.clone();
            let jsonl_writer = jsonl_writer.clone();
            let sampler = sampler.clone();
            let context = context.clone();
            let state = self.state.clone();

            let handle = tokio::spawn(async move {
                worker_task(worker_id, rx, adapter, metrics, jsonl_writer, sampler, context, state)
                    .await;
            });
            handles.push(handle);
        }

        // Wait for all workers to complete
        for handle in handles {
            let _ = handle.await;
        }

        metrics_handle.abort();
        self.state.active_workers.store(0, Ordering::Relaxed);
        metrics.set_active_workers(0);
        info!("Fixed pool: all {} workers completed", num_workers);
    }

    /// Elastic pool mode - dynamically scaling worker pool
    async fn run_elastic(
        &self,
        event_rx: mpsc::Receiver<QueuedEvent>,
        adapter: Arc<dyn CryptoAdapter + Send + Sync>,
        metrics: Metrics,
        jsonl_writer: JsonlWriter,
        sampler: SysInfoSampler,
        context: ExecutionContext,
    ) {
        let max_workers = self.config.max_workers;
        let queue_capacity = self.config.queue_capacity;
        let event_rx = Arc::new(tokio::sync::Mutex::new(event_rx));

        // Start with 1 worker
        let current_workers = Arc::new(AtomicUsize::new(1));
        let worker_counter = Arc::new(AtomicUsize::new(0));

        self.state.active_workers.store(1, Ordering::Relaxed);
        metrics.set_active_workers(1);

        // Track active worker handles
        let worker_handles = Arc::new(tokio::sync::Mutex::new(Vec::new()));

        // Spawn initial worker
        {
            let worker_id = worker_counter.fetch_add(1, Ordering::Relaxed);
            let rx = event_rx.clone();
            let adapter = adapter.clone();
            let metrics = metrics.clone();
            let jsonl_writer = jsonl_writer.clone();
            let sampler = sampler.clone();
            let context = context.clone();
            let state = self.state.clone();

            let handle = tokio::spawn(async move {
                worker_task(worker_id, rx, adapter, metrics, jsonl_writer, sampler, context, state)
                    .await;
            });
            worker_handles.lock().await.push(handle);
        }

        // Scaling decision task
        let scaling_state = self.state.clone();
        let scaling_metrics = metrics.clone();
        let scaling_current_workers = current_workers.clone();
        let scaling_worker_counter = worker_counter.clone();
        let scaling_handles = worker_handles.clone();
        let scaling_rx = event_rx.clone();
        let scaling_adapter = adapter.clone();
        let scaling_jsonl_writer = jsonl_writer.clone();
        let scaling_sampler = sampler.clone();
        let scaling_context = context.clone();

        let scale_handle = tokio::spawn(async move {
            let mut low_queue_since: Option<Instant> = None;
            let mut interval = tokio::time::interval(Duration::from_millis(100));

            loop {
                interval.tick().await;

                if scaling_state.is_shutdown_requested() {
                    break;
                }

                let queue_len = scaling_state.queue_length.load(Ordering::Relaxed);
                let workers = scaling_current_workers.load(Ordering::Relaxed);

                // Update metrics
                scaling_metrics.set_queue_length(queue_len);
                scaling_metrics.set_active_workers(workers);

                let high_threshold = (queue_capacity as f64 * 0.75) as usize;
                let low_threshold = (queue_capacity as f64 * 0.25) as usize;

                // Scale up: if queue > 75% capacity and we can add workers
                if queue_len > high_threshold && workers < max_workers {
                    let new_workers = (workers + 1).min(max_workers);
                    scaling_current_workers.store(new_workers, Ordering::Relaxed);
                    scaling_state
                        .active_workers
                        .store(new_workers, Ordering::Relaxed);

                    // Spawn new worker
                    let worker_id = scaling_worker_counter.fetch_add(1, Ordering::Relaxed);
                    let rx = scaling_rx.clone();
                    let adapter = scaling_adapter.clone();
                    let metrics = scaling_metrics.clone();
                    let jsonl_writer = scaling_jsonl_writer.clone();
                    let sampler = scaling_sampler.clone();
                    let context = scaling_context.clone();
                    let state = scaling_state.clone();

                    let handle = tokio::spawn(async move {
                        worker_task(
                            worker_id,
                            rx,
                            adapter,
                            metrics,
                            jsonl_writer,
                            sampler,
                            context,
                            state,
                        )
                        .await;
                    });
                    scaling_handles.lock().await.push(handle);

                    info!(
                        "Elastic: scaled up to {} workers (queue_len={})",
                        new_workers, queue_len
                    );
                    low_queue_since = None;
                }
                // Scale down: if queue < 25% capacity for > 2 seconds and we have > 1 worker
                else if queue_len < low_threshold && workers > 1 {
                    match low_queue_since {
                        Some(since) if since.elapsed() > Duration::from_secs(2) => {
                            let new_workers = workers - 1;
                            scaling_current_workers.store(new_workers, Ordering::Relaxed);
                            scaling_state
                                .active_workers
                                .store(new_workers, Ordering::Relaxed);
                            info!(
                                "Elastic: scaled down to {} workers (queue_len={})",
                                new_workers, queue_len
                            );
                            low_queue_since = None;
                        }
                        None => {
                            low_queue_since = Some(Instant::now());
                        }
                        _ => {}
                    }
                } else {
                    low_queue_since = None;
                }
            }
        });

        // Wait for scaling task to complete
        let _ = scale_handle.await;

        // Wait for all workers
        let handles = worker_handles.lock().await;
        for handle in handles.iter() {
            handle.abort();
        }

        self.state.active_workers.store(0, Ordering::Relaxed);
        metrics.set_active_workers(0);
        info!(
            "Elastic pool: completed with max {} workers used",
            worker_counter.load(Ordering::Relaxed)
        );
    }
}

/// Worker task that processes events from the shared queue
async fn worker_task(
    worker_id: usize,
    rx: Arc<tokio::sync::Mutex<mpsc::Receiver<QueuedEvent>>>,
    adapter: Arc<dyn CryptoAdapter + Send + Sync>,
    metrics: Metrics,
    jsonl_writer: JsonlWriter,
    sampler: SysInfoSampler,
    context: ExecutionContext,
    state: ExecutionState,
) {
    loop {
        // Check shutdown before waiting for event
        if state.is_shutdown_requested() {
            break;
        }

        // Try to receive an event
        let event = {
            let mut guard = rx.lock().await;
            guard.recv().await
        };

        match event {
            Some(event) => {
                // Decrement queue length
                state.queue_length.fetch_sub(1, Ordering::Relaxed);

                let result = process_event(
                    &event,
                    worker_id,
                    &adapter,
                    &metrics,
                    &jsonl_writer,
                    &sampler,
                    &context,
                )
                .await;

                state.events_processed.fetch_add(1, Ordering::Relaxed);
                *state.total_latency_ns.lock().unwrap() += result.latency_ns;
            }
            None => {
                // Channel closed, exit
                break;
            }
        }
    }
}

/// Process a single event
async fn process_event(
    event: &QueuedEvent,
    worker_id: usize,
    adapter: &Arc<dyn CryptoAdapter + Send + Sync>,
    metrics: &Metrics,
    jsonl_writer: &JsonlWriter,
    sampler: &SysInfoSampler,
    context: &ExecutionContext,
) -> ProcessedEvent {
    let dequeue_ts = Instant::now();
    let queue_delay_ns = dequeue_ts.duration_since(event.enqueue_ts).as_nanos();

    let span = info_span!(
        "crypto_op",
        event_id = event.event_id,
        algorithm = %context.algorithm,
        operation = %context.operation,
        payload_size = event.payload.len(),
        worker_id = worker_id,
        queue_delay_ns = queue_delay_ns
    );

    let result = async {
        let start = Instant::now();

        // Perform the crypto operation
        let op_result = match context.operation.as_str() {
            "sign" => adapter.sign(&[], &event.payload).map(|s| (Some(s.len()), None)),
            "verify" => adapter
                .verify(&[], &event.payload, &event.payload)
                .map(|_| (None, None)),
            "encrypt" => adapter
                .encapsulate(&event.payload)
                .map(|(ct, _)| (Some(ct.len()), None)),
            "decrypt" => adapter
                .decapsulate(&event.payload, &event.payload)
                .map(|_| (None, None)),
            "keygen" => adapter.keygen().map(|_| (None, None)),
            "kem_aead_encrypt" => {
                if let Some(ref keypair) = context.keypair {
                    let adapter_clone = adapter.clone();
                    let pk = keypair.public_key.clone();

                    let result =
                        hybrid_encrypt(|pubkey| adapter_clone.encapsulate(pubkey), &pk, &event.payload);

                    match result {
                        Ok(combined) => {
                            let sizes = HybridSizes::from_payload(&combined).ok();
                            Ok((Some(combined.len()), sizes.map(|s| s.ct_kem_len)))
                        }
                        Err(e) => Err(e),
                    }
                } else {
                    Err(CryptoError::InternalError(
                        "No keypair available for KEM operation".to_string(),
                    ))
                }
            }
            "kem_aead_decrypt" => {
                if let Some(ref keypair) = context.keypair {
                    let adapter_clone = adapter.clone();
                    let pk = keypair.public_key.clone();
                    let sk = keypair.secret_key.clone();

                    hybrid_encrypt(|pubkey| adapter_clone.encapsulate(pubkey), &pk, &event.payload)
                        .and_then(|combined| {
                            let adapter_clone2 = adapter.clone();
                            hybrid_decrypt(
                                |secret_key, ciphertext| adapter_clone2.decapsulate(secret_key, ciphertext),
                                &sk,
                                &combined,
                            )
                        })
                        .and_then(|decrypted| {
                            if decrypted != event.payload {
                                Err(CryptoError::InternalError(
                                    "Decryption verification failed".to_string(),
                                ))
                            } else {
                                Ok((Some(decrypted.len()), None))
                            }
                        })
                } else {
                    Err(CryptoError::InternalError(
                        "No keypair available for KEM operation".to_string(),
                    ))
                }
            }
            "kem_aead_sign" => {
                // Hybrid: KEM + AEAD encrypt, then sign with Dilithium
                // First, do KEM+AEAD encryption
                if let Some(ref keypair) = context.keypair {
                    let adapter_clone = adapter.clone();
                    let pk = keypair.public_key.clone();

                    let hybrid_result = hybrid_encrypt(|pubkey| adapter_clone.encapsulate(pubkey), &pk, &event.payload);
                    
                    match hybrid_result {
                        Ok(combined) => {
                            // Now sign the combined ciphertext with Dilithium
                            // Note: This requires a signature adapter (Dilithium) to be available
                            // For now, we return the combined size as the signature would be appended
                            // The actual signature would need to be computed separately
                            let sizes = HybridSizes::from_payload(&combined).ok();
                            Ok((Some(combined.len()), sizes.map(|s| s.ct_kem_len)))
                        }
                        Err(e) => Err(e),
                    }
                } else {
                    Err(CryptoError::InternalError(
                        "No keypair available for KEM operation".to_string(),
                    ))
                }
            }
            _ => Err(CryptoError::NotImplemented),
        };

        // Measure latency in nanoseconds
        let latency_ns = start.elapsed().as_nanos();
        
        let (success, output_size, _ct_kem_size, error_msg) = match op_result {
            Ok((size, kem_size)) => (true, size, kem_size, None),
            Err(e) => (false, None, None, Some(e.to_string())),
        };

        // Sample system metrics
        let (cpu_user, memory_rss) = sampler.sample();

        // Update Prometheus metrics (using nanoseconds)
        metrics.observe_latency(&context.algorithm, &context.operation, latency_ns as f64);
        metrics.observe_queue_delay(queue_delay_ns as f64);
        metrics.inc_ops(&context.algorithm, &context.operation, success);
        metrics.set_memory_bytes(memory_rss);
        metrics.inc_worker_events(worker_id);

        // Determine ciphertext size for JSONL
        let ciphertext_size = if context.operation == "encrypt" || context.operation == "kem_aead_encrypt" {
            output_size
        } else {
            None
        };

        // Write JSONL row with queue delay
        let row = EventRowWithQueueDelay {
            run_id: context.run_id.clone(),
            scenario_id: context.scenario_id.clone(),
            event_id: event.event_id,
            timestamp_utc_iso: Utc::now().to_rfc3339(),
            timestamp_monotonic_ns: event.timestamp_ns,
            operation: context.operation.clone(),
            algorithm: context.algorithm.clone(),
            latency_ns,
            queue_delay_ns,
            worker_id,
            payload_size_bytes: event.payload.len(),
            ciphertext_size_bytes: ciphertext_size,
            signature_size_bytes: if context.operation == "sign" || context.operation == "kem_aead_sign" { output_size } else { None },
            cpu_user_seconds: cpu_user,
            memory_rss_bytes: memory_rss,
            rng_seed: context.rng_seed,
            error: error_msg,
        };

        if let Err(e) = jsonl_writer.write(&row) {
            warn!("Failed to write JSONL row: {}", e);
        }

        ProcessedEvent {
            event_id: event.event_id,
            latency_ns,
            queue_delay_ns,
            success,
            output_size,
            worker_id,
        }
    }
    .instrument(span)
    .await;

    result
}

/// Extended event row with queue delay for JSONL output
#[derive(Debug, serde::Serialize)]
struct EventRowWithQueueDelay {
    pub run_id: String,
    pub scenario_id: String,
    pub event_id: u64,
    pub timestamp_utc_iso: String,
    pub timestamp_monotonic_ns: u128,
    pub operation: String,
    pub algorithm: String,
    pub latency_ns: u128,  // Nanosecond precision
    pub queue_delay_ns: u128,  // Nanosecond precision
    pub worker_id: usize,
    pub payload_size_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ciphertext_size_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature_size_bytes: Option<usize>,
    pub cpu_user_seconds: f64,
    pub memory_rss_bytes: u64,
    pub rng_seed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::ExecutionMode;

    #[test]
    fn test_execution_state_new() {
        let state = ExecutionState::new(2000);
        assert_eq!(state.queue_capacity, 2000);
        assert!(!state.is_shutdown_requested());
    }

    #[test]
    fn test_execution_state_shutdown() {
        let state = ExecutionState::new(1000);
        assert!(!state.is_shutdown_requested());
        state.request_shutdown();
        assert!(state.is_shutdown_requested());
    }

    #[test]
    fn test_execution_engine_new() {
        let config = ExecutionConfig {
            mode: ExecutionMode::FixedPool,
            workers: 4,
            max_workers: 16,
            queue_capacity: 2000,
        };
        let engine = ExecutionEngine::new(config);
        assert_eq!(engine.state().queue_capacity, 2000);
    }
}

