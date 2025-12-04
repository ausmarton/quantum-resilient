//! Pipeline Module
//!
//! This module provides the async streaming pipeline infrastructure for running
//! cryptographic benchmarks in real-time data processing scenarios.

pub mod execution;
pub mod workload;

pub use execution::{ExecutionContext, ExecutionEngine, ExecutionState, ProcessedEvent, QueuedEvent};
pub use workload::{create_workload_model, WorkloadModel};

use crate::crypto_adapter::{CryptoAdapter, CryptoError, KeypairWithSecret};
use crate::scenario::Scenario;
use crate::telemetry::{JsonlWriter, Metrics, SysInfoSampler};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{info, warn};

/// Error type for pipeline operations
#[derive(Debug)]
pub enum PipelineError {
    /// Pipeline initialization failed
    InitializationError(String),
    /// Pipeline execution failed
    ExecutionError(String),
    /// Pipeline shutdown failed
    ShutdownError(String),
}

/// Configuration for the benchmark pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of worker threads
    pub num_workers: usize,
    /// Buffer size for streaming data
    pub buffer_size: usize,
    /// Whether to enable detailed metrics
    pub enable_metrics: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            num_workers: 4,
            buffer_size: 1000,
            enable_metrics: true,
        }
    }
}

/// Event passed through the pipeline (legacy format)
#[derive(Debug, Clone)]
pub struct PipelineEvent {
    pub event_id: u64,
    pub payload: Vec<u8>,
    pub timestamp_ns: u128,
}

/// Pipeline context containing shared state for KEM operations
#[derive(Clone)]
pub struct PipelineContext {
    /// Cached keypair for KEM operations
    pub keypair: Option<Arc<KeypairWithSecret>>,
}

impl Default for PipelineContext {
    fn default() -> Self {
        Self { keypair: None }
    }
}

/// Statistics from a pipeline run
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub total_events: u64,
    pub events_processed: u64,
    pub duration: Duration,
    pub avg_latency_us: f64,
    /// Maximum number of active workers during the run
    pub max_active_workers: usize,
}

/// The main pipeline struct for orchestrating benchmark runs
pub struct Pipeline {
    config: PipelineConfig,
}

impl Pipeline {
    /// Creates a new Pipeline with default configuration
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
        }
    }

    /// Creates a new Pipeline with the given configuration
    pub fn with_config(config: PipelineConfig) -> Self {
        Self { config }
    }

    /// Runs the benchmark pipeline asynchronously with the new execution models
    /// 
    /// # Arguments
    /// * `scenario` - The scenario configuration
    /// * `adapter` - The crypto adapter to use
    /// * `metrics` - Prometheus metrics instance
    /// * `jsonl_writer` - JSONL output writer
    /// * `sampler` - System info sampler
    /// * `shared_state` - Optional shared execution state for control plane integration
    pub async fn run_async(
        &self,
        scenario: &Scenario,
        adapter: Arc<dyn CryptoAdapter + Send + Sync>,
        metrics: Metrics,
        jsonl_writer: JsonlWriter,
        sampler: SysInfoSampler,
        shared_state: Option<ExecutionState>,
    ) -> Result<PipelineStats, PipelineError> {
        // Generate keypair if needed for KEM operations
        let context = if scenario.requires_keypair() {
            info!("Generating keypair for KEM operations...");
            let meta = adapter
                .keygen()
                .map_err(|e| PipelineError::InitializationError(e.to_string()))?;

            // For Kyber, we need the full keypair including secret key
            let (pk, sk) = generate_keypair_with_secret(&adapter)?;

            let keypair = KeypairWithSecret {
                public_key: pk,
                secret_key: sk,
                params: meta.params,
            };

            info!(
                "Keypair generated: pk_len={}, sk_len={}",
                keypair.public_key.len(),
                keypair.secret_key.len()
            );

            PipelineContext {
                keypair: Some(Arc::new(keypair)),
            }
        } else {
            PipelineContext::default()
        };

        // Create execution context
        let exec_context = ExecutionContext {
            keypair: context.keypair.clone(),
            sig_keypair: None, // Will be set for hybrid operations
            algorithm: adapter.name().to_string(),
            operation: scenario.algorithm.operation.clone(),
            run_id: scenario.id.clone(),
            scenario_id: scenario.id.clone(),
            rng_seed: scenario.effective_rng_seed(),
        };

        // Create workload model
        let mut workload_model = create_workload_model(&scenario.workload);
        info!("Workload model: {}", workload_model.description());

        // Use shared state if provided, otherwise create new execution engine
        let (execution_engine, state) = if let Some(shared) = shared_state {
            // Create engine that uses the shared state
            let engine = ExecutionEngine::with_state(scenario.execution.clone(), shared.clone());
            (engine, shared)
        } else {
            let engine = ExecutionEngine::new(scenario.execution.clone());
            let state = engine.shared_state();
            (engine, state)
        };

        // Create the event channel
        let queue_capacity = scenario.execution.queue_capacity;
        let (producer_tx, consumer_rx) = mpsc::channel::<QueuedEvent>(queue_capacity);

        // Calculate total events based on workload
        let total_events = calculate_total_events(scenario, &mut *workload_model);
        info!("Total events to generate: {}", total_events);

        // Spawn the producer task
        let producer_state = state.clone();
        let producer_scenario = scenario.clone();
        let producer_handle = tokio::spawn(async move {
            producer_task_advanced(
                producer_tx,
                &producer_scenario,
                workload_model,
                producer_state,
            )
            .await;
        });

        // Run the execution engine
        let (events_processed, total_latency_us, elapsed) = execution_engine
            .run(
                scenario,
                adapter.clone(),
                metrics.clone(),
                jsonl_writer.clone(),
                sampler,
                exec_context,
                consumer_rx,
            )
            .await;

        // Wait for producer to finish
        let _ = producer_handle.await;

        let avg_latency = if events_processed > 0 {
            total_latency_us as f64 / events_processed as f64
        } else {
            0.0
        };

        Ok(PipelineStats {
            total_events,
            events_processed,
            duration: elapsed,
            avg_latency_us: avg_latency,
            max_active_workers: scenario.execution.workers.max(1),
        })
    }

    /// Runs a single timed cryptographic operation (legacy sync method)
    pub fn run_timed_operation(
        adapter: &dyn CryptoAdapter,
        operation: &str,
        payload: &[u8],
    ) -> Result<u128, CryptoError> {
        let start = Instant::now();

        match operation {
            "sign" => {
                adapter.sign(&[], payload)?;
            }
            "verify" => {
                adapter.verify(&[], payload, payload)?;
            }
            "encrypt" => {
                adapter.encapsulate(payload)?;
            }
            "decrypt" => {
                adapter.decapsulate(payload, payload)?;
            }
            "keygen" => {
                adapter.keygen()?;
            }
            _ => return Err(CryptoError::NotImplemented),
        }

        let duration = start.elapsed().as_micros();
        Ok(duration)
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Generates a keypair with both public and secret key for KEM operations
fn generate_keypair_with_secret(
    adapter: &Arc<dyn CryptoAdapter + Send + Sync>,
) -> Result<(Vec<u8>, Vec<u8>), PipelineError> {
    // For Kyber adapter, we need to call keygen and get both keys
    if adapter.name() == "kyber" {
        #[cfg(feature = "pqcrypto_fallback")]
        {
            use pqcrypto_kyber::kyber512;
            use pqcrypto_traits::kem::{PublicKey, SecretKey};

            let (pk, sk) = kyber512::keypair();
            return Ok((pk.as_bytes().to_vec(), sk.as_bytes().to_vec()));
        }

        #[cfg(not(feature = "pqcrypto_fallback"))]
        {
            return Err(PipelineError::InitializationError(
                "No PQC implementation available".to_string(),
            ));
        }
    }

    // For other adapters, use the standard keygen
    let meta = adapter
        .keygen()
        .map_err(|e| PipelineError::InitializationError(e.to_string()))?;

    // For non-KEM adapters, return dummy secret key
    Ok((meta.public_key, vec![0u8; meta.secret_key_length]))
}

/// Calculate total events based on workload model
fn calculate_total_events(scenario: &Scenario, workload_model: &mut dyn WorkloadModel) -> u64 {
    let duration_ms = workload_model.duration_ms().unwrap_or(scenario.workload.duration_sec * 1000);
    let mut total_events = 0u64;
    let step_ms = 100; // Sample every 100ms

    for elapsed in (0..duration_ms).step_by(step_ms as usize) {
        let rps = workload_model.next_rps(elapsed);
        total_events += rps * step_ms / 1000;
    }

    // Reset the workload model for actual use
    // Note: For trace workloads, we recreate from config in the actual producer
    total_events.max(1)
}

/// Advanced producer task that uses workload models
async fn producer_task_advanced(
    tx: mpsc::Sender<QueuedEvent>,
    scenario: &Scenario,
    mut workload_model: Box<dyn WorkloadModel>,
    state: ExecutionState,
) {
    let payload_size = scenario.workload.msg_size_bytes;
    let duration_ms = workload_model.duration_ms().unwrap_or(scenario.workload.duration_sec * 1000);

    let start = Instant::now();
    let mut event_id = 0u64;
    let mut events_this_second = 0u64;
    let mut last_second = 0u64;

    while start.elapsed().as_millis() < duration_ms as u128 {
        // Check for shutdown
        if state.is_shutdown_requested() {
            info!("Producer: shutdown requested, stopping");
            break;
        }

        let elapsed_ms = start.elapsed().as_millis() as u64;
        let current_second = elapsed_ms / 1000;

        // Get current target RPS
        let target_rps = workload_model.next_rps(elapsed_ms);

        // Reset counter on new second
        if current_second != last_second {
            events_this_second = 0;
            last_second = current_second;
        }

        // Check if we should emit an event
        let events_per_second = target_rps;
        let interval_us = if events_per_second > 0 {
            1_000_000 / events_per_second
        } else {
            1_000_000 // 1 second if 0 RPS
        };

        // Have we emitted enough events this second?
        let expected_events = (elapsed_ms % 1000) * events_per_second / 1000;
        if events_this_second < expected_events || events_this_second == 0 {
            event_id += 1;
            events_this_second += 1;

            let payload = vec![0xAB; payload_size];
            let event = QueuedEvent {
                event_id,
                payload,
                timestamp_ns: start.elapsed().as_nanos(),
                enqueue_ts: Instant::now(),
            };

            // Update queue length before sending
            state.queue_length.fetch_add(1, Ordering::Relaxed);

            if tx.send(event).await.is_err() {
                state.queue_length.fetch_sub(1, Ordering::Relaxed);
                warn!("Producer: receiver dropped, stopping");
                break;
            }
        } else {
            // Sleep for a small interval before checking again
            tokio::time::sleep(Duration::from_micros(interval_us.min(1000))).await;
        }
    }

    info!("Producer: completed {} events", event_id);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_adapter::NoOpCryptoAdapter;

    #[test]
    fn test_pipeline_new() {
        let pipeline = Pipeline::new();
        assert_eq!(pipeline.config.num_workers, 4);
    }

    #[test]
    fn test_run_timed_operation_sign() {
        let adapter = NoOpCryptoAdapter;
        let payload = vec![0u8; 64];
        let result = Pipeline::run_timed_operation(&adapter, "sign", &payload);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_timed_operation_keygen() {
        let adapter = NoOpCryptoAdapter;
        let payload = vec![0u8; 64];
        let result = Pipeline::run_timed_operation(&adapter, "keygen", &payload);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_timed_operation_unknown() {
        let adapter = NoOpCryptoAdapter;
        let payload = vec![0u8; 64];
        let result = Pipeline::run_timed_operation(&adapter, "unknown_op", &payload);
        assert!(matches!(result, Err(CryptoError::NotImplemented)));
    }

    #[tokio::test]
    async fn test_pipeline_async_noop() {
        let scenario = Scenario {
            id: "test".to_string(),
            description: None,
            workload: crate::scenario::WorkloadConfig {
                msgs_per_sec: 10,
                msg_size_bytes: 64,
                duration_sec: 1,
                pattern: crate::scenario::WorkloadPattern::Constant,
                burst: None,
                ramp: None,
                trace_file: None,
            },
            algorithm: crate::scenario::AlgorithmConfig {
                adapter: "noop".to_string(),
                operation: "sign".to_string(),
            },
            metrics: crate::scenario::MetricsConfig::default(),
            execution: crate::scenario::ExecutionConfig::default(),
            rng_seed: Some(12345),
        };

        let pipeline = Pipeline::new();
        let adapter: Arc<dyn CryptoAdapter + Send + Sync> = Arc::new(NoOpCryptoAdapter);
        let metrics = Metrics::new().unwrap();
        let jsonl_writer = JsonlWriter::new("/tmp/test_pipeline.jsonl").unwrap();
        let sampler = SysInfoSampler::new();

        let stats = pipeline
            .run_async(&scenario, adapter, metrics, jsonl_writer, sampler, None)
            .await
            .unwrap();

        assert!(stats.events_processed > 0);
    }

    #[tokio::test]
    async fn test_pipeline_fixed_pool() {
        use crate::scenario::ExecutionMode;

        let scenario = Scenario {
            id: "test_fixed_pool".to_string(),
            description: None,
            workload: crate::scenario::WorkloadConfig {
                msgs_per_sec: 20,
                msg_size_bytes: 64,
                duration_sec: 1,
                pattern: crate::scenario::WorkloadPattern::Constant,
                burst: None,
                ramp: None,
                trace_file: None,
            },
            algorithm: crate::scenario::AlgorithmConfig {
                adapter: "noop".to_string(),
                operation: "sign".to_string(),
            },
            metrics: crate::scenario::MetricsConfig::default(),
            execution: crate::scenario::ExecutionConfig {
                mode: ExecutionMode::FixedPool,
                workers: 2,
                max_workers: 4,
                queue_capacity: 100,
            },
            rng_seed: Some(12345),
        };

        let pipeline = Pipeline::new();
        let adapter: Arc<dyn CryptoAdapter + Send + Sync> = Arc::new(NoOpCryptoAdapter);
        let metrics = Metrics::new().unwrap();
        let jsonl_writer = JsonlWriter::new("/tmp/test_fixed_pool.jsonl").unwrap();
        let sampler = SysInfoSampler::new();

        let stats = pipeline
            .run_async(&scenario, adapter, metrics, jsonl_writer, sampler, None)
            .await
            .unwrap();

        assert!(stats.events_processed > 0);
    }

    #[tokio::test]
    #[cfg(feature = "pqcrypto_fallback")]
    async fn test_pipeline_kyber_kem_aead_encrypt() {
        use crate::crypto_adapter::KyberAdapter;

        let scenario = Scenario {
            id: "kyber_test".to_string(),
            description: None,
            workload: crate::scenario::WorkloadConfig {
                msgs_per_sec: 5,
                msg_size_bytes: 64,
                duration_sec: 1,
                pattern: crate::scenario::WorkloadPattern::Constant,
                burst: None,
                ramp: None,
                trace_file: None,
            },
            algorithm: crate::scenario::AlgorithmConfig {
                adapter: "kyber".to_string(),
                operation: "kem_aead_encrypt".to_string(),
            },
            metrics: crate::scenario::MetricsConfig::default(),
            execution: crate::scenario::ExecutionConfig::default(),
            rng_seed: Some(12345),
        };

        let pipeline = Pipeline::new();
        let adapter: Arc<dyn CryptoAdapter + Send + Sync> =
            Arc::new(KyberAdapter::new("kyber512").unwrap());
        let metrics = Metrics::new().unwrap();
        let jsonl_writer = JsonlWriter::new("/tmp/test_kyber_pipeline.jsonl").unwrap();
        let sampler = SysInfoSampler::new();

        let stats = pipeline
            .run_async(&scenario, adapter, metrics, jsonl_writer, sampler, None)
            .await
            .unwrap();

        assert!(stats.events_processed > 0);
    }
}


