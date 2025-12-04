//! Pipeline Module
//!
//! This module provides the async streaming pipeline infrastructure for running
//! cryptographic benchmarks in real-time data processing scenarios.

use crate::crypto_adapter::{
    hybrid_decrypt, hybrid_encrypt, CryptoAdapter, CryptoError, HybridSizes, KeypairWithSecret,
};
use crate::scenario::Scenario;
use crate::telemetry::jsonl_logger::EventRow;
use crate::telemetry::{JsonlWriter, Metrics, SysInfoSampler};
use chrono::Utc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{info, info_span, warn, Instrument};

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

/// Event passed through the pipeline
#[derive(Debug, Clone)]
pub struct PipelineEvent {
    pub event_id: u64,
    pub payload: Vec<u8>,
    pub timestamp_ns: u128,
}

/// Result of processing an event
#[derive(Debug)]
pub struct ProcessedEvent {
    pub event_id: u64,
    pub latency_us: u128,
    pub success: bool,
    pub output_size: Option<usize>,
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

    /// Runs the benchmark pipeline asynchronously
    pub async fn run_async(
        &self,
        scenario: &Scenario,
        adapter: Arc<dyn CryptoAdapter + Send + Sync>,
        metrics: Metrics,
        jsonl_writer: JsonlWriter,
        sampler: SysInfoSampler,
    ) -> Result<PipelineStats, PipelineError> {
        // Generate keypair if needed for KEM operations
        let context = if scenario.requires_keypair() {
            info!("Generating keypair for KEM operations...");
            let meta = adapter
                .keygen()
                .map_err(|e| PipelineError::InitializationError(e.to_string()))?;

            // For Kyber, we need the full keypair including secret key
            // Generate a fresh keypair and store both parts
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

        let (producer_tx, processor_rx) =
            mpsc::channel::<PipelineEvent>(self.config.buffer_size);
        let (processor_tx, mut consumer_rx) =
            mpsc::channel::<ProcessedEvent>(self.config.buffer_size);

        let total_events =
            scenario.workload.duration_sec as u64 * scenario.workload.msgs_per_sec as u64;
        let events_processed = Arc::new(AtomicU64::new(0));
        let total_latency_us = Arc::new(AtomicU64::new(0));

        let start_time = Instant::now();

        // Spawn producer
        let producer_handle = {
            let scenario = scenario.clone();
            let context = context.clone();
            tokio::spawn(async move {
                producer_task(producer_tx, &scenario, &context).await;
            })
        };

        // Spawn processor
        let processor_handle = {
            let scenario = scenario.clone();
            let metrics = metrics.clone();
            let sampler = sampler.clone();
            let context = context.clone();
            tokio::spawn(async move {
                processor_task(
                    processor_rx,
                    processor_tx,
                    adapter,
                    &scenario,
                    metrics,
                    jsonl_writer,
                    sampler,
                    context,
                )
                .await;
            })
        };

        // Spawn consumer
        let consumer_handle = {
            let events_processed = events_processed.clone();
            let total_latency = total_latency_us.clone();
            tokio::spawn(async move {
                while let Some(result) = consumer_rx.recv().await {
                    events_processed.fetch_add(1, Ordering::Relaxed);
                    total_latency.fetch_add(result.latency_us as u64, Ordering::Relaxed);
                }
            })
        };

        // Wait for all tasks to complete
        let _ = producer_handle.await;
        let _ = processor_handle.await;
        let _ = consumer_handle.await;

        let elapsed = start_time.elapsed();
        let processed = events_processed.load(Ordering::Relaxed);
        let total_lat = total_latency_us.load(Ordering::Relaxed);
        let avg_latency = if processed > 0 {
            total_lat as f64 / processed as f64
        } else {
            0.0
        };

        Ok(PipelineStats {
            total_events,
            events_processed: processed,
            duration: elapsed,
            avg_latency_us: avg_latency,
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

/// Statistics from a pipeline run
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub total_events: u64,
    pub events_processed: u64,
    pub duration: Duration,
    pub avg_latency_us: f64,
}

/// Generates a keypair with both public and secret key for KEM operations
fn generate_keypair_with_secret(
    adapter: &Arc<dyn CryptoAdapter + Send + Sync>,
) -> Result<(Vec<u8>, Vec<u8>), PipelineError> {
    // For Kyber adapter, we need to call keygen and get both keys
    // The adapter's keygen returns KeypairMeta which only has public key
    // We need to use the internal generation that gives us both
    
    // First, check if adapter is Kyber by name
    if adapter.name() == "kyber" {
        // Use encapsulate with a dummy to trigger key generation
        // Actually, we need a different approach - generate keys properly
        
        // For pqcrypto-kyber, we generate directly here
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

/// Producer task: generates events at the specified rate
async fn producer_task(
    tx: mpsc::Sender<PipelineEvent>,
    scenario: &Scenario,
    _context: &PipelineContext,
) {
    let total_events =
        scenario.workload.duration_sec as u64 * scenario.workload.msgs_per_sec as u64;
    let interval_ns = 1_000_000_000u64 / scenario.workload.msgs_per_sec as u64;
    let payload_size = scenario.workload.msg_size_bytes;

    let start = Instant::now();

    for event_id in 1..=total_events {
        // For kem_aead_decrypt, we'd normally load pre-encrypted payloads
        // For this iteration, we generate plaintext and let processor handle encryption
        let payload = vec![0xAB; payload_size];
        let event = PipelineEvent {
            event_id,
            payload,
            timestamp_ns: start.elapsed().as_nanos(),
        };

        if tx.send(event).await.is_err() {
            warn!("Producer: receiver dropped, stopping");
            break;
        }

        // Pace the output
        let expected_time = Duration::from_nanos(event_id * interval_ns);
        let actual_time = start.elapsed();
        if expected_time > actual_time {
            tokio::time::sleep(expected_time - actual_time).await;
        }
    }

    info!("Producer: completed {} events", total_events);
}

/// Processor task: performs crypto operations and records metrics
async fn processor_task(
    mut rx: mpsc::Receiver<PipelineEvent>,
    tx: mpsc::Sender<ProcessedEvent>,
    adapter: Arc<dyn CryptoAdapter + Send + Sync>,
    scenario: &Scenario,
    metrics: Metrics,
    jsonl_writer: JsonlWriter,
    sampler: SysInfoSampler,
    context: PipelineContext,
) {
    let run_id = scenario.id.clone();
    let operation = scenario.algorithm.operation.clone();
    let algorithm = adapter.name().to_string();

    while let Some(event) = rx.recv().await {
        let span = info_span!(
            "crypto_op",
            event_id = event.event_id,
            algorithm = %algorithm,
            operation = %operation,
            payload_size = event.payload.len()
        );

        let result = async {
            let start = Instant::now();

            // Perform the crypto operation
            let op_result = match operation.as_str() {
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
                    // Hybrid encryption: KEM + AES-GCM
                    if let Some(ref keypair) = context.keypair {
                        let adapter_clone = adapter.clone();
                        let pk = keypair.public_key.clone();
                        
                        let result = hybrid_encrypt(
                            |pubkey| adapter_clone.encapsulate(pubkey),
                            &pk,
                            &event.payload,
                        );
                        
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
                    // For decrypt benchmark, we first encrypt then decrypt
                    // This measures the full roundtrip
                    if let Some(ref keypair) = context.keypair {
                        let adapter_clone = adapter.clone();
                        let pk = keypair.public_key.clone();
                        let sk = keypair.secret_key.clone();
                        
                        // First encrypt, then decrypt
                        hybrid_encrypt(
                            |pubkey| adapter_clone.encapsulate(pubkey),
                            &pk,
                            &event.payload,
                        )
                        .and_then(|combined| {
                            let adapter_clone2 = adapter.clone();
                            hybrid_decrypt(
                                |secret_key, ciphertext| adapter_clone2.decapsulate(secret_key, ciphertext),
                                &sk,
                                &combined,
                            )
                        })
                        .and_then(|decrypted| {
                            // Verify decryption succeeded
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
                _ => Err(CryptoError::NotImplemented),
            };

            let latency_us = start.elapsed().as_micros();
            let (success, output_size, _ct_kem_size, error_msg) = match op_result {
                Ok((size, kem_size)) => (true, size, kem_size, None),
                Err(e) => (false, None, None, Some(e.to_string())),
            };

            // Sample system metrics
            let (cpu_user, memory_rss) = sampler.sample();

            // Update Prometheus metrics
            metrics.observe_latency(&algorithm, &operation, latency_us as f64);
            metrics.inc_ops(&algorithm, &operation, success);
            metrics.set_memory_bytes(memory_rss);

            // Determine ciphertext size for JSONL
            let ciphertext_size = if operation == "encrypt" || operation == "kem_aead_encrypt" {
                output_size
            } else {
                None
            };

            // Write JSONL row
            let row = EventRow {
                run_id: run_id.clone(),
                event_id: event.event_id,
                timestamp_utc_iso: Utc::now().to_rfc3339(),
                timestamp_monotonic_ns: event.timestamp_ns,
                operation: operation.clone(),
                algorithm: algorithm.clone(),
                latency_us,
                payload_size_bytes: event.payload.len(),
                ciphertext_size_bytes: ciphertext_size,
                signature_size_bytes: if operation == "sign" { output_size } else { None },
                cpu_user_seconds: cpu_user,
                memory_rss_bytes: memory_rss,
                error: error_msg,
            };

            if let Err(e) = jsonl_writer.write(&row) {
                warn!("Failed to write JSONL row: {}", e);
            }

            ProcessedEvent {
                event_id: event.event_id,
                latency_us,
                success,
                output_size,
            }
        }
        .instrument(span)
        .await;

        if tx.send(result).await.is_err() {
            warn!("Processor: consumer dropped, stopping");
            break;
        }
    }

    info!("Processor: completed");
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
            },
            algorithm: crate::scenario::AlgorithmConfig {
                adapter: "noop".to_string(),
                operation: "sign".to_string(),
            },
            metrics: crate::scenario::MetricsConfig::default(),
        };

        let pipeline = Pipeline::new();
        let adapter: Arc<dyn CryptoAdapter + Send + Sync> = Arc::new(NoOpCryptoAdapter);
        let metrics = Metrics::new().unwrap();
        let jsonl_writer = JsonlWriter::new("/tmp/test_pipeline.jsonl").unwrap();
        let sampler = SysInfoSampler::new();

        let stats = pipeline
            .run_async(&scenario, adapter, metrics, jsonl_writer, sampler)
            .await
            .unwrap();

        assert_eq!(stats.total_events, 10);
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
            },
            algorithm: crate::scenario::AlgorithmConfig {
                adapter: "kyber".to_string(),
                operation: "kem_aead_encrypt".to_string(),
            },
            metrics: crate::scenario::MetricsConfig::default(),
        };

        let pipeline = Pipeline::new();
        let adapter: Arc<dyn CryptoAdapter + Send + Sync> =
            Arc::new(KyberAdapter::new("kyber512").unwrap());
        let metrics = Metrics::new().unwrap();
        let jsonl_writer = JsonlWriter::new("/tmp/test_kyber_pipeline.jsonl").unwrap();
        let sampler = SysInfoSampler::new();

        let stats = pipeline
            .run_async(&scenario, adapter, metrics, jsonl_writer, sampler)
            .await
            .unwrap();

        assert_eq!(stats.total_events, 5);
        assert!(stats.events_processed > 0);
    }
}
