//! Rust Core Library for Quantum-Resilient Cryptography Benchmarking
//!
//! This crate provides the core functionality for benchmarking Post-Quantum Cryptography (PQC)
//! algorithms against classical cryptography in real-time streaming pipelines.

pub mod controlplane;
pub mod crypto_adapter;
pub mod pipeline;
pub mod scenario;
pub mod telemetry;
pub mod workload;

// Re-export main types for convenience
pub use crypto_adapter::{
    aead_decrypt, aead_encrypt, derive_aead_key, get_adapter, hybrid_decrypt, hybrid_encrypt,
    supported_adapters, CryptoAdapter, CryptoError, EcdsaP256Adapter, HybridSizes, KeypairMeta,
    KeypairWithSecret, KyberAdapter, NoOpCryptoAdapter, Rsa2048Adapter, KEY_SIZE, NONCE_SIZE,
    TAG_SIZE,
};
pub use pipeline::{
    create_workload_model, ExecutionContext, ExecutionEngine, ExecutionState, Pipeline,
    PipelineConfig, PipelineContext, PipelineStats, ProcessedEvent, QueuedEvent, WorkloadModel,
};
pub use scenario::{
    load_scenario, supported_operations, BurstConfig, ExecutionConfig, ExecutionMode, RampConfig,
    Scenario, WorkloadPattern,
};
pub use telemetry::{init_tracing, JsonlWriter, Metrics, SysInfoSampler, Telemetry};
pub use workload::Workload;
pub use controlplane::{start_control_plane_server, ControlPlaneState, OrchestrationState};
