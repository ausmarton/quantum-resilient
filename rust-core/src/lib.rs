//! Rust Core Library for Quantum-Resilient Cryptography Benchmarking
//!
//! This crate provides the core functionality for benchmarking Post-Quantum Cryptography (PQC)
//! algorithms against classical cryptography in real-time streaming pipelines.

pub mod crypto_adapter;
pub mod pipeline;
pub mod scenario;
pub mod telemetry;
pub mod workload;

// Re-export main types for convenience
pub use crypto_adapter::{CryptoAdapter, CryptoError, KeypairMeta, NoOpCryptoAdapter};
pub use pipeline::Pipeline;
pub use scenario::{load_scenario, Scenario};
pub use telemetry::Telemetry;
pub use workload::Workload;
