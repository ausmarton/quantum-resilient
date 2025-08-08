//! Rust Core Library for Quantum-Resilient Cryptography Benchmarking
//!
//! This crate provides the core functionality for benchmarking Post-Quantum Cryptography (PQC)
//! algorithms against classical cryptography in real-time streaming pipelines.

pub mod crypto_adapter;
pub mod pipeline;
pub mod telemetry;
pub mod workload;

// Re-export main types for convenience
pub use crypto_adapter::CryptoAdapter;
pub use pipeline::Pipeline;
pub use telemetry::Telemetry;
pub use workload::Workload;

