//! Pipeline Module
//!
//! This module provides the streaming pipeline infrastructure for running
//! cryptographic benchmarks in real-time data processing scenarios.

use crate::crypto_adapter::{CryptoAdapter, CryptoError};
use std::time::Instant;

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
#[derive(Debug, Default)]
pub struct PipelineConfig {
    /// Number of worker threads
    pub num_workers: usize,
    /// Buffer size for streaming data
    pub buffer_size: usize,
    /// Whether to enable detailed metrics
    pub enable_metrics: bool,
}

/// The main pipeline struct for orchestrating benchmark runs
#[derive(Debug)]
pub struct Pipeline {
    config: PipelineConfig,
    is_running: bool,
}

impl Pipeline {
    /// Creates a new Pipeline with default configuration
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
            is_running: false,
        }
    }

    /// Creates a new Pipeline with the given configuration
    pub fn with_config(config: PipelineConfig) -> Self {
        Self {
            config,
            is_running: false,
        }
    }

    /// Initializes the pipeline
    pub fn init(&mut self) -> Result<(), PipelineError> {
        // Placeholder: initialize resources
        Ok(())
    }

    /// Runs the benchmark pipeline
    pub fn run(&mut self) -> Result<(), PipelineError> {
        self.is_running = true;
        // Placeholder: execute pipeline stages
        self.is_running = false;
        Ok(())
    }

    /// Shuts down the pipeline gracefully
    pub fn shutdown(&mut self) -> Result<(), PipelineError> {
        self.is_running = false;
        // Placeholder: cleanup resources
        Ok(())
    }

    /// Returns whether the pipeline is currently running
    pub fn is_running(&self) -> bool {
        self.is_running
    }

    /// Returns the pipeline configuration
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Runs a timed cryptographic operation
    ///
    /// # Arguments
    /// * `adapter` - The crypto adapter to use
    /// * `operation` - The operation to perform: "sign", "verify", "encrypt", "decrypt", "keygen"
    /// * `payload` - The payload data for the operation
    ///
    /// # Returns
    /// The duration of the operation in microseconds
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_adapter::NoOpCryptoAdapter;

    #[test]
    fn test_pipeline_new() {
        let pipeline = Pipeline::new();
        assert!(!pipeline.is_running());
    }

    #[test]
    fn test_pipeline_run() {
        let mut pipeline = Pipeline::new();
        assert!(pipeline.run().is_ok());
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
}
