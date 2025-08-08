//! Workload Module
//!
//! This module defines workload generators and configurations for simulating
//! real-world cryptographic operation patterns in benchmark scenarios.

/// Error type for workload operations
#[derive(Debug)]
pub enum WorkloadError {
    /// Workload generation failed
    GenerationError(String),
    /// Invalid workload configuration
    ConfigurationError(String),
}

/// Types of workload patterns
#[derive(Debug, Clone, Copy)]
pub enum WorkloadPattern {
    /// Constant rate of operations
    Constant,
    /// Burst pattern with periodic spikes
    Burst,
    /// Gradually increasing load
    Ramp,
    /// Random distribution of operations
    Random,
}

/// Configuration for workload generation
#[derive(Debug)]
pub struct WorkloadConfig {
    /// The pattern to use for generating workload
    pub pattern: WorkloadPattern,
    /// Target operations per second
    pub ops_per_second: u64,
    /// Duration of the workload in seconds
    pub duration_secs: u64,
    /// Size of each payload in bytes
    pub payload_size: usize,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            pattern: WorkloadPattern::Constant,
            ops_per_second: 1000,
            duration_secs: 60,
            payload_size: 1024,
        }
    }
}

/// Workload generator for benchmark scenarios
#[derive(Debug)]
pub struct Workload {
    config: WorkloadConfig,
    operations_generated: u64,
}

impl Workload {
    /// Creates a new Workload with default configuration
    pub fn new() -> Self {
        Self {
            config: WorkloadConfig::default(),
            operations_generated: 0,
        }
    }

    /// Creates a new Workload with the given configuration
    pub fn with_config(config: WorkloadConfig) -> Self {
        Self {
            config,
            operations_generated: 0,
        }
    }

    /// Generates the next batch of workload data
    pub fn generate_batch(&mut self, batch_size: usize) -> Result<Vec<Vec<u8>>, WorkloadError> {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            batch.push(vec![0u8; self.config.payload_size]);
            self.operations_generated += 1;
        }
        Ok(batch)
    }

    /// Resets the workload generator state
    pub fn reset(&mut self) {
        self.operations_generated = 0;
    }

    /// Returns the number of operations generated so far
    pub fn operations_generated(&self) -> u64 {
        self.operations_generated
    }

    /// Returns the workload configuration
    pub fn config(&self) -> &WorkloadConfig {
        &self.config
    }
}

impl Default for Workload {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workload_new() {
        let workload = Workload::new();
        assert_eq!(workload.operations_generated(), 0);
    }

    #[test]
    fn test_workload_generate_batch() {
        let mut workload = Workload::new();
        let batch = workload.generate_batch(10).unwrap();
        assert_eq!(batch.len(), 10);
        assert_eq!(workload.operations_generated(), 10);
    }
}

