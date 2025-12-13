//! Scenario Module
//!
//! This module provides scenario definitions and YAML loading capabilities
//! for configuring benchmark runs.

use serde::Deserialize;
use std::fs::File;
use std::io::Read;

/// A benchmark scenario configuration
#[derive(Debug, Deserialize, Clone)]
pub struct Scenario {
    /// Unique identifier for the scenario
    pub id: String,
    /// Optional human-readable description
    pub description: Option<String>,
    /// Workload configuration
    pub workload: WorkloadConfig,
    /// Algorithm configuration
    pub algorithm: AlgorithmConfig,
    /// Metrics configuration (optional)
    #[serde(default)]
    pub metrics: MetricsConfig,
    /// Execution model configuration (optional)
    #[serde(default)]
    pub execution: ExecutionConfig,
    /// RNG seed for deterministic reproducibility (optional)
    /// If not specified, uses current unix nanoseconds
    #[serde(default)]
    pub rng_seed: Option<u64>,
}

/// Workload configuration for a scenario
#[derive(Debug, Deserialize, Clone)]
pub struct WorkloadConfig {
    /// Target messages per second (baseline for burst/ramp patterns)
    #[serde(default = "default_msgs_per_sec")]
    pub msgs_per_sec: u32,
    /// Size of each message in bytes
    pub msg_size_bytes: usize,
    /// Duration of the benchmark in seconds
    pub duration_sec: u64,

    // NEW: Workload pattern configuration
    /// Workload pattern: constant, burst, ramp, or trace
    #[serde(default)]
    pub pattern: WorkloadPattern,

    /// Burst pattern configuration
    #[serde(default)]
    pub burst: Option<BurstConfig>,

    /// Ramp pattern configuration
    #[serde(default)]
    pub ramp: Option<RampConfig>,

    /// Path to trace file for trace pattern (CSV: timestamp_ms, rps)
    #[serde(default)]
    pub trace_file: Option<String>,
}

/// Workload pattern types
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum WorkloadPattern {
    /// Constant rate of operations (default)
    #[default]
    Constant,
    /// Burst pattern with periodic spikes
    Burst,
    /// Gradually ramping load
    Ramp,
    /// Trace-driven replay from CSV file
    Trace,
}

/// Configuration for burst workload pattern
#[derive(Debug, Deserialize, Clone)]
pub struct BurstConfig {
    /// Multiplier for burst RPS (e.g., 5 = 5× baseline)
    #[serde(default = "default_burst_factor")]
    pub factor: u32,
    /// Duration of each burst in milliseconds
    #[serde(default = "default_burst_duration_ms")]
    pub duration_ms: u64,
    /// Interval between bursts in milliseconds
    #[serde(default = "default_burst_interval_ms")]
    pub interval_ms: u64,
}

impl Default for BurstConfig {
    fn default() -> Self {
        Self {
            factor: default_burst_factor(),
            duration_ms: default_burst_duration_ms(),
            interval_ms: default_burst_interval_ms(),
        }
    }
}

/// Configuration for ramp workload pattern
#[derive(Debug, Deserialize, Clone)]
pub struct RampConfig {
    /// Starting RPS
    #[serde(default = "default_ramp_from")]
    pub from: u32,
    /// Ending RPS
    #[serde(default = "default_ramp_to")]
    pub to: u32,
    /// Duration of the ramp in seconds
    #[serde(default = "default_ramp_duration_sec")]
    pub duration_sec: u64,
}

impl Default for RampConfig {
    fn default() -> Self {
        Self {
            from: default_ramp_from(),
            to: default_ramp_to(),
            duration_sec: default_ramp_duration_sec(),
        }
    }
}

/// Execution model configuration
#[derive(Debug, Deserialize, Clone)]
pub struct ExecutionConfig {
    /// Execution mode: single, fixed_pool, or elastic
    #[serde(default)]
    pub mode: ExecutionMode,
    /// Number of workers for fixed_pool mode
    #[serde(default = "default_workers")]
    pub workers: usize,
    /// Maximum workers for elastic mode
    #[serde(default = "default_max_workers")]
    pub max_workers: usize,
    /// Bounded queue capacity
    #[serde(default = "default_queue_capacity")]
    pub queue_capacity: usize,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            mode: ExecutionMode::default(),
            workers: default_workers(),
            max_workers: default_max_workers(),
            queue_capacity: default_queue_capacity(),
        }
    }
}

/// Execution mode types
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    /// Single processor task (default, existing behavior)
    #[default]
    Single,
    /// Fixed number of processor tasks
    FixedPool,
    /// Elastic pool that scales based on queue length
    Elastic,
}

/// Algorithm configuration for a scenario
#[derive(Debug, Deserialize, Clone)]
pub struct AlgorithmConfig {
    /// Name of the crypto adapter to use (e.g., "noop", "rsa2048", "ecdsa_p256", "kyber")
    pub adapter: String,
    /// Operation to perform: "sign", "verify", "encrypt", "decrypt", "keygen",
    /// "kem_aead_encrypt", "kem_aead_decrypt"
    #[serde(default = "default_operation")]
    pub operation: String,
}

/// Metrics configuration for a scenario
#[derive(Debug, Deserialize, Clone)]
pub struct MetricsConfig {
    /// Prometheus endpoint address (default: "0.0.0.0:9898")
    #[serde(default = "default_prometheus_endpoint")]
    pub prometheus_endpoint: String,
    /// Path to JSONL output file (default: "./results/run_<id>.jsonl")
    #[serde(default)]
    pub jsonl_out: Option<String>,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            prometheus_endpoint: default_prometheus_endpoint(),
            jsonl_out: None,
        }
    }
}

// Default value functions
fn default_msgs_per_sec() -> u32 {
    100
}

fn default_operation() -> String {
    "sign".to_string()
}

fn default_prometheus_endpoint() -> String {
    "0.0.0.0:9898".to_string()
}

fn default_burst_factor() -> u32 {
    5
}

fn default_burst_duration_ms() -> u64 {
    200
}

fn default_burst_interval_ms() -> u64 {
    1000
}

fn default_ramp_from() -> u32 {
    10
}

fn default_ramp_to() -> u32 {
    200
}

fn default_ramp_duration_sec() -> u64 {
    5
}

fn default_workers() -> usize {
    4
}

fn default_max_workers() -> usize {
    16
}

fn default_queue_capacity() -> usize {
    2000
}

impl Scenario {
    /// Returns the JSONL output path, defaulting to ./results/<id>.jsonl
    pub fn jsonl_output_path(&self) -> String {
        self.metrics
            .jsonl_out
            .clone()
            .unwrap_or_else(|| format!("./results/{}.jsonl", self.id))
    }

    /// Returns true if the operation requires a cached keypair
    pub fn requires_keypair(&self) -> bool {
        matches!(
            self.algorithm.operation.as_str(),
            "kem_aead_encrypt" | "kem_aead_decrypt" | "kem_aead_sign"
        )
    }

    /// Returns true if this is a KEM-based hybrid operation
    pub fn is_kem_hybrid_operation(&self) -> bool {
        matches!(
            self.algorithm.operation.as_str(),
            "kem_aead_encrypt" | "kem_aead_decrypt" | "kem_aead_sign"
        )
    }

    /// Returns the effective RNG seed (provided or generated from current time)
    pub fn effective_rng_seed(&self) -> u64 {
        self.rng_seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        })
    }
}

/// Loads a scenario from a YAML file
///
/// # Arguments
/// * `path` - Path to the YAML scenario file
///
/// # Returns
/// The parsed Scenario on success, or an error if loading/parsing fails
///
/// # Example
/// ```ignore
/// let scenario = load_scenario("scenarios/smoke_noop.yaml")?;
/// println!("Loaded scenario: {}", scenario.id);
/// ```
pub fn load_scenario(path: &str) -> Result<Scenario, Box<dyn std::error::Error>> {
    let mut f = File::open(path)
        .map_err(|e| format!("Failed to open scenario file '{}': {}", path, e))?;

    let mut contents = String::new();
    f.read_to_string(&mut contents)
        .map_err(|e| format!("Failed to read scenario file '{}': {}", path, e))?;

    let scenario: Scenario = serde_yaml::from_str(&contents)
        .map_err(|e| format!("Failed to parse scenario YAML '{}': {}", path, e))?;

    Ok(scenario)
}

/// Returns a list of supported operations
pub fn supported_operations() -> &'static [&'static str] {
    &[
        "sign",
        "verify",
        "encrypt",
        "decrypt",
        "keygen",
        "kem_aead_encrypt",
        "kem_aead_decrypt",
        "kem_aead_sign", // Hybrid: Kyber KEM + AES-GCM + Dilithium sign
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scenario_deserialize() {
        let yaml = r#"
id: test_scenario
description: "A test scenario"
workload:
  msgs_per_sec: 100
  msg_size_bytes: 256
  duration_sec: 10
algorithm:
  adapter: noop
  operation: sign
"#;
        let scenario: Scenario = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(scenario.id, "test_scenario");
        assert_eq!(scenario.description, Some("A test scenario".to_string()));
        assert_eq!(scenario.workload.msgs_per_sec, 100);
        assert_eq!(scenario.workload.msg_size_bytes, 256);
        assert_eq!(scenario.workload.duration_sec, 10);
        assert_eq!(scenario.algorithm.adapter, "noop");
        assert_eq!(scenario.algorithm.operation, "sign");
    }

    #[test]
    fn test_scenario_kem_aead() {
        let yaml = r#"
id: kyber_test
workload:
  msgs_per_sec: 10
  msg_size_bytes: 64
  duration_sec: 1
algorithm:
  adapter: kyber
  operation: kem_aead_encrypt
"#;
        let scenario: Scenario = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(scenario.algorithm.adapter, "kyber");
        assert_eq!(scenario.algorithm.operation, "kem_aead_encrypt");
        assert!(scenario.is_kem_hybrid_operation());
        assert!(scenario.requires_keypair());
    }

    #[test]
    fn test_scenario_with_metrics() {
        let yaml = r#"
id: test_with_metrics
workload:
  msgs_per_sec: 10
  msg_size_bytes: 64
  duration_sec: 1
algorithm:
  adapter: noop
  operation: sign
metrics:
  prometheus_endpoint: "0.0.0.0:9999"
  jsonl_out: "./custom/path.jsonl"
"#;
        let scenario: Scenario = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(scenario.metrics.prometheus_endpoint, "0.0.0.0:9999");
        assert_eq!(
            scenario.metrics.jsonl_out,
            Some("./custom/path.jsonl".to_string())
        );
    }

    #[test]
    fn test_scenario_default_metrics() {
        let yaml = r#"
id: minimal
workload:
  msgs_per_sec: 10
  msg_size_bytes: 64
  duration_sec: 1
algorithm:
  adapter: noop
"#;
        let scenario: Scenario = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(scenario.metrics.prometheus_endpoint, "0.0.0.0:9898");
        assert!(scenario.metrics.jsonl_out.is_none());
        assert_eq!(scenario.jsonl_output_path(), "./results/minimal.jsonl");
    }

    #[test]
    fn test_scenario_default_operation() {
        let yaml = r#"
id: default_op
workload:
  msgs_per_sec: 10
  msg_size_bytes: 64
  duration_sec: 1
algorithm:
  adapter: noop
"#;
        let scenario: Scenario = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(scenario.algorithm.operation, "sign");
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = load_scenario("nonexistent_file.yaml");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Failed to open"));
    }

    #[test]
    fn test_supported_operations_includes_kem() {
        let ops = supported_operations();
        assert!(ops.contains(&"kem_aead_encrypt"));
        assert!(ops.contains(&"kem_aead_decrypt"));
    }

    #[test]
    fn test_scenario_with_execution_config() {
        let yaml = r#"
id: execution_test
workload:
  msgs_per_sec: 100
  msg_size_bytes: 64
  duration_sec: 5
algorithm:
  adapter: noop
execution:
  mode: fixed_pool
  workers: 8
  queue_capacity: 3000
"#;
        let scenario: Scenario = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(scenario.execution.mode, ExecutionMode::FixedPool);
        assert_eq!(scenario.execution.workers, 8);
        assert_eq!(scenario.execution.queue_capacity, 3000);
    }

    #[test]
    fn test_scenario_with_burst_pattern() {
        let yaml = r#"
id: burst_test
workload:
  msgs_per_sec: 100
  msg_size_bytes: 128
  duration_sec: 5
  pattern: burst
  burst:
    factor: 4
    duration_ms: 200
    interval_ms: 1000
algorithm:
  adapter: noop
"#;
        let scenario: Scenario = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(scenario.workload.pattern, WorkloadPattern::Burst);
        let burst = scenario.workload.burst.unwrap();
        assert_eq!(burst.factor, 4);
        assert_eq!(burst.duration_ms, 200);
        assert_eq!(burst.interval_ms, 1000);
    }

    #[test]
    fn test_scenario_with_ramp_pattern() {
        let yaml = r#"
id: ramp_test
workload:
  msg_size_bytes: 256
  duration_sec: 5
  pattern: ramp
  ramp:
    from: 20
    to: 300
    duration_sec: 5
algorithm:
  adapter: kyber
  operation: kem_aead_encrypt
"#;
        let scenario: Scenario = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(scenario.workload.pattern, WorkloadPattern::Ramp);
        let ramp = scenario.workload.ramp.unwrap();
        assert_eq!(ramp.from, 20);
        assert_eq!(ramp.to, 300);
        assert_eq!(ramp.duration_sec, 5);
    }

    #[test]
    fn test_scenario_default_execution() {
        let yaml = r#"
id: default_exec
workload:
  msgs_per_sec: 10
  msg_size_bytes: 64
  duration_sec: 1
algorithm:
  adapter: noop
"#;
        let scenario: Scenario = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(scenario.execution.mode, ExecutionMode::Single);
        assert_eq!(scenario.execution.workers, 4);
        assert_eq!(scenario.execution.max_workers, 16);
        assert_eq!(scenario.execution.queue_capacity, 2000);
    }

    #[test]
    fn test_scenario_elastic_mode() {
        let yaml = r#"
id: elastic_test
workload:
  msgs_per_sec: 100
  msg_size_bytes: 64
  duration_sec: 5
algorithm:
  adapter: noop
execution:
  mode: elastic
  max_workers: 32
  queue_capacity: 5000
"#;
        let scenario: Scenario = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(scenario.execution.mode, ExecutionMode::Elastic);
        assert_eq!(scenario.execution.max_workers, 32);
        assert_eq!(scenario.execution.queue_capacity, 5000);
    }
}
