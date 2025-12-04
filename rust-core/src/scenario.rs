//! Scenario Module
//!
//! This module provides scenario definitions and YAML loading capabilities
//! for configuring benchmark runs.

use serde::Deserialize;
use std::fs::File;
use std::io::Read;

/// A benchmark scenario configuration
#[derive(Debug, Deserialize)]
pub struct Scenario {
    /// Unique identifier for the scenario
    pub id: String,
    /// Optional human-readable description
    pub description: Option<String>,
    /// Workload configuration
    pub workload: WorkloadConfig,
    /// Algorithm configuration
    pub algorithm: AlgorithmConfig,
}

/// Workload configuration for a scenario
#[derive(Debug, Deserialize)]
pub struct WorkloadConfig {
    /// Target messages per second
    pub msgs_per_sec: u32,
    /// Size of each message in bytes
    pub msg_size_bytes: usize,
    /// Duration of the benchmark in seconds
    pub duration_sec: u64,
}

/// Algorithm configuration for a scenario
#[derive(Debug, Deserialize)]
pub struct AlgorithmConfig {
    /// Name of the crypto adapter to use (e.g., "noop", "kyber", "dilithium")
    pub adapter: String,
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
    let mut f = File::open(path).map_err(|e| {
        format!("Failed to open scenario file '{}': {}", path, e)
    })?;
    
    let mut contents = String::new();
    f.read_to_string(&mut contents).map_err(|e| {
        format!("Failed to read scenario file '{}': {}", path, e)
    })?;

    let scenario: Scenario = serde_yaml::from_str(&contents).map_err(|e| {
        format!("Failed to parse scenario YAML '{}': {}", path, e)
    })?;
    
    Ok(scenario)
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
"#;
        let scenario: Scenario = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(scenario.id, "test_scenario");
        assert_eq!(scenario.description, Some("A test scenario".to_string()));
        assert_eq!(scenario.workload.msgs_per_sec, 100);
        assert_eq!(scenario.workload.msg_size_bytes, 256);
        assert_eq!(scenario.workload.duration_sec, 10);
        assert_eq!(scenario.algorithm.adapter, "noop");
    }

    #[test]
    fn test_scenario_without_description() {
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
        assert_eq!(scenario.id, "minimal");
        assert!(scenario.description.is_none());
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = load_scenario("nonexistent_file.yaml");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Failed to open"));
    }
}

