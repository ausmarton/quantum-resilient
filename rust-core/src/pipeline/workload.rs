//! Workload Generator Module
//!
//! This module provides advanced workload generation models for benchmark scenarios:
//! - Constant: Fixed rate (existing behavior)
//! - Burst: Periodic spikes in load
//! - Ramp: Linear interpolation from start to end RPS
//! - Trace: Replay from CSV file

use crate::scenario::{BurstConfig, RampConfig, WorkloadConfig, WorkloadPattern};
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Trait for workload generation models
pub trait WorkloadModel: Send + Sync {
    /// Returns the target RPS at the given elapsed time in milliseconds
    fn next_rps(&mut self, elapsed_ms: u64) -> u64;

    /// Returns the total duration in milliseconds (if bounded)
    fn duration_ms(&self) -> Option<u64>;

    /// Returns a description of the workload model
    fn description(&self) -> String;
}

/// Constant rate workload model
pub struct ConstantWorkload {
    rps: u64,
    duration_ms: u64,
}

impl ConstantWorkload {
    pub fn new(rps: u64, duration_sec: u64) -> Self {
        Self {
            rps,
            duration_ms: duration_sec * 1000,
        }
    }
}

impl WorkloadModel for ConstantWorkload {
    fn next_rps(&mut self, _elapsed_ms: u64) -> u64 {
        self.rps
    }

    fn duration_ms(&self) -> Option<u64> {
        Some(self.duration_ms)
    }

    fn description(&self) -> String {
        format!("Constant {} RPS for {} ms", self.rps, self.duration_ms)
    }
}

/// Burst pattern workload model
pub struct BurstWorkload {
    /// Baseline RPS outside of bursts
    baseline_rps: u64,
    /// Multiplier factor for burst RPS
    factor: u32,
    /// Duration of each burst in milliseconds
    burst_duration_ms: u64,
    /// Interval between burst starts in milliseconds
    burst_interval_ms: u64,
    /// Total duration in milliseconds
    total_duration_ms: u64,
}

impl BurstWorkload {
    pub fn new(baseline_rps: u64, config: &BurstConfig, duration_sec: u64) -> Self {
        Self {
            baseline_rps,
            factor: config.factor,
            burst_duration_ms: config.duration_ms,
            burst_interval_ms: config.interval_ms,
            total_duration_ms: duration_sec * 1000,
        }
    }

    /// Returns whether we're currently in a burst period
    fn is_in_burst(&self, elapsed_ms: u64) -> bool {
        if self.burst_interval_ms == 0 {
            return false;
        }
        let position_in_interval = elapsed_ms % self.burst_interval_ms;
        position_in_interval < self.burst_duration_ms
    }
}

impl WorkloadModel for BurstWorkload {
    fn next_rps(&mut self, elapsed_ms: u64) -> u64 {
        if self.is_in_burst(elapsed_ms) {
            self.baseline_rps * self.factor as u64
        } else {
            self.baseline_rps
        }
    }

    fn duration_ms(&self) -> Option<u64> {
        Some(self.total_duration_ms)
    }

    fn description(&self) -> String {
        format!(
            "Burst: baseline {} RPS, {}× factor for {} ms every {} ms",
            self.baseline_rps, self.factor, self.burst_duration_ms, self.burst_interval_ms
        )
    }
}

/// Ramp pattern workload model (linear interpolation)
pub struct RampWorkload {
    /// Starting RPS
    from_rps: u64,
    /// Ending RPS
    to_rps: u64,
    /// Duration of the ramp in milliseconds
    ramp_duration_ms: u64,
    /// Total workload duration in milliseconds
    total_duration_ms: u64,
}

impl RampWorkload {
    pub fn new(config: &RampConfig, total_duration_sec: u64) -> Self {
        Self {
            from_rps: config.from as u64,
            to_rps: config.to as u64,
            ramp_duration_ms: config.duration_sec * 1000,
            total_duration_ms: total_duration_sec * 1000,
        }
    }
}

impl WorkloadModel for RampWorkload {
    fn next_rps(&mut self, elapsed_ms: u64) -> u64 {
        if elapsed_ms >= self.ramp_duration_ms {
            // After ramp completes, hold at final rate
            self.to_rps
        } else {
            // Linear interpolation
            let progress = elapsed_ms as f64 / self.ramp_duration_ms as f64;
            let range = self.to_rps as f64 - self.from_rps as f64;
            (self.from_rps as f64 + range * progress) as u64
        }
    }

    fn duration_ms(&self) -> Option<u64> {
        Some(self.total_duration_ms)
    }

    fn description(&self) -> String {
        format!(
            "Ramp: {} to {} RPS over {} ms",
            self.from_rps, self.to_rps, self.ramp_duration_ms
        )
    }
}

/// Trace entry from CSV
#[derive(Debug, Clone)]
struct TraceEntry {
    timestamp_ms: u64,
    rps: u64,
}

/// Trace-driven workload model (replay from CSV)
pub struct TraceWorkload {
    /// Trace entries sorted by timestamp
    entries: Vec<TraceEntry>,
    /// Current index in the trace
    current_index: usize,
    /// Duration based on last trace entry
    total_duration_ms: u64,
}

impl TraceWorkload {
    /// Creates a new trace workload from a CSV file
    /// CSV format: timestamp_ms,rps (no header)
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();
        let mut last_timestamp: Option<u64> = None;

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            let line = line.trim();
            
            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() != 2 {
                return Err(format!(
                    "Invalid trace format at line {}: expected 'timestamp_ms,rps'",
                    line_num + 1
                )
                .into());
            }

            let timestamp_ms: u64 = parts[0].trim().parse().map_err(|_| {
                format!(
                    "Invalid timestamp at line {}: '{}'",
                    line_num + 1,
                    parts[0]
                )
            })?;

            let rps: u64 = parts[1].trim().parse().map_err(|_| {
                format!("Invalid RPS at line {}: '{}'", line_num + 1, parts[1])
            })?;

            // Validate monotonic timestamps
            if let Some(last) = last_timestamp {
                if timestamp_ms < last {
                    return Err(format!(
                        "Timestamps must be monotonically increasing: {} < {} at line {}",
                        timestamp_ms,
                        last,
                        line_num + 1
                    )
                    .into());
                }
            }
            last_timestamp = Some(timestamp_ms);

            entries.push(TraceEntry { timestamp_ms, rps });
        }

        if entries.is_empty() {
            return Err("Trace file is empty or contains no valid entries".into());
        }

        let total_duration_ms = entries.last().map(|e| e.timestamp_ms).unwrap_or(0);

        Ok(Self {
            entries,
            current_index: 0,
            total_duration_ms,
        })
    }

    /// Creates a trace workload from entries directly (for testing)
    pub fn from_entries(entries: Vec<(u64, u64)>) -> Self {
        let entries: Vec<TraceEntry> = entries
            .into_iter()
            .map(|(timestamp_ms, rps)| TraceEntry { timestamp_ms, rps })
            .collect();
        let total_duration_ms = entries.last().map(|e| e.timestamp_ms).unwrap_or(0);
        Self {
            entries,
            current_index: 0,
            total_duration_ms,
        }
    }
}

impl WorkloadModel for TraceWorkload {
    fn next_rps(&mut self, elapsed_ms: u64) -> u64 {
        // Find the appropriate entry for this timestamp
        while self.current_index + 1 < self.entries.len()
            && self.entries[self.current_index + 1].timestamp_ms <= elapsed_ms
        {
            self.current_index += 1;
        }

        // If we're past the trace, return the last RPS
        if self.entries.is_empty() {
            return 0;
        }

        self.entries[self.current_index].rps
    }

    fn duration_ms(&self) -> Option<u64> {
        Some(self.total_duration_ms)
    }

    fn description(&self) -> String {
        format!(
            "Trace: {} entries over {} ms",
            self.entries.len(),
            self.total_duration_ms
        )
    }
}

/// Creates a workload model from scenario configuration
pub fn create_workload_model(config: &WorkloadConfig) -> Box<dyn WorkloadModel> {
    match config.pattern {
        WorkloadPattern::Constant => {
            Box::new(ConstantWorkload::new(config.msgs_per_sec as u64, config.duration_sec))
        }
        WorkloadPattern::Burst => {
            let burst_config = config.burst.clone().unwrap_or_default();
            Box::new(BurstWorkload::new(
                config.msgs_per_sec as u64,
                &burst_config,
                config.duration_sec,
            ))
        }
        WorkloadPattern::Ramp => {
            let ramp_config = config.ramp.clone().unwrap_or_default();
            Box::new(RampWorkload::new(&ramp_config, config.duration_sec))
        }
        WorkloadPattern::Trace => {
            if let Some(ref trace_file) = config.trace_file {
                match TraceWorkload::from_file(trace_file) {
                    Ok(trace) => Box::new(trace),
                    Err(e) => {
                        tracing::warn!("Failed to load trace file '{}': {}, falling back to constant", trace_file, e);
                        Box::new(ConstantWorkload::new(config.msgs_per_sec as u64, config.duration_sec))
                    }
                }
            } else {
                tracing::warn!("Trace pattern specified but no trace_file provided, falling back to constant");
                Box::new(ConstantWorkload::new(config.msgs_per_sec as u64, config.duration_sec))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_workload() {
        let mut workload = ConstantWorkload::new(100, 10);
        assert_eq!(workload.next_rps(0), 100);
        assert_eq!(workload.next_rps(5000), 100);
        assert_eq!(workload.next_rps(10000), 100);
        assert_eq!(workload.duration_ms(), Some(10000));
    }

    #[test]
    fn test_burst_workload() {
        let config = BurstConfig {
            factor: 5,
            duration_ms: 200,
            interval_ms: 1000,
        };
        let mut workload = BurstWorkload::new(100, &config, 5);

        // At start (in burst)
        assert_eq!(workload.next_rps(0), 500);
        assert_eq!(workload.next_rps(100), 500);
        assert_eq!(workload.next_rps(199), 500);

        // Outside burst
        assert_eq!(workload.next_rps(200), 100);
        assert_eq!(workload.next_rps(500), 100);
        assert_eq!(workload.next_rps(999), 100);

        // Next burst
        assert_eq!(workload.next_rps(1000), 500);
        assert_eq!(workload.next_rps(1100), 500);
    }

    #[test]
    fn test_ramp_workload() {
        let config = RampConfig {
            from: 20,
            to: 200,
            duration_sec: 5,
        };
        let mut workload = RampWorkload::new(&config, 10);

        // At start
        assert_eq!(workload.next_rps(0), 20);

        // Halfway through ramp
        let halfway = workload.next_rps(2500);
        assert!(halfway >= 100 && halfway <= 120); // ~110

        // At end of ramp
        assert_eq!(workload.next_rps(5000), 200);

        // After ramp (hold at final)
        assert_eq!(workload.next_rps(7000), 200);
        assert_eq!(workload.next_rps(10000), 200);
    }

    #[test]
    fn test_trace_workload_from_entries() {
        let entries = vec![
            (0, 100),
            (1000, 200),
            (2000, 150),
            (3000, 300),
        ];
        let mut workload = TraceWorkload::from_entries(entries);

        assert_eq!(workload.next_rps(0), 100);
        assert_eq!(workload.next_rps(500), 100);
        assert_eq!(workload.next_rps(1000), 200);
        assert_eq!(workload.next_rps(1500), 200);
        assert_eq!(workload.next_rps(2000), 150);
        assert_eq!(workload.next_rps(3000), 300);
        assert_eq!(workload.next_rps(5000), 300); // Past end, hold last value
    }

    #[test]
    fn test_burst_workload_is_in_burst() {
        let config = BurstConfig {
            factor: 4,
            duration_ms: 200,
            interval_ms: 1000,
        };
        let workload = BurstWorkload::new(100, &config, 5);

        assert!(workload.is_in_burst(0));
        assert!(workload.is_in_burst(100));
        assert!(workload.is_in_burst(199));
        assert!(!workload.is_in_burst(200));
        assert!(!workload.is_in_burst(500));
        assert!(!workload.is_in_burst(999));
        assert!(workload.is_in_burst(1000));
        assert!(workload.is_in_burst(1100));
    }

    #[test]
    fn test_create_workload_model_constant() {
        let config = WorkloadConfig {
            msgs_per_sec: 100,
            msg_size_bytes: 256,
            duration_sec: 10,
            pattern: WorkloadPattern::Constant,
            burst: None,
            ramp: None,
            trace_file: None,
        };
        let mut model = create_workload_model(&config);
        assert_eq!(model.next_rps(0), 100);
        assert!(model.description().contains("Constant"));
    }

    #[test]
    fn test_create_workload_model_burst() {
        let config = WorkloadConfig {
            msgs_per_sec: 100,
            msg_size_bytes: 256,
            duration_sec: 10,
            pattern: WorkloadPattern::Burst,
            burst: Some(BurstConfig {
                factor: 3,
                duration_ms: 100,
                interval_ms: 500,
            }),
            ramp: None,
            trace_file: None,
        };
        let mut model = create_workload_model(&config);
        assert_eq!(model.next_rps(0), 300); // In burst
        assert_eq!(model.next_rps(200), 100); // Outside burst
        assert!(model.description().contains("Burst"));
    }

    #[test]
    fn test_create_workload_model_ramp() {
        let config = WorkloadConfig {
            msgs_per_sec: 100, // Not used for ramp
            msg_size_bytes: 256,
            duration_sec: 10,
            pattern: WorkloadPattern::Ramp,
            burst: None,
            ramp: Some(RampConfig {
                from: 10,
                to: 100,
                duration_sec: 5,
            }),
            trace_file: None,
        };
        let mut model = create_workload_model(&config);
        assert_eq!(model.next_rps(0), 10);
        assert_eq!(model.next_rps(5000), 100);
        assert!(model.description().contains("Ramp"));
    }
}


