//! Telemetry Module
//!
//! This module provides telemetry collection and reporting capabilities
//! for benchmark measurements and performance analysis.

pub mod jsonl_logger;
pub mod metrics;
pub mod sysinfo_sampler;
pub mod tracing_setup;

pub use jsonl_logger::JsonlWriter;
pub use metrics::{start_metrics_server, Metrics};
pub use sysinfo_sampler::SysInfoSampler;
pub use tracing_setup::init_tracing;

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Error type for telemetry operations
#[derive(Debug)]
pub enum TelemetryError {
    /// Metric recording failed
    RecordError(String),
    /// Report generation failed
    ReportError(String),
}

/// A single metric measurement
#[derive(Debug, Clone)]
pub struct Metric {
    /// Name of the metric
    pub name: String,
    /// Value of the metric
    pub value: f64,
    /// Unit of measurement
    pub unit: String,
    /// Timestamp when the metric was recorded
    pub timestamp: Instant,
}

/// Telemetry collector for benchmark metrics
#[derive(Debug)]
pub struct Telemetry {
    metrics: HashMap<String, Vec<Metric>>,
    start_time: Option<Instant>,
}

impl Telemetry {
    /// Creates a new Telemetry collector
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            start_time: None,
        }
    }

    /// Starts the telemetry collection session
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.metrics.clear();
    }

    /// Stops the telemetry collection session
    pub fn stop(&mut self) -> Option<Duration> {
        self.start_time.map(|start| start.elapsed())
    }

    /// Records a metric value
    pub fn record(&mut self, name: &str, value: f64, unit: &str) -> Result<(), TelemetryError> {
        let metric = Metric {
            name: name.to_string(),
            value,
            unit: unit.to_string(),
            timestamp: Instant::now(),
        };

        self.metrics
            .entry(name.to_string())
            .or_default()
            .push(metric);

        Ok(())
    }

    /// Records a latency measurement
    pub fn record_latency(&mut self, name: &str, duration: Duration) -> Result<(), TelemetryError> {
        // Record in nanoseconds for precision, convert to microseconds for display
        self.record(name, duration.as_nanos() as f64 / 1000.0, "μs")
    }

    /// Returns all recorded metrics
    pub fn metrics(&self) -> &HashMap<String, Vec<Metric>> {
        &self.metrics
    }

    /// Calculates the average value for a given metric
    pub fn average(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).map(|metrics| {
            if metrics.is_empty() {
                0.0
            } else {
                metrics.iter().map(|m| m.value).sum::<f64>() / metrics.len() as f64
            }
        })
    }

    /// Generates a summary report
    pub fn summary(&self) -> Result<String, TelemetryError> {
        let mut report = String::from("Telemetry Summary\n");
        report.push_str("=================\n");

        for (name, metrics) in &self.metrics {
            if !metrics.is_empty() {
                let avg = metrics.iter().map(|m| m.value).sum::<f64>() / metrics.len() as f64;
                let unit = &metrics[0].unit;
                report.push_str(&format!(
                    "{}: {:.2} {} (avg, n={})\n",
                    name,
                    avg,
                    unit,
                    metrics.len()
                ));
            }
        }

        Ok(report)
    }
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_new() {
        let telemetry = Telemetry::new();
        assert!(telemetry.metrics().is_empty());
    }

    #[test]
    fn test_telemetry_record() {
        let mut telemetry = Telemetry::new();
        telemetry.start();
        telemetry.record("test_metric", 42.0, "ops").unwrap();
        assert_eq!(telemetry.metrics().len(), 1);
    }

    #[test]
    fn test_telemetry_average() {
        let mut telemetry = Telemetry::new();
        telemetry.record("latency", 10.0, "ms").unwrap();
        telemetry.record("latency", 20.0, "ms").unwrap();
        telemetry.record("latency", 30.0, "ms").unwrap();
        assert_eq!(telemetry.average("latency"), Some(20.0));
    }
}

