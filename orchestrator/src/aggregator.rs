//! Result Aggregator
//!
//! Merges JSONL results from multiple workers and computes summary statistics.

use crate::controller::AggregationSummary;
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tracing::{info, warn};

#[derive(Error, Debug)]
pub enum AggregatorError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("No data to aggregate")]
    NoData,
}

/// Event record from JSONL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRecord {
    pub run_id: String,
    pub event_id: u64,
    pub timestamp_utc_iso: String,
    pub timestamp_monotonic_ns: u128,
    pub operation: String,
    pub algorithm: String,
    pub latency_us: u128,
    #[serde(default)]
    pub queue_delay_us: u128,
    #[serde(default)]
    pub worker_id: usize,
    pub payload_size_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ciphertext_size_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature_size_bytes: Option<usize>,
    pub cpu_user_seconds: f64,
    pub memory_rss_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Result aggregator for merging and analyzing benchmark results
pub struct ResultAggregator;

impl ResultAggregator {
    pub fn new() -> Self {
        Self
    }

    /// Aggregate multiple JSONL files into a single merged output
    pub async fn aggregate(
        &self,
        input_files: &[std::path::PathBuf],
        output_path: &Path,
    ) -> Result<AggregationSummary, AggregatorError> {
        info!("Aggregating {} files into {}", input_files.len(), output_path.display());

        let mut all_events: Vec<EventRecord> = Vec::new();

        // Read all events from all files
        for file_path in input_files {
            if !file_path.exists() {
                warn!("File does not exist: {}", file_path.display());
                continue;
            }

            let file = File::open(file_path).await?;
            let reader = BufReader::new(file);
            let mut lines = reader.lines();

            while let Some(line) = lines.next_line().await? {
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<EventRecord>(&line) {
                    Ok(event) => all_events.push(event),
                    Err(e) => {
                        warn!("Failed to parse line in {}: {}", file_path.display(), e);
                    }
                }
            }
        }

        if all_events.is_empty() {
            return Err(AggregatorError::NoData);
        }

        info!("Loaded {} total events", all_events.len());

        // Sort by timestamp
        all_events.sort_by_key(|e| e.timestamp_monotonic_ns);

        // Write merged output
        let output_file = File::create(output_path).await?;
        let mut writer = BufWriter::new(output_file);

        for event in &all_events {
            let line = serde_json::to_string(event)?;
            writer.write_all(line.as_bytes()).await?;
            writer.write_all(b"\n").await?;
        }
        writer.flush().await?;

        // Compute summary statistics
        let summary = self.compute_summary(&all_events);

        info!(
            "Aggregation complete: {} events, {:.2} ops/sec",
            summary.total_events, summary.throughput_ops_sec
        );

        Ok(summary)
    }

    /// Compute summary statistics from events
    fn compute_summary(&self, events: &[EventRecord]) -> AggregationSummary {
        if events.is_empty() {
            return AggregationSummary::default();
        }

        let total_events = events.len() as u64;

        // Calculate duration from first to last event
        let first_ts = events.first().map(|e| e.timestamp_monotonic_ns).unwrap_or(0);
        let last_ts = events.last().map(|e| e.timestamp_monotonic_ns).unwrap_or(0);
        let duration_ns = last_ts.saturating_sub(first_ts);
        let total_duration_sec = duration_ns as f64 / 1_000_000_000.0;

        // Throughput
        let throughput_ops_sec = if total_duration_sec > 0.0 {
            total_events as f64 / total_duration_sec
        } else {
            0.0
        };

        // Collect latency values
        let mut latencies: Vec<u128> = events.iter().map(|e| e.latency_us).collect();
        latencies.sort_unstable();

        // Collect queue delays
        let mut queue_delays: Vec<u128> = events.iter().map(|e| e.queue_delay_us).collect();
        queue_delays.sort_unstable();

        AggregationSummary {
            total_events,
            total_duration_sec,
            throughput_ops_sec,
            latency_p50_us: percentile(&latencies, 50.0),
            latency_p90_us: percentile(&latencies, 90.0),
            latency_p95_us: percentile(&latencies, 95.0),
            latency_p99_us: percentile(&latencies, 99.0),
            queue_delay_p50_us: percentile(&queue_delays, 50.0),
            queue_delay_p99_us: percentile(&queue_delays, 99.0),
        }
    }
}

/// Calculate percentile from sorted values
fn percentile(sorted_values: &[u128], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }

    let n = sorted_values.len();
    let idx = ((p / 100.0) * (n - 1) as f64) as usize;
    sorted_values[idx.min(n - 1)] as f64
}

impl Default for ResultAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile() {
        let values = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(percentile(&values, 50.0), 5.0);
        assert_eq!(percentile(&values, 90.0), 9.0);
    }

    #[test]
    fn test_percentile_empty() {
        let values: Vec<u128> = vec![];
        assert_eq!(percentile(&values, 50.0), 0.0);
    }
}

