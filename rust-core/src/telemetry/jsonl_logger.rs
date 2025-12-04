//! JSONL Logger
//!
//! Thread-safe JSON Lines (JSONL) file writer for event logging.

use parking_lot::Mutex;
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

/// Thread-safe JSONL writer that appends JSON objects (one per line)
#[derive(Clone)]
pub struct JsonlWriter {
    inner: Arc<Mutex<JsonlWriterInner>>,
}

struct JsonlWriterInner {
    file: File,
    path: String,
}

impl JsonlWriter {
    /// Creates a new JSONL writer at the specified path
    ///
    /// Creates parent directories if they don't exist.
    pub fn new(path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Create parent directories if needed
        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;

        Ok(Self {
            inner: Arc::new(Mutex::new(JsonlWriterInner {
                file,
                path: path.to_string(),
            })),
        })
    }

    /// Writes a JSON event as a single line
    pub fn write_event(
        &self,
        value: &serde_json::Value,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let json_line = serde_json::to_string(value)?;
        let mut inner = self.inner.lock();
        writeln!(inner.file, "{}", json_line)?;
        inner.file.flush()?;
        Ok(())
    }

    /// Writes a serializable event as a single line
    pub fn write<T: Serialize>(
        &self,
        event: &T,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let json_line = serde_json::to_string(event)?;
        let mut inner = self.inner.lock();
        writeln!(inner.file, "{}", json_line)?;
        inner.file.flush()?;
        Ok(())
    }

    /// Returns the path to the JSONL file
    pub fn path(&self) -> String {
        self.inner.lock().path.clone()
    }
}

/// Event row for JSONL output with all mandatory fields
#[derive(Debug, Serialize)]
pub struct EventRow {
    pub run_id: String,
    pub scenario_id: String,
    pub event_id: u64,
    pub timestamp_utc_iso: String,
    pub timestamp_monotonic_ns: u128,
    pub operation: String,
    pub algorithm: String,
    pub latency_us: u128,
    pub payload_size_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ciphertext_size_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature_size_bytes: Option<usize>,
    pub cpu_user_seconds: f64,
    pub memory_rss_bytes: u64,
    pub rng_seed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Extended event row with queue delay for JSONL output (used by execution engine)
#[derive(Debug, Serialize)]
pub struct EventRowFull {
    pub run_id: String,
    pub scenario_id: String,
    pub event_id: u64,
    pub timestamp_utc_iso: String,
    pub timestamp_monotonic_ns: u128,
    pub operation: String,
    pub algorithm: String,
    pub latency_us: u128,
    pub queue_delay_us: u128,
    pub worker_id: usize,
    pub payload_size_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ciphertext_size_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature_size_bytes: Option<usize>,
    pub cpu_user_seconds: f64,
    pub memory_rss_bytes: u64,
    pub rng_seed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::BufRead;

    #[test]
    fn test_jsonl_writer_creates_file() {
        let path = "/tmp/test_jsonl_writer.jsonl";
        let _ = fs::remove_file(path); // Clean up from previous runs

        let writer = JsonlWriter::new(path).unwrap();
        assert!(Path::new(path).exists());
        assert_eq!(writer.path(), path);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_jsonl_writer_appends_lines() {
        let path = "/tmp/test_jsonl_append.jsonl";
        let _ = fs::remove_file(path);

        let writer = JsonlWriter::new(path).unwrap();

        // Write multiple events
        let event1 = serde_json::json!({"event_id": 1, "value": "first"});
        let event2 = serde_json::json!({"event_id": 2, "value": "second"});

        writer.write_event(&event1).unwrap();
        writer.write_event(&event2).unwrap();

        // Read and verify
        let file = File::open(path).unwrap();
        let reader = std::io::BufReader::new(file);
        let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();

        assert_eq!(lines.len(), 2);

        let parsed1: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(parsed1["event_id"], 1);

        let parsed2: serde_json::Value = serde_json::from_str(&lines[1]).unwrap();
        assert_eq!(parsed2["event_id"], 2);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_jsonl_writer_creates_directories() {
        let path = "/tmp/nested/dir/test.jsonl";
        let _ = fs::remove_file(path);
        let _ = fs::remove_dir_all("/tmp/nested");

        let writer = JsonlWriter::new(path).unwrap();
        assert!(Path::new(path).exists());

        fs::remove_file(path).unwrap();
        fs::remove_dir_all("/tmp/nested").unwrap();
    }
}

