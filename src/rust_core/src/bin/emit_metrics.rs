use std::path::PathBuf;
use std::sync::Arc;

use rust_core::adapters::kyber512::Kyber512;
use rust_core::metrics::JsonFileMetricsCollector;
use rust_core::{CryptoAdapter, InstrumentedAdapter, MetricsCollector, OperationKind};

fn main() {
	let out_dir = PathBuf::from("results");
	std::fs::create_dir_all(&out_dir).ok();
	let jsonl = out_dir.join("metrics.jsonl");
	let collector = Arc::new(JsonFileMetricsCollector::new(jsonl).expect("open metrics file"));

	let adapter = Kyber512;
	let inst = InstrumentedAdapter::new(Box::new(adapter), collector.clone());

	// Generate a few events to validate structured logs exist
	let _ = inst.keygen();
	// Use the public key from keygen to exercise encapsulate
	if let Ok((pk, _sk)) = inst.keygen() {
		let _ = inst.encapsulate(&pk);
	}

	// Emit a synthetic dropped event for validation
	let drop_evt = rust_core::OperationMetrics {
		timestamp_seconds_utc: Some(chrono::Utc::now()),
		operation: OperationKind::Dropped,
		latency_micros: 0,
		attempts: Some(0),
		error: None,
		cpu_user_micros: None,
		cpu_system_micros: None,
		max_rss_bytes: None,
		algorithm: Some("synthetic".into()),
		parameter_set: None,
		public_key_bytes: None,
		secret_key_bytes: None,
		signature_bytes: None,
		ciphertext_bytes: None,
		storage_overhead_pct: None,
		keygen_time_ms: None,
		encapsulate_time_ms: None,
		decapsulate_time_ms: None,
		encrypt_time_ms: None,
		decrypt_time_ms: None,
		sign_time_ms: None,
		verify_time_ms: None,
		throughput_ops_per_sec: None,
		avg_cpu_percent: None,
		avg_memory_mb: None,
		disk_io_bytes: None,
		net_tx_bytes: None,
		net_rx_bytes: None,
	};
	collector.record(&drop_evt);
}


