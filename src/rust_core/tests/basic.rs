use rust_core::{adapters::kyber512::Kyber512, metrics::JsonFileMetricsCollector, workload::{WorkloadConfig, WorkloadOp, BackpressureMode}, CryptoAdapter};
use std::{path::PathBuf, sync::Arc};

#[test]
fn serialize_metrics_and_backpressure_compile() {
	let adapter = Kyber512;
	let tmp = tempfile::NamedTempFile::new().unwrap();
	let collector = Arc::new(JsonFileMetricsCollector::new(PathBuf::from(tmp.path())).unwrap());
	let cfg = WorkloadConfig {
		payload_bytes: 128,
		tps: 1000,
		duration_secs: 1,
		chunk_size_bytes: 128,
		repetitions: 1,
		seed: 42,
		op: WorkloadOp::Sign,
		retries: 1,
		retry_backoff_ms: 1,
		backpressure_mode: BackpressureMode::Drop,
		max_lag_ms: 0,
	};
	// Just ensure it runs without panic; metrics written to file
	let _ = rust_core::workload::run_streaming_workload(&adapter, &cfg, &Default::default(), collector);
}


