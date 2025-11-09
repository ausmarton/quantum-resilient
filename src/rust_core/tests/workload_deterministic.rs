use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rust_core::workload::{run_streaming_workload, BackpressureMode, WorkloadConfig, WorkloadHooks, WorkloadOp};
use rust_core::{adapters::kyber512::Kyber512, MetricsCollector, NoopMetricsCollector};
use std::sync::Arc;

#[test]
fn workload_generates_deterministic_payloads() {
	let seed = 12345u64;
	let payload_bytes = 128usize;
	let mut rng = ChaCha20Rng::seed_from_u64(seed);
	let mut expected = vec![0u8; payload_bytes];
	use rand::RngCore;
	rng.fill_bytes(&mut expected);

	let hooks = WorkloadHooks {
		encrypt: Some(Arc::new(move |payload: &[u8]| {
			assert_eq!(payload.len(), expected.len());
			assert_eq!(&payload[0..payload.len()], &expected[0..payload.len()]);
			Ok(Vec::new())
		})),
		decrypt: None,
	};
	let adapter = Kyber512;
	let cfg = WorkloadConfig {
		payload_bytes,
		tps: 1,
		duration_secs: 1,
		chunk_size_bytes: payload_bytes,
		repetitions: 1,
		seed,
		op: WorkloadOp::Encrypt,
		retries: 0,
		retry_backoff_ms: 0,
		backpressure_mode: BackpressureMode::Block,
		max_lag_ms: 0,
	};
	let collector: Arc<dyn MetricsCollector> = Arc::new(NoopMetricsCollector);
	let _summary = run_streaming_workload(&adapter, &cfg, &hooks, collector).unwrap();
}


