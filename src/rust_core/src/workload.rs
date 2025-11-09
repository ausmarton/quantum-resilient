use crate::{CryptoAdapter, CryptoError, CryptoResult, MetricsCollector, OperationKind, OperationMetrics};
use rand::RngCore;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread::sleep;

#[derive(Clone, Debug)]
pub enum WorkloadOp {
	Sign,
	Verify,
	KemEncapsulate,
	KemDecapsulate,
	Encrypt,
	Decrypt,
}

#[derive(Clone, Debug)]
pub struct WorkloadConfig {
	pub payload_bytes: usize,
	pub tps: u32,
	pub duration_secs: u64,
	pub chunk_size_bytes: usize,
	pub repetitions: u32,
	pub seed: u64,
	pub op: WorkloadOp,
}

#[derive(Default, Clone, Debug)]
pub struct WorkloadSummary {
	pub total_ops: u64,
	pub total_bytes: u64,
	pub elapsed_ms: u128,
}

pub struct WorkloadHooks {
	pub encrypt: Option<Arc<dyn Fn(&[u8]) -> CryptoResult<Vec<u8>> + Send + Sync>>,
	pub decrypt: Option<Arc<dyn Fn(&[u8]) -> CryptoResult<Vec<u8>> + Send + Sync>>,
}

impl Default for WorkloadHooks {
	fn default() -> Self {
		Self { encrypt: None, decrypt: None }
	}
}

pub fn run_streaming_workload(
	adapter: &dyn CryptoAdapter,
	cfg: &WorkloadConfig,
	hooks: &WorkloadHooks,
	collector: Arc<dyn MetricsCollector>,
) -> CryptoResult<WorkloadSummary> {
	let total_iterations_per_rep = compute_iterations(cfg.tps, cfg.duration_secs);
	let per_op_target_ns = if cfg.tps == 0 { 0 } else { 1_000_000_000u64 / (cfg.tps as u64) };

	let mut rng = ChaCha20Rng::seed_from_u64(cfg.seed);

	// Pre-generate key material as needed for the chosen operation
	let (pk, sk) = adapter.keygen()?;

	// Precompute artifacts for verify/decapsulate to avoid measuring unrelated ops
	let (verify_msg, verify_sig) = if matches!(cfg.op, WorkloadOp::Verify) {
		let msg = deterministic_payload(&mut rng, cfg.payload_bytes);
		let sig = adapter.sign(&sk, &msg)?;
		(Some(msg), Some(sig))
	} else {
		(None, None)
	};

	let kem_ciphertext = if matches!(cfg.op, WorkloadOp::KemDecapsulate) {
		let (ct, _ss) = adapter.encapsulate(&pk)?;
		Some(ct)
	} else {
		None
	};

	let start_overall = Instant::now();
	let mut total_ops: u64 = 0;
	let mut total_bytes: u64 = 0;

	for rep in 0..cfg.repetitions {
		let rep_start = Instant::now();
		for i in 0..total_iterations_per_rep {
			let iter_start_target = if per_op_target_ns > 0 {
				let target_elapsed_ns = (i as u64).saturating_mul(per_op_target_ns);
				Some(rep_start + Duration::from_nanos(target_elapsed_ns))
			} else {
				None
			};

			let payload = match cfg.op {
				WorkloadOp::Verify => verify_msg.as_ref().unwrap().clone(),
				_ => deterministic_payload(&mut rng, cfg.payload_bytes),
			};

			match cfg.op {
				WorkloadOp::Sign => {
					record_latency(&collector, OperationKind::Sign, || adapter.sign(&sk, &payload))?;
				}
				WorkloadOp::Verify => {
					record_latency(&collector, OperationKind::Verify, || {
						adapter.verify(&pk, &payload, verify_sig.as_ref().unwrap())
					})?;
				}
				WorkloadOp::KemEncapsulate => {
					record_latency(&collector, OperationKind::Encapsulate, || adapter.encapsulate(&pk))?;
				}
				WorkloadOp::KemDecapsulate => {
					let ct = kem_ciphertext.as_ref().expect("ciphertext for decapsulate");
					record_latency(&collector, OperationKind::Decapsulate, || adapter.decapsulate(&sk, ct))?;
				}
				WorkloadOp::Encrypt => {
					let encrypt = hooks.encrypt.as_ref().ok_or_else(|| CryptoError::UnsupportedOperation("encrypt"))?;
					record_latency(&collector, OperationKind::BulkEncrypt, || encrypt(&payload))?;
				}
				WorkloadOp::Decrypt => {
					let decrypt = hooks.decrypt.as_ref().ok_or_else(|| CryptoError::UnsupportedOperation("decrypt"))?;
					record_latency(&collector, OperationKind::BulkDecrypt, || decrypt(&payload))?;
				}
			}

			total_ops += 1;
			total_bytes = total_bytes.saturating_add(cfg.payload_bytes as u64);

			if let Some(target) = iter_start_target {
				let now = Instant::now();
				if target > now {
					sleep(target - now);
				}
			}
		}
		// Advance RNG deterministically across repetitions
		let rep_mix = (rep as u64).wrapping_mul(0x9E3779B97F4A7C15);
		rng = ChaCha20Rng::seed_from_u64(cfg.seed ^ rep_mix);
	}

	let elapsed_ms = start_overall.elapsed().as_millis();
	Ok(WorkloadSummary { total_ops, total_bytes, elapsed_ms })
}

fn deterministic_payload(rng: &mut ChaCha20Rng, size: usize) -> Vec<u8> {
	let mut buf = vec![0u8; size];
	rng.fill_bytes(&mut buf);
	buf
}

fn record_latency<T, F: FnOnce() -> CryptoResult<T>>(collector: &Arc<dyn MetricsCollector>, op: OperationKind, f: F) -> CryptoResult<T> {
	let start = Instant::now();
	let res = f();
	let elapsed = start.elapsed();
	let metrics = OperationMetrics {
		operation: op,
		latency_micros: elapsed.as_micros() as u64,
		cpu_user_micros: None,
		cpu_system_micros: None,
		max_rss_bytes: None,
	};
	collector.record(&metrics);
	res
}

fn compute_iterations(tps: u32, duration_secs: u64) -> u64 {
	if tps == 0 || duration_secs == 0 {
		1
	} else {
		(tps as u64).saturating_mul(duration_secs)
	}
}


