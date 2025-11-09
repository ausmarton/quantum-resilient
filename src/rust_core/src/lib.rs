//! Core traits and types for PQC benchmarking.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::time::Instant;

pub type CryptoResult<T> = Result<T, CryptoError>;

#[derive(Debug)]
pub enum CryptoError {
	UnsupportedOperation(&'static str),
	InvalidKey(&'static str),
	VerificationFailed,
	KemFailure,
	SignFailure,
	SerializationError(&'static str),
	RandomnessError(&'static str),
	InternalError(String),
}

impl Display for CryptoError {
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
		match self {
			CryptoError::UnsupportedOperation(op) => write!(f, "unsupported operation: {}", op),
			CryptoError::InvalidKey(msg) => write!(f, "invalid key: {}", msg),
			CryptoError::VerificationFailed => write!(f, "verification failed"),
			CryptoError::KemFailure => write!(f, "kem operation failed"),
			CryptoError::SignFailure => write!(f, "sign operation failed"),
			CryptoError::SerializationError(msg) => write!(f, "serialization error: {}", msg),
			CryptoError::RandomnessError(msg) => write!(f, "randomness error: {}", msg),
			CryptoError::InternalError(msg) => write!(f, "internal error: {}", msg),
		}
	}
}

impl Error for CryptoError {}

#[derive(Clone, Debug, serde::Serialize)]
pub enum OperationKind {
	Keygen,
	Encapsulate,
	Decapsulate,
	Sign,
	Verify,
	BulkEncrypt,
	BulkDecrypt,
	Dropped,
}

impl Default for OperationKind {
	fn default() -> Self { OperationKind::Keygen }
}
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct OperationMetrics {
	#[serde(with = "chrono::serde::ts_seconds_option", skip_serializing_if = "Option::is_none")]
	pub timestamp_seconds_utc: Option<chrono::DateTime<chrono::Utc>>,
	pub operation: OperationKind,
	pub latency_micros: u64,
	pub attempts: Option<u32>,
	pub error: Option<String>,
	pub cpu_user_micros: Option<u64>,
	pub cpu_system_micros: Option<u64>,
	pub max_rss_bytes: Option<u64>,
	// context
	pub algorithm: Option<String>,
	pub parameter_set: Option<String>,
	// sizes
	pub public_key_bytes: Option<u64>,
	pub secret_key_bytes: Option<u64>,
	pub signature_bytes: Option<u64>,
	pub ciphertext_bytes: Option<u64>,
	pub storage_overhead_pct: Option<f64>,
	// per-op named times (ms)
	pub keygen_time_ms: Option<f64>,
	pub encapsulate_time_ms: Option<f64>,
	pub decapsulate_time_ms: Option<f64>,
	pub encrypt_time_ms: Option<f64>,
	pub decrypt_time_ms: Option<f64>,
	pub sign_time_ms: Option<f64>,
	pub verify_time_ms: Option<f64>,
	// instantaneous performance/resource hints
	pub throughput_ops_per_sec: Option<f64>,
	pub avg_cpu_percent: Option<f64>,
	pub avg_memory_mb: Option<f64>,
	pub disk_io_bytes: Option<u64>,
	pub net_tx_bytes: Option<u64>,
	pub net_rx_bytes: Option<u64>,
}

pub trait MetricsCollector: Send + Sync {
	fn record(&self, metrics: &OperationMetrics);
}

pub struct NoopMetricsCollector;

impl MetricsCollector for NoopMetricsCollector {
	fn record(&self, _metrics: &OperationMetrics) {}
}

pub trait CryptoAdapter: Send + Sync {
	fn name(&self) -> &str;

	fn public_key_size(&self) -> usize;
	fn secret_key_size(&self) -> usize;
	fn signature_size(&self) -> usize;

	fn keygen(&self) -> CryptoResult<(Vec<u8>, Vec<u8>)>;

	fn encapsulate(&self, public_key: &[u8]) -> CryptoResult<(Vec<u8>, Vec<u8>)>;

	fn decapsulate(&self, secret_key: &[u8], ciphertext: &[u8]) -> CryptoResult<Vec<u8>>;

	fn sign(&self, secret_key: &[u8], message: &[u8]) -> CryptoResult<Vec<u8>>;

	fn verify(&self, public_key: &[u8], message: &[u8], signature: &[u8]) -> CryptoResult<()>;
}

impl<T: CryptoAdapter + ?Sized> CryptoAdapter for Box<T> {
	fn name(&self) -> &str {
		(**self).name()
	}
	fn public_key_size(&self) -> usize {
		(**self).public_key_size()
	}
	fn secret_key_size(&self) -> usize {
		(**self).secret_key_size()
	}
	fn signature_size(&self) -> usize {
		(**self).signature_size()
	}
	fn keygen(&self) -> CryptoResult<(Vec<u8>, Vec<u8>)> {
		(**self).keygen()
	}
	fn encapsulate(&self, public_key: &[u8]) -> CryptoResult<(Vec<u8>, Vec<u8>)> {
		(**self).encapsulate(public_key)
	}
	fn decapsulate(&self, secret_key: &[u8], ciphertext: &[u8]) -> CryptoResult<Vec<u8>> {
		(**self).decapsulate(secret_key, ciphertext)
	}
	fn sign(&self, secret_key: &[u8], message: &[u8]) -> CryptoResult<Vec<u8>> {
		(**self).sign(secret_key, message)
	}
	fn verify(&self, public_key: &[u8], message: &[u8], signature: &[u8]) -> CryptoResult<()> {
		(**self).verify(public_key, message, signature)
	}
}

pub struct InstrumentedAdapter<A: CryptoAdapter + ?Sized> {
	inner: Box<A>,
	collector: Arc<dyn MetricsCollector>,
}

impl<A: CryptoAdapter + ?Sized> InstrumentedAdapter<A> {
	pub fn new(inner: Box<A>, collector: Arc<dyn MetricsCollector>) -> Self {
		Self { inner, collector }
	}

	fn with_metrics<R>(
		&self,
		operation: OperationKind,
		f: impl FnOnce(&A) -> CryptoResult<R>,
	) -> CryptoResult<R> {
		let start = Instant::now();
		let result = f(&self.inner);
		let elapsed = start.elapsed();
		let (cpu_user_micros, cpu_system_micros, max_rss_bytes, disk_bytes, net_tx, net_rx) = sample_resources();
		let latency_micros = elapsed.as_micros() as u64;
		let latency_ms = (latency_micros as f64) / 1000.0;
		let throughput = if latency_micros > 0 { 1_000_000.0 / (latency_micros as f64) } else { 0.0 };
		let op_clone = operation.clone();
		let metrics = OperationMetrics {
			timestamp_seconds_utc: Some(chrono::Utc::now()),
			operation: operation,
			latency_micros,
			attempts: Some(1),
			error: None,
			cpu_user_micros,
			cpu_system_micros,
			max_rss_bytes,
			algorithm: Some(self.inner.name().to_string()),
			parameter_set: None,
			public_key_bytes: Some(self.inner.public_key_size() as u64),
			secret_key_bytes: Some(self.inner.secret_key_size() as u64),
			signature_bytes: Some(self.inner.signature_size() as u64),
			ciphertext_bytes: None,
			storage_overhead_pct: None,
			keygen_time_ms: if matches!(op_clone, OperationKind::Keygen) { Some(latency_ms) } else { None },
			encapsulate_time_ms: if matches!(op_clone, OperationKind::Encapsulate) { Some(latency_ms) } else { None },
			decapsulate_time_ms: if matches!(op_clone, OperationKind::Decapsulate) { Some(latency_ms) } else { None },
			encrypt_time_ms: if matches!(op_clone, OperationKind::BulkEncrypt) { Some(latency_ms) } else { None },
			decrypt_time_ms: if matches!(op_clone, OperationKind::BulkDecrypt) { Some(latency_ms) } else { None },
			sign_time_ms: if matches!(op_clone, OperationKind::Sign) { Some(latency_ms) } else { None },
			verify_time_ms: if matches!(op_clone, OperationKind::Verify) { Some(latency_ms) } else { None },
			throughput_ops_per_sec: Some(throughput),
			avg_cpu_percent: None,
			avg_memory_mb: max_rss_bytes.map(|b| (b as f64) / (1024.0 * 1024.0)),
			disk_io_bytes: disk_bytes,
			net_tx_bytes: net_tx,
			net_rx_bytes: net_rx,
		};
		self.collector.record(&metrics);
		result
	}
}

impl<A: CryptoAdapter + ?Sized> CryptoAdapter for InstrumentedAdapter<A> {
	fn name(&self) -> &str {
		self.inner.name()
	}
	fn public_key_size(&self) -> usize {
		self.inner.public_key_size()
	}
	fn secret_key_size(&self) -> usize {
		self.inner.secret_key_size()
	}
	fn signature_size(&self) -> usize {
		self.inner.signature_size()
	}

	fn keygen(&self) -> CryptoResult<(Vec<u8>, Vec<u8>)> {
		self.with_metrics(OperationKind::Keygen, |inner| inner.keygen())
	}
	fn encapsulate(&self, public_key: &[u8]) -> CryptoResult<(Vec<u8>, Vec<u8>)> {
		self.with_metrics(OperationKind::Encapsulate, |inner| inner.encapsulate(public_key))
	}
	fn decapsulate(&self, secret_key: &[u8], ciphertext: &[u8]) -> CryptoResult<Vec<u8>> {
		self.with_metrics(OperationKind::Decapsulate, |inner| inner.decapsulate(secret_key, ciphertext))
	}
	fn sign(&self, secret_key: &[u8], message: &[u8]) -> CryptoResult<Vec<u8>> {
		self.with_metrics(OperationKind::Sign, |inner| inner.sign(secret_key, message))
	}
	fn verify(&self, public_key: &[u8], message: &[u8], signature: &[u8]) -> CryptoResult<()> {
		self.with_metrics(OperationKind::Verify, |inner| inner.verify(public_key, message, signature))
	}
}

fn sample_resources() -> (Option<u64>, Option<u64>, Option<u64>, Option<u64>, Option<u64>, Option<u64>) {
	unsafe {
		let mut usage: libc::rusage = std::mem::zeroed();
		if libc::getrusage(libc::RUSAGE_SELF, &mut usage as *mut _) == 0 {
			let user_us = (usage.ru_utime.tv_sec as u64)
				.saturating_mul(1_000_000)
				.saturating_add(usage.ru_utime.tv_usec as u64);
			let sys_us = (usage.ru_stime.tv_sec as u64)
				.saturating_mul(1_000_000)
				.saturating_add(usage.ru_stime.tv_usec as u64);
			// ru_maxrss is in kilobytes on Linux
			let rss_bytes = (usage.ru_maxrss as u64).saturating_mul(1024);
			let disk = read_proc_self_io_bytes();
			let (net_rx, net_tx) = read_proc_net_dev_bytes();
			(Some(user_us), Some(sys_us), Some(rss_bytes), disk, net_tx, net_rx)
		} else {
			(None, None, None, None, None, None)
		}
	}
}

fn read_proc_self_io_bytes() -> Option<u64> {
	if cfg!(target_os = "linux") {
		if let Ok(content) = std::fs::read_to_string("/proc/self/io") {
			let mut sum: u128 = 0;
			for line in content.lines() {
				if let Some((k, v)) = line.split_once(":") {
					let key = k.trim();
					let val_str = v.trim();
					if key == "read_bytes" || key == "write_bytes" {
						if let Ok(val) = val_str.parse::<u128>() {
							sum = sum.saturating_add(val);
						}
					}
				}
			}
			return Some(sum as u64);
		}
	}
	None
}

fn read_proc_net_dev_bytes() -> (Option<u64>, Option<u64>) {
	if cfg!(target_os = "linux") {
		if let Ok(content) = std::fs::read_to_string("/proc/net/dev") {
			let mut rx: u128 = 0;
			let mut tx: u128 = 0;
			for line in content.lines().skip(2) {
				// format: iface:  rx_bytes ... tx_bytes ...
				if let Some((_iface, rest)) = line.split_once(":") {
					let parts: Vec<&str> = rest.split_whitespace().collect();
					if parts.len() >= 16 {
						if let Ok(rxb) = parts[0].parse::<u128>() {
							rx = rx.saturating_add(rxb);
						}
						if let Ok(txb) = parts[8].parse::<u128>() {
							tx = tx.saturating_add(txb);
						}
					}
				}
			}
			return (Some(rx as u64), Some(tx as u64));
		}
	}
	(None, None)
}

pub mod metrics;
pub mod adapters;
pub mod workload;
pub mod modes;
pub mod pyo3_mod;


