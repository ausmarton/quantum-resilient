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
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct OperationMetrics {
	pub operation: OperationKind,
	pub latency_micros: u64,
	pub cpu_user_micros: Option<u64>,
	pub cpu_system_micros: Option<u64>,
	pub max_rss_bytes: Option<u64>,
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
		let (cpu_user_micros, cpu_system_micros, max_rss_bytes) = sample_resources();
		let metrics = OperationMetrics {
			operation,
			latency_micros: elapsed.as_micros() as u64,
			cpu_user_micros,
			cpu_system_micros,
			max_rss_bytes,
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

fn sample_resources() -> (Option<u64>, Option<u64>, Option<u64>) {
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
			(Some(user_us), Some(sys_us), Some(rss_bytes))
		} else {
			(None, None, None)
		}
	}
}

pub mod metrics;
pub mod adapters;
pub mod workload;


