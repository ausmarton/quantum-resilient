use crate::{CryptoAdapter, CryptoResult, MetricsCollector, workload::{WorkloadConfig, WorkloadHooks, WorkloadOp, run_streaming_workload}};
use crate::{OperationMetrics, OperationKind};
use std::sync::Arc;
use aes_gcm::{Aes256Gcm, KeyInit, aead::{Aead, Key, generic_array::GenericArray}};
use std::time::Instant;

fn expand_seed_to_key(seed: u64) -> [u8; 32] {
	let mut x = seed;
	let mut out = [0u8; 32];
	for i in 0..32 {
		// xorshift64*
		x ^= x >> 12;
		x ^= x << 25;
		x ^= x >> 27;
		let v = x.wrapping_mul(0x2545F4914F6CDD1D);
		out[i] = (v as u8) ^ (((v >> 32) & 0xFF) as u8);
	}
	out
}

fn derive_nonce(seed: u64, payload: &[u8]) -> [u8; 12] {
	let mut acc = seed;
	for &b in payload.iter() {
		acc = acc.wrapping_mul(16777619) ^ (b as u64);
	}
	let mut nonce = [0u8; 12];
	for i in 0..12 {
		nonce[i] = ((acc >> ((i % 8) * 8)) & 0xFF) as u8;
	}
	nonce
}

pub fn aes_gcm_encrypt(seed: u64, plaintext: &[u8]) -> CryptoResult<Vec<u8>> {
	let key_bytes = expand_seed_to_key(seed);
	let key: Key<Aes256Gcm> = Key::<Aes256Gcm>::from_slice(&key_bytes).clone();
	let cipher = Aes256Gcm::new(&key);
	let nonce_bytes = derive_nonce(seed, plaintext);
	let nonce = GenericArray::from_slice(&nonce_bytes);
	let ct = cipher.encrypt(nonce, plaintext)
		.map_err(|_| crate::CryptoError::InternalError("aes-gcm encrypt failed".into()))?;
	Ok(ct)
}

pub fn aes_gcm_decrypt(seed: u64, ciphertext: &[u8], associated_plain_example: &[u8]) -> CryptoResult<Vec<u8>> {
	let key_bytes = expand_seed_to_key(seed);
	let key: Key<Aes256Gcm> = Key::<Aes256Gcm>::from_slice(&key_bytes).clone();
	let cipher = Aes256Gcm::new(&key);
	let nonce_bytes = derive_nonce(seed, associated_plain_example);
	let nonce = GenericArray::from_slice(&nonce_bytes);
	let pt = cipher.decrypt(nonce, ciphertext)
		.map_err(|_| crate::CryptoError::InternalError("aes-gcm decrypt failed".into()))?;
	Ok(pt)
}

pub fn make_aes_gcm_hooks(seed: u64) -> WorkloadHooks {
	let enc_seed = seed;
	let dec_seed = seed;
	WorkloadHooks {
		encrypt: Some(Arc::new(move |payload: &[u8]| aes_gcm_encrypt(enc_seed, payload))),
		decrypt: Some(Arc::new(move |payload: &[u8]| aes_gcm_decrypt(dec_seed, payload, payload))),
	}
}

pub fn run_hybrid_tls_pqc_vs_classical(
	pqc_kem: &dyn CryptoAdapter,
	classical_kex: &dyn CryptoAdapter,
	classical_sig: &dyn CryptoAdapter,
	collector: Arc<dyn MetricsCollector>,
) -> CryptoResult<()> {
	// PQC handshake simulation: keygen + encapsulate + decapsulate
	let (pqc_pk, pqc_sk) = pqc_kem.keygen()?;
	let (ct, _ss) = pqc_kem.encapsulate(&pqc_pk)?;
	let _ss2 = pqc_kem.decapsulate(&pqc_sk, &ct)?;

	// Classical handshake simulation: ECDHE + RSA sign/verify
	let (kex_pk, kex_sk) = classical_kex.keygen()?;
	let (_transcript, _ss) = classical_kex.encapsulate(&kex_pk)?;
	let message = b"server_certificate_transcript";
	let sig = classical_sig.sign(&kex_sk, message)?;
	let _ = classical_sig.verify(&kex_pk, message, &sig)?;

	// Nothing more to do; operations were instrumented individually
	Ok(())
}

pub fn run_app_streaming_aes_gcm(
	adapter: &dyn CryptoAdapter,
	payload_bytes: usize,
	tps: u32,
	duration_secs: u64,
	repetitions: u32,
	seed: u64,
	collector: Arc<dyn MetricsCollector>,
) -> CryptoResult<crate::workload::WorkloadSummary> {
	let hooks = make_aes_gcm_hooks(seed);
	let cfg = WorkloadConfig {
		payload_bytes,
		tps,
		duration_secs,
		chunk_size_bytes: payload_bytes,
		repetitions,
		seed,
		op: WorkloadOp::Encrypt,
	};
	run_streaming_workload(adapter, &cfg, &hooks, collector)
}

pub fn run_key_wrap_like_for_like(
	pqc_kem: &dyn CryptoAdapter,
	rsa: &dyn CryptoAdapter,
	payload_len: usize,
	chunk_size_bytes: usize,
	repetitions: u32,
	seed: u64,
	collector: Arc<dyn MetricsCollector>,
) -> CryptoResult<()> {
	let plaintext = vec![0u8; payload_len];
	// PQC path (Kyber*)
	let (pqc_pk, pqc_sk) = {
		let t0 = Instant::now();
		let keys = pqc_kem.keygen()?;
		let dt = t0.elapsed();
		let m = OperationMetrics {
			timestamp_seconds_utc: Some(chrono::Utc::now().timestamp()),
			operation: OperationKind::Keygen,
			latency_micros: dt.as_micros() as u64,
			cpu_user_micros: None, cpu_system_micros: None, max_rss_bytes: None,
			algorithm: Some(pqc_kem.name().to_string()),
			parameter_set: None,
			public_key_bytes: Some(pqc_kem.public_key_size() as u64),
			secret_key_bytes: Some(pqc_kem.secret_key_size() as u64),
			signature_bytes: None, ciphertext_bytes: None, storage_overhead_pct: None,
			keygen_time_ms: Some(dt.as_secs_f64() * 1000.0),
			encapsulate_time_ms: None, decapsulate_time_ms: None,
			encrypt_time_ms: None, decrypt_time_ms: None,
			sign_time_ms: None, verify_time_ms: None,
			throughput_ops_per_sec: None,
			avg_cpu_percent: None, avg_memory_mb: None,
			disk_io_bytes: None, net_tx_bytes: None, net_rx_bytes: None,
		};
		collector.record(&m);
		keys
	};
	let (pqc_ct, pqc_ss) = {
		let t0 = Instant::now();
		let (ct, ss) = pqc_kem.encapsulate(&pqc_pk)?;
		let dt = t0.elapsed();
		let overhead = if payload_len > 0 {
			((ct.len() as f64) / (payload_len as f64)) * 100.0
		} else { 0.0 };
		let m = OperationMetrics {
			timestamp_seconds_utc: Some(chrono::Utc::now().timestamp()),
			operation: OperationKind::Encapsulate,
			latency_micros: dt.as_micros() as u64,
			cpu_user_micros: None, cpu_system_micros: None, max_rss_bytes: None,
			algorithm: Some(pqc_kem.name().to_string()),
			parameter_set: None,
			public_key_bytes: None, secret_key_bytes: None,
			signature_bytes: None, ciphertext_bytes: Some(ct.len() as u64),
			storage_overhead_pct: Some(overhead),
			keygen_time_ms: None, encapsulate_time_ms: Some(dt.as_secs_f64() * 1000.0),
			decapsulate_time_ms: None, encrypt_time_ms: None, decrypt_time_ms: None,
			sign_time_ms: None, verify_time_ms: None,
			throughput_ops_per_sec: if dt.as_micros() > 0 { Some(1_000_000.0 / dt.as_micros() as f64) } else { Some(0.0) },
			avg_cpu_percent: None, avg_memory_mb: None,
			disk_io_bytes: None, net_tx_bytes: None, net_rx_bytes: None,
		};
		collector.record(&m);
		(ct, ss)
	};
	// Encrypt payload using AES-GCM with ss-derived seed
	let enc_seed = {
		let mut acc: u64 = seed;
		for &b in pqc_ss.iter() { acc = acc.wrapping_mul(1315423911) ^ (b as u64); }
		acc
	};
	let mut offset = 0usize;
	while offset < plaintext.len() {
		let end = std::cmp::min(offset + chunk_size_bytes, plaintext.len());
		let slice = &plaintext[offset..end];
		let t0 = Instant::now();
		let ct = aes_gcm_encrypt(enc_seed, slice)?;
		let dt = t0.elapsed();
		let overhead = if slice.len() > 0 { ((ct.len() as f64 - slice.len() as f64) / slice.len() as f64) * 100.0 } else { 0.0 };
		let m = OperationMetrics {
			timestamp_seconds_utc: Some(chrono::Utc::now().timestamp()),
			operation: OperationKind::BulkEncrypt,
			latency_micros: dt.as_micros() as u64,
			cpu_user_micros: None, cpu_system_micros: None, max_rss_bytes: None,
			algorithm: Some("AES-GCM-256".into()),
			parameter_set: None,
			public_key_bytes: None, secret_key_bytes: None,
			signature_bytes: None, ciphertext_bytes: Some(ct.len() as u64),
			storage_overhead_pct: Some(overhead),
			keygen_time_ms: None, encapsulate_time_ms: None, decapsulate_time_ms: None,
			encrypt_time_ms: Some(dt.as_secs_f64() * 1000.0), decrypt_time_ms: None,
			sign_time_ms: None, verify_time_ms: None,
			throughput_ops_per_sec: if dt.as_micros() > 0 { Some(1_000_000.0 / dt.as_micros() as f64) } else { Some(0.0) },
			avg_cpu_percent: None, avg_memory_mb: None,
			disk_io_bytes: None, net_tx_bytes: None, net_rx_bytes: None,
		};
		collector.record(&m);
		offset = end;
	}

	// Classical RSA path
	let (rsa_pk, rsa_sk) = rsa.keygen()?;
	let (rsa_ct, rsa_ss) = rsa.encapsulate(&rsa_pk)?;
	let rsa_overhead = if payload_len > 0 { ((rsa_ct.len() as f64) / (payload_len as f64)) * 100.0 } else { 0.0 };
	let m_ct = OperationMetrics {
		timestamp_seconds_utc: Some(chrono::Utc::now().timestamp()),
		operation: OperationKind::Encapsulate,
		latency_micros: 0,
		cpu_user_micros: None, cpu_system_micros: None, max_rss_bytes: None,
		algorithm: Some(rsa.name().to_string()),
		parameter_set: None,
		public_key_bytes: None, secret_key_bytes: None,
		signature_bytes: None, ciphertext_bytes: Some(rsa_ct.len() as u64),
		storage_overhead_pct: Some(rsa_overhead),
		keygen_time_ms: None, encapsulate_time_ms: None, decapsulate_time_ms: None,
		encrypt_time_ms: None, decrypt_time_ms: None, sign_time_ms: None, verify_time_ms: None,
		throughput_ops_per_sec: None,
		avg_cpu_percent: None, avg_memory_mb: None,
		disk_io_bytes: None, net_tx_bytes: None, net_rx_bytes: None,
	};
	collector.record(&m_ct);
	let enc_seed2 = {
		let mut acc: u64 = seed ^ 0xA5A5A5A5A5A5A5A5;
		for &b in rsa_ss.iter() { acc = acc.wrapping_mul(2654435761) ^ (b as u64); }
		acc
	};
	let mut offset = 0usize;
	while offset < plaintext.len() {
		let end = std::cmp::min(offset + chunk_size_bytes, plaintext.len());
		let slice = &plaintext[offset..end];
		let t0 = Instant::now();
		let ct = aes_gcm_encrypt(enc_seed2, slice)?;
		let dt = t0.elapsed();
		let overhead = if slice.len() > 0 { ((ct.len() as f64 - slice.len() as f64) / slice.len() as f64) * 100.0 } else { 0.0 };
		let m = OperationMetrics {
			timestamp_seconds_utc: Some(chrono::Utc::now().timestamp()),
			operation: OperationKind::BulkEncrypt,
			latency_micros: dt.as_micros() as u64,
			cpu_user_micros: None, cpu_system_micros: None, max_rss_bytes: None,
			algorithm: Some("AES-GCM-256".into()),
			parameter_set: None,
			public_key_bytes: None, secret_key_bytes: None,
			signature_bytes: None, ciphertext_bytes: Some(ct.len() as u64),
			storage_overhead_pct: Some(overhead),
			keygen_time_ms: None, encapsulate_time_ms: None, decapsulate_time_ms: None,
			encrypt_time_ms: Some(dt.as_secs_f64() * 1000.0), decrypt_time_ms: None,
			sign_time_ms: None, verify_time_ms: None,
			throughput_ops_per_sec: if dt.as_micros() > 0 { Some(1_000_000.0 / dt.as_micros() as f64) } else { Some(0.0) },
			avg_cpu_percent: None, avg_memory_mb: None,
			disk_io_bytes: None, net_tx_bytes: None, net_rx_bytes: None,
		};
		collector.record(&m);
		offset = end;
	}
	Ok(())
}

pub fn run_integrity_like_for_like(
	dilithium: &dyn CryptoAdapter,
	ecdsa: &dyn CryptoAdapter,
	payload_len: usize,
	repetitions: u32,
	collector: Arc<dyn MetricsCollector>,
) -> CryptoResult<()> {
	let payload = vec![0u8; payload_len];
	let (_dpk, dsk) = dilithium.keygen()?;
	let (_epk, esk) = ecdsa.keygen()?;
	for _ in 0..repetitions {
		// Dilithium sign/verify
		let t0 = Instant::now();
		let dsig = dilithium.sign(&dsk, &payload)?;
		let dt = t0.elapsed();
		let m = OperationMetrics {
			timestamp_seconds_utc: Some(chrono::Utc::now().timestamp()),
			operation: OperationKind::Sign,
			latency_micros: dt.as_micros() as u64,
			cpu_user_micros: None, cpu_system_micros: None, max_rss_bytes: None,
			algorithm: Some(dilithium.name().to_string()),
			parameter_set: None,
			public_key_bytes: None, secret_key_bytes: None,
			signature_bytes: Some(dsig.len() as u64), ciphertext_bytes: None,
			storage_overhead_pct: if payload_len > 0 { Some((dsig.len() as f64 / payload_len as f64) * 100.0) } else { None },
			keygen_time_ms: None, encapsulate_time_ms: None, decapsulate_time_ms: None,
			encrypt_time_ms: None, decrypt_time_ms: None, sign_time_ms: Some(dt.as_secs_f64() * 1000.0),
			verify_time_ms: None,
			throughput_ops_per_sec: if dt.as_micros() > 0 { Some(1_000_000.0 / dt.as_micros() as f64) } else { Some(0.0) },
			avg_cpu_percent: None, avg_memory_mb: None,
			disk_io_bytes: None, net_tx_bytes: None, net_rx_bytes: None,
		};
		collector.record(&m);
		let t1 = Instant::now();
		let _ = dilithium.verify(&[_dpk.len() as u8], &payload, &dsig);
		let dv = t1.elapsed();
		let m2 = OperationMetrics {
			timestamp_seconds_utc: Some(chrono::Utc::now().timestamp()),
			operation: OperationKind::Verify,
			latency_micros: dv.as_micros() as u64,
			cpu_user_micros: None, cpu_system_micros: None, max_rss_bytes: None,
			algorithm: Some(dilithium.name().to_string()),
			parameter_set: None,
			public_key_bytes: None, secret_key_bytes: None,
			signature_bytes: Some(dsig.len() as u64), ciphertext_bytes: None,
			storage_overhead_pct: None,
			keygen_time_ms: None, encapsulate_time_ms: None, decapsulate_time_ms: None,
			encrypt_time_ms: None, decrypt_time_ms: None, sign_time_ms: None,
			verify_time_ms: Some(dv.as_secs_f64() * 1000.0),
			throughput_ops_per_sec: if dv.as_micros() > 0 { Some(1_000_000.0 / dv.as_micros() as f64) } else { Some(0.0) },
			avg_cpu_percent: None, avg_memory_mb: None,
			disk_io_bytes: None, net_tx_bytes: None, net_rx_bytes: None,
		};
		collector.record(&m2);

		// ECDSA sign/verify
		let t0 = Instant::now();
		let esig = ecdsa.sign(&esk, &payload)?;
		let dt = t0.elapsed();
		let m = OperationMetrics {
			timestamp_seconds_utc: Some(chrono::Utc::now().timestamp()),
			operation: OperationKind::Sign,
			latency_micros: dt.as_micros() as u64,
			cpu_user_micros: None, cpu_system_micros: None, max_rss_bytes: None,
			algorithm: Some(ecdsa.name().to_string()),
			parameter_set: None,
			public_key_bytes: None, secret_key_bytes: None,
			signature_bytes: Some(esig.len() as u64), ciphertext_bytes: None,
			storage_overhead_pct: if payload_len > 0 { Some((esig.len() as f64 / payload_len as f64) * 100.0) } else { None },
			keygen_time_ms: None, encapsulate_time_ms: None, decapsulate_time_ms: None,
			encrypt_time_ms: None, decrypt_time_ms: None, sign_time_ms: Some(dt.as_secs_f64() * 1000.0),
			verify_time_ms: None,
			throughput_ops_per_sec: if dt.as_micros() > 0 { Some(1_000_000.0 / dt.as_micros() as f64) } else { Some(0.0) },
			avg_cpu_percent: None, avg_memory_mb: None,
			disk_io_bytes: None, net_tx_bytes: None, net_rx_bytes: None,
		};
		collector.record(&m);
		let t1 = Instant::now();
		let _ = ecdsa.verify(&[_epk.len() as u8], &payload, &esig);
		let dv = t1.elapsed();
		let m2 = OperationMetrics {
			timestamp_seconds_utc: Some(chrono::Utc::now().timestamp()),
			operation: OperationKind::Verify,
			latency_micros: dv.as_micros() as u64,
			cpu_user_micros: None, cpu_system_micros: None, max_rss_bytes: None,
			algorithm: Some(ecdsa.name().to_string()),
			parameter_set: None,
			public_key_bytes: None, secret_key_bytes: None,
			signature_bytes: Some(esig.len() as u64), ciphertext_bytes: None,
			storage_overhead_pct: None,
			keygen_time_ms: None, encapsulate_time_ms: None, decapsulate_time_ms: None,
			encrypt_time_ms: None, decrypt_time_ms: None, sign_time_ms: None,
			verify_time_ms: Some(dv.as_secs_f64() * 1000.0),
			throughput_ops_per_sec: if dv.as_micros() > 0 { Some(1_000_000.0 / dv.as_micros() as f64) } else { Some(0.0) },
			avg_cpu_percent: None, avg_memory_mb: None,
			disk_io_bytes: None, net_tx_bytes: None, net_rx_bytes: None,
		};
		collector.record(&m2);
	}
	Ok(())
}


