use crate::{CryptoAdapter, CryptoResult, MetricsCollector, workload::{WorkloadConfig, WorkloadHooks, WorkloadOp, run_streaming_workload}};
use std::sync::Arc;
use aes_gcm::{Aes256Gcm, KeyInit, aead::{Aead, Key, generic_array::GenericArray}};

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


