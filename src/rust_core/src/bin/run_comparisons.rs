use std::fs::{create_dir_all, OpenOptions};
use std::path::PathBuf;
use std::sync::Arc;

use rust_core::adapters::{ecdhe_p256::EcdheP256, ecdsa_p256::EcdsaP256, kyber512::Kyber512, rsa2048::Rsa2048};
use rust_core::metrics::JsonFileMetricsCollector;
use rust_core::modes::{run_hybrid_tls_pqc_vs_classical, run_integrity_like_for_like, run_key_wrap_like_for_like};
use rust_core::{CryptoAdapter, MetricsCollector};

fn main() {
	let results_dir = PathBuf::from("results");
	create_dir_all(&results_dir).ok();
	let jsonl_path = results_dir.join("metrics.jsonl");
	// truncate on start to avoid mixing previous runs
	let _ = OpenOptions::new().create(true).write(true).truncate(true).open(&jsonl_path);
	let collector: Arc<dyn MetricsCollector> =
		Arc::new(JsonFileMetricsCollector::new(jsonl_path).expect("open metrics output"));

	// Hybrid TLS-like comparison: Kyber vs ECDHE + RSA
	let kyber = Kyber512;
	let ecdhe = EcdheP256;
	let rsa = Rsa2048;
	let _ = run_hybrid_tls_pqc_vs_classical(&kyber, &ecdhe, &rsa, collector.clone());

	// Key-wrap like-for-like (identical AES-GCM chunking/payload)
	let payload_len = 1024usize;
	let chunk_size = 1024usize;
	let repetitions = 1u32;
	let seed = 42u64;
	let _ = run_key_wrap_like_for_like(&kyber, &rsa, payload_len, chunk_size, repetitions, seed, collector.clone());

	// Integrity like-for-like: Dilithium vs ECDSA
	let ecdsa = EcdsaP256;
	let dilithium = rust_core::adapters::dilithium2::Dilithium2;
	let _ = run_integrity_like_for_like(&dilithium, &ecdsa, 1024, 1, collector.clone());
}


