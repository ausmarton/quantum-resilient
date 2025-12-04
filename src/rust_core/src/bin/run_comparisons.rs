use std::fs::{create_dir_all, OpenOptions};
use std::path::PathBuf;
use std::sync::Arc;

use rust_core::adapters::{
	dilithium2::Dilithium2, dilithium3::Dilithium3,
	ecdhe_p256::EcdheP256, ecdsa_p256::EcdsaP256,
	kyber512::Kyber512, kyber768::Kyber768,
	rsa2048::Rsa2048,
};
use rust_core::metrics::JsonFileMetricsCollector;
use rust_core::{CryptoAdapter, InstrumentedAdapter, MetricsCollector};

fn main() {
	let results_dir = PathBuf::from("results");
	create_dir_all(&results_dir).ok();
	let jsonl_path = results_dir.join("metrics.jsonl");
	// truncate on start to avoid mixing previous runs
	let _ = OpenOptions::new().create(true).write(true).truncate(true).open(&jsonl_path);
	let collector: Arc<dyn MetricsCollector> =
		Arc::new(JsonFileMetricsCollector::new(jsonl_path).expect("open metrics output"));

	// Use InstrumentedAdapter to capture resource metrics (CPU, memory, I/O)
	
	// PQC Algorithms
	let kyber512 = InstrumentedAdapter::new(Box::new(Kyber512), collector.clone());
	let kyber768 = InstrumentedAdapter::new(Box::new(Kyber768), collector.clone());
	let dilithium2 = InstrumentedAdapter::new(Box::new(Dilithium2), collector.clone());
	let dilithium3 = InstrumentedAdapter::new(Box::new(Dilithium3), collector.clone());
	
	// Classical Algorithms
	let rsa = InstrumentedAdapter::new(Box::new(Rsa2048), collector.clone());
	let ecdsa = InstrumentedAdapter::new(Box::new(EcdsaP256), collector.clone());
	let ecdhe = InstrumentedAdapter::new(Box::new(EcdheP256), collector.clone());

	// === KEM Operations (Key Exchange) ===
	
	// Kyber512 KEM
	let _ = kyber512.keygen();
	if let Ok((pk, sk)) = kyber512.keygen() {
		let _ = kyber512.encapsulate(&pk);
		if let Ok((ct, _ss)) = kyber512.encapsulate(&pk) {
			let _ = kyber512.decapsulate(&sk, &ct);
		}
	}
	
	// Kyber768 KEM (higher security level)
	let _ = kyber768.keygen();
	if let Ok((pk, sk)) = kyber768.keygen() {
		let _ = kyber768.encapsulate(&pk);
		if let Ok((ct, _ss)) = kyber768.encapsulate(&pk) {
			let _ = kyber768.decapsulate(&sk, &ct);
		}
	}
	
	// RSA-2048 KEM (classical)
	let _ = rsa.keygen();
	if let Ok((pk, _sk)) = rsa.keygen() {
		let _ = rsa.encapsulate(&pk);
	}
	
	// ECDHE-P256 (classical key exchange)
	let _ = ecdhe.keygen();
	if let Ok((pk, _sk)) = ecdhe.keygen() {
		let _ = ecdhe.encapsulate(&pk);
	}

	// === Signature Operations ===
	
	let message = vec![0u8; 1024];
	
	// Dilithium2 (PQC signature)
	if let Ok((_pk, sk)) = dilithium2.keygen() {
		if let Ok(sig) = dilithium2.sign(&sk, &message) {
			let _ = dilithium2.verify(&[0u8; 800], &message, &sig);
		}
	}
	
	// Dilithium3 (PQC signature, higher security)
	if let Ok((_pk, sk)) = dilithium3.keygen() {
		if let Ok(sig) = dilithium3.sign(&sk, &message) {
			let _ = dilithium3.verify(&[0u8; 1200], &message, &sig);
		}
	}
	
	// ECDSA-P256 (classical signature)
	if let Ok((_pk, sk)) = ecdsa.keygen() {
		if let Ok(sig) = ecdsa.sign(&sk, &message) {
			let _ = ecdsa.verify(&[0u8; 64], &message, &sig);
		}
	}

	// === AES-GCM Symmetric Encryption ===
	use std::time::Instant;
	let plaintext = vec![0u8; 1024];
	let seed = 42u64;
	
	// Encrypt
	let t0 = Instant::now();
	let ciphertext = rust_core::modes::aes_gcm_encrypt(seed, &plaintext);
	let encrypt_time = t0.elapsed();
	
	if ciphertext.is_ok() {
		let (cpu_user, cpu_sys, max_rss, disk, net_tx, net_rx) = rust_core::sample_resources_public();
		let metrics = rust_core::OperationMetrics {
			timestamp_seconds_utc: Some(chrono::Utc::now()),
			operation: rust_core::OperationKind::BulkEncrypt,
			latency_micros: encrypt_time.as_micros() as u64,
			attempts: Some(1),
			error: None,
			cpu_user_micros: cpu_user,
			cpu_system_micros: cpu_sys,
			max_rss_bytes: max_rss,
			algorithm: Some("AES-GCM-256".to_string()),
			parameter_set: None,
			public_key_bytes: None,
			secret_key_bytes: None,
			signature_bytes: None,
			ciphertext_bytes: ciphertext.as_ref().ok().map(|c| c.len() as u64),
			storage_overhead_pct: ciphertext.as_ref().ok().map(|c| (c.len() as f64 / plaintext.len() as f64) * 100.0),
			keygen_time_ms: None,
			encapsulate_time_ms: None,
			decapsulate_time_ms: None,
			encrypt_time_ms: Some(encrypt_time.as_secs_f64() * 1000.0),
			decrypt_time_ms: None,
			sign_time_ms: None,
			verify_time_ms: None,
			throughput_ops_per_sec: if encrypt_time.as_micros() > 0 { Some(1_000_000.0 / encrypt_time.as_micros() as f64) } else { Some(0.0) },
			avg_cpu_percent: None,
			avg_memory_mb: max_rss.map(|r| r as f64 / 1024.0 / 1024.0),
			disk_io_bytes: disk,
			net_tx_bytes: net_tx,
			net_rx_bytes: net_rx,
		};
		collector.record(&metrics);
		
		// Decrypt
		if let Ok(ct) = ciphertext {
			let t1 = Instant::now();
			let _decrypted = rust_core::modes::aes_gcm_decrypt(seed, &ct, &plaintext);
			let decrypt_time = t1.elapsed();
			
			let (cpu_user, cpu_sys, max_rss, disk, net_tx, net_rx) = rust_core::sample_resources_public();
			let metrics = rust_core::OperationMetrics {
				timestamp_seconds_utc: Some(chrono::Utc::now()),
				operation: rust_core::OperationKind::BulkDecrypt,
				latency_micros: decrypt_time.as_micros() as u64,
				attempts: Some(1),
				error: None,
				cpu_user_micros: cpu_user,
				cpu_system_micros: cpu_sys,
				max_rss_bytes: max_rss,
				algorithm: Some("AES-GCM-256".to_string()),
				parameter_set: None,
				public_key_bytes: None,
				secret_key_bytes: None,
				signature_bytes: None,
				ciphertext_bytes: Some(ct.len() as u64),
				storage_overhead_pct: Some((ct.len() as f64 / plaintext.len() as f64) * 100.0),
				keygen_time_ms: None,
				encapsulate_time_ms: None,
				decapsulate_time_ms: None,
				encrypt_time_ms: None,
				decrypt_time_ms: Some(decrypt_time.as_secs_f64() * 1000.0),
				sign_time_ms: None,
				verify_time_ms: None,
				throughput_ops_per_sec: if decrypt_time.as_micros() > 0 { Some(1_000_000.0 / decrypt_time.as_micros() as f64) } else { Some(0.0) },
				avg_cpu_percent: None,
				avg_memory_mb: max_rss.map(|r| r as f64 / 1024.0 / 1024.0),
				disk_io_bytes: disk,
				net_tx_bytes: net_tx,
				net_rx_bytes: net_rx,
			};
			collector.record(&metrics);
		}
	}
}


