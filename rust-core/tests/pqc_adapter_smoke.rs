//! PQC Adapter Smoke Tests
//!
//! Tests that verify Kyber and Dilithium adapters work correctly.

use rust_core::crypto_adapter::{CryptoAdapter, KyberAdapter};

#[cfg(feature = "pqcrypto_dilithium")]
use rust_core::crypto_adapter::DilithiumAdapter;

/// Test Kyber encapsulate/decapsulate round-trip
#[test]
#[cfg(feature = "pqcrypto_fallback")]
fn test_kyber_encapsulate_decapsulate_roundtrip() {
    let adapter = KyberAdapter::new("kyber512").expect("Failed to create Kyber adapter");
    
    // Generate and cache keypair
    let meta = adapter.generate_and_cache_keypair()
        .expect("Failed to generate keypair");
    
    let public_key = meta.public_key.clone();
    let secret_key = adapter.get_secret_key().expect("Failed to get secret key");
    
    // Encapsulate
    let (ciphertext, shared_secret1) = adapter
        .encapsulate(&public_key)
        .expect("Encapsulation failed");
    
    // Verify sizes
    assert!(!ciphertext.is_empty(), "Ciphertext should not be empty");
    assert!(!shared_secret1.is_empty(), "Shared secret should not be empty");
    assert_eq!(shared_secret1.len(), 32, "Kyber shared secret should be 32 bytes");
    
    // Decapsulate
    let shared_secret2 = adapter
        .decapsulate(&secret_key, &ciphertext)
        .expect("Decapsulation failed");
    
    // Verify shared secrets match
    assert_eq!(
        shared_secret1, shared_secret2,
        "Shared secrets must match after encapsulate/decapsulate"
    );
    
    println!("Kyber round-trip test passed!");
    println!("  Public key size: {} bytes", public_key.len());
    println!("  Secret key size: {} bytes", secret_key.len());
    println!("  Ciphertext size: {} bytes", ciphertext.len());
    println!("  Shared secret size: {} bytes", shared_secret1.len());
}

/// Test Kyber with different encapsulations produce different ciphertexts
#[test]
#[cfg(feature = "pqcrypto_fallback")]
fn test_kyber_different_encapsulations() {
    let adapter = KyberAdapter::new("kyber512").expect("Failed to create adapter");
    
    let meta = adapter.generate_and_cache_keypair().expect("Keygen failed");
    let pk = meta.public_key;
    
    // Two encapsulations should produce different ciphertexts but same shared secrets
    // when decapsulated with the same secret key
    let (ct1, _ss1) = adapter.encapsulate(&pk).expect("Encap 1 failed");
    let (ct2, _ss2) = adapter.encapsulate(&pk).expect("Encap 2 failed");
    
    // Ciphertexts should be different (randomized KEM)
    assert_ne!(ct1, ct2, "Ciphertexts should be different due to randomization");
}

/// Test Dilithium sign/verify round-trip
#[test]
#[cfg(feature = "pqcrypto_dilithium")]
fn test_dilithium_sign_verify_roundtrip() {
    let adapter = DilithiumAdapter::new("dilithium2").expect("Failed to create Dilithium adapter");
    
    // Generate keypair
    let keypair = adapter.generate_and_cache_keypair().expect("Keygen failed");
    let public_key = keypair.public_key;
    let secret_key = keypair.secret_key;
    
    // Sign a message
    let message = b"Hello, quantum-resilient world!";
    let signature = adapter
        .sign(&secret_key, message)
        .expect("Signing failed");
    
    // Verify sizes
    assert!(!signature.is_empty(), "Signature should not be empty");
    
    // Verify signature
    let is_valid = adapter
        .verify(&public_key, message, &signature)
        .expect("Verification failed");
    
    assert!(is_valid, "Signature should be valid");
    
    println!("Dilithium round-trip test passed!");
    println!("  Public key size: {} bytes", public_key.len());
    println!("  Secret key size: {} bytes", secret_key.len());
    println!("  Signature size: {} bytes", signature.len());
}

/// Test Dilithium verification fails with wrong message
#[test]
#[cfg(feature = "pqcrypto_dilithium")]
fn test_dilithium_verify_wrong_message() {
    let adapter = DilithiumAdapter::new("dilithium2").expect("Failed to create adapter");
    
    let keypair = adapter.generate_and_cache_keypair().expect("Keygen failed");
    let public_key = keypair.public_key;
    let secret_key = keypair.secret_key;
    
    // Sign a message
    let message = b"Original message";
    let signature = adapter.sign(&secret_key, message).expect("Signing failed");
    
    // Try to verify with wrong message
    let wrong_message = b"Tampered message";
    let is_valid = adapter
        .verify(&public_key, wrong_message, &signature)
        .expect("Verification call failed");
    
    assert!(!is_valid, "Verification should fail with wrong message");
}

/// Test Dilithium verification fails with wrong public key
#[test]
#[cfg(feature = "pqcrypto_dilithium")]
fn test_dilithium_verify_wrong_key() {
    let adapter1 = DilithiumAdapter::new("dilithium2").expect("Failed to create adapter 1");
    let adapter2 = DilithiumAdapter::new("dilithium2").expect("Failed to create adapter 2");
    
    // Generate two different keypairs
    let keypair1 = adapter1.generate_and_cache_keypair().expect("Keygen 1 failed");
    let keypair2 = adapter2.generate_and_cache_keypair().expect("Keygen 2 failed");
    
    // Sign with keypair1
    let message = b"Test message";
    let signature = adapter1.sign(&keypair1.secret_key, message).expect("Signing failed");
    
    // Try to verify with keypair2's public key - should fail
    let is_valid = adapter2
        .verify(&keypair2.public_key, message, &signature)
        .expect("Verification call failed");
    
    assert!(!is_valid, "Verification should fail with wrong public key");
}

/// Test adapter names are correct
#[test]
fn test_adapter_names() {
    let kyber = KyberAdapter::new("kyber512").expect("Kyber creation failed");
    assert_eq!(kyber.name(), "kyber");
    
    #[cfg(feature = "pqcrypto_dilithium")]
    {
        let dilithium = DilithiumAdapter::new("dilithium2").expect("Dilithium creation failed");
        assert_eq!(dilithium.name(), "dilithium");
    }
}

/// Test invalid parameter sets are rejected
#[test]
fn test_invalid_paramsets_rejected() {
    let result = KyberAdapter::new("kyber999");
    assert!(result.is_err(), "Invalid Kyber paramset should be rejected");
    
    #[cfg(feature = "pqcrypto_dilithium")]
    {
        let result = DilithiumAdapter::new("dilithium999");
        assert!(result.is_err(), "Invalid Dilithium paramset should be rejected");
    }
}

