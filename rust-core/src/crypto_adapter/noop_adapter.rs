//! NoOp Crypto Adapter
//!
//! A zero-cost cryptographic adapter for baseline benchmarking.
//! All operations return hard-coded dummy values with minimal overhead.

use super::{CryptoAdapter, CryptoError, KeypairMeta};

/// A no-operation cryptographic adapter for baseline benchmarking.
///
/// This adapter performs zero actual cryptographic operations.
/// It returns hard-coded byte arrays and always succeeds, making it
/// useful for measuring pipeline overhead without crypto costs.
#[derive(Debug, Clone, Default)]
pub struct NoOpCryptoAdapter;

impl NoOpCryptoAdapter {
    /// Creates a new NoOpCryptoAdapter
    pub fn new() -> Self {
        Self
    }
}

/// Hard-coded dummy public key (32 bytes of 0xAA)
const DUMMY_PUBLIC_KEY: [u8; 32] = [0xAA; 32];

/// Hard-coded dummy secret key length
const DUMMY_SECRET_KEY_LENGTH: usize = 32;

/// Hard-coded dummy ciphertext (48 bytes of 0xBB)
const DUMMY_CIPHERTEXT: [u8; 48] = [0xBB; 48];

/// Hard-coded dummy shared secret (32 bytes of 0xCC)
const DUMMY_SHARED_SECRET: [u8; 32] = [0xCC; 32];

/// Hard-coded dummy signature (64 bytes of 0xDD)
const DUMMY_SIGNATURE: [u8; 64] = [0xDD; 64];

impl CryptoAdapter for NoOpCryptoAdapter {
    fn name(&self) -> &'static str {
        "noop"
    }

    fn keygen(&self) -> Result<KeypairMeta, CryptoError> {
        Ok(KeypairMeta {
            public_key: DUMMY_PUBLIC_KEY.to_vec(),
            secret_key_length: DUMMY_SECRET_KEY_LENGTH,
            params: "noop-default".to_string(),
        })
    }

    fn encapsulate(&self, _public_key: &[u8]) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
        // Return dummy ciphertext and shared secret
        Ok((DUMMY_CIPHERTEXT.to_vec(), DUMMY_SHARED_SECRET.to_vec()))
    }

    fn decapsulate(&self, _secret_key: &[u8], _ciphertext: &[u8]) -> Result<Vec<u8>, CryptoError> {
        // Return dummy shared secret
        Ok(DUMMY_SHARED_SECRET.to_vec())
    }

    fn sign(&self, _secret_key: &[u8], _msg: &[u8]) -> Result<Vec<u8>, CryptoError> {
        // Return dummy signature
        Ok(DUMMY_SIGNATURE.to_vec())
    }

    fn verify(&self, _public_key: &[u8], _msg: &[u8], _sig: &[u8]) -> Result<bool, CryptoError> {
        // Always return true for noop adapter
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_name() {
        let adapter = NoOpCryptoAdapter::new();
        assert_eq!(adapter.name(), "noop");
    }

    #[test]
    fn test_noop_keygen() {
        let adapter = NoOpCryptoAdapter::new();
        let result = adapter.keygen();
        assert!(result.is_ok());

        let meta = result.unwrap();
        assert_eq!(meta.public_key.len(), 32);
        assert_eq!(meta.secret_key_length, 32);
        assert_eq!(meta.params, "noop-default");
    }

    #[test]
    fn test_noop_encapsulate() {
        let adapter = NoOpCryptoAdapter::new();
        let result = adapter.encapsulate(&[0u8; 32]);
        assert!(result.is_ok());

        let (ciphertext, shared_secret) = result.unwrap();
        assert_eq!(ciphertext.len(), 48);
        assert_eq!(shared_secret.len(), 32);
    }

    #[test]
    fn test_noop_decapsulate() {
        let adapter = NoOpCryptoAdapter::new();
        let result = adapter.decapsulate(&[0u8; 32], &[0u8; 48]);
        assert!(result.is_ok());

        let shared_secret = result.unwrap();
        assert_eq!(shared_secret.len(), 32);
    }

    #[test]
    fn test_noop_sign() {
        let adapter = NoOpCryptoAdapter::new();
        let result = adapter.sign(&[0u8; 32], b"test message");
        assert!(result.is_ok());

        let signature = result.unwrap();
        assert_eq!(signature.len(), 64);
    }

    #[test]
    fn test_noop_verify() {
        let adapter = NoOpCryptoAdapter::new();
        let result = adapter.verify(&[0u8; 32], b"test message", &[0u8; 64]);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_noop_roundtrip() {
        let adapter = NoOpCryptoAdapter::new();

        // Generate keypair
        let meta = adapter.keygen().unwrap();

        // Encapsulate
        let (ciphertext, shared_secret_enc) = adapter.encapsulate(&meta.public_key).unwrap();

        // Decapsulate
        let shared_secret_dec = adapter.decapsulate(&[0u8; 32], &ciphertext).unwrap();

        // Both should return the same dummy shared secret
        assert_eq!(shared_secret_enc, shared_secret_dec);
    }
}

