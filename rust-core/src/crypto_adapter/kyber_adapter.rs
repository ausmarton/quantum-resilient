//! Kyber KEM Adapter
//!
//! Provides Kyber key encapsulation mechanism for post-quantum cryptography.
//! Uses pqcrypto-kyber as the implementation (pure Rust fallback).

use crate::crypto_adapter::{CryptoAdapter, CryptoError, KeypairMeta};
use parking_lot::RwLock;
use zeroize::Zeroizing;

#[cfg(feature = "pqcrypto_fallback")]
use pqcrypto_kyber::kyber512;
#[cfg(feature = "pqcrypto_fallback")]
use pqcrypto_traits::kem::{Ciphertext, PublicKey, SecretKey, SharedSecret};

/// Kyber KEM adapter supporting multiple parameter sets
///
/// Currently supports:
/// - kyber512 (default, NIST security level 1)
///
/// Future support planned for:
/// - kyber768 (NIST security level 3)
/// - kyber1024 (NIST security level 5)
pub struct KyberAdapter {
    /// Parameter set name (e.g., "kyber512")
    pub paramset: String,
    /// Cached keypair for performance (optional)
    keypair: RwLock<Option<KyberKeypair>>,
}

/// Internal keypair storage with zeroization
struct KyberKeypair {
    public_key: Vec<u8>,
    secret_key: Zeroizing<Vec<u8>>,
}

impl KyberAdapter {
    /// Creates a new Kyber adapter with the specified parameter set
    ///
    /// # Arguments
    /// * `paramset` - Parameter set name: "kyber512" (default), "kyber768", "kyber1024"
    ///
    /// # Errors
    /// Returns `CryptoError::InvalidKey` if the parameter set is not supported
    pub fn new(paramset: &str) -> Result<Self, CryptoError> {
        match paramset {
            "kyber512" => Ok(Self {
                paramset: paramset.to_string(),
                keypair: RwLock::new(None),
            }),
            // Future: "kyber768" | "kyber1024"
            _ => Err(CryptoError::InternalError(format!(
                "Unsupported Kyber parameter set: {}. Supported: kyber512",
                paramset
            ))),
        }
    }

    /// Generates a keypair and caches it for subsequent operations
    pub fn generate_and_cache_keypair(&self) -> Result<KeypairMeta, CryptoError> {
        let (pk, sk) = self.keygen_internal()?;
        let meta = KeypairMeta {
            public_key: pk.clone(),
            secret_key_length: sk.len(),
            params: self.paramset.clone(),
        };

        let mut keypair_guard = self.keypair.write();
        *keypair_guard = Some(KyberKeypair {
            public_key: pk,
            secret_key: Zeroizing::new(sk),
        });

        Ok(meta)
    }

    /// Returns the cached public key, if available
    pub fn get_public_key(&self) -> Option<Vec<u8>> {
        self.keypair.read().as_ref().map(|kp| kp.public_key.clone())
    }

    /// Returns the cached secret key for decapsulation
    /// Warning: Handle with care - secret key material
    pub fn get_secret_key(&self) -> Option<Vec<u8>> {
        self.keypair
            .read()
            .as_ref()
            .map(|kp| kp.secret_key.to_vec())
    }

    /// Internal keypair generation using pqcrypto fallback
    #[cfg(feature = "pqcrypto_fallback")]
    fn keygen_internal(&self) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
        match self.paramset.as_str() {
            "kyber512" => {
                let (pk, sk) = kyber512::keypair();
                Ok((pk.as_bytes().to_vec(), sk.as_bytes().to_vec()))
            }
            _ => Err(CryptoError::InternalError(format!(
                "Unsupported paramset: {}",
                self.paramset
            ))),
        }
    }

    /// Fallback when no PQC feature is enabled
    #[cfg(not(feature = "pqcrypto_fallback"))]
    fn keygen_internal(&self) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
        Err(CryptoError::InternalError(
            "No PQC implementation available. Enable pqcrypto_fallback feature.".to_string(),
        ))
    }

    /// Internal encapsulation using pqcrypto fallback
    #[cfg(feature = "pqcrypto_fallback")]
    fn encapsulate_internal(&self, public_key: &[u8]) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
        match self.paramset.as_str() {
            "kyber512" => {
                let pk = kyber512::PublicKey::from_bytes(public_key)
                    .map_err(|_e| CryptoError::InvalidKey)?;
                let (ss, ct) = kyber512::encapsulate(&pk);
                Ok((ct.as_bytes().to_vec(), ss.as_bytes().to_vec()))
            }
            _ => Err(CryptoError::InternalError(format!(
                "Unsupported paramset: {}",
                self.paramset
            ))),
        }
    }

    #[cfg(not(feature = "pqcrypto_fallback"))]
    fn encapsulate_internal(&self, _public_key: &[u8]) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
        Err(CryptoError::InternalError(
            "No PQC implementation available. Enable pqcrypto_fallback feature.".to_string(),
        ))
    }

    /// Internal decapsulation using pqcrypto fallback
    #[cfg(feature = "pqcrypto_fallback")]
    fn decapsulate_internal(
        &self,
        secret_key: &[u8],
        ciphertext: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        match self.paramset.as_str() {
            "kyber512" => {
                let sk = kyber512::SecretKey::from_bytes(secret_key)
                    .map_err(|_e| CryptoError::InvalidKey)?;
                let ct = kyber512::Ciphertext::from_bytes(ciphertext)
                    .map_err(|_e| CryptoError::InvalidCiphertext)?;
                let ss = kyber512::decapsulate(&ct, &sk);
                Ok(ss.as_bytes().to_vec())
            }
            _ => Err(CryptoError::InternalError(format!(
                "Unsupported paramset: {}",
                self.paramset
            ))),
        }
    }

    #[cfg(not(feature = "pqcrypto_fallback"))]
    fn decapsulate_internal(
        &self,
        _secret_key: &[u8],
        _ciphertext: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        Err(CryptoError::InternalError(
            "No PQC implementation available. Enable pqcrypto_fallback feature.".to_string(),
        ))
    }
}

impl CryptoAdapter for KyberAdapter {
    fn name(&self) -> &'static str {
        "kyber"
    }

    fn keygen(&self) -> Result<KeypairMeta, CryptoError> {
        let (pk, sk) = self.keygen_internal()?;
        Ok(KeypairMeta {
            public_key: pk,
            secret_key_length: sk.len(),
            params: self.paramset.clone(),
        })
    }

    fn encapsulate(&self, public_key: &[u8]) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
        self.encapsulate_internal(public_key)
    }

    fn decapsulate(&self, secret_key: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>, CryptoError> {
        self.decapsulate_internal(secret_key, ciphertext)
    }

    fn sign(&self, _secret_key: &[u8], _msg: &[u8]) -> Result<Vec<u8>, CryptoError> {
        // Kyber is a KEM, not a signature scheme
        Err(CryptoError::NotImplemented)
    }

    fn verify(&self, _public_key: &[u8], _msg: &[u8], _sig: &[u8]) -> Result<bool, CryptoError> {
        // Kyber is a KEM, not a signature scheme
        Err(CryptoError::NotImplemented)
    }
}

// Implement Send + Sync for KyberAdapter
unsafe impl Send for KyberAdapter {}
unsafe impl Sync for KyberAdapter {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kyber_adapter_new() {
        let adapter = KyberAdapter::new("kyber512");
        assert!(adapter.is_ok());
        assert_eq!(adapter.unwrap().paramset, "kyber512");
    }

    #[test]
    fn test_kyber_adapter_invalid_paramset() {
        let adapter = KyberAdapter::new("kyber999");
        assert!(adapter.is_err());
    }

    #[test]
    fn test_kyber_adapter_name() {
        let adapter = KyberAdapter::new("kyber512").unwrap();
        assert_eq!(adapter.name(), "kyber");
    }

    #[test]
    #[cfg(feature = "pqcrypto_fallback")]
    fn test_kyber_keygen() {
        let adapter = KyberAdapter::new("kyber512").unwrap();
        let meta = adapter.keygen();
        assert!(meta.is_ok());
        let meta = meta.unwrap();
        assert!(!meta.public_key.is_empty());
        assert!(meta.secret_key_length > 0);
        assert_eq!(meta.params, "kyber512");
    }

    #[test]
    #[cfg(feature = "pqcrypto_fallback")]
    fn test_kyber_encapsulate_decapsulate() {
        let adapter = KyberAdapter::new("kyber512").unwrap();

        // Generate keypair
        let meta = adapter.generate_and_cache_keypair().unwrap();
        let pk = meta.public_key;
        let sk = adapter.get_secret_key().unwrap();

        // Encapsulate
        let (ciphertext, shared_secret1) = adapter.encapsulate(&pk).unwrap();
        assert!(!ciphertext.is_empty());
        assert!(!shared_secret1.is_empty());

        // Decapsulate
        let shared_secret2 = adapter.decapsulate(&sk, &ciphertext).unwrap();

        // Shared secrets must match
        assert_eq!(shared_secret1, shared_secret2);
    }

    #[test]
    fn test_kyber_sign_not_implemented() {
        let adapter = KyberAdapter::new("kyber512").unwrap();
        let result = adapter.sign(&[], b"test");
        assert!(matches!(result, Err(CryptoError::NotImplemented)));
    }

    #[test]
    fn test_kyber_verify_not_implemented() {
        let adapter = KyberAdapter::new("kyber512").unwrap();
        let result = adapter.verify(&[], b"test", &[]);
        assert!(matches!(result, Err(CryptoError::NotImplemented)));
    }
}

