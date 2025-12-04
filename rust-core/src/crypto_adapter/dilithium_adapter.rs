//! Dilithium Signature Adapter
//!
//! Provides Dilithium digital signature scheme for post-quantum cryptography.
//! Uses pqcrypto-dilithium as the implementation (pure Rust).

use crate::crypto_adapter::{CryptoAdapter, CryptoError, KeypairMeta, KeypairWithSecret};
use parking_lot::RwLock;
use zeroize::Zeroizing;

#[cfg(feature = "pqcrypto_dilithium")]
use pqcrypto_dilithium::dilithium2;
#[cfg(feature = "pqcrypto_dilithium")]
use pqcrypto_traits::sign::{DetachedSignature, PublicKey, SecretKey};

/// Dilithium digital signature adapter supporting multiple parameter sets
///
/// Currently supports:
/// - dilithium2 (default, NIST security level 2)
///
/// Future support planned for:
/// - dilithium3 (NIST security level 3)
/// - dilithium5 (NIST security level 5)
pub struct DilithiumAdapter {
    /// Parameter set name (e.g., "dilithium2")
    pub paramset: String,
    /// Cached keypair for performance (optional)
    keypair: RwLock<Option<DilithiumKeypair>>,
}

/// Internal keypair storage with zeroization
struct DilithiumKeypair {
    public_key: Vec<u8>,
    secret_key: Zeroizing<Vec<u8>>,
}

impl DilithiumAdapter {
    /// Creates a new Dilithium adapter with the specified parameter set
    ///
    /// # Arguments
    /// * `paramset` - Parameter set name: "dilithium2" (default), "dilithium3", "dilithium5"
    ///
    /// # Errors
    /// Returns `CryptoError::InternalError` if the parameter set is not supported
    pub fn new(paramset: &str) -> Result<Self, CryptoError> {
        match paramset {
            "dilithium2" => Ok(Self {
                paramset: paramset.to_string(),
                keypair: RwLock::new(None),
            }),
            // Future: "dilithium3" | "dilithium5"
            _ => Err(CryptoError::InternalError(format!(
                "Unsupported Dilithium parameter set: {}. Supported: dilithium2",
                paramset
            ))),
        }
    }

    /// Generates a keypair and caches it for subsequent operations
    pub fn generate_and_cache_keypair(&self) -> Result<KeypairWithSecret, CryptoError> {
        let (pk, sk) = self.keygen_internal()?;
        
        let keypair_with_secret = KeypairWithSecret {
            public_key: pk.clone(),
            secret_key: sk.clone(),
            params: self.paramset.clone(),
        };

        let mut keypair_guard = self.keypair.write();
        *keypair_guard = Some(DilithiumKeypair {
            public_key: pk,
            secret_key: Zeroizing::new(sk),
        });

        Ok(keypair_with_secret)
    }

    /// Returns the cached public key, if available
    pub fn get_public_key(&self) -> Option<Vec<u8>> {
        self.keypair.read().as_ref().map(|kp| kp.public_key.clone())
    }

    /// Returns the cached secret key for signing
    /// Warning: Handle with care - secret key material
    pub fn get_secret_key(&self) -> Option<Vec<u8>> {
        self.keypair
            .read()
            .as_ref()
            .map(|kp| kp.secret_key.to_vec())
    }

    /// Internal keypair generation using pqcrypto
    #[cfg(feature = "pqcrypto_dilithium")]
    fn keygen_internal(&self) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
        match self.paramset.as_str() {
            "dilithium2" => {
                let (pk, sk) = dilithium2::keypair();
                Ok((pk.as_bytes().to_vec(), sk.as_bytes().to_vec()))
            }
            _ => Err(CryptoError::InternalError(format!(
                "Unsupported paramset: {}",
                self.paramset
            ))),
        }
    }

    /// Fallback when no PQC feature is enabled
    #[cfg(not(feature = "pqcrypto_dilithium"))]
    fn keygen_internal(&self) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
        Err(CryptoError::InternalError(
            "No Dilithium implementation available. Enable pqcrypto_dilithium feature.".to_string(),
        ))
    }

    /// Internal sign operation using pqcrypto
    #[cfg(feature = "pqcrypto_dilithium")]
    fn sign_internal(&self, secret_key: &[u8], msg: &[u8]) -> Result<Vec<u8>, CryptoError> {
        match self.paramset.as_str() {
            "dilithium2" => {
                let sk = dilithium2::SecretKey::from_bytes(secret_key)
                    .map_err(|_e| CryptoError::InvalidKey)?;
                let sig = dilithium2::detached_sign(msg, &sk);
                Ok(sig.as_bytes().to_vec())
            }
            _ => Err(CryptoError::InternalError(format!(
                "Unsupported paramset: {}",
                self.paramset
            ))),
        }
    }

    #[cfg(not(feature = "pqcrypto_dilithium"))]
    fn sign_internal(&self, _secret_key: &[u8], _msg: &[u8]) -> Result<Vec<u8>, CryptoError> {
        Err(CryptoError::InternalError(
            "No Dilithium implementation available. Enable pqcrypto_dilithium feature.".to_string(),
        ))
    }

    /// Internal verify operation using pqcrypto
    #[cfg(feature = "pqcrypto_dilithium")]
    fn verify_internal(&self, public_key: &[u8], msg: &[u8], sig: &[u8]) -> Result<bool, CryptoError> {
        match self.paramset.as_str() {
            "dilithium2" => {
                let pk = dilithium2::PublicKey::from_bytes(public_key)
                    .map_err(|_e| CryptoError::InvalidKey)?;
                let signature = dilithium2::DetachedSignature::from_bytes(sig)
                    .map_err(|_e| CryptoError::InvalidSignature)?;
                
                match dilithium2::verify_detached_signature(&signature, msg, &pk) {
                    Ok(()) => Ok(true),
                    Err(_) => Ok(false),
                }
            }
            _ => Err(CryptoError::InternalError(format!(
                "Unsupported paramset: {}",
                self.paramset
            ))),
        }
    }

    #[cfg(not(feature = "pqcrypto_dilithium"))]
    fn verify_internal(&self, _public_key: &[u8], _msg: &[u8], _sig: &[u8]) -> Result<bool, CryptoError> {
        Err(CryptoError::InternalError(
            "No Dilithium implementation available. Enable pqcrypto_dilithium feature.".to_string(),
        ))
    }
}

impl CryptoAdapter for DilithiumAdapter {
    fn name(&self) -> &'static str {
        "dilithium"
    }

    fn keygen(&self) -> Result<KeypairMeta, CryptoError> {
        let (pk, sk) = self.keygen_internal()?;
        Ok(KeypairMeta {
            public_key: pk,
            secret_key_length: sk.len(),
            params: self.paramset.clone(),
        })
    }

    fn encapsulate(&self, _public_key: &[u8]) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
        // Dilithium is a signature scheme, not a KEM
        Err(CryptoError::NotImplemented)
    }

    fn decapsulate(&self, _secret_key: &[u8], _ciphertext: &[u8]) -> Result<Vec<u8>, CryptoError> {
        // Dilithium is a signature scheme, not a KEM
        Err(CryptoError::NotImplemented)
    }

    fn sign(&self, secret_key: &[u8], msg: &[u8]) -> Result<Vec<u8>, CryptoError> {
        self.sign_internal(secret_key, msg)
    }

    fn verify(&self, public_key: &[u8], msg: &[u8], sig: &[u8]) -> Result<bool, CryptoError> {
        self.verify_internal(public_key, msg, sig)
    }
}

// Implement Send + Sync for DilithiumAdapter
unsafe impl Send for DilithiumAdapter {}
unsafe impl Sync for DilithiumAdapter {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dilithium_adapter_new() {
        let adapter = DilithiumAdapter::new("dilithium2");
        assert!(adapter.is_ok());
        assert_eq!(adapter.unwrap().paramset, "dilithium2");
    }

    #[test]
    fn test_dilithium_adapter_invalid_paramset() {
        let adapter = DilithiumAdapter::new("dilithium999");
        assert!(adapter.is_err());
    }

    #[test]
    fn test_dilithium_adapter_name() {
        let adapter = DilithiumAdapter::new("dilithium2").unwrap();
        assert_eq!(adapter.name(), "dilithium");
    }

    #[test]
    #[cfg(feature = "pqcrypto_dilithium")]
    fn test_dilithium_keygen() {
        let adapter = DilithiumAdapter::new("dilithium2").unwrap();
        let meta = adapter.keygen();
        assert!(meta.is_ok());
        let meta = meta.unwrap();
        assert!(!meta.public_key.is_empty());
        assert!(meta.secret_key_length > 0);
        assert_eq!(meta.params, "dilithium2");
    }

    #[test]
    #[cfg(feature = "pqcrypto_dilithium")]
    fn test_dilithium_sign_verify() {
        let adapter = DilithiumAdapter::new("dilithium2").unwrap();

        // Generate keypair
        let keypair = adapter.generate_and_cache_keypair().unwrap();
        let pk = keypair.public_key;
        let sk = keypair.secret_key;

        // Sign message
        let message = b"Hello, Dilithium!";
        let signature = adapter.sign(&sk, message).unwrap();
        assert!(!signature.is_empty());

        // Verify signature
        let valid = adapter.verify(&pk, message, &signature).unwrap();
        assert!(valid);

        // Verify with wrong message
        let wrong_message = b"Wrong message!";
        let invalid = adapter.verify(&pk, wrong_message, &signature).unwrap();
        assert!(!invalid);
    }

    #[test]
    fn test_dilithium_encapsulate_not_implemented() {
        let adapter = DilithiumAdapter::new("dilithium2").unwrap();
        let result = adapter.encapsulate(&[]);
        assert!(matches!(result, Err(CryptoError::NotImplemented)));
    }

    #[test]
    fn test_dilithium_decapsulate_not_implemented() {
        let adapter = DilithiumAdapter::new("dilithium2").unwrap();
        let result = adapter.decapsulate(&[], &[]);
        assert!(matches!(result, Err(CryptoError::NotImplemented)));
    }
}

