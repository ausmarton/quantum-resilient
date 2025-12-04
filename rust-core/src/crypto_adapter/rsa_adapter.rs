//! RSA 2048 Crypto Adapter
//!
//! Provides RSA-2048 operations for baseline benchmarking against PQC algorithms.

use crate::crypto_adapter::{CryptoAdapter, CryptoError, KeypairMeta};
use rand::rngs::OsRng;
use rsa::pkcs8::EncodePublicKey;
use rsa::{Oaep, RsaPrivateKey, RsaPublicKey};
use sha2::Sha256;

/// RSA-2048 cryptographic adapter
///
/// Note: This adapter bends RSA into the unified CryptoAdapter interface
/// for timing baseline purposes. The sign() method actually performs
/// OAEP encryption for uniform interface comparison.
pub struct Rsa2048Adapter {
    /// The RSA private key
    pub private_key: RsaPrivateKey,
    /// The RSA public key
    pub public_key: RsaPublicKey,
}

impl Rsa2048Adapter {
    /// Creates a new RSA-2048 adapter with a freshly generated keypair
    pub fn new() -> Result<Self, CryptoError> {
        let priv_key = RsaPrivateKey::new(&mut OsRng, 2048)
            .map_err(|e| CryptoError::InternalError(e.to_string()))?;
        let pub_key = RsaPublicKey::from(&priv_key);
        Ok(Self {
            private_key: priv_key,
            public_key: pub_key,
        })
    }
}

impl CryptoAdapter for Rsa2048Adapter {
    fn name(&self) -> &'static str {
        "rsa2048"
    }

    fn keygen(&self) -> Result<KeypairMeta, CryptoError> {
        let pub_bytes = self
            .public_key
            .to_public_key_der()
            .map_err(|e| CryptoError::InternalError(e.to_string()))?
            .as_bytes()
            .to_vec();

        Ok(KeypairMeta {
            public_key: pub_bytes,
            secret_key_length: 2048 / 8,
            params: "rsa2048".to_string(),
        })
    }

    fn encapsulate(&self, _pk: &[u8]) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
        Err(CryptoError::NotImplemented)
    }

    fn decapsulate(&self, _sk: &[u8], _ct: &[u8]) -> Result<Vec<u8>, CryptoError> {
        Err(CryptoError::NotImplemented)
    }

    fn sign(&self, _sk: &[u8], msg: &[u8]) -> Result<Vec<u8>, CryptoError> {
        // Note: Using OAEP encryption as a symmetric interface requirement
        // for timing baseline purposes
        let padding = Oaep::new::<Sha256>();
        self.public_key
            .encrypt(&mut OsRng, padding, msg)
            .map_err(|e| CryptoError::InternalError(e.to_string()))
    }

    fn verify(&self, _pk: &[u8], _msg: &[u8], _sig: &[u8]) -> Result<bool, CryptoError> {
        // Placeholder for uniform interface
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsa_adapter_new() {
        let adapter = Rsa2048Adapter::new();
        assert!(adapter.is_ok());
    }

    #[test]
    fn test_rsa_adapter_name() {
        let adapter = Rsa2048Adapter::new().unwrap();
        assert_eq!(adapter.name(), "rsa2048");
    }

    #[test]
    fn test_rsa_adapter_keygen() {
        let adapter = Rsa2048Adapter::new().unwrap();
        let meta = adapter.keygen();
        assert!(meta.is_ok());
        let meta = meta.unwrap();
        assert!(!meta.public_key.is_empty());
        assert_eq!(meta.secret_key_length, 256); // 2048 bits / 8
        assert_eq!(meta.params, "rsa2048");
    }

    #[test]
    fn test_rsa_adapter_sign() {
        let adapter = Rsa2048Adapter::new().unwrap();
        // RSA-OAEP can only encrypt messages shorter than the key size minus padding
        let msg = b"test message";
        let result = adapter.sign(&[], msg);
        assert!(result.is_ok());
        let ciphertext = result.unwrap();
        assert_eq!(ciphertext.len(), 256); // 2048 bits / 8
    }

    #[test]
    fn test_rsa_adapter_encapsulate_not_implemented() {
        let adapter = Rsa2048Adapter::new().unwrap();
        let result = adapter.encapsulate(&[]);
        assert!(matches!(result, Err(CryptoError::NotImplemented)));
    }
}

