//! Cryptographic Adapter Module
//!
//! This module defines the `CryptoAdapter` trait which provides a unified interface
//! for both classical and post-quantum cryptographic operations.

pub mod ecdsa_adapter;
pub mod noop_adapter;
pub mod registry;
pub mod rsa_adapter;

pub use ecdsa_adapter::EcdsaP256Adapter;
pub use noop_adapter::NoOpCryptoAdapter;
pub use registry::{get_adapter, supported_adapters};
pub use rsa_adapter::Rsa2048Adapter;

/// Metadata about a generated keypair
#[derive(Debug, Clone)]
pub struct KeypairMeta {
    /// The public key bytes
    pub public_key: Vec<u8>,
    /// Length of the secret key in bytes
    pub secret_key_length: usize,
    /// Parameter set name or description
    pub params: String,
}

/// Error type for cryptographic operations
#[derive(Debug)]
pub enum CryptoError {
    /// Operation not implemented for this adapter
    NotImplemented,
    /// Invalid key format or length
    InvalidKey,
    /// Invalid ciphertext format
    InvalidCiphertext,
    /// Invalid signature format
    InvalidSignature,
    /// Internal error with description
    InternalError(String),
}

impl std::fmt::Display for CryptoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CryptoError::NotImplemented => write!(f, "Operation not implemented"),
            CryptoError::InvalidKey => write!(f, "Invalid key"),
            CryptoError::InvalidCiphertext => write!(f, "Invalid ciphertext"),
            CryptoError::InvalidSignature => write!(f, "Invalid signature"),
            CryptoError::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for CryptoError {}

/// Trait defining the interface for cryptographic adapters.
///
/// Implementors of this trait provide specific cryptographic algorithm implementations,
/// whether classical (RSA, ECDSA) or post-quantum (Kyber, Dilithium, etc.).
///
/// The trait supports both Key Encapsulation Mechanisms (KEM) for key exchange
/// and digital signatures.
///
/// Implementations must be thread-safe (Send + Sync) to support async pipelines.
pub trait CryptoAdapter: Send + Sync {
    /// Returns the name of the cryptographic algorithm
    fn name(&self) -> &'static str;

    /// Generates a new keypair
    ///
    /// Returns metadata about the generated keypair including the public key
    /// and secret key length. The secret key itself should be stored securely
    /// by the implementation.
    fn keygen(&self) -> Result<KeypairMeta, CryptoError>;

    /// Encapsulates a shared secret using the given public key (KEM operation)
    ///
    /// # Arguments
    /// * `public_key` - The recipient's public key
    ///
    /// # Returns
    /// A tuple of (ciphertext, shared_secret) on success
    fn encapsulate(&self, public_key: &[u8]) -> Result<(Vec<u8>, Vec<u8>), CryptoError>;

    /// Decapsulates a shared secret using the secret key (KEM operation)
    ///
    /// # Arguments
    /// * `secret_key` - The recipient's secret key
    /// * `ciphertext` - The ciphertext from encapsulation
    ///
    /// # Returns
    /// The shared secret on success
    fn decapsulate(&self, secret_key: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>, CryptoError>;

    /// Signs a message using the secret key
    ///
    /// # Arguments
    /// * `secret_key` - The signer's secret key
    /// * `msg` - The message to sign
    ///
    /// # Returns
    /// The signature bytes on success
    fn sign(&self, secret_key: &[u8], msg: &[u8]) -> Result<Vec<u8>, CryptoError>;

    /// Verifies a signature against a message
    ///
    /// # Arguments
    /// * `public_key` - The signer's public key
    /// * `msg` - The original message
    /// * `sig` - The signature to verify
    ///
    /// # Returns
    /// `true` if the signature is valid, `false` otherwise
    fn verify(&self, public_key: &[u8], msg: &[u8], sig: &[u8]) -> Result<bool, CryptoError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crypto_error_display() {
        let error = CryptoError::NotImplemented;
        assert_eq!(format!("{}", error), "Operation not implemented");

        let error = CryptoError::InternalError("test error".to_string());
        assert!(format!("{}", error).contains("test error"));
    }

    #[test]
    fn test_keypair_meta_clone() {
        let meta = KeypairMeta {
            public_key: vec![1, 2, 3],
            secret_key_length: 32,
            params: "test".to_string(),
        };
        let cloned = meta.clone();
        assert_eq!(cloned.public_key, meta.public_key);
        assert_eq!(cloned.secret_key_length, meta.secret_key_length);
    }
}
