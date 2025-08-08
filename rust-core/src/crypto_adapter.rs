//! Cryptographic Adapter Module
//!
//! This module defines the `CryptoAdapter` trait which provides a unified interface
//! for both classical and post-quantum cryptographic operations.

/// Error type for cryptographic operations
#[derive(Debug)]
pub enum CryptoError {
    /// Key generation failed
    KeyGenerationError(String),
    /// Encryption failed
    EncryptionError(String),
    /// Decryption failed
    DecryptionError(String),
    /// Signature generation failed
    SignatureError(String),
    /// Signature verification failed
    VerificationError(String),
}

/// Trait defining the interface for cryptographic adapters.
///
/// Implementors of this trait provide specific cryptographic algorithm implementations,
/// whether classical (RSA, ECDSA) or post-quantum (Kyber, Dilithium, etc.).
pub trait CryptoAdapter {
    /// Returns the name of the cryptographic algorithm
    fn algorithm_name(&self) -> &str;

    /// Generates a new keypair
    fn generate_keypair(&mut self) -> Result<(), CryptoError>;

    /// Encrypts the given plaintext
    fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>, CryptoError>;

    /// Decrypts the given ciphertext
    fn decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>, CryptoError>;

    /// Signs the given message
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, CryptoError>;

    /// Verifies a signature against a message
    fn verify(&self, message: &[u8], signature: &[u8]) -> Result<bool, CryptoError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crypto_error_debug() {
        let error = CryptoError::KeyGenerationError("test".to_string());
        assert!(format!("{:?}", error).contains("KeyGenerationError"));
    }
}

