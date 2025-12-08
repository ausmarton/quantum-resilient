//! KEM → AEAD Hybrid Encryption
//!
//! Provides helper functions for hybrid encryption using:
//! - KEM (Key Encapsulation Mechanism) for key exchange
//! - HKDF for key derivation
//! - AES-256-GCM for symmetric encryption
//!
//! ## Combined Payload Format
//!
//! The hybrid_encrypt function produces a combined payload with this format:
//! ```text
//! [ct_kem_len: u16 BE] [ct_kem: bytes] [nonce: 12 bytes] [ct_aead: bytes] [tag: 16 bytes]
//! ```
//!
//! - ct_kem_len: 2-byte big-endian length of the KEM ciphertext
//! - ct_kem: KEM ciphertext (encapsulated key)
//! - nonce: 12-byte random nonce for AES-GCM
//! - ct_aead: AES-GCM encrypted ciphertext (same length as plaintext)
//! - tag: 16-byte AES-GCM authentication tag

use crate::crypto_adapter::CryptoError;
use aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use hkdf::Hkdf;
use rand::RngCore;
use sha2::Sha256;
use zeroize::Zeroizing;

/// AES-GCM nonce size (12 bytes)
pub const NONCE_SIZE: usize = 12;

/// AES-GCM tag size (16 bytes)
pub const TAG_SIZE: usize = 16;

/// AES-256 key size (32 bytes)
pub const KEY_SIZE: usize = 32;

/// Derives a 32-byte AES key from a shared secret using HKDF-SHA256
///
/// # Arguments
/// * `shared_secret` - The KEM shared secret
///
/// # Returns
/// A 32-byte key suitable for AES-256-GCM
pub fn derive_aead_key(shared_secret: &[u8]) -> Zeroizing<[u8; KEY_SIZE]> {
    let hk = Hkdf::<Sha256>::new(None, shared_secret);
    let mut key = Zeroizing::new([0u8; KEY_SIZE]);
    // Use "pqc-bench-aead" as the info string for domain separation
    hk.expand(b"pqc-bench-aead", key.as_mut())
        .expect("HKDF expand should not fail with valid parameters");
    key
}

/// Encrypts plaintext using AES-256-GCM
///
/// # Arguments
/// * `aes_key` - 32-byte AES key
/// * `plaintext` - Data to encrypt
/// * `aad` - Additional authenticated data (can be empty)
///
/// # Returns
/// Tuple of (nonce, ciphertext_with_tag) on success
pub fn aead_encrypt(
    aes_key: &[u8; KEY_SIZE],
    plaintext: &[u8],
    aad: &[u8],
) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
    let cipher = Aes256Gcm::new_from_slice(aes_key)
        .map_err(|e| CryptoError::InternalError(format!("Failed to create cipher: {}", e)))?;

    // Generate random nonce
    let mut nonce_bytes = [0u8; NONCE_SIZE];
    rand::thread_rng().fill_bytes(&mut nonce_bytes);
    // Suppress deprecation warning - from_slice still works with current aead version
    #[allow(deprecated)]
    let nonce = Nonce::from_slice(&nonce_bytes);

    // Encrypt with AAD
    let ciphertext = cipher
        .encrypt(nonce, aead::Payload { msg: plaintext, aad })
        .map_err(|e| CryptoError::InternalError(format!("Encryption failed: {}", e)))?;

    Ok((nonce_bytes.to_vec(), ciphertext))
}

/// Decrypts ciphertext using AES-256-GCM
///
/// # Arguments
/// * `aes_key` - 32-byte AES key
/// * `nonce` - 12-byte nonce used during encryption
/// * `ciphertext_with_tag` - Ciphertext including the authentication tag
/// * `aad` - Additional authenticated data (must match encryption)
///
/// # Returns
/// Decrypted plaintext on success
pub fn aead_decrypt(
    aes_key: &[u8; KEY_SIZE],
    nonce: &[u8],
    ciphertext_with_tag: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    if nonce.len() != NONCE_SIZE {
        return Err(CryptoError::InvalidCiphertext);
    }

    let cipher = Aes256Gcm::new_from_slice(aes_key)
        .map_err(|e| CryptoError::InternalError(format!("Failed to create cipher: {}", e)))?;

    // Suppress deprecation warning - from_slice still works with current aead version
    #[allow(deprecated)]
    let nonce = Nonce::from_slice(nonce);

    cipher
        .decrypt(
            nonce,
            aead::Payload {
                msg: ciphertext_with_tag,
                aad,
            },
        )
        .map_err(|e| CryptoError::InternalError(format!("Decryption failed: {}", e)))
}

/// Performs hybrid encryption: KEM encapsulation + AES-GCM encryption
///
/// # Arguments
/// * `kem_encapsulate` - Function to perform KEM encapsulation
/// * `public_key` - Recipient's public key
/// * `plaintext` - Data to encrypt
///
/// # Returns
/// Combined payload containing KEM ciphertext and AEAD ciphertext
///
/// # Payload Format
/// ```text
/// [ct_kem_len: u16 BE] [ct_kem: bytes] [nonce: 12 bytes] [ct_aead_with_tag: bytes]
/// ```
pub fn hybrid_encrypt<F>(
    kem_encapsulate: F,
    public_key: &[u8],
    plaintext: &[u8],
) -> Result<Vec<u8>, CryptoError>
where
    F: FnOnce(&[u8]) -> Result<(Vec<u8>, Vec<u8>), CryptoError>,
{
    // Step 1: KEM encapsulation
    let (ct_kem, shared_secret) = kem_encapsulate(public_key)?;

    // Step 2: Derive AES key from shared secret
    let aes_key = derive_aead_key(&shared_secret);

    // Step 3: AES-GCM encrypt (no AAD for simplicity)
    let (nonce, ct_aead) = aead_encrypt(&aes_key, plaintext, &[])?;

    // Step 4: Package combined payload
    let ct_kem_len = ct_kem.len() as u16;
    let mut combined = Vec::with_capacity(2 + ct_kem.len() + NONCE_SIZE + ct_aead.len());
    combined.extend_from_slice(&ct_kem_len.to_be_bytes());
    combined.extend_from_slice(&ct_kem);
    combined.extend_from_slice(&nonce);
    combined.extend_from_slice(&ct_aead);

    Ok(combined)
}

/// Performs hybrid decryption: KEM decapsulation + AES-GCM decryption
///
/// # Arguments
/// * `kem_decapsulate` - Function to perform KEM decapsulation
/// * `secret_key` - Recipient's secret key
/// * `combined_payload` - Combined payload from hybrid_encrypt
///
/// # Returns
/// Decrypted plaintext on success
pub fn hybrid_decrypt<F>(
    kem_decapsulate: F,
    secret_key: &[u8],
    combined_payload: &[u8],
) -> Result<Vec<u8>, CryptoError>
where
    F: FnOnce(&[u8], &[u8]) -> Result<Vec<u8>, CryptoError>,
{
    // Parse combined payload
    if combined_payload.len() < 2 + NONCE_SIZE {
        return Err(CryptoError::InvalidCiphertext);
    }

    // Step 1: Extract ct_kem_len
    let ct_kem_len = u16::from_be_bytes([combined_payload[0], combined_payload[1]]) as usize;

    if combined_payload.len() < 2 + ct_kem_len + NONCE_SIZE {
        return Err(CryptoError::InvalidCiphertext);
    }

    // Step 2: Extract ct_kem
    let ct_kem = &combined_payload[2..2 + ct_kem_len];

    // Step 3: Extract nonce
    let nonce_start = 2 + ct_kem_len;
    let nonce = &combined_payload[nonce_start..nonce_start + NONCE_SIZE];

    // Step 4: Extract ct_aead
    let ct_aead = &combined_payload[nonce_start + NONCE_SIZE..];

    // Step 5: KEM decapsulation
    let shared_secret = kem_decapsulate(secret_key, ct_kem)?;

    // Step 6: Derive AES key
    let aes_key = derive_aead_key(&shared_secret);

    // Step 7: AES-GCM decrypt
    aead_decrypt(&aes_key, nonce, ct_aead, &[])
}

/// Helper struct to track sizes for metrics/logging
#[derive(Debug, Clone)]
pub struct HybridSizes {
    pub ct_kem_len: usize,
    pub nonce_len: usize,
    pub ct_aead_len: usize,
    pub total_len: usize,
}

impl HybridSizes {
    /// Parses sizes from a combined payload without decrypting
    pub fn from_payload(combined_payload: &[u8]) -> Result<Self, CryptoError> {
        if combined_payload.len() < 2 {
            return Err(CryptoError::InvalidCiphertext);
        }

        let ct_kem_len = u16::from_be_bytes([combined_payload[0], combined_payload[1]]) as usize;

        if combined_payload.len() < 2 + ct_kem_len + NONCE_SIZE {
            return Err(CryptoError::InvalidCiphertext);
        }

        let ct_aead_len = combined_payload.len() - 2 - ct_kem_len - NONCE_SIZE;

        Ok(Self {
            ct_kem_len,
            nonce_len: NONCE_SIZE,
            ct_aead_len,
            total_len: combined_payload.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_aead_key_length() {
        let shared_secret = vec![0x42; 32];
        let key = derive_aead_key(&shared_secret);
        assert_eq!(key.len(), KEY_SIZE);
    }

    #[test]
    fn test_derive_aead_key_deterministic() {
        let shared_secret = vec![0x42; 32];
        let key1 = derive_aead_key(&shared_secret);
        let key2 = derive_aead_key(&shared_secret);
        assert_eq!(key1.as_slice(), key2.as_slice());
    }

    #[test]
    fn test_derive_aead_key_different_inputs() {
        let ss1 = vec![0x42; 32];
        let ss2 = vec![0x43; 32];
        let key1 = derive_aead_key(&ss1);
        let key2 = derive_aead_key(&ss2);
        assert_ne!(key1.as_slice(), key2.as_slice());
    }

    #[test]
    fn test_aead_encrypt_decrypt() {
        let key = [0x42u8; KEY_SIZE];
        let plaintext = b"Hello, quantum-resilient world!";
        let aad = b"additional data";

        let (nonce, ciphertext) = aead_encrypt(&key, plaintext, aad).unwrap();
        assert_eq!(nonce.len(), NONCE_SIZE);
        assert!(!ciphertext.is_empty());

        let decrypted = aead_decrypt(&key, &nonce, &ciphertext, aad).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_aead_wrong_key_fails() {
        let key1 = [0x42u8; KEY_SIZE];
        let key2 = [0x43u8; KEY_SIZE];
        let plaintext = b"secret message";

        let (nonce, ciphertext) = aead_encrypt(&key1, plaintext, &[]).unwrap();
        let result = aead_decrypt(&key2, &nonce, &ciphertext, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_aead_wrong_aad_fails() {
        let key = [0x42u8; KEY_SIZE];
        let plaintext = b"secret message";

        let (nonce, ciphertext) = aead_encrypt(&key, plaintext, b"aad1").unwrap();
        let result = aead_decrypt(&key, &nonce, &ciphertext, b"aad2");
        assert!(result.is_err());
    }

    #[test]
    fn test_hybrid_encrypt_decrypt() {
        // Mock KEM functions
        let mock_shared_secret = vec![0xAB; 32];
        let mock_ct_kem = vec![0xCD; 768]; // Typical Kyber512 ciphertext size

        let encapsulate = |_pk: &[u8]| -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
            Ok((mock_ct_kem.clone(), mock_shared_secret.clone()))
        };

        let decapsulate = |_sk: &[u8], _ct: &[u8]| -> Result<Vec<u8>, CryptoError> {
            Ok(mock_shared_secret.clone())
        };

        let public_key = vec![0x01; 800];
        let secret_key = vec![0x02; 1632];
        let plaintext = b"Test message for hybrid encryption";

        // Encrypt
        let combined = hybrid_encrypt(encapsulate, &public_key, plaintext).unwrap();
        assert!(!combined.is_empty());

        // Verify sizes
        let sizes = HybridSizes::from_payload(&combined).unwrap();
        assert_eq!(sizes.ct_kem_len, mock_ct_kem.len());
        assert_eq!(sizes.nonce_len, NONCE_SIZE);

        // Decrypt
        let decrypted = hybrid_decrypt(decapsulate, &secret_key, &combined).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_hybrid_sizes_from_payload() {
        let combined = vec![
            0x00, 0x10, // ct_kem_len = 16
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, // ct_kem (16 bytes)
            0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, //
            0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, // nonce (12 bytes)
            0x19, 0x1A, 0x1B, 0x1C, //
            0xA1, 0xA2, 0xA3, 0xA4, // ct_aead (some bytes)
        ];

        let sizes = HybridSizes::from_payload(&combined).unwrap();
        assert_eq!(sizes.ct_kem_len, 16);
        assert_eq!(sizes.nonce_len, 12);
        assert_eq!(sizes.ct_aead_len, 4);
    }
}

