//! ECDHE/ECDH P-256 KEM Adapter
//!
//! Provides ECDH key exchange using NIST P-256 curve for baseline KEM benchmarking.
//! This implements a Key Encapsulation Mechanism (KEM) interface using ECDH.

use crate::crypto_adapter::{CryptoAdapter, CryptoError, KeypairMeta};
use p256::{
    ecdh::EphemeralSecret,
    EncodedPoint, PublicKey, SecretKey,
};
use rand::rngs::OsRng;
use zeroize::Zeroizing;

/// ECDHE/ECDH P-256 KEM adapter
///
/// Provides key exchange operations using the NIST P-256 curve.
/// Implements the KEM interface where:
/// - `encapsulate` generates an ephemeral keypair and computes shared secret
/// - `decapsulate` uses the receiver's secret key and ephemeral public key to compute shared secret
pub struct EcdheP256Adapter {
    /// The ECDH secret key bytes (private key) for decapsulation
    pub secret_key_bytes: Zeroizing<Vec<u8>>,
    /// The ECDH secret key for ECDH operations
    pub secret_key: SecretKey,
    /// The ECDH public key encoded point for encapsulation
    pub public_key: EncodedPoint,
}

impl EcdheP256Adapter {
    /// Creates a new ECDHE P-256 adapter with a freshly generated keypair
    pub fn new() -> Result<Self, CryptoError> {
        let sk = SecretKey::random(&mut OsRng);
        let sk_bytes = Zeroizing::new(sk.to_bytes().to_vec());
        let pk = EncodedPoint::from(sk.public_key());
        Ok(Self {
            secret_key_bytes: sk_bytes,
            secret_key: sk,
            public_key: pk,
        })
    }

    /// Returns the secret key bytes (for pipeline keypair generation)
    /// Warning: Handle with care - secret key material
    pub fn get_secret_key_bytes(&self) -> Vec<u8> {
        self.secret_key_bytes.to_vec()
    }
}

impl CryptoAdapter for EcdheP256Adapter {
    fn name(&self) -> &'static str {
        "ecdhe_p256"
    }

    fn keygen(&self) -> Result<KeypairMeta, CryptoError> {
        Ok(KeypairMeta {
            public_key: self.public_key.as_bytes().to_vec(),
            secret_key_length: 32, // P-256 secret key is 32 bytes
            params: "ecdh_p256".to_string(),
        })
    }

    fn encapsulate(&self, public_key: &[u8]) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
        // Parse the recipient's public key
        let recipient_pk = PublicKey::from_sec1_bytes(public_key)
            .map_err(|_| CryptoError::InvalidKey)?;

        // Generate ephemeral keypair for this encapsulation
        let ephemeral_secret = EphemeralSecret::random(&mut OsRng);
        let ephemeral_pk = EncodedPoint::from(ephemeral_secret.public_key());

        // Compute shared secret: ECDH(ephemeral_secret, recipient_pk)
        let shared_secret = ephemeral_secret.diffie_hellman(&recipient_pk);

        // Return ciphertext (ephemeral public key) and shared secret
        let ciphertext = ephemeral_pk.as_bytes().to_vec();
        let shared_secret_bytes = shared_secret.raw_secret_bytes().to_vec();

        Ok((ciphertext, shared_secret_bytes))
    }

    fn decapsulate(&self, _secret_key: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>, CryptoError> {
        // Parse the ephemeral public key from ciphertext
        let ephemeral_pk = PublicKey::from_sec1_bytes(ciphertext)
            .map_err(|_| CryptoError::InvalidCiphertext)?;

        // Compute shared secret: ECDH(self.secret_key, ephemeral_pk)
        // Note: We use the adapter's internal secret key, ignoring the parameter
        // For ECDHE, the receiver uses their static secret key with the ephemeral public key
        //
        // Since p256's EphemeralSecret API doesn't support creating from SecretKey,
        // we'll compute the shared secret using point multiplication and encode it
        use p256::elliptic_curve::ops::Mul;
        
        // Get the scalar from our secret key
        let scalar = self.secret_key.to_nonzero_scalar();
        
        // Convert ephemeral public key to projective point and multiply by our scalar
        let ephemeral_point = ephemeral_pk.to_projective();
        let shared_point = ephemeral_point * scalar.as_ref();
        
        // Convert to affine point first, then to encoded point to extract coordinates
        let shared_affine = shared_point.to_affine();
        let shared_encoded = EncodedPoint::from(shared_affine);
        
        // The shared secret in ECDH is typically the x-coordinate
        // EncodedPoint format: [0x04/0x02/0x03] [x (32 bytes)] [y (32 bytes, if uncompressed)]
        // For compressed: [0x02/0x03] [x (32 bytes)]
        // For uncompressed: [0x04] [x (32 bytes)] [y (32 bytes)]
        let shared_bytes = shared_encoded.as_bytes();
        
        // Extract x-coordinate (skip the first byte which is the format indicator)
        // Take 32 bytes for the x-coordinate
        if shared_bytes.len() >= 33 {
            Ok(shared_bytes[1..33].to_vec())
        } else if shared_bytes.len() >= 65 {
            // Uncompressed format
            Ok(shared_bytes[1..33].to_vec())
        } else {
            Err(CryptoError::InternalError("Invalid shared point encoding".to_string()))
        }
    }

    fn sign(&self, _secret_key: &[u8], _msg: &[u8]) -> Result<Vec<u8>, CryptoError> {
        // ECDHE is a KEM, not a signature scheme
        Err(CryptoError::NotImplemented)
    }

    fn verify(&self, _public_key: &[u8], _msg: &[u8], _sig: &[u8]) -> Result<bool, CryptoError> {
        // ECDHE is a KEM, not a signature scheme
        Err(CryptoError::NotImplemented)
    }
}

// Implement Send + Sync for EcdheP256Adapter
unsafe impl Send for EcdheP256Adapter {}
unsafe impl Sync for EcdheP256Adapter {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ecdhe_adapter_new() {
        let adapter = EcdheP256Adapter::new();
        assert!(adapter.is_ok());
    }

    #[test]
    fn test_ecdhe_adapter_name() {
        let adapter = EcdheP256Adapter::new().unwrap();
        assert_eq!(adapter.name(), "ecdhe_p256");
    }

    #[test]
    fn test_ecdhe_adapter_keygen() {
        let adapter = EcdheP256Adapter::new().unwrap();
        let meta = adapter.keygen();
        assert!(meta.is_ok());
        let meta = meta.unwrap();
        assert!(!meta.public_key.is_empty());
        assert_eq!(meta.secret_key_length, 32);
        assert_eq!(meta.params, "ecdh_p256");
    }

    #[test]
    fn test_ecdhe_encapsulate_decapsulate() {
        // Create two adapters (simulating Alice and Bob)
        let alice = EcdheP256Adapter::new().unwrap();
        let bob = EcdheP256Adapter::new().unwrap();

        // Get Bob's public key
        let bob_pk_meta = bob.keygen().unwrap();
        let bob_pk = bob_pk_meta.public_key;

        // Alice encapsulates to Bob (generates ephemeral keypair, computes shared secret)
        let (ciphertext, alice_shared_secret) = alice.encapsulate(&bob_pk).unwrap();
        assert!(!ciphertext.is_empty());
        assert!(!alice_shared_secret.is_empty());

        // Bob decapsulates (uses his secret key and ephemeral public key to compute shared secret)
        let bob_shared_secret = bob.decapsulate(&[], &ciphertext).unwrap();

        // Shared secrets must match
        assert_eq!(alice_shared_secret, bob_shared_secret);
    }

    #[test]
    fn test_ecdhe_sign_not_implemented() {
        let adapter = EcdheP256Adapter::new().unwrap();
        let result = adapter.sign(&[], b"test");
        assert!(matches!(result, Err(CryptoError::NotImplemented)));
    }

    #[test]
    fn test_ecdhe_verify_not_implemented() {
        let adapter = EcdheP256Adapter::new().unwrap();
        let result = adapter.verify(&[], b"test", &[]);
        assert!(matches!(result, Err(CryptoError::NotImplemented)));
    }
}
