//! ECDSA P-256 Crypto Adapter
//!
//! Provides ECDSA P-256 signing and verification for baseline benchmarking.

use crate::crypto_adapter::{CryptoAdapter, CryptoError, KeypairMeta};
use p256::ecdsa::{
    signature::{Signer, Verifier},
    Signature, SigningKey, VerifyingKey,
};
use rand::rngs::OsRng;

/// ECDSA P-256 cryptographic adapter
///
/// Provides digital signature operations using the NIST P-256 curve.
pub struct EcdsaP256Adapter {
    /// The ECDSA signing key (private key)
    pub signing_key: SigningKey,
    /// The ECDSA verifying key (public key)
    pub verifying_key: VerifyingKey,
}

impl EcdsaP256Adapter {
    /// Creates a new ECDSA P-256 adapter with a freshly generated keypair
    pub fn new() -> Result<Self, CryptoError> {
        let sk = SigningKey::random(&mut OsRng);
        let vk = VerifyingKey::from(&sk);
        Ok(Self {
            signing_key: sk,
            verifying_key: vk,
        })
    }
}

impl CryptoAdapter for EcdsaP256Adapter {
    fn name(&self) -> &'static str {
        "ecdsa_p256"
    }

    fn keygen(&self) -> Result<KeypairMeta, CryptoError> {
        Ok(KeypairMeta {
            public_key: self
                .verifying_key
                .to_encoded_point(false)
                .as_bytes()
                .to_vec(),
            secret_key_length: 32,
            params: "ecdsa_p256".to_string(),
        })
    }

    fn encapsulate(&self, _pk: &[u8]) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
        Err(CryptoError::NotImplemented)
    }

    fn decapsulate(&self, _sk: &[u8], _ct: &[u8]) -> Result<Vec<u8>, CryptoError> {
        Err(CryptoError::NotImplemented)
    }

    fn sign(&self, _sk: &[u8], msg: &[u8]) -> Result<Vec<u8>, CryptoError> {
        let sig: Signature = self.signing_key.sign(msg);
        Ok(sig.to_der().as_bytes().to_vec())
    }

    fn verify(&self, _pk: &[u8], msg: &[u8], sig: &[u8]) -> Result<bool, CryptoError> {
        let sig =
            Signature::from_der(sig).map_err(|e| CryptoError::InternalError(e.to_string()))?;
        Ok(self.verifying_key.verify(msg, &sig).is_ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ecdsa_adapter_new() {
        let adapter = EcdsaP256Adapter::new();
        assert!(adapter.is_ok());
    }

    #[test]
    fn test_ecdsa_adapter_name() {
        let adapter = EcdsaP256Adapter::new().unwrap();
        assert_eq!(adapter.name(), "ecdsa_p256");
    }

    #[test]
    fn test_ecdsa_adapter_keygen() {
        let adapter = EcdsaP256Adapter::new().unwrap();
        let meta = adapter.keygen();
        assert!(meta.is_ok());
        let meta = meta.unwrap();
        assert!(!meta.public_key.is_empty());
        assert_eq!(meta.secret_key_length, 32);
        assert_eq!(meta.params, "ecdsa_p256");
    }

    #[test]
    fn test_ecdsa_adapter_sign() {
        let adapter = EcdsaP256Adapter::new().unwrap();
        let msg = b"test message for signing";
        let result = adapter.sign(&[], msg);
        assert!(result.is_ok());
        let signature = result.unwrap();
        assert!(!signature.is_empty());
    }

    #[test]
    fn test_ecdsa_adapter_verify() {
        let adapter = EcdsaP256Adapter::new().unwrap();
        let msg = b"test message for signing";

        // Sign the message
        let signature = adapter.sign(&[], msg).unwrap();

        // Verify the signature
        let result = adapter.verify(&[], msg, &signature);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_ecdsa_adapter_verify_invalid_signature() {
        let adapter = EcdsaP256Adapter::new().unwrap();
        let msg = b"test message";

        // Create a valid signature first, then tamper with it
        let mut signature = adapter.sign(&[], msg).unwrap();
        if !signature.is_empty() {
            signature[0] ^= 0xFF; // Flip bits to invalidate
        }

        // Verification should fail (return false or error)
        let result = adapter.verify(&[], msg, &signature);
        // Either returns Ok(false) or Err (invalid DER)
        assert!(result.is_err() || !result.unwrap());
    }

    #[test]
    fn test_ecdsa_adapter_encapsulate_not_implemented() {
        let adapter = EcdsaP256Adapter::new().unwrap();
        let result = adapter.encapsulate(&[]);
        assert!(matches!(result, Err(CryptoError::NotImplemented)));
    }
}

