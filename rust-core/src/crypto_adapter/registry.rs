//! Crypto Adapter Registry
//!
//! Provides a factory function to instantiate crypto adapters by name.

use super::{
    ecdsa_adapter::EcdsaP256Adapter, kyber_adapter::KyberAdapter, noop_adapter::NoOpCryptoAdapter,
    rsa_adapter::Rsa2048Adapter, CryptoAdapter, CryptoError,
};
use std::sync::Arc;

/// Gets a crypto adapter instance by name
///
/// # Arguments
/// * `name` - The name of the adapter to instantiate
///
/// # Returns
/// An Arc-wrapped CryptoAdapter instance on success
///
/// # Supported Adapters
/// - `"noop"` - NoOp baseline adapter (zero-cost operations)
/// - `"rsa2048"` - RSA-2048 adapter
/// - `"ecdsa_p256"` - ECDSA P-256 adapter
/// - `"kyber"` - Kyber-512 PQC KEM adapter
///
/// # Errors
/// Returns `CryptoError::InvalidKey` if the adapter name is not recognized
pub fn get_adapter(name: &str) -> Result<Arc<dyn CryptoAdapter>, CryptoError> {
    match name {
        "noop" => Ok(Arc::new(NoOpCryptoAdapter)),
        "rsa2048" => Ok(Arc::new(Rsa2048Adapter::new()?)),
        "ecdsa_p256" => Ok(Arc::new(EcdsaP256Adapter::new()?)),
        "kyber" => Ok(Arc::new(KyberAdapter::new("kyber512")?)),
        _ => Err(CryptoError::InvalidKey),
    }
}

/// Returns a list of all supported adapter names
pub fn supported_adapters() -> &'static [&'static str] {
    &["noop", "rsa2048", "ecdsa_p256", "kyber"]
}

/// Returns whether the adapter supports KEM operations
pub fn adapter_supports_kem(name: &str) -> bool {
    matches!(name, "kyber" | "noop")
}

/// Returns whether the adapter supports signature operations
pub fn adapter_supports_signatures(name: &str) -> bool {
    matches!(name, "ecdsa_p256" | "rsa2048" | "noop")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_noop_adapter() {
        let adapter = get_adapter("noop");
        assert!(adapter.is_ok());
        assert_eq!(adapter.unwrap().name(), "noop");
    }

    #[test]
    fn test_get_rsa_adapter() {
        let adapter = get_adapter("rsa2048");
        assert!(adapter.is_ok());
        assert_eq!(adapter.unwrap().name(), "rsa2048");
    }

    #[test]
    fn test_get_ecdsa_adapter() {
        let adapter = get_adapter("ecdsa_p256");
        assert!(adapter.is_ok());
        assert_eq!(adapter.unwrap().name(), "ecdsa_p256");
    }

    #[test]
    fn test_get_kyber_adapter() {
        let adapter = get_adapter("kyber");
        assert!(adapter.is_ok());
        assert_eq!(adapter.unwrap().name(), "kyber");
    }

    #[test]
    fn test_get_unknown_adapter() {
        let adapter = get_adapter("unknown");
        assert!(adapter.is_err());
        assert!(matches!(adapter, Err(CryptoError::InvalidKey)));
    }

    #[test]
    fn test_supported_adapters() {
        let adapters = supported_adapters();
        assert!(adapters.contains(&"noop"));
        assert!(adapters.contains(&"rsa2048"));
        assert!(adapters.contains(&"ecdsa_p256"));
        assert!(adapters.contains(&"kyber"));
    }

    #[test]
    fn test_adapter_supports_kem() {
        assert!(adapter_supports_kem("kyber"));
        assert!(adapter_supports_kem("noop"));
        assert!(!adapter_supports_kem("ecdsa_p256"));
        assert!(!adapter_supports_kem("rsa2048"));
    }

    #[test]
    fn test_adapter_supports_signatures() {
        assert!(adapter_supports_signatures("ecdsa_p256"));
        assert!(adapter_supports_signatures("rsa2048"));
        assert!(adapter_supports_signatures("noop"));
        assert!(!adapter_supports_signatures("kyber"));
    }
}
