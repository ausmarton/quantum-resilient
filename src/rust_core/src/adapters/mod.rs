pub mod kyber512;
pub mod kyber768;
pub mod dilithium2;
pub mod dilithium3;
pub mod rsa2048;
pub mod ecdhe_p256;
pub mod ecdsa_p256;

use crate::CryptoAdapter;

pub fn all_adapters() -> Vec<Box<dyn CryptoAdapter>> {
	vec![
		Box::new(kyber512::Kyber512),
		Box::new(kyber768::Kyber768),
		Box::new(dilithium2::Dilithium2),
		Box::new(dilithium3::Dilithium3),
		Box::new(rsa2048::Rsa2048),
		Box::new(ecdhe_p256::EcdheP256),
		Box::new(ecdsa_p256::EcdsaP256),
	]
}


