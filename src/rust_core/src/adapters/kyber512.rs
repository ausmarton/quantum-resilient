use crate::{CryptoAdapter, CryptoError, CryptoResult};
use rand::RngCore;

pub struct Kyber512;

impl CryptoAdapter for Kyber512 {
	fn name(&self) -> &str { "Kyber512" }

	fn public_key_size(&self) -> usize { 800 }
	fn secret_key_size(&self) -> usize { 1632 }
	fn signature_size(&self) -> usize { 0 }

	fn keygen(&self) -> CryptoResult<(Vec<u8>, Vec<u8>)> {
		let mut pk = vec![0u8; self.public_key_size()];
		let mut sk = vec![0u8; self.secret_key_size()];
		rand::thread_rng().fill_bytes(&mut pk);
		rand::thread_rng().fill_bytes(&mut sk);
		Ok((pk, sk))
	}

	fn encapsulate(&self, _public_key: &[u8]) -> CryptoResult<(Vec<u8>, Vec<u8>)> {
		let mut shared_secret = vec![0u8; 32];
		let mut ciphertext = vec![0u8; 768];
		rand::thread_rng().fill_bytes(&mut shared_secret);
		ciphertext[..32].copy_from_slice(&shared_secret);
		Ok((ciphertext, shared_secret))
	}

	fn decapsulate(&self, _secret_key: &[u8], ciphertext: &[u8]) -> CryptoResult<Vec<u8>> {
		if ciphertext.len() < 32 {
			return Err(CryptoError::KemFailure);
		}
		Ok(ciphertext[..32].to_vec())
	}

	fn sign(&self, _secret_key: &[u8], _message: &[u8]) -> CryptoResult<Vec<u8>> {
		Err(CryptoError::UnsupportedOperation("sign"))
	}

	fn verify(&self, _public_key: &[u8], _message: &[u8], _signature: &[u8]) -> CryptoResult<()> {
		Err(CryptoError::UnsupportedOperation("verify"))
	}
}


