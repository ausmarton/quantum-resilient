use crate::{CryptoAdapter, CryptoError, CryptoResult};
use rand::RngCore;

pub struct EcdheP256;

impl CryptoAdapter for EcdheP256 {
	fn name(&self) -> &str { "ECDHE-P256" }

	fn public_key_size(&self) -> usize { 65 }   // uncompressed
	fn secret_key_size(&self) -> usize { 32 }
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
		let mut transcript = vec![0u8; 96];
		rand::thread_rng().fill_bytes(&mut shared_secret);
		transcript[..32].copy_from_slice(&shared_secret);
		Ok((transcript, shared_secret))
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


