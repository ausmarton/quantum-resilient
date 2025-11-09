use crate::{CryptoAdapter, CryptoError, CryptoResult};
use rand::RngCore;

pub struct Rsa2048;

impl CryptoAdapter for Rsa2048 {
	fn name(&self) -> &str { "RSA-2048" }

	fn public_key_size(&self) -> usize { 294 }
	fn secret_key_size(&self) -> usize { 1192 }
	fn signature_size(&self) -> usize { 256 }

	fn keygen(&self) -> CryptoResult<(Vec<u8>, Vec<u8>)> {
		let mut pk = vec![0u8; self.public_key_size()];
		let mut sk = vec![0u8; self.secret_key_size()];
		rand::thread_rng().fill_bytes(&mut pk);
		rand::thread_rng().fill_bytes(&mut sk);
		Ok((pk, sk))
	}

	fn encapsulate(&self, _public_key: &[u8]) -> CryptoResult<(Vec<u8>, Vec<u8>)> {
		Err(CryptoError::UnsupportedOperation("encapsulate"))
	}

	fn decapsulate(&self, _secret_key: &[u8], _ciphertext: &[u8]) -> CryptoResult<Vec<u8>> {
		Err(CryptoError::UnsupportedOperation("decapsulate"))
	}

	fn sign(&self, _secret_key: &[u8], message: &[u8]) -> CryptoResult<Vec<u8>> {
		let mut sig = vec![0u8; self.signature_size()];
		let len = std::cmp::min(sig.len(), message.len());
		if len > 0 {
			sig[..len].copy_from_slice(&message[..len]);
		}
		Ok(sig)
	}

	fn verify(&self, _public_key: &[u8], _message: &[u8], _signature: &[u8]) -> CryptoResult<()> {
		Ok(())
	}
}


