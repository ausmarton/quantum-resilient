use rust_core::adapters::{
	dilithium2::Dilithium2, dilithium3::Dilithium3, ecdhe_p256::EcdheP256, ecdsa_p256::EcdsaP256,
	kyber512::Kyber512, kyber768::Kyber768, rsa2048::Rsa2048,
};
use rust_core::{CryptoAdapter, CryptoError};

#[test]
fn kyber_adapters_compile_and_shapes() {
	let k512 = Kyber512;
	let (pk, sk) = k512.keygen().unwrap();
	assert_eq!(pk.len(), k512.public_key_size());
	assert_eq!(sk.len(), k512.secret_key_size());
	let (ct, ss) = k512.encapsulate(&pk).unwrap();
	assert!(ct.len() >= 32);
	assert_eq!(ss.len(), 32);
	let ss2 = k512.decapsulate(&sk, &ct).unwrap();
	assert_eq!(ss2.len(), 32);
	assert!(k512.sign(&sk, b"x").is_err());
	assert!(matches!(k512.sign(&sk, b"x"), Err(CryptoError::UnsupportedOperation(_))));

	let k768 = Kyber768;
	let (pk, sk) = k768.keygen().unwrap();
	assert_eq!(pk.len(), k768.public_key_size());
	assert_eq!(sk.len(), k768.secret_key_size());
	let (ct, ss) = k768.encapsulate(&pk).unwrap();
	assert!(ct.len() >= 32);
	assert_eq!(ss.len(), 32);
	let ss2 = k768.decapsulate(&sk, &ct).unwrap();
	assert_eq!(ss2.len(), 32);
	assert!(matches!(k768.sign(&sk, b"x"), Err(CryptoError::UnsupportedOperation(_))));
}

#[test]
fn dilithium_adapters_compile_and_shapes() {
	let d2 = Dilithium2;
	let (pk, sk) = d2.keygen().unwrap();
	assert_eq!(pk.len(), d2.public_key_size());
	assert_eq!(sk.len(), d2.secret_key_size());
	let sig = d2.sign(&sk, b"hello").unwrap();
	assert_eq!(sig.len(), d2.signature_size());
	assert!(d2.verify(&pk, b"hello", &sig).is_ok());
	assert!(matches!(d2.encapsulate(&pk), Err(CryptoError::UnsupportedOperation(_))));

	let d3 = Dilithium3;
	let (pk, sk) = d3.keygen().unwrap();
	assert_eq!(pk.len(), d3.public_key_size());
	assert_eq!(sk.len(), d3.secret_key_size());
	let sig = d3.sign(&sk, b"hello").unwrap();
	assert_eq!(sig.len(), d3.signature_size());
	assert!(d3.verify(&pk, b"hello", &sig).is_ok());
}

#[test]
fn classical_adapters_compile_and_shapes() {
	let rsa = Rsa2048;
	let (pk, sk) = rsa.keygen().unwrap();
	assert_eq!(pk.len(), rsa.public_key_size());
	assert_eq!(sk.len(), rsa.secret_key_size());
	let (ct, ss) = rsa.encapsulate(&pk).unwrap();
	assert_eq!(ct.len(), 256);
	assert_eq!(ss.len(), 32);
	let ss2 = rsa.decapsulate(&sk, &ct).unwrap();
	assert_eq!(ss2.len(), 32);
	let sig = rsa.sign(&sk, b"msg").unwrap();
	assert_eq!(sig.len(), rsa.signature_size());
	assert!(rsa.verify(&pk, b"msg", &sig).is_ok());

	let ecdhe = EcdheP256;
	let (pk, sk) = ecdhe.keygen().unwrap();
	let (ct, ss) = ecdhe.encapsulate(&pk).unwrap();
	assert!(ct.len() >= 32);
	assert_eq!(ss.len(), 32);
	let ss2 = ecdhe.decapsulate(&sk, &ct).unwrap();
	assert_eq!(ss2.len(), 32);

	let ecdsa = EcdsaP256;
	let (pk, sk) = ecdsa.keygen().unwrap();
	let sig = ecdsa.sign(&sk, b"msg").unwrap();
	assert!(sig.len() >= 60 && sig.len() <= 80);
	assert!(ecdsa.verify(&pk, b"msg", &sig).is_ok());
}


