use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::adapters::{
	dilithium2::Dilithium2, dilithium3::Dilithium3, ecdhe_p256::EcdheP256, ecdsa_p256::EcdsaP256,
	kyber512::Kyber512, kyber768::Kyber768, rsa2048::Rsa2048,
};
use crate::{CryptoAdapter, CryptoResult};

#[pyclass(name = "Adapter")]
pub struct PyAdapter {
	inner: Box<dyn CryptoAdapter>,
}

impl PyAdapter {
	fn new(inner: Box<dyn CryptoAdapter>) -> Self {
		Self { inner }
	}
}

#[pymethods]
impl PyAdapter {
	fn name(&self) -> &str {
		self.inner.name()
	}
	fn public_key_size(&self) -> usize {
		self.inner.public_key_size()
	}
	fn secret_key_size(&self) -> usize {
		self.inner.secret_key_size()
	}
	fn signature_size(&self) -> usize {
		self.inner.signature_size()
	}
	fn keygen<'py>(&self, py: Python<'py>) -> PyResult<(&'py PyBytes, &'py PyBytes)> {
		let (pk, sk) = self.inner.keygen().map_err(to_py)?;
		Ok((PyBytes::new(py, &pk), PyBytes::new(py, &sk)))
	}
	fn encapsulate<'py>(&self, py: Python<'py>, public_key: &[u8]) -> PyResult<(&'py PyBytes, &'py PyBytes)> {
		let (ct, ss) = self.inner.encapsulate(public_key).map_err(to_py)?;
		Ok((PyBytes::new(py, &ct), PyBytes::new(py, &ss)))
	}
	fn decapsulate<'py>(&self, py: Python<'py>, secret_key: &[u8], ciphertext: &[u8]) -> PyResult<&'py PyBytes> {
		let ss = self.inner.decapsulate(secret_key, ciphertext).map_err(to_py)?;
		Ok(PyBytes::new(py, &ss))
	}
	fn sign<'py>(&self, py: Python<'py>, secret_key: &[u8], message: &[u8]) -> PyResult<&'py PyBytes> {
		let sig = self.inner.sign(secret_key, message).map_err(to_py)?;
		Ok(PyBytes::new(py, &sig))
	}
	fn verify(&self, public_key: &[u8], message: &[u8], signature: &[u8]) -> PyResult<()> {
		self.inner.verify(public_key, message, signature).map_err(to_py)?;
		Ok(())
	}
}

fn to_py(err: crate::CryptoError) -> PyErr {
	pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", err))
}

#[pyfunction]
pub fn list_adapters(py: Python<'_>) -> PyResult<Vec<Py<PyAdapter>>> {
	let mut out: Vec<Py<PyAdapter>> = Vec::new();
	let adapters: Vec<Box<dyn CryptoAdapter>> = vec![
		Box::new(Kyber512),
		Box::new(Kyber768),
		Box::new(Dilithium2),
		Box::new(Dilithium3),
		Box::new(Rsa2048),
		Box::new(EcdheP256),
		Box::new(EcdsaP256),
	];
	for a in adapters {
		let py_adapter = Py::new(py, PyAdapter::new(a))?;
		out.push(py_adapter);
	}
	Ok(out)
}

#[pymodule]
pub fn pqc_core(py: Python<'_>, m: &PyModule) -> PyResult<()> {
	m.add_class::<PyAdapter>()?;
	m.add_function(pyo3::wrap_pyfunction!(list_adapters, m)?)?;
	Ok(())
}


