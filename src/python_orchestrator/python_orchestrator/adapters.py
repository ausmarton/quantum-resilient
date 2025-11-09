from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class CryptoAdapter(Protocol):
	def name(self) -> str: ...
	def public_key_size(self) -> int: ...
	def secret_key_size(self) -> int: ...
	def signature_size(self) -> int: ...
	def keygen(self) -> tuple[bytes, bytes]: ...
	def encapsulate(self, public_key: bytes) -> tuple[bytes, bytes]: ...
	def decapsulate(self, secret_key: bytes, ciphertext: bytes) -> bytes: ...
	def sign(self, secret_key: bytes, message: bytes) -> bytes: ...
	def verify(self, public_key: bytes, message: bytes, signature: bytes) -> None: ...


def load_rust_adapters() -> List[CryptoAdapter]:
	try:
		import pqc_core  # type: ignore
	except Exception:
		return []
	try:
		return list(pqc_core.list_adapters())  # type: ignore[attr-defined]
	except Exception:
		return []


def select_adapters(all_adapters: List[CryptoAdapter], names: List[str]) -> List[CryptoAdapter]:
	name_set = {n.lower() for n in names}
	return [a for a in all_adapters if a.name().lower() in name_set]


