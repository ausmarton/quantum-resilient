# ECDHE P-256 Reference

**Date**: 2025-01-27  
**Status**: Implementation Complete  
**Purpose**: Single source of truth for ECDHE P-256 implementation details

---

## Quick Reference

**What we have**: **One-sided ephemeral ECDH** (commonly called "ECDHE")
- **Sender (encapsulate)**: Generates new ephemeral keypair each time ✅
- **Receiver (decapsulate)**: Uses static keypair ⚠️
- **Security**: Provides forward secrecy (ephemeral sender keys protect past sessions)
- **Terminology**: Called "ECDHE" in practice (matches TLS/HTTPS terminology)

**Why this is correct**:
- Standard pattern for KEM interfaces
- Matches Kyber's security model exactly
- Provides forward secrecy
- Industry standard (TLS ECDHE uses same pattern)

---

## Implementation Details

### Adapter
- **Name**: `EcdheP256Adapter`
- **Adapter ID**: `"ecdhe_p256"`
- **Operation**: `kem_aead_encrypt` (KEM + AES-GCM encryption)
- **Category**: `classical`

### Key Exchange Pattern
- **Encapsulate**: 
  - Generates ephemeral keypair (`EphemeralSecret::random`)
  - Computes shared secret with recipient's static public key
  - Returns (ephemeral public key, shared secret)
  
- **Decapsulate**:
  - Uses receiver's static secret key
  - Computes shared secret with sender's ephemeral public key
  - Returns shared secret

### Comparison to Kyber
- **Kyber**: Ephemeral on sender, static on receiver ✅
- **ECDHE**: Ephemeral on sender, static on receiver ✅
- **Security model**: Identical
- **Comparison**: True apples-to-apples KEM comparison

---

## Experiment Count

**Additional experiments**: 66 total
- Native: 20 experiments
- Minikube: 23 experiments
- GCP: 23 experiments

**Total experiments** (with ECDHE):
- Native: 120 (was 100)
- Minikube: 138 (was 115)
- GCP: 138 (was 115)
- **Total**: 396 (was 330)

---

## Technical Details

### ECDH vs ECDHE Distinction

**ECDH (Static)**:
- Uses static (long-lived) keys on both sides
- No forward secrecy
- Keys are reused across multiple sessions

**ECDHE (Ephemeral)**:
- Uses ephemeral (temporary) keys for each session
- Provides forward secrecy (compromised keys don't affect past sessions)
- Each key exchange uses fresh keys

**Our Implementation**: One-sided ephemeral ECDH (commonly called "ECDHE")
- Sender (encapsulate): Ephemeral keys ✅
- Receiver (decapsulate): Static key ⚠️
- This is the standard pattern for KEM interfaces and matches TLS ECDHE

### Why This Pattern?

1. **KEM Interface**: Matches the security model of Kyber exactly
2. **Forward Secrecy**: Ephemeral sender keys protect past sessions
3. **Efficiency**: Receiver doesn't need to generate new keys for each message
4. **Industry Standard**: TLS ECDHE uses the same pattern (client ephemeral, server static)

### Integration Status

✅ **Complete**:
- Rust adapter implemented (`rust-core/src/crypto_adapter/ecdhe_adapter.rs`)
- Registered in adapter registry
- Added to experiment matrix (all experiment types)
- Analysis scripts updated
- Cargo.toml updated (added `ecdh` feature to `p256`)

### Impact on Existing Data

**No impact** - Adding ECDHE adapter does not affect existing raw data:
- Each adapter is a separate, independent module
- No shared mutable state between adapters
- New adapter only used for new ECDHE experiments
- Existing data remains unchanged

---

**Status**: ✅ Implementation complete, documentation updated, ready for data collection.
