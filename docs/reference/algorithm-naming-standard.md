# Algorithm Naming Standard

**Date**: 2025-12-15  
**Purpose**: Standardize algorithm names throughout dissertation

---

## ALGORITHMS ACTUALLY TESTED

Based on code analysis (`experiment_matrix.yaml`):
- **RSA-2048** (rsa2048)
- **ECDSA P-256** (ecdsa_p256) 
- **ECDHE P-256** (ecdhe_p256)
- **Kyber-512** (kyber512)
- **Dilithium-2** (dilithium2)
- **Hybrid** (hybrid_kyber_dilithium) - also referred to as "Hybrid Kyber-Dilithium"

---

## NAMING STANDARD

### First Mention (with full context)
- **PQC**: "Kyber-512 (ML-KEM)" and "Dilithium-2 (ML-DSA)" - establish NIST standard names
- **Classical**: "RSA-2048", "ECDSA P-256", "ECDHE P-256"

### Subsequent Mentions
- **PQC**: Use "Kyber-512" and "Dilithium-2" (drop ML-KEM/ML-DSA after first mention)
- **Classical**: Use "RSA-2048", "ECDSA P-256", "ECDHE P-256" consistently
- **Hybrid**: Use "Hybrid" (can expand to "Hybrid Kyber-Dilithium" if needed for clarity)

### When to Use Full Names
- Abstract: Use full names with variants
- Section 2.2 (ERK): First mention should include NIST standard names
- Section 2.2.3: Algorithm selection - use full names
- Section 3.3.2: Experimental matrix - use standard names
- Section 4.1.1: Results - use standard names
- Tables: Use standard names

### When to Use Short Names
- After first definition in each major section
- In comparisons and analysis
- In repeated references

---

## INCONSISTENCIES TO FIX

1. **Abstract**: "ML-KEM/Kyber and ML-DSA/Dilithium" → Should be "Kyber-512 (ML-KEM) and Dilithium-2 (ML-DSA)"
2. **Abstract**: "RSA-2048, ECDSA-P256, ECDHE-P256" → Should be "RSA-2048, ECDSA P-256, ECDHE P-256"
3. **Section 1.1**: "ML-KEM (Kyber) and ML-DSA (Dilithium)" → After first mention, use "Kyber-512" and "Dilithium-2"
4. **Section 2.2.2**: "ML-KEM (Kyber)" and "ML-DSA (Dilithium)" → Use "Kyber-512" and "Dilithium-2" after first mention
5. **Section 2.2.2**: "ECDH-P256" → Should be "ECDHE P-256"
6. **Section 2.2.3**: "Kyber-512" and "Dilithium-2" - ✅ CORRECT
7. **Section 4.1.1**: "Kyber-512, Dilithium-2" - ✅ CORRECT

---

## STANDARDIZATION RULES

1. **Always include variant numbers**: Kyber-512 (not just "Kyber"), Dilithium-2 (not just "Dilithium")
2. **Consistent spacing**: "ECDSA P-256" (space before P), "ECDHE P-256" (space before P)
3. **NIST standard names**: Mention ML-KEM/ML-DSA only on first mention in major sections
4. **Hybrid naming**: Use "Hybrid" as primary name, expand when needed for clarity

