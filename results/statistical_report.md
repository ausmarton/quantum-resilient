# Statistical Comparison Report
## PQC vs Classical Algorithms

### Kyber512 vs RSA-2048 (Keygen)

**Sample sizes:** n_A=60, n_B=60

**Means:** 2.533333 vs 0.116667 µs

**Medians:** 2.000000 vs 0.000000 µs

**Faster algorithm:** RSA-2048

**Mean difference:** 2.416667 µs (2071.43%)

**Statistical Tests:**
- Independent t-test: t=7.0743, p=0.000000 ***  (p < 0.001, highly significant)
- Mann-Whitney U: U=2805.0000, p=0.000000 ***  (p < 0.001, highly significant)

**Effect Sizes:**
- Cohen's d: 1.2916 (large)
- Rank-biserial r: -0.5583

---

### Kyber512 vs RSA-2048 (Encapsulate)

**Sample sizes:** n_A=60, n_B=30

**Means:** 0.016667 vs 0.000000 µs

**Medians:** 0.000000 vs 0.000000 µs

**Faster algorithm:** RSA-2048

**Mean difference:** 0.016667 µs (nan%)

**Statistical Tests:**
- Independent t-test: t=0.7051, p=0.482606 ns   (not significant)
- Mann-Whitney U: U=915.0000, p=0.494268 ns   (not significant)

**Effect Sizes:**
- Cohen's d: 0.1577 (negligible)
- Rank-biserial r: -0.0167

---

### Kyber768 vs RSA-2048 (Keygen)

**Sample sizes:** n_A=60, n_B=60

**Means:** 1.233333 vs 0.116667 µs

**Medians:** 1.000000 vs 0.000000 µs

**Faster algorithm:** RSA-2048

**Mean difference:** 1.116667 µs (957.14%)

**Statistical Tests:**
- Independent t-test: t=7.7806, p=0.000000 ***  (p < 0.001, highly significant)
- Mann-Whitney U: U=3414.5000, p=0.000000 ***  (p < 0.001, highly significant)

**Effect Sizes:**
- Cohen's d: 1.4205 (large)
- Rank-biserial r: -0.8969

---

### Kyber768 vs RSA-2048 (Encapsulate)

**Sample sizes:** n_A=60, n_B=30

**Means:** 0.000000 vs 0.000000 µs

**Medians:** 0.000000 vs 0.000000 µs

**Faster algorithm:** RSA-2048

**Mean difference:** 0.000000 µs (nan%)

**Statistical Tests:**
- Mann-Whitney U: U=900.0000, p=1.000000 ns   (not significant)

**Effect Sizes:**
- Cohen's d: 0.0000 (negligible)
- Rank-biserial r: 0.0000

---

### Kyber512 vs ECDHE-P256 (Keygen)

**Sample sizes:** n_A=60, n_B=60

**Means:** 2.533333 vs 0.000000 µs

**Medians:** 2.000000 vs 0.000000 µs

**Faster algorithm:** ECDHE-P256

**Mean difference:** 2.533333 µs (nan%)

**Statistical Tests:**
- Independent t-test: t=7.4719, p=0.000000 ***  (p < 0.001, highly significant)
- Mann-Whitney U: U=2910.0000, p=0.000000 ***  (p < 0.001, highly significant)

**Effect Sizes:**
- Cohen's d: 1.3642 (large)
- Rank-biserial r: -0.6167

---

### Kyber768 vs ECDHE-P256 (Keygen)

**Sample sizes:** n_A=60, n_B=60

**Means:** 1.233333 vs 0.000000 µs

**Medians:** 1.000000 vs 0.000000 µs

**Faster algorithm:** ECDHE-P256

**Mean difference:** 1.233333 µs (nan%)

**Statistical Tests:**
- Independent t-test: t=8.9828, p=0.000000 ***  (p < 0.001, highly significant)
- Mann-Whitney U: U=3600.0000, p=0.000000 ***  (p < 0.001, highly significant)

**Effect Sizes:**
- Cohen's d: 1.6400 (large)
- Rank-biserial r: -1.0000

---

### Dilithium2 vs ECDSA-P256 (Sign)

**Sample sizes:** n_A=30, n_B=30

**Means:** 0.000000 vs 0.000000 µs

**Medians:** 0.000000 vs 0.000000 µs

**Faster algorithm:** ECDSA-P256

**Mean difference:** 0.000000 µs (nan%)

**Statistical Tests:**
- Mann-Whitney U: U=450.0000, p=1.000000 ns   (not significant)

**Effect Sizes:**
- Cohen's d: 0.0000 (negligible)
- Rank-biserial r: 0.0000

---

### Dilithium2 vs ECDSA-P256 (Verify)

**Sample sizes:** n_A=30, n_B=30

**Means:** 0.000000 vs 0.000000 µs

**Medians:** 0.000000 vs 0.000000 µs

**Faster algorithm:** ECDSA-P256

**Mean difference:** 0.000000 µs (nan%)

**Statistical Tests:**
- Mann-Whitney U: U=450.0000, p=1.000000 ns   (not significant)

**Effect Sizes:**
- Cohen's d: 0.0000 (negligible)
- Rank-biserial r: 0.0000

---

### Dilithium3 vs ECDSA-P256 (Sign)

**Sample sizes:** n_A=30, n_B=30

**Means:** 0.300000 vs 0.000000 µs

**Medians:** 0.000000 vs 0.000000 µs

**Faster algorithm:** ECDSA-P256

**Mean difference:** 0.300000 µs (nan%)

**Statistical Tests:**
- Independent t-test: t=1.6075, p=0.113369 ns   (not significant)
- Mann-Whitney U: U=495.0000, p=0.081493 .    (p < 0.10, marginally significant)

**Effect Sizes:**
- Cohen's d: 0.4151 (small)
- Rank-biserial r: -0.1000

---

### Dilithium3 vs ECDSA-P256 (Verify)

**Sample sizes:** n_A=30, n_B=30

**Means:** 0.000000 vs 0.000000 µs

**Medians:** 0.000000 vs 0.000000 µs

**Faster algorithm:** ECDSA-P256

**Mean difference:** 0.000000 µs (nan%)

**Statistical Tests:**
- Mann-Whitney U: U=450.0000, p=1.000000 ns   (not significant)

**Effect Sizes:**
- Cohen's d: 0.0000 (negligible)
- Rank-biserial r: 0.0000

---

