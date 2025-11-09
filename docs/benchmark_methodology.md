Benchmark Methodology
=====================

Workload Model
--------------
- Operations: keygen, encapsulate/decapsulate (KEM), sign/verify (signature), encrypt/decrypt (streaming).
- Streaming: deterministic payload generation using `ChaCha20Rng` seeded with `seed`, with:
  - TPS pacing and run duration
  - Chunk size control
  - Backpressure mode: Block or Drop with `max_lag_ms`
  - Graceful retries with `retries` and `retry_backoff_ms`
- Hybrid TLS-like:
  - PQC: Kyber keygen + encapsulate + decapsulate; AES-GCM for record protection (seed from shared secret).
  - Classical: ECDHE + RSA/ECDSA sign/verify; AES-GCM for record protection.
- Like-for-like comparisons:
  - Key-wrap: Kyber vs RSA-OAEP simulated wrap; identical AES-GCM, chunking, and payloads.
  - Integrity: Dilithium vs ECDSA; identical payloads and repetitions.

Measurements
------------
- Per-operation event fields (JSONL):
  - Timestamps, latency (µs), attempts, error
  - Sizes: public/secret key, signature, ciphertext; storage_overhead_pct
  - Performance: throughput_ops_per_sec (instantaneous)
  - Resources: CPU user/system time, max RSS; disk I/O (read/write_bytes), net tx/rx (Linux)
- Aggregation:
  - Latency percentiles: p50, p95, p99; mean; stddev
  - 95% CI (t-based): mean ± t_crit * (std / sqrt(n))
  - Throughput: 1 / latency_s (per event), mean reported per group
  - CPU/memory: mean per group (if present)

Statistical Tests
-----------------
- Paired comparisons (per operation):
  - Paired t-test (means); Wilcoxon signed-rank (medians, non-parametric)
  - Effect sizes: Cohen’s dz (paired); rank-biserial r (approximation)
- Pairing:
  - If `pair_id` available in events, align exact pairs; else align by index up to min length.

Controls & Reproducibility
--------------------------
- Deterministic RNG for payloads and AES-GCM seed derivations.
- Fixed environment toggles in CI: TZ=UTC, LC_ALL=C, PYTHONHASHSEED=0, single-threaded BLAS.
- Environment snapshot: CPU model/count, OS, Python, liboqs version (if present), git hash.
- Reproducibility script runs two identical experiments and compares:
  - environment.json, metrics, summary files, charts.

Caveats
-------
- Linux resource readings rely on `/proc` (best-effort).
- The simulated classical wrap (RSA) uses placeholder sizes; tune when wiring real crypto libs.
- For strict comparability, ensure identical symmetric cipher, chunking, and storage backend.


