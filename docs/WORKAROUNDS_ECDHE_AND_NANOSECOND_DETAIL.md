# Why We Did the ECDHE and Nanosecond Workarounds

Detailed rationale and evidence for why each change was **necessary** (not just desirable).

---

## 1. Nanosecond precision (microsecond → nanosecond)

### What was the problem?

**Issue**: Many cryptographic operations complete in **under 1 microsecond**. The code used Rust’s `elapsed().as_micros()`, which returns an **integer** number of microseconds. Any duration &lt; 1 μs is truncated to **0**.

**Root cause** (from `docs/reference/precision-implementation.md`):

```rust
let latency_us = start.elapsed().as_micros();  // Truncates <1μs to 0
```

So we were **recording 0** for all sub-microsecond operations instead of their real latencies.

### Evidence that it was a real problem

From the project’s own data and docs:

- **Sample data**: Many rows had `latency_us: 0`.
- **Logs vs stored data**: Logs showed “Average latency: 0.02 μs” but the **stored** value was 0.
- **Proportion affected**: In some experiments, **94% of operations** had `latency_us = 0`.
- **Actual range**: Operations were in the range **0.01 μs to 0.82 μs** (about **800 ns** spread), all of which were being collapsed to 0 when stored in integer microseconds.

So the issue was not rare: a large fraction of operations (and in some configs almost all) were mis-measured.

### Why was the fix necessary (not optional)?

1. **Research claims**  
   The methodology claims comparison of algorithm performance (latency, percentiles, effect sizes). If most latencies are stored as 0:
   - Percentiles (p50, p95, p99) are wrong or degenerate.
   - Mean latency is biased (systematically too low).
   - Effect sizes and statistical tests are invalid when one “algorithm” is mostly zeros and another is not.

2. **Which algorithms were affected**  
   The docs cite values like 0.02 μs, 0.04 μs, 0.55 μs, 0.63 μs, 0.74 μs, 0.77 μs, 0.82 μs. The **fastest** algorithms (e.g. certain KEM or symmetric-style operations) are exactly the ones that fall below 1 μs. So the workaround was necessary to **measure the fastest operations correctly** and to compare them to slower ones (e.g. RSA/ECDSA) without artefact.

3. **Statistical validity**  
   The design uses run-level aggregates and hypothesis testing. Those aggregates (means, percentiles) must be computed from **real** latencies, not from truncated zeros. Without nanosecond precision (or an equivalent like float μs), the data for sub-μs operations was wrong by construction, so the fix was **necessary** for valid inference.

4. **Alternative considered**  
   Option 2 (float microseconds) was documented (`docs/reference/option2-precision.md`) but **not** chosen; the project went with **integer nanoseconds** (Option 1) to avoid float precision issues at very small values and to keep a clear, reproducible schema. So the “workaround” was really: **store in nanoseconds (integer)** and convert to μs only for display/analysis.

### Summary: why nanosecond was necessary

| Aspect | Without fix | With fix |
|--------|-------------|----------|
| Sub-μs ops | Stored as 0 | Stored as true ns (e.g. 20–820 ns) |
| Percentiles / means | Wrong or degenerate | Correct for all ops |
| Fast vs slow comparison | Biased (fast algo “zeros”) | Fair comparison |
| Statistical tests | Invalid for sub-μs data | Valid |

**Conclusion**: The nanosecond workaround was **necessary** because integer microsecond storage **truncated** sub-μs latencies to 0, corrupting a large share of the data and invalidating the latency comparisons and statistics the research relies on.

---

## 2. ECDHE (classical KEM) in the mix

### What was the problem?

**Issue**: Kyber-512 is a **KEM** (key encapsulation) algorithm. The only classical algorithms initially benchmarked were **RSA-2048** and **ECDSA P-256**, which are **signature** algorithms. So we were comparing:

- **Kyber-512**: KEM operation (`kem_aead_encrypt`).
- **RSA-2048 / ECDSA P-256**: Signature operations (`sign`).

That is **different cryptographic primitives** (KEM vs signatures). A reviewer (and the project’s own assessment) correctly called this **“apples to oranges”**.

### Evidence that it was a real problem

From `docs/analysis/comparison-issue-assessment.md`:

1. **Reviewer concern**  
   “Kyber-512 (KEM operations) is being compared to RSA-2048 and ECDSA P-256 (signature operations) without explicit operational framing.”

2. **Missing baseline**  
   “Without a classical KEM baseline, we can’t say if Kyber is better than classical KEMs. We can only say ‘Kyber KEM is faster than classical signatures’ – which is not a meaningful comparison.”

3. **Misleading wording**  
   The research had statements like “Kyber-512 significantly outperforms classical algorithms, with 8.3x lower latency than RSA-2048 and ECDSA P-256.” That sounds like a **general** PQC-vs-classical claim, but it was really **KEM vs signatures**. So the comparison was not only structurally wrong but also **easily misinterpreted**.

4. **Scale of the issue**  
   The assessment document lists **19 instances** where Kyber was compared to classical algorithms without proper operational framing (KEM vs signatures).

### Why was adding ECDHE necessary (not optional)?

1. **Apples-to-apples comparison**  
   Kyber is a KEM. To answer “how does post-quantum KEM compare to classical KEM?” we need a **classical KEM** in the benchmark set. ECDHE P-256 (one-sided ephemeral ECDH, used as KEM) is the standard classical KEM (e.g. TLS). Adding it gave a **true KEM-vs-KEM** comparison (ECDHE vs Kyber).

2. **Reviewer validity**  
   The assessment states: “The reviewer is **correct** that (1) comparing KEM to signatures is apples-to-oranges, (2) KEM operations are structurally simpler, (3) the comparison needs explicit operational framing, (4) statements sound like general crypto claims when they’re operation-specific.” So the concern was **valid**; adding ECDHE was the way to **fix** the design, not just reframe the text.

3. **What ECDHE provides**  
   - Same **operation type** as Kyber (KEM: encapsulate/decapsulate).  
   - Same **security pattern** in the benchmark: one-sided ephemeral (sender ephemeral, receiver static), matching Kyber and TLS ECDHE.  
   - So ECDHE vs Kyber is a **like-for-like** classical-vs-PQC KEM comparison.

4. **Could we have avoided it?**  
   Option A was “add explicit operational framing” (always say “KEM” vs “signatures”) and keep only RSA/ECDSA. That would have **qualified** the existing comparisons but would **not** have provided a classical KEM baseline. The assessment recommended **both**: framing (Option A) **and** adding ECDHE (Option B). So ECDHE was necessary to **support** a proper classical KEM baseline claim; framing alone was not enough.

### Summary: why ECDHE was necessary

| Aspect | Without ECDHE | With ECDHE |
|--------|----------------|------------|
| Kyber comparison | Only vs RSA/ECDSA (signatures) | Vs ECDHE (KEM) + vs RSA/ECDSA |
| “Classical KEM” claim | Cannot make it | ECDHE vs Kyber is KEM–KEM |
| Reviewer concern | Valid: apples-to-oranges | Addressed by design |
| Misleading “8.3x vs classical” | Sounds like all classical | Can separate KEM vs signature baselines |

**Conclusion**: The ECDHE workaround was **necessary** because Kyber is a KEM and we had **no classical KEM** in the benchmark; without ECDHE, we could not make a defensible “Kyber vs classical KEM” comparison and the reviewer’s “apples-to-oranges” criticism remained valid.

---

## Quick reference

| Workaround | Trigger | Why necessary |
|------------|--------|----------------|
| **Nanosecond precision** | Sub-μs ops stored as 0 (94% in some runs); `as_micros()` truncation | Latency data was wrong for fast ops; percentiles, means, and statistical tests were invalid without true sub-μs values. |
| **ECDHE P-256** | Reviewer: Kyber (KEM) compared to RSA/ECDSA (signatures) = apples-to-oranges | No classical KEM in the mix; adding ECDHE provided a proper KEM-vs-KEM baseline (ECDHE vs Kyber) and addressed the validity of the comparison. |
