# Project Workarounds Summary

Short list of design decisions and changes made to work around specific issues during the quantum-resilient benchmarking project.

---

## Measurement & Precision

1. **Microsecond → nanosecond precision for latency**  
   Many crypto operations complete in &lt;1 μs; `as_micros()` truncated to 0 and lost data (e.g. 94% of operations recorded as 0). Switched to `as_nanos()` and store/analyse in ns (convert to μs only for display) so sub-microsecond latencies are captured.

2. **CPU sampling: `/proc/self/stat` instead of sysinfo percentage**  
   `sysinfo::Process::cpu_usage()` is percentage-based and gave zero/near-zero for fast ops. Switched to cumulative CPU time from `/proc/self/stat` (utime + stime in clock ticks) so CPU utilisation can be computed from deltas between events.

---

## Algorithm Comparison

3. **Added ECDHE P-256 to the mix**  
   Kyber (KEM) was only comparable to RSA/ECDSA (signatures), so reviewers could see it as apples-to-oranges. Implemented ECDHE P-256 as a classical KEM so Kyber has a true classical KEM counterpart; ECDHE vs Kyber is now a direct KEM-to-KEM comparison.

---

## Statistical & Experimental Design

4. **Run-level analysis to avoid pseudo-replication**  
   Event-level data would treat many measurements from the same run as independent. Hypothesis testing is done on run-level aggregates (e.g. mean latency per run), with runs as the unit of replication, so conclusions are not inflated by pseudo-replication.

5. **Holm–Bonferroni correction for multiple comparisons**  
   Many pairwise comparisons (algorithms × metrics × environments) inflate Type I error. Holm–Bonferroni is applied so family-wise error rate is controlled and “significant” results are not overclaimed.

6. **Within-environment comparisons only for inference; cross-environment descriptive**  
   Native/Minikube vs GCP differ in hardware (AMD vs Intel, core count, memory), so absolute latency comparisons across environments are confounded. Inferential claims (e.g. “algorithm A faster than B”) are restricted to within-environment baselines; cross-environment comparisons are reported descriptively only.

---

## Methodology & Scope

7. **Closed-system prototype instead of live production**  
   Using a live AML/production platform was ruled out due to operational risk, GDPR/data protection, and inability to instrument at the required precision without affecting performance. A closed-system prototype with synthetic workload and full instrumentation was used instead.

8. **Scope boundaries: exclude ML, network jitter, business logic**  
   End-to-end latency would mix crypto cost with ML inference, network jitter, and business logic. The framework measures only cryptographic operation performance so that differences can be attributed to algorithm choice, not system-level factors.

9. **Deterministic RNG seed from scenario parameters**  
   Workload (event schedule, payloads) must be reproducible across runs and machines. RNG seed is computed deterministically from algorithm, payload size, rate, run index (and pattern/duration where relevant) so identical scenario ⇒ identical workload.

---

## Infrastructure & Tooling

10. **GCP Terraform: directory path, variable names, `-target` in ephemeral mode**  
    `deploy_gcp.sh` pointed at the wrong Terraform dir and used names (`cluster_name`, `machine_type`, etc.) that didn’t match `variables.tf` (`gke_name`, `gke_node_machine_type`, etc.). Ephemeral mode tried to create K8s resources before the cluster was ready; `-target` was added so cluster (and dependencies) are created first, then kubectl configured, then K8s resources applied.

11. **Containerized analysis with host-Python fallback**  
    To get consistent Python version and deps across machines (and identical analysis outputs), analysis scripts run in a container by default. `QR_USE_CONTAINER=false` allows fallback to host Python for development/debugging when Docker/Podman isn’t desired.

12. **Container path conversion for analysis scripts**  
    When analysis runs in a container, the project root is mounted at `/workspace`. Scripts were given absolute host paths, so they looked in the wrong place inside the container. Path arguments (e.g. `--input`, `--output`) are now converted to container paths (`/workspace/relative/...`) so merge_jsonl and downstream steps find the files.

13. **Podman volume mount: `:z` for SELinux**  
    On Fedora with SELinux, volume mounts without a context caused permission errors. The container wrapper uses the `:z` (shared context) flag on volume mounts so the container can read/write mounted project directories.

---

## Scaling & Environments

14. **Native: no replicas; scaling only on Minikube/GCP**  
    Native runs a single process; there is no “replicas” concept. Replica counts &gt; 1 are skipped for native. Scaling experiments (replicas 2, 4, 8) run only on Minikube and GCP, where they test orchestration overhead (Minikube, single node) or multi-node scaling (GCP).

15. **Fewer runs for scaling and 5‑minute experiments**  
    Baseline configs use 5 runs; scaling and 5‑minute sustained-load runs use 3 runs to keep resource and time cost manageable, while still giving usable power for medium/large effects.

---

## Summary Table

| # | Workaround | Issue addressed |
|---|------------|------------------|
| 1 | Nanosecond latency precision | Sub-μs ops truncated to 0 with μs |
| 2 | `/proc/self/stat` for CPU | sysinfo percentage useless for fast ops |
| 3 | ECDHE P-256 added | Kyber needed classical KEM counterpart |
| 4 | Run-level statistical unit | Pseudo-replication from event-level tests |
| 5 | Holm–Bonferroni | Type I error from many comparisons |
| 6 | Within-environment inference only | Hardware confounding across envs |
| 7 | Closed-system prototype | Live production impractical (risk, GDPR, instrumentation) |
| 8 | Scope: crypto-only | Attribution to algorithm vs system factors |
| 9 | Deterministic RNG seed | Reproducible workload across runs/machines |
| 10 | GCP Terraform path/vars/`-target` | Wrong dir, wrong vars, K8s before cluster ready |
| 11 | Containerized analysis + fallback | Consistent deps; dev without container |
| 12 | Container path conversion | Scripts received host paths inside container |
| 13 | Podman `:z` mount | SELinux blocking container access |
| 14 | No native replicas | Native is single process |
| 15 | 3 runs for scaling/5‑min | Resource/time trade-off |
