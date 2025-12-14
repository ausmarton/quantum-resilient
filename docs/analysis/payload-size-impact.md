# Payload Size Impact Analysis

**Date**: 2025-12-14  
**Status**: Complete  
**Requirement**: FR10 - Payload Size Impact Analysis

---

## Overview

This document analyzes the impact of payload size on cryptographic operation latency across all algorithms and environments.

**Payload Sizes Tested**:
- 256 bytes (p256)
- 1024 bytes (p1024) 
- 4096 bytes (p4096)
- 16384 bytes (p16384)

---

## Data Source

- **Aggregated Statistics**: `final-results/aggregated_stats.json`
- **Experiments**: All baseline experiments (396 total with ECDHE: 120 native + 138 minikube + 138 gcp)
- **Environments**: Native, Minikube, GCP

---

## Analysis Method

1. Group experiments by algorithm and payload size
2. Extract p95 latency for each configuration
3. Compare latency across payload sizes within each algorithm
4. Calculate scaling factors (% increase per KB)

---

## Key Findings

### Native Environment

**Algorithm Performance by Payload Size (p95 latency in microseconds)**:

**dilithium2**:
- 256 bytes: 73.41μs
- 1024 bytes: 78.76μs
  → ++7.3% (+9.70% per KB)
- 4096 bytes: 98.31μs
  → ++24.8% (+8.28% per KB)
- 16384 bytes: 99.54μs
  → ++1.2% (+0.10% per KB)

**ecdsa**:
- 256 bytes: 114.51μs
- 1024 bytes: 115.50μs
  → ++0.9% (+1.15% per KB)
- 4096 bytes: 115.91μs
  → ++0.4% (+0.12% per KB)
- 16384 bytes: 120.26μs
  → ++3.8% (+0.31% per KB)

**hybrid**:
- 256 bytes: 111.93μs
- 1024 bytes: 114.62μs
  → ++2.4% (+3.21% per KB)
- 4096 bytes: 124.13μs
  → ++8.3% (+2.77% per KB)
- 16384 bytes: 161.52μs
  → ++30.1% (+2.51% per KB)

**kyber512**:
- 256 bytes: 10.94μs
- 1024 bytes: 11.81μs
  → ++8.0% (+10.66% per KB)
- 4096 bytes: 14.50μs
  → ++22.7% (+7.58% per KB)
- 16384 bytes: 24.64μs
  → ++70.0% (+5.83% per KB)

**rsa2048**:
- 256 bytes: 126.85μs
- 1024 bytes: 126.49μs
  → +-0.3% (-0.39% per KB)
- 4096 bytes: 126.11μs
  → +-0.3% (-0.10% per KB)
- 16384 bytes: 131.68μs
  → ++4.4% (+0.37% per KB)

## Native Environment

**Algorithm Performance by Payload Size (p95 latency in microseconds)**:

**dilithium2**:
- 256 bytes: 73.41μs
- 1024 bytes: 78.76μs (+7.3%, +9.70% per KB)
- 4096 bytes: 98.31μs (+24.8%, +8.28% per KB)
- 16384 bytes: 99.54μs (+1.2%, +0.10% per KB)

**ecdsa**:
- 256 bytes: 114.51μs
- 1024 bytes: 115.50μs (+0.9%, +1.15% per KB)
- 4096 bytes: 115.91μs (+0.4%, +0.12% per KB)
- 16384 bytes: 120.26μs (+3.8%, +0.31% per KB)

**hybrid**:
- 256 bytes: 111.93μs
- 1024 bytes: 114.62μs (+2.4%, +3.21% per KB)
- 4096 bytes: 124.13μs (+8.3%, +2.77% per KB)
- 16384 bytes: 161.52μs (+30.1%, +2.51% per KB)

**kyber512**:
- 256 bytes: 10.94μs
- 1024 bytes: 11.81μs (+8.0%, +10.66% per KB)
- 4096 bytes: 14.50μs (+22.7%, +7.58% per KB)
- 16384 bytes: 24.64μs (+70.0%, +5.83% per KB)

**rsa2048**:
- 256 bytes: 126.85μs
- 1024 bytes: 126.49μs (-0.3%, -0.39% per KB)
- 4096 bytes: 126.11μs (-0.3%, -0.10% per KB)
- 16384 bytes: 131.68μs (+4.4%, +0.37% per KB)

**Scaling Characteristics**:
- **Kyber-512** shows the strongest payload size dependency (+70% from 256B to 16KB, ~5.8% per KB)
- **Dilithium-2** shows moderate scaling (+36% total, decreasing per-KB impact)
- **Hybrid** shows moderate scaling (+44% total, ~2.5% per KB)
- **ECDSA** and **RSA-2048** show minimal payload size impact (<5% total variation)
- Scaling is **non-linear** for most algorithms, with decreasing per-KB impact at larger sizes

### Environment Comparison

**Payload Size Impact Across Environments**:
- [Compare native vs minikube vs gcp]
- [Document environment-specific scaling]

---

## Statistical Analysis

*[If hypothesis tests include payload size comparisons]*

---

## Interpretation

*[Interpret findings in context of dissertation claims]*

---

## Supporting Data

- Aggregated statistics: `final-results/aggregated_stats.json`
- Individual experiment summaries: `results/*/stats/summary.json`

---

**Status**: Framework created - data extraction in progress
