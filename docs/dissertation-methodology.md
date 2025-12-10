# Dissertation Methodology: Measurement and Analysis

**Date**: 2025-12-10  
**Status**: Complete  
**Purpose**: Methodology documentation for dissertation Chapter 3 (Methodology)

---

## Overview

This document describes the measurement methodology, precision requirements, and resource utilization analysis approach used in the quantum-resilient cryptographic benchmarking framework. This methodology ensures accurate, reproducible, and statistically rigorous performance measurements across multiple deployment environments.

---

## Measurement Precision

### Latency Measurement

**Implementation**: Latencies are measured using Rust's `Instant::now()` with nanosecond precision (1ns resolution). The measurement captures the time elapsed between the start and completion of each cryptographic operation.

**Precision Details**:
- **Primary Measurement**: Nanosecond precision (`latency_ns: u128`)
- **Derived Measurement**: Microsecond precision (`latency_us: u128` or `f64`) derived from nanoseconds
- **Resolution**: 1 nanosecond (system clock dependent, typically 1-100ns on modern hardware)
- **Range**: Supports operations from sub-microsecond (<1μs) to millisecond+ (>1000μs)

**Sub-Microsecond Operations**:
Operations completing in <1 microsecond are accurately recorded in nanoseconds and converted to microseconds with decimal precision for analysis. This ensures no data loss for very fast cryptographic operations and enables precise statistical analysis.

**Evidence**: Sample data shows operations ranging from 0.02μs (20ns) to 0.82μs (820ns), demonstrating the framework's ability to capture sub-microsecond latencies accurately.

**Analysis Impact**: Nanosecond precision enables:
- Accurate percentile calculations (p50, p95, p99) for sub-microsecond operations
- Statistical significance testing for small performance differences
- Precise environment overhead quantification
- Distribution analysis without truncation artifacts

### Queue Delay Measurement

**Implementation**: Queue delays are measured with nanosecond precision but stored in microseconds for analysis.

**Precision Details**:
- **Measurement**: Nanosecond precision (`queue_delay_ns: u128`)
- **Storage**: Microsecond precision (`queue_delay_us: u128`)
- **Rationale**: Queue delays are typically ≥1μs, so microsecond precision is sufficient for analysis
- **Separation**: Queue delay is measured separately from cryptographic operation latency to enable analysis of queuing overhead

**Analysis**: Queue delay analysis enables:
- Separation of cryptographic latency from queuing overhead
- Quantification of queuing contribution to total latency
- Analysis of queuing behavior under different load conditions

### Throughput Measurement

**Implementation**: Throughput is calculated using timestamp-based 1-second buckets.

**Timestamp Precision**:
- **ISO Timestamp**: `timestamp_utc_iso` in ISO 8601 format with millisecond precision
- **Monotonic Timestamp**: `timestamp_monotonic_ns` in nanoseconds for detailed time-series analysis
- **Bucket Size**: 1 second (millisecond precision sufficient)
- **Calculation**: Events per second calculated by grouping events into 1-second intervals

**Precision Requirements**:
- Millisecond timestamp precision is sufficient for 1-second bucket calculations
- Supports throughput rates from 10 to 10,000+ operations per second
- Monotonic timestamps provide nanosecond precision for detailed analysis if needed

**Analysis**: Throughput analysis enables:
- Rate-based performance comparison
- Scaling efficiency calculation
- Load-dependent performance analysis

---

## Resource Utilization Measurement

### Memory Utilization

**Implementation**: Instantaneous RSS (Resident Set Size) is captured per event using the `sysinfo` crate.

**Measurement Details**:
- **Metric**: `memory_rss_bytes: u64` - Resident Set Size in bytes
- **Sampling**: Per-event sampling (captured for each cryptographic operation)
- **Analysis**: Direct analysis of RSS values (mean, max, min, percentiles)
- **Units**: Bytes (also reported in MB for readability)

**Analysis Capabilities**:
- Mean and maximum memory usage per algorithm
- Memory efficiency comparison across environments
- Per-algorithm memory statistics
- Memory overhead analysis for horizontal scaling

**Limitations**: 
- Instantaneous RSS may vary slightly between samples
- Memory measurements reflect process memory, not system-wide memory
- Container overhead (Minikube, GCP) may affect absolute values but relative comparisons remain valid

### CPU Utilization

**Implementation**: Cumulative CPU time is captured per event by reading `/proc/self/stat` on Linux systems.

**Measurement Details**:
- **Metric**: `cpu_user_seconds: f64` - Cumulative CPU time (user + system) in seconds
- **Source**: `/proc/self/stat` fields 14 (utime) and 15 (stime) in clock ticks
- **Conversion**: Clock ticks converted to seconds (dividing by `sysconf(_SC_CLK_TCK)`, typically 100)
- **Sampling**: Per-event sampling (cumulative since process start)

**CPU Utilization Calculation**:
CPU utilization requires delta calculation between consecutive events:
```
cpu_utilization = (cpu_delta) / (time_delta)
```
where:
- `cpu_delta` = difference in cumulative CPU time between events
- `time_delta` = difference in wall-clock time between events

**Analysis Capabilities**:
- Mean and maximum CPU utilization per algorithm
- CPU efficiency comparison (operations per CPU-second)
- Per-algorithm CPU statistics
- CPU utilization analysis for resource efficiency claims

**Limitations**:
- Very fast operations (<1ms) may have zero or near-zero CPU deltas
- CPU sampling resolution limited by system clock ticks (typically 10ms)
- Cumulative CPU time enables accurate utilization calculation for operations >10ms
- CPU utilization analysis may be limited for sub-millisecond operations

**Edge Cases Handled**:
- Zero or negative deltas (system clock adjustment, skipped)
- Zero time deltas (concurrent events, skipped)
- All zeros (documented limitation in summary)

---

## Timestamp Precision

### Timestamp Formats

Event timestamps are captured in two formats:

1. **ISO 8601 Timestamp** (`timestamp_utc_iso`):
   - Format: UTC timestamp in ISO 8601 format
   - Precision: Millisecond precision
   - Use: Throughput calculation (1-second buckets)
   - Example: `"2025-01-01T00:00:00.123Z"`

2. **Monotonic Timestamp** (`timestamp_monotonic_ns`):
   - Format: Monotonic timestamp in nanoseconds
   - Precision: Nanosecond precision enables accurate measurement of sub-microsecond operations, which is critical for comparing fast cryptographic algorithms. The framework uses Rust's `Instant::now()` which provides nanosecond precision on modern systems.

**Throughput Calculation**: Millisecond precision is sufficient for 1-second bucket calculations, enabling accurate throughput analysis across environments.

**Resource Utilization**: CPU and memory metrics enable resource efficiency analysis.

---

## Summary

This methodology document describes the complete measurement approach for the quantum-resilient cryptographic benchmarking framework:

- **Latency**: Nanosecond precision enables accurate measurement of sub-microsecond operations
- **Throughput**: Millisecond timestamp precision sufficient for 1-second bucket calculations
- **Memory**: Instantaneous RSS captured per event, enabling memory efficiency analysis
- **CPU**: Cumulative CPU time captured per event, enabling CPU utilization analysis via delta calculation
- **Queue Delay**: Nanosecond precision measurement with microsecond storage for queuing overhead analysis

All measurements are designed to support statistically rigorous performance comparisons across native, containerized (Minikube), and cloud (GCP) deployment environments.
