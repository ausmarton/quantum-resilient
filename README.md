# Quantum-Resilient Cryptography Benchmark Framework

A modular benchmark test framework for comparing Post-Quantum Cryptography (PQC) algorithms against classical cryptography in real-time streaming pipelines.

## Overview

This framework provides tools for:
- Benchmarking PQC vs classical cryptographic operations
- Simulating real-time streaming pipeline workloads
- Collecting and analyzing performance telemetry

## Project Structure

```
quantum-resilient/
├── Cargo.toml              # Workspace definition
├── rust-core/              # Core library and binary
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs          # Library entry point
│       ├── main.rs         # pqc-bench binary
│       ├── crypto_adapter.rs   # Cryptographic adapter trait
│       ├── pipeline.rs     # Streaming pipeline
│       ├── workload.rs     # Workload generator
│       └── telemetry.rs    # Metrics collection
├── Makefile
├── Dockerfile.podman
└── README.md
```

## Prerequisites

- Rust (stable toolchain, pinned via `.rust-toolchain.toml`)
- Make
- Podman (optional, for containerized builds)

## Quick Start

### Local Build and Run

1. **Build the project:**

```bash
make build
```

2. **Run the benchmark:**

```bash
make run
```

Expected output:
```
Starting PQC Benchmark Framework...
pipeline OK
```

3. **Run tests:**

```bash
make test
```

### Container Build and Run (Podman)

1. **Build the container:**

```bash
podman build -f Dockerfile.podman -t pqc-bench:latest .
```

Or using Make:
```bash
make container
```

2. **Run the container:**

```bash
podman run --rm pqc-bench:latest
```

Or using Make:
```bash
make container-run
```

## Development

### Available Make Targets

| Target | Description |
|--------|-------------|
| `build` | Build the project (debug) |
| `release` | Build the project (release) |
| `run` | Run pqc-bench binary |
| `test` | Run all tests |
| `fmt` | Format code |
| `clippy` | Run clippy lints |
| `check` | Run all checks (fmt, clippy, test) |
| `clean` | Clean build artifacts |
| `container` | Build Podman container |
| `container-run` | Run Podman container |

### Code Style

Format code before committing:
```bash
make fmt
```

Run lints:
```bash
make clippy
```

## Architecture

### Modules

- **crypto_adapter**: Defines the `CryptoAdapter` trait for unified cryptographic operations
- **pipeline**: Streaming pipeline infrastructure for benchmark execution
- **workload**: Workload generators for simulating real-world patterns
- **telemetry**: Metrics collection and performance analysis

## License

[To be determined]

