# Quantum-Resilient Cryptography Benchmark Framework

A modular benchmark test framework for comparing Post-Quantum Cryptography (PQC) algorithms against classical cryptography in real-time streaming pipelines.

## Overview

This framework provides tools for:
- Benchmarking PQC vs classical cryptographic operations
- Simulating real-time streaming pipeline workloads
- Collecting and analyzing performance telemetry
- Scenario-based test configuration via YAML

## Project Structure

```
quantum-resilient/
├── Cargo.toml              # Workspace definition
├── rust-core/              # Core library and binary
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs              # Library entry point
│       ├── main.rs             # pqc-bench binary
│       ├── crypto_adapter/     # Cryptographic adapters
│       │   ├── mod.rs          # CryptoAdapter trait
│       │   └── noop_adapter.rs # NoOp baseline adapter
│       ├── pipeline.rs         # Streaming pipeline
│       ├── scenario.rs         # Scenario loading
│       ├── workload.rs         # Workload generator
│       └── telemetry.rs        # Metrics collection
├── scenarios/              # Benchmark scenario definitions
│   └── smoke_noop.yaml     # Basic NoOp smoke test
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

2. **Run the NoOp smoke test scenario:**

```bash
make run ARGS="--scenario scenarios/smoke_noop.yaml"
```

Or use the convenience target:

```bash
make smoke
```

Expected output:
```
Starting PQC Benchmark Framework...
Loaded scenario: smoke_noop
Using adapter: noop
Pipeline ready — running warm-up...
Pipeline OK (noop)
```

3. **Run tests:**

```bash
make test
```

### Running with Custom Scenarios

Create a YAML scenario file (see `scenarios/smoke_noop.yaml` for reference):

```yaml
id: my_scenario
description: "My custom benchmark scenario"

workload:
  msgs_per_sec: 100
  msg_size_bytes: 256
  duration_sec: 10

algorithm:
  adapter: noop  # Currently only "noop" is supported
```

Then run:

```bash
cargo run --bin pqc-bench -- --scenario path/to/my_scenario.yaml
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

2. **Run the container with scenario:**

```bash
podman run --rm -v ./scenarios:/app/scenarios:ro,Z pqc-bench:latest --scenario /app/scenarios/smoke_noop.yaml
```

> Note: The `:Z` flag enables SELinux relabeling for the volume (required on Fedora/RHEL systems).

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
| `run` | Run pqc-bench binary (use `ARGS` for options) |
| `smoke` | Run the smoke_noop scenario |
| `test` | Run all tests |
| `fmt` | Format code |
| `clippy` | Run clippy lints |
| `check` | Run all checks (fmt, clippy, test) |
| `clean` | Clean build artifacts |
| `container` | Build Podman container |
| `container-run` | Run Podman container with smoke test |

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
  - `NoOpCryptoAdapter`: Zero-cost baseline adapter for measuring pipeline overhead
- **pipeline**: Streaming pipeline infrastructure for benchmark execution
- **scenario**: YAML-based scenario configuration and loading
- **workload**: Workload generators for simulating real-world patterns
- **telemetry**: Metrics collection and performance analysis

### CryptoAdapter Trait

The `CryptoAdapter` trait provides a unified interface for both KEM (Key Encapsulation Mechanism) and digital signature operations:

```rust
pub trait CryptoAdapter {
    fn name(&self) -> &'static str;
    fn keygen(&self) -> Result<KeypairMeta, CryptoError>;
    fn encapsulate(&self, public_key: &[u8]) -> Result<(Vec<u8>, Vec<u8>), CryptoError>;
    fn decapsulate(&self, secret_key: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>, CryptoError>;
    fn sign(&self, secret_key: &[u8], msg: &[u8]) -> Result<Vec<u8>, CryptoError>;
    fn verify(&self, public_key: &[u8], msg: &[u8], sig: &[u8]) -> Result<bool, CryptoError>;
}
```

### Supported Adapters

| Adapter | Description | Status |
|---------|-------------|--------|
| `noop` | Zero-cost baseline | ✅ Implemented |
| `kyber` | NIST PQC KEM | 🔜 Planned |
| `dilithium` | NIST PQC Signature | 🔜 Planned |

## License

[To be determined]
