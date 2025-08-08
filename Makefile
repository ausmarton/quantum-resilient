# Quantum-Resilient PQC Benchmark Framework
# Makefile for building, testing, and running the framework

.PHONY: build run test clean fmt clippy check all container

# Default target
all: build

# Build the project
build:
	cargo build

# Build in release mode
release:
	cargo build --release

# Run the pqc-bench binary
run:
	cargo run --bin pqc-bench

# Run in release mode
run-release:
	cargo run --release --bin pqc-bench

# Run all tests
test:
	cargo test

# Run tests with output
test-verbose:
	cargo test -- --nocapture

# Format code
fmt:
	cargo fmt

# Check formatting
fmt-check:
	cargo fmt -- --check

# Run clippy lints
clippy:
	cargo clippy -- -D warnings

# Run all checks (format, clippy, test)
check: fmt-check clippy test

# Clean build artifacts
clean:
	cargo clean

# Build container with Podman
container:
	podman build -f Dockerfile.podman -t pqc-bench:latest .

# Run container with Podman
container-run:
	podman run --rm pqc-bench:latest

# Help target
help:
	@echo "Available targets:"
	@echo "  build         - Build the project (debug)"
	@echo "  release       - Build the project (release)"
	@echo "  run           - Run pqc-bench binary"
	@echo "  run-release   - Run pqc-bench binary (release)"
	@echo "  test          - Run all tests"
	@echo "  test-verbose  - Run tests with output"
	@echo "  fmt           - Format code"
	@echo "  fmt-check     - Check code formatting"
	@echo "  clippy        - Run clippy lints"
	@echo "  check         - Run all checks"
	@echo "  clean         - Clean build artifacts"
	@echo "  container     - Build Podman container"
	@echo "  container-run - Run Podman container"

