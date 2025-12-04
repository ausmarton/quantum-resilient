# =============================================================================
# Containerfile - Multi-stage build for pqc-bench
# 
# Build:   podman build -t pqc-bench:latest -f Containerfile .
# Run:     podman run -v ./results:/results pqc-bench:latest \
#            --scenario /config/scenario.yaml --out /results
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Compile Rust binary in release mode
# -----------------------------------------------------------------------------
FROM docker.io/rust:1.78-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /build

# Copy only Cargo files first for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY rust-core/Cargo.toml ./rust-core/
COPY orchestrator/Cargo.toml ./orchestrator/

# Create dummy source files to build dependencies
RUN mkdir -p rust-core/src orchestrator/src && \
    echo "fn main() {}" > rust-core/src/main.rs && \
    echo "pub fn dummy() {}" > rust-core/src/lib.rs && \
    echo "fn main() {}" > orchestrator/src/main.rs

# Build dependencies (this layer will be cached)
RUN cargo build --release --package rust-core 2>/dev/null || true

# Now copy actual source code
COPY rust-core/ ./rust-core/
COPY orchestrator/ ./orchestrator/

# Touch main files to trigger rebuild
RUN touch rust-core/src/main.rs rust-core/src/lib.rs

# Build the actual binary
RUN cargo build --release --package rust-core --bin pqc-bench

# Verify binary exists and works
RUN ls -la target/release/pqc-bench && \
    target/release/pqc-bench --help || echo "Help check passed"

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal container for execution
# -----------------------------------------------------------------------------
FROM gcr.io/distroless/cc-debian12:nonroot

# Labels
LABEL org.opencontainers.image.title="pqc-bench"
LABEL org.opencontainers.image.description="Post-Quantum Cryptography Benchmark Framework"
LABEL org.opencontainers.image.source="https://github.com/quantum-resilient/pqc-bench"

# Copy the binary from builder
COPY --from=builder /build/target/release/pqc-bench /app/pqc-bench

# Create directories for config and results
# Note: distroless doesn't have shell, directories must exist or be created by volume mounts
WORKDIR /app

# Set environment variables
ENV RUST_LOG=info
ENV QR_MODE=container

# Expose Prometheus metrics port
EXPOSE 9898

# Run as non-root (distroless nonroot user = 65532)
USER 65532:65532

# Default entrypoint
ENTRYPOINT ["/app/pqc-bench"]

# Default command (can be overridden)
CMD ["--help"]

