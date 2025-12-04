# Build stage
FROM rust:1.75-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    make \
    clang \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY rust-core ./rust-core
COPY orchestrator ./orchestrator

# Build the orchestrator
RUN cargo build --release -p orchestrator

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the binary
COPY --from=builder /app/target/release/qr-orchestrator /app/qr-orchestrator

# Create results directory
RUN mkdir -p /data/results

# Expose port
EXPOSE 7070

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7070/healthz || exit 1

# Run the orchestrator
CMD ["/app/qr-orchestrator"]

