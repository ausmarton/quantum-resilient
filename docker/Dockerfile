FROM python:3.11-slim
LABEL org.opencontainers.image.title="pqc-benchmarking"
LABEL org.opencontainers.image.description="PQC benchmarking orchestrator with Rust core (PyO3)"
LABEL org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYO3_PYTHON=python3

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential pkg-config \
    gcc g++ libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Install rustup + stable toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable && \
    . /root/.cargo/env && rustc --version && cargo --version
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy repository
COPY . /app

# Build and install PyO3 extension and orchestrator
RUN python3 -m pip install --upgrade pip maturin && \
    maturin build --release -m src/rust_core/Cargo.toml --interpreter python3 && \
    python3 -m pip install --no-cache-dir /app/src/rust_core/target/wheels/*.whl && \
    python3 -m pip install -e src/python_orchestrator -c src/python_orchestrator/constraints.txt

# Add a small entrypoint to tee logs to /results for easy retrieval
RUN printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -uo pipefail' \
  'mkdir -p /results || true' \
  'echo "[entrypoint] PWD=$(pwd)"; echo "[entrypoint] Python: $(python3 -V)";' \
  'echo "[entrypoint] Args: $*"' \
  'echo "[entrypoint] Listing /app/configs:"; ls -la /app/configs || true' \
  'echo "[entrypoint] Starting orchestrator..."' \
  'python3 -m python_orchestrator.cli "$@" 2>&1 | tee /results/orchestrator.log' \
  'ec=${PIPESTATUS[0]}' \
  'echo "[entrypoint] orchestrator exited with code ${ec}"' \
  'exit ${ec}' \
  > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Default entrypoint; pass --config via k8s args
ENTRYPOINT ["/app/entrypoint.sh"]


