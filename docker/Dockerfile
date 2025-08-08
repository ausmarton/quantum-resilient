# Multi-stage build for Rust + Python framework
FROM rust:1.70 as rust-builder

# Install Python and build tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy Rust source
COPY src/rust_core /app/rust_core
WORKDIR /app/rust_core

# Build Rust library with Python bindings
RUN pip3 install maturin
RUN maturin build --release

# Python stage
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy Rust library from builder stage
COPY --from=rust-builder /app/rust_core/target/wheels/*.whl /tmp/

# Install Python dependencies
COPY requirements.txt /app/
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Install the Rust library
RUN pip install /tmp/*.whl

# Copy Python application
COPY src/python_orchestrator /app/python_orchestrator
COPY src/real_world /app/real_world
COPY src/reporting /app/reporting

# Set Python path
ENV PYTHONPATH=/app
ENV LD_LIBRARY_PATH=/usr/local/lib

# Create results directory
RUN mkdir -p /app/results

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app/src
ENV QUANTUMRESILIENT_VERSION=0.1.0

# Run the application
CMD ["python", "python_orchestrator/main.py"]
