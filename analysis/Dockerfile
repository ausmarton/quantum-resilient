# Quantum-Resilient Analysis Pipeline Container
# Python environment for analyzing benchmark results
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy analysis scripts and notebooks
COPY scripts /app/scripts
COPY notebooks /app/notebooks
COPY *.py /app/

# Note: orchestration/ directory is available via volume mount at /workspace/orchestration
# The container wrapper mounts the entire project root as /workspace

# Set Python path to include scripts directory
ENV PYTHONPATH=/app:/app/scripts

# Default entrypoint (can be overridden)
ENTRYPOINT ["python3"]
