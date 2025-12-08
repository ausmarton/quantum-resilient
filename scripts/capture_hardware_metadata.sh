#!/usr/bin/env bash
# =============================================================================
# capture_hardware_metadata.sh - Capture hardware metadata for native runs
#
# Creates hardware_metadata.json file with system specifications.
# This enables hardware-aware analysis and comparison validation.
#
# Usage:
#   ./capture_hardware_metadata.sh <output_dir>
# =============================================================================

set -euo pipefail

OUTPUT_DIR="${1:-}"

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 <output_dir>"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Capture hardware metadata
CPU_MODEL=$(grep 'model name' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs || echo "unknown")
CPU_COUNT=$(nproc || echo "unknown")
MEMORY_TOTAL_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}' || echo "unknown")
KERNEL_VERSION=$(uname -r || echo "unknown")
ARCH=$(uname -m || echo "unknown")
HOSTNAME=$(hostname || echo "unknown")

# Create metadata JSON
cat > "$OUTPUT_DIR/hardware_metadata.json" << EOF
{
  "type": "native",
  "cpu_model": "$CPU_MODEL",
  "cpu_count": $CPU_COUNT,
  "memory_total_kb": $MEMORY_TOTAL_KB,
  "memory_total_gb": $(echo "scale=2; $MEMORY_TOTAL_KB / 1024 / 1024" | bc 2>/dev/null || echo "unknown"),
  "kernel_version": "$KERNEL_VERSION",
  "arch": "$ARCH",
  "hostname": "$HOSTNAME",
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "Hardware metadata captured: $OUTPUT_DIR/hardware_metadata.json"
cat "$OUTPUT_DIR/hardware_metadata.json"

