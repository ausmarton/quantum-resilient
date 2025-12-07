#!/usr/bin/env bash
# =============================================================================
# check_system_load.sh - Check system load before running benchmarks
#
# Provides recommendations on whether it's safe to run benchmarks based on
# current system load.
#
# Usage:
#   ./scripts/check_system_load.sh [--warn-threshold N] [--fail-threshold N]
# =============================================================================

set -euo pipefail

WARN_THRESHOLD=1.0  # Warn if load average > 1.0 per CPU
FAIL_THRESHOLD=2.0  # Fail if load average > 2.0 per CPU

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Check system load and provide recommendations for running benchmarks.

OPTIONS:
    --warn-threshold N    Warn if load > N per CPU (default: 1.0)
    --fail-threshold N    Fail if load > N per CPU (default: 2.0)
    -h, --help           Show this help message
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --warn-threshold)
            WARN_THRESHOLD="$2"
            shift 2
            ;;
        --fail-threshold)
            FAIL_THRESHOLD="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

log_info "Checking system load..."
echo ""

# Get CPU count
CPU_COUNT=$(nproc 2>/dev/null || echo "1")

# Get load average (1 minute)
LOAD_1MIN=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')

# Calculate load per CPU (format to 2 decimal places)
LOAD_PER_CPU=$(echo "scale=2; $LOAD_1MIN / $CPU_COUNT" | bc -l)

# Get memory usage
MEM_TOTAL=$(free -m | awk '/^Mem:/ {print $2}')
MEM_USED=$(free -m | awk '/^Mem:/ {print $3}')
MEM_PERCENT=$(echo "scale=1; $MEM_USED * 100 / $MEM_TOTAL" | bc -l)

# Get top CPU-consuming processes
TOP_PROCESSES=$(ps aux --sort=-%cpu | head -6 | tail -5)

echo "System Information:"
echo "  CPUs: $CPU_COUNT"
echo "  Load average (1min): $LOAD_1MIN"
echo "  Load per CPU: $(printf "%.2f" $LOAD_PER_CPU)"
echo "  Memory: ${MEM_USED}MB / ${MEM_TOTAL}MB (${MEM_PERCENT}%)"
echo ""

# Check load (compare as floats)
if (( $(echo "$LOAD_PER_CPU > $FAIL_THRESHOLD" | bc -l) )); then
    log_error "System load is HIGH ($(printf "%.2f" $LOAD_PER_CPU) per CPU)"
    echo ""
    echo "Top CPU-consuming processes:"
    echo "$TOP_PROCESSES"
    echo ""
    log_error "NOT RECOMMENDED to run benchmarks now."
    log_info "Consider:"
    echo "  - Closing heavy applications (browser, IDE)"
    echo "  - Waiting for system load to decrease"
    echo "  - Using GCP mode instead (isolated from local load)"
    exit 1
elif (( $(echo "$LOAD_PER_CPU > $WARN_THRESHOLD" | bc -l) )); then
    log_warn "System load is MODERATE ($(printf "%.2f" $LOAD_PER_CPU) per CPU)"
    echo ""
    echo "Top CPU-consuming processes:"
    echo "$TOP_PROCESSES"
    echo ""
    log_warn "Benchmarks may be affected by system load."
    log_info "Recommendations:"
    echo "  - Close unnecessary applications"
    echo "  - Monitor system load during runs"
    echo "  - Consider using GCP mode for better isolation"
    exit 0
else
    log_success "System load is LOW ($(printf "%.2f" $LOAD_PER_CPU) per CPU)"
    echo ""
    if (( $(echo "$MEM_PERCENT > 80" | bc -l) )); then
        log_warn "Memory usage is high (${MEM_PERCENT}%)"
        echo "  Consider closing applications to free memory"
    else
        log_success "Memory usage is acceptable (${MEM_PERCENT}%)"
    fi
    echo ""
    log_success "System is ready for benchmarks"
    exit 0
fi

