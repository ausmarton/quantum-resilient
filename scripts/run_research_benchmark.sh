#!/usr/bin/env bash
set -euo pipefail

usage() {
	echo "Usage: $0 [--repetitions <N>] [--output-dir <path>]" >&2
	echo "  --repetitions: Number of times to run each comparison (default: 30)" >&2
	echo "  --output-dir:  Output directory for results (default: ./results)" >&2
	exit 1
}

REPETITIONS=30
OUTPUT_DIR="./results"

while [[ $# -gt 0 ]]; do
	case "$1" in
		-r|--repetitions) REPETITIONS="${2:-30}"; shift 2;;
		-o|--output-dir) OUTPUT_DIR="${2:-}"; shift 2;;
		-h|--help) usage;;
		*) echo "Unknown arg: $1" >&2; usage;;
	esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

echo "==========================================="
echo "PQC Research Benchmark - Full Comparison"
echo "==========================================="
echo "Repetitions: ${REPETITIONS}"
echo "Output: ${OUTPUT_DIR}"
echo "==========================================="
echo ""

# Check prerequisites
if ! command -v cargo >/dev/null 2>&1; then
	echo "ERROR: cargo not found. Please install Rust toolchain:" >&2
	echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh" >&2
	exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build release binary
echo "[build] Compiling Rust benchmarks..."
cd "${ROOT}"
cargo build --release --manifest-path src/rust_core/Cargo.toml --bin run_comparisons

# Clean previous metrics
METRICS_FILE="${OUTPUT_DIR}/metrics.jsonl"
: > "${METRICS_FILE}"  # Truncate file

echo "[benchmark] Running ${REPETITIONS} repetitions..."
echo ""

# Run the comparison binary N times to get adequate sample size
TEMP_RESULTS=$(mktemp -d)
for i in $(seq 1 "${REPETITIONS}"); do
	printf "\r[%3d/%3d] Running comparison suite..." "$i" "${REPETITIONS}"
	
	# Run in temp directory
	(cd "${TEMP_RESULTS}" && "${ROOT}/src/rust_core/target/release/run_comparisons" 2>/dev/null)
	
	# Append to our accumulated file
	if [[ -f "${TEMP_RESULTS}/results/metrics.jsonl" ]]; then
		cat "${TEMP_RESULTS}/results/metrics.jsonl" >> "${METRICS_FILE}"
		rm -f "${TEMP_RESULTS}/results/metrics.jsonl"
	fi
done
rm -rf "${TEMP_RESULTS}"

echo ""
echo ""
echo "[analysis] Generating reports..."

# If venv exists, use it to generate reports
if [[ -d "${ROOT}/.venv" ]]; then
	source "${ROOT}/.venv/bin/activate"
	
	# Create a temporary config that points to our output dir
	TEMP_CONFIG=$(mktemp)
	cat > "${TEMP_CONFIG}" <<EOF
benchmark:
  comparison_mode: "pqc_vs_classical"
  repetitions: ${REPETITIONS}
  warmup_seconds: 0
  workload:
    type: "kem_encapsulate"
    concurrency: 1
    chunk_size_bytes: 16384
  algorithms:
    pqc:
      - "Kyber512"
      - "Dilithium2"
    classical:
      key_exchange:
        - "RSA-2048"
      signature:
        - "ECDSA-P256"
    symmetric:
      - "AES-GCM-256"
output:
  directory: "${OUTPUT_DIR}"
  format: "csv"
EOF
	
	pqc-orchestrator --config "${TEMP_CONFIG}" 2>/dev/null || true
	rm -f "${TEMP_CONFIG}"
else
	echo "[analysis] No venv found. Run 'bash scripts/run_local_direct.sh' first to set up environment."
fi

# Count events per algorithm
echo ""
echo "==========================================="
echo "Benchmark Complete!"
echo "==========================================="
echo ""
echo "Events captured by algorithm:"
if command -v jq >/dev/null 2>&1; then
	cat "${METRICS_FILE}" | jq -r '.algorithm' | sort | uniq -c
else
	echo "  (install jq to see breakdown)"
	wc -l < "${METRICS_FILE}"
	echo "  total events"
fi

echo ""
echo "Total events: $(wc -l < "${METRICS_FILE}")"
echo ""
echo "Results directory: ${OUTPUT_DIR}"
echo ""
echo "Available outputs:"
[[ -f "${OUTPUT_DIR}/metrics.jsonl" ]] && echo "  ✓ metrics.jsonl ($(wc -l < "${OUTPUT_DIR}/metrics.jsonl") lines)"
[[ -f "${OUTPUT_DIR}/metrics.csv" ]] && echo "  ✓ metrics.csv"
[[ -f "${OUTPUT_DIR}/summary.csv" ]] && echo "  ✓ summary.csv"
[[ -f "${OUTPUT_DIR}/summary.json" ]] && echo "  ✓ summary.json"
[[ -f "${OUTPUT_DIR}/summary.md" ]] && echo "  ✓ summary.md"
[[ -d "${OUTPUT_DIR}/charts" ]] && echo "  ✓ charts/ ($(find "${OUTPUT_DIR}/charts" -name "*.png" 2>/dev/null | wc -l) charts)"
[[ -f "${OUTPUT_DIR}/analysis.ipynb" ]] && echo "  ✓ analysis.ipynb"
[[ -f "${OUTPUT_DIR}/report.zip" ]] && echo "  ✓ report.zip"

echo ""
echo "To view summary:"
echo "  cat ${OUTPUT_DIR}/summary.md"
echo ""

