#!/usr/bin/env bash
set -euo pipefail

usage() {
	echo "Usage: $0 [--config <experiment.yaml>] [--clean]" >&2
	echo "  --config: Path to experiment config (default: ./configs/default.yaml)" >&2
	echo "  --clean:  Remove venv and rebuild from scratch" >&2
	exit 1
}

CONFIG="./configs/default.yaml"
CLEAN=false

while [[ $# -gt 0 ]]; do
	case "$1" in
		-c|--config) CONFIG="${2:-}"; shift 2;;
		--clean) CLEAN=true; shift;;
		-h|--help) usage;;
		*) echo "Unknown arg: $1" >&2; usage;;
	esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
VENV_DIR="${ROOT}/.venv"

echo "==========================================="
echo "PQC Benchmark - Direct Local Run (Isolated)"
echo "==========================================="
echo "Root: ${ROOT}"
echo "Config: ${CONFIG}"
echo "Venv: ${VENV_DIR}"
echo "==========================================="

# Check prerequisites
if ! command -v python3 >/dev/null 2>&1; then
	echo "ERROR: python3 not found. Please install Python 3.11+" >&2
	exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
	echo "ERROR: cargo not found. Please install Rust toolchain:" >&2
	echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh" >&2
	exit 1
fi

# Check for C++ compiler (needed for pandas and other Python packages)
if ! command -v g++ >/dev/null 2>&1 && ! command -v c++ >/dev/null 2>&1; then
	echo "WARNING: C++ compiler not found. Installing from prebuilt wheels..." >&2
	echo "  If wheel installation fails, install g++:" >&2
	echo "  - Fedora/RHEL: sudo dnf install gcc-c++" >&2
	echo "  - Ubuntu/Debian: sudo apt install g++" >&2
	USE_BINARY_WHEELS=true
else
	USE_BINARY_WHEELS=false
fi

# Clean if requested
if [[ "${CLEAN}" == "true" ]]; then
	echo "[setup] Cleaning existing venv..."
	rm -rf "${VENV_DIR}"
fi

# Create Python virtual environment
if [[ ! -d "${VENV_DIR}" ]]; then
	echo "[setup] Creating Python virtual environment..."
	python3 -m venv "${VENV_DIR}"
	echo "[setup] Virtual environment created at ${VENV_DIR}"
else
	echo "[setup] Using existing virtual environment at ${VENV_DIR}"
fi

# Activate venv
echo "[setup] Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

# Verify we're in the venv
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
	echo "ERROR: Failed to activate virtual environment" >&2
	exit 1
fi
echo "[setup] ✓ Virtual environment active: ${VIRTUAL_ENV}"

# Upgrade pip and install build tools
echo "[setup] Installing/upgrading build tools..."
pip install --upgrade pip setuptools wheel
pip install maturin

# Build Rust core with PyO3 bindings
echo "[build] Building Rust core (this may take a few minutes)..."
cd "${ROOT}"

# Check Python version and set compatibility flag if needed
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[build] Python version: ${PYTHON_VERSION}"

# Try to build, with fallback for newer Python versions
if ! maturin develop --release -m src/rust_core/Cargo.toml; then
	echo "[build] Initial build failed, trying with ABI3 forward compatibility..."
	export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
	maturin develop --release -m src/rust_core/Cargo.toml
fi

# Install Python orchestrator
echo "[build] Installing Python orchestrator..."
pip install -e src/python_orchestrator

# Verify installations
echo "[verify] Verifying installations..."
if python3 -c "import pqc_core" 2>/dev/null; then
	echo "[verify] ✓ pqc_core (Rust PyO3) imported successfully"
else
	echo "[verify] ⚠ WARNING: pqc_core import failed (may still work)"
fi

if command -v pqc-orchestrator >/dev/null 2>&1; then
	echo "[verify] ✓ pqc-orchestrator command available"
else
	echo "[verify] ✗ ERROR: pqc-orchestrator command not found" >&2
	exit 1
fi

# Ensure results directory exists
RESULTS_DIR="${ROOT}/results"
mkdir -p "${RESULTS_DIR}"

# Run the benchmark
echo ""
echo "==========================================="
echo "Running Benchmark"
echo "==========================================="
echo "Config: ${CONFIG}"
echo "Output: ${RESULTS_DIR}"
echo "==========================================="
echo ""

if [[ ! -f "${CONFIG}" ]]; then
	echo "ERROR: Config file not found: ${CONFIG}" >&2
	exit 1
fi

# Run orchestrator
pqc-orchestrator --config "${CONFIG}"

echo ""
echo "==========================================="
echo "Benchmark Complete!"
echo "==========================================="
echo "Results directory: ${RESULTS_DIR}"
echo ""
echo "Available outputs:"
if [[ -f "${RESULTS_DIR}/metrics.jsonl" ]]; then
	echo "  ✓ metrics.jsonl ($(wc -l < "${RESULTS_DIR}/metrics.jsonl") lines)"
fi
if [[ -f "${RESULTS_DIR}/metrics.csv" ]]; then
	echo "  ✓ metrics.csv"
fi
if [[ -f "${RESULTS_DIR}/summary.csv" ]]; then
	echo "  ✓ summary.csv"
fi
if [[ -f "${RESULTS_DIR}/summary.json" ]]; then
	echo "  ✓ summary.json"
fi
if [[ -f "${RESULTS_DIR}/summary.md" ]]; then
	echo "  ✓ summary.md"
fi
if [[ -d "${RESULTS_DIR}/charts" ]]; then
	CHART_COUNT=$(find "${RESULTS_DIR}/charts" -name "*.png" 2>/dev/null | wc -l)
	echo "  ✓ charts/ (${CHART_COUNT} charts)"
fi
if [[ -f "${RESULTS_DIR}/analysis.ipynb" ]]; then
	echo "  ✓ analysis.ipynb"
fi
if [[ -f "${RESULTS_DIR}/environment.json" ]]; then
	echo "  ✓ environment.json"
fi
if [[ -f "${RESULTS_DIR}/report.zip" ]]; then
	echo "  ✓ report.zip"
fi

echo ""
echo "To view results:"
echo "  cat ${RESULTS_DIR}/summary.md"
echo "  cat ${RESULTS_DIR}/summary.json"
echo "  open ${RESULTS_DIR}/charts/"
echo ""
echo "To deactivate virtual environment:"
echo "  deactivate"
echo ""

