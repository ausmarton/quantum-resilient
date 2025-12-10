#!/usr/bin/env bash
# =============================================================================
# scripts/lib/common.sh - Common utilities for all experiment scripts
#
# Provides shared logging, color output, and common helper functions.
# This is the first library extracted from duplicate code across run_local.sh,
# run_minikube.sh, and deploy_gcp.sh.
#
# Usage:
#   source "$SCRIPT_DIR/scripts/lib/common.sh"
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# Logging Functions
# -----------------------------------------------------------------------------

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

log_step() {
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

log_run() {
    echo -e "${CYAN}[RUN $1/$2]${NC} $3"
}

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get absolute path
abs_path() {
    local path="$1"
    if [[ -d "$path" ]]; then
        cd "$path" && pwd
    else
        echo "$(cd "$(dirname "$path")" && pwd)/$(basename "$path")"
    fi
}

# Sanitize string for use in filenames
sanitize_filename() {
    echo "$1" | sed 's/[^a-zA-Z0-9._-]/_/g'
}

# Sanitize string for Kubernetes resource names (RFC 1123)
sanitize_k8s_name() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/_/-/g' | sed 's/[^a-z0-9-]/-/g' | sed 's/--*/-/g' | sed 's/^-\|-$//g'
}

