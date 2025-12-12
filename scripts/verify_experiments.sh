#!/usr/bin/env bash
# =============================================================================
# verify_experiments.sh - Verify experiment results and analysis outputs
#
# Checks that:
# 1. Experiments completed successfully (per index.json)
# 2. Output files exist for successful experiments
# 3. Analysis outputs were generated
# 4. Figures and statistics are present
#
# Usage:
#   ./scripts/verify_experiments.sh [results_dir]
#
# Default: final-results (unified for both smoke-test and full-scale)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

# Determine results directory
if [[ $# -ge 1 ]]; then
    RESULTS_DIR="$1"
else
    # Default to unified final-results directory
    if [[ -d "$SCRIPT_DIR/final-results" ]]; then
        RESULTS_DIR="$SCRIPT_DIR/final-results"
    else
        log_error "No results directory found. Specify one or run experiments first."
        exit 1
    fi
fi

RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"

if [[ ! -d "$RESULTS_DIR" ]]; then
    log_error "Results directory not found: $RESULTS_DIR"
    exit 1
fi

log_info "Verifying experiments in: $RESULTS_DIR"
echo ""

# =============================================================================
# 1. Check index.json
# =============================================================================
log_info "1. Checking experiment index..."

INDEX_FILE="$RESULTS_DIR/index.json"
if [[ ! -f "$INDEX_FILE" ]]; then
    log_error "index.json not found!"
    exit 1
fi

log_success "index.json found"

# Parse index
# Note: These are experiment counts (unique configs), not scenario counts (all runs)
# Each experiment handles multiple runs internally
TOTAL_EXPERIMENTS=$(python3 -c "import json; print(json.load(open('$INDEX_FILE'))['total_scenarios'])" 2>/dev/null || echo "?")
COMPLETED=$(python3 -c "import json; print(json.load(open('$INDEX_FILE'))['completed_scenarios'])" 2>/dev/null || echo "?")
FAILED=$(python3 -c "import json; print(json.load(open('$INDEX_FILE'))['failed_scenarios'])" 2>/dev/null || echo "?")

echo "  Total experiments: $TOTAL_EXPERIMENTS"
echo "  Completed: $COMPLETED"
echo "  Failed: $FAILED"
echo ""

# Count by environment and status
log_info "2. Checking experiment status by environment..."

python3 <<EOF
import json
from collections import defaultdict

with open('$INDEX_FILE') as f:
    data = json.load(f)

by_env_status = defaultdict(lambda: {'success': 0, 'failed': 0, 'cached': 0, 'dry_run': 0})

for exp in data.get('experiments', []):
    env = exp.get('environment', 'unknown')
    status = exp.get('status', 'unknown')
    by_env_status[env][status] = by_env_status[env][status] + 1

print("  Status breakdown:")
for env in sorted(by_env_status.keys()):
    stats = by_env_status[env]
    total = sum(stats.values())
    success = stats.get('success', 0)
    failed = stats.get('failed', 0)
    cached = stats.get('cached', 0)
    
    if failed > 0:
        print(f"    {env}: {success} success, {failed} failed, {cached} cached (total: {total})")
    else:
        print(f"    {env}: {success} success, {cached} cached (total: {total})")
EOF

echo ""

# =============================================================================
# 3. Verify output files for successful experiments
# =============================================================================
log_info "3. Verifying output files for successful experiments..."

SUCCESS_COUNT=0
MISSING_COUNT=0

python3 <<EOF
import json
import os
from pathlib import Path

with open('$INDEX_FILE') as f:
    data = json.load(f)

success_exps = [e for e in data.get('experiments', []) if e.get('status') == 'success']
print(f"  Checking {len(success_exps)} successful experiments...")

missing = []
for exp in success_exps:
    output_dir = exp.get('output_dir', '')
    if not output_dir:
        continue
    
    # Check for key files
    merged_file = os.path.join(output_dir, 'merged', 'merged.jsonl')
    stats_file = os.path.join(output_dir, 'stats', 'summary.json')
    
    has_merged = os.path.exists(merged_file) and os.path.getsize(merged_file) > 0
    has_stats = os.path.exists(stats_file)
    
    if not has_merged and not has_stats:
        missing.append({
            'id': exp.get('scenario_id', 'unknown'),
            'env': exp.get('environment', 'unknown'),
            'dir': output_dir
        })

if missing:
    print(f"  ⚠️  {len(missing)} experiments missing output files:")
    for m in missing:
        print(f"      - {m['id']} ({m['env']})")
        print(f"        Expected: {m['dir']}/merged/merged.jsonl or stats/summary.json")
else:
    print(f"  ✓ All successful experiments have output files")
EOF

echo ""

# =============================================================================
# 4. Check analysis outputs
# =============================================================================
log_info "4. Checking analysis outputs..."

ANALYSIS_OK=true

# Required files
REQUIRED_FILES=(
    "aggregated_stats.json"
    "hypothesis_tests.json"
)

OPTIONAL_FILES=(
    "aggregated_stats.csv"
    "hypothesis_table.csv"
    "hypothesis_interpretation.txt"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$RESULTS_DIR/$file" ]]; then
        SIZE=$(du -h "$RESULTS_DIR/$file" | cut -f1)
        log_success "$file ($SIZE)"
    else
        log_error "$file missing!"
        ANALYSIS_OK=false
    fi
done

for file in "${OPTIONAL_FILES[@]}"; do
    if [[ -f "$RESULTS_DIR/$file" ]]; then
        SIZE=$(du -h "$RESULTS_DIR/$file" | cut -f1)
        echo "  ✓ $file ($SIZE)"
    else
        echo "  - $file (optional, not found)"
    fi
done

echo ""

# =============================================================================
# 5. Check figures
# =============================================================================
log_info "5. Checking generated figures..."

FIGURES_DIR="$RESULTS_DIR/figures"
if [[ -d "$FIGURES_DIR" ]]; then
    FIGURE_COUNT=$(find "$FIGURES_DIR" -name "*.png" -type f 2>/dev/null | wc -l)
    if [[ $FIGURE_COUNT -gt 0 ]]; then
        log_success "Found $FIGURE_COUNT figure(s) in figures/"
        
        # List key figures
        KEY_FIGURES=(
            "combined_ecdf.png"
            "classical_vs_pqc.png"
            "scaling_curves.png"
        )
        
        echo "  Key figures:"
        for fig in "${KEY_FIGURES[@]}"; do
            if [[ -f "$FIGURES_DIR/$fig" ]]; then
                SIZE=$(du -h "$FIGURES_DIR/$fig" | cut -f1)
                echo "    ✓ $fig ($SIZE)"
            else
                echo "    - $fig (not found)"
            fi
        done
    else
        log_warn "Figures directory exists but no PNG files found"
    fi
else
    log_warn "Figures directory not found"
fi

echo ""

# =============================================================================
# 6. Check hypothesis test results
# =============================================================================
if [[ -f "$RESULTS_DIR/hypothesis_tests.json" ]]; then
    log_info "6. Hypothesis test summary..."
    
    python3 <<EOF
import json

try:
    with open('$RESULTS_DIR/hypothesis_tests.json') as f:
        data = json.load(f)
    
    total = data.get('total_comparisons', 0)
    significant = data.get('significant_comparisons', 0)
    alpha = data.get('alpha', 0.05)
    
    print(f"  Total comparisons: {total}")
    print(f"  Significant (α={alpha}): {significant}")
    
    if total > 0:
        pct = (significant / total) * 100
        print(f"  Percentage: {pct:.1f}%")
    
    # Effect sizes
    effect_sizes = data.get('summary', {}).get('effect_sizes', {})
    if effect_sizes:
        print(f"  Effect sizes:")
        print(f"    Large: {effect_sizes.get('large', 0)}")
        print(f"    Medium: {effect_sizes.get('medium', 0)}")
        print(f"    Small: {effect_sizes.get('small', 0)}")
        print(f"    Negligible: {effect_sizes.get('negligible', 0)}")
except Exception as e:
    print(f"  ⚠️  Could not parse hypothesis_tests.json: {e}")
EOF
    echo ""
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  VERIFICATION SUMMARY${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Overall status
if [[ "$FAILED" == "0" ]] || [[ "$FAILED" == "?" ]]; then
    if [[ "$COMPLETED" != "0" ]] && [[ "$COMPLETED" != "?" ]]; then
        log_success "All experiments completed successfully!"
    fi
else
    log_warn "$FAILED experiment(s) failed (see index.json for details)"
fi

if [[ "$ANALYSIS_OK" == "true" ]]; then
    log_success "Analysis outputs are complete"
else
    log_warn "Some analysis outputs are missing"
fi

echo ""
log_info "Results location: $RESULTS_DIR"
echo ""

# Quick access commands
echo "To inspect results:"
echo "  cat $RESULTS_DIR/index.json | python3 -m json.tool"
echo "  cat $RESULTS_DIR/hypothesis_tests.json | python3 -m json.tool"
echo "  ls -lh $RESULTS_DIR/figures/"
echo ""

exit 0

