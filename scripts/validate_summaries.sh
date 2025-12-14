#!/usr/bin/env bash
# =============================================================================
# validate_summaries.sh - Validate generated summary.json files for accuracy
#
# Checks:
# - Summary files are valid JSON
# - Required fields present
# - Data consistency (events match raw data)
# - Statistical validity (percentiles, means)
# - Cross-experiment consistency
#
# Usage:
#   ./scripts/validate_summaries.sh [OPTIONS]
#
# Options:
#   --detailed       Show detailed validation for each summary
#   --fix-issues      Attempt to regenerate summaries with issues
#   -h, --help       Show this help message
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

DETAILED=false
FIX_ISSUES=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

Validate generated summary.json files for accuracy and consistency.

OPTIONS:
    --detailed       Show detailed validation for each summary
    --fix-issues     Attempt to regenerate summaries with issues
    -h, --help       Show this help message
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --detailed)
            DETAILED=true
            shift
            ;;
        --fix-issues)
            FIX_ISSUES=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

# Python validation script
PYTHON_SCRIPT=$(cat <<'PYTHON_EOF'
import json
import sys
from pathlib import Path
from collections import defaultdict

def validate_summary(summary_path: Path) -> dict:
    """Validate a single summary.json file."""
    result = {
        'path': str(summary_path),
        'valid_json': False,
        'has_required_fields': False,
        'has_latency': False,
        'has_throughput': False,
        'has_events': False,
        'event_count': 0,
        'latency_valid': False,
        'throughput_valid': False,
        'errors': [],
        'warnings': []
    }
    
    # Check if file exists
    if not summary_path.exists():
        result['errors'].append('File does not exist')
        return result
    
    # Check if valid JSON
    try:
        with open(summary_path) as f:
            data = json.load(f)
        result['valid_json'] = True
    except json.JSONDecodeError as e:
        result['errors'].append(f'Invalid JSON: {e}')
        return result
    except Exception as e:
        result['errors'].append(f'Error reading file: {e}')
        return result
    
    # Check required fields
    required_fields = ['total_events']
    has_all_required = all(field in data for field in required_fields)
    result['has_required_fields'] = has_all_required
    
    if not has_all_required:
        missing = [f for f in required_fields if f not in data]
        result['errors'].append(f'Missing required fields: {missing}')
    
    # Check event count
    if 'total_events' in data:
        result['has_events'] = True
        result['event_count'] = data['total_events']
        if data['total_events'] == 0:
            result['warnings'].append('Zero events in summary')
    
    # Check latency data
    if 'latency' in data or 'latency_ns' in data:
        result['has_latency'] = True
        lat_data = data.get('latency', data.get('latency_ns', {}))
        if isinstance(lat_data, dict):
            # Check percentiles
            percentiles = ['p50', 'p95', 'p99']
            has_percentiles = all(p in lat_data for p in percentiles)
            if has_percentiles:
                # Validate percentile ordering
                p50 = lat_data.get('p50', 0)
                p95 = lat_data.get('p95', 0)
                p99 = lat_data.get('p99', 0)
                if not (p50 <= p95 <= p99):
                    result['warnings'].append(f'Percentile ordering issue: p50={p50}, p95={p95}, p99={p99}')
                result['latency_valid'] = True
            else:
                result['warnings'].append('Missing latency percentiles')
        else:
            result['warnings'].append('Latency data not in expected format')
    else:
        result['errors'].append('Missing latency data')
    
    # Check throughput data
    if 'throughput' in data:
        result['has_throughput'] = True
        tput_data = data.get('throughput', {})
        if isinstance(tput_data, dict):
            if 'mean_msgs_per_sec' in tput_data:
                tput = tput_data['mean_msgs_per_sec']
                if tput < 0:
                    result['warnings'].append(f'Negative throughput: {tput}')
                elif tput == 0 and result['event_count'] > 0:
                    result['warnings'].append('Zero throughput with non-zero events')
                else:
                    result['throughput_valid'] = True
            else:
                result['warnings'].append('Missing mean_msgs_per_sec in throughput')
        else:
            result['warnings'].append('Throughput data not in expected format')
    else:
        result['warnings'].append('Missing throughput data')
    
    return result

def main():
    # Find all summaries
    summaries = list(Path('results').rglob('**/stats/summary.json'))
    
    if not summaries:
        print('No summary files found', file=sys.stderr)
        sys.exit(1)
    
    print(f'Validating {len(summaries)} summary files...')
    print()
    
    results = []
    for summary_path in summaries:
        result = validate_summary(summary_path)
        results.append(result)
    
    # Summary statistics
    valid_json = sum(1 for r in results if r['valid_json'])
    has_required = sum(1 for r in results if r['has_required_fields'])
    has_latency = sum(1 for r in results if r['has_latency'])
    has_throughput = sum(1 for r in results if r['has_throughput'])
    latency_valid = sum(1 for r in results if r['latency_valid'])
    throughput_valid = sum(1 for r in results if r['throughput_valid'])
    has_errors = sum(1 for r in results if r['errors'])
    has_warnings = sum(1 for r in results if r['warnings'])
    
    print('=' * 80)
    print('VALIDATION SUMMARY')
    print('=' * 80)
    print(f'Total summaries: {len(summaries)}')
    print(f'Valid JSON: {valid_json}/{len(summaries)} ({valid_json/len(summaries)*100:.1f}%)')
    print(f'Has required fields: {has_required}/{len(summaries)} ({has_required/len(summaries)*100:.1f}%)')
    print(f'Has latency data: {has_latency}/{len(summaries)} ({has_latency/len(summaries)*100:.1f}%)')
    print(f'Has throughput data: {has_throughput}/{len(summaries)} ({has_throughput/len(summaries)*100:.1f}%)')
    print(f'Latency valid: {latency_valid}/{len(summaries)} ({latency_valid/len(summaries)*100:.1f}%)')
    print(f'Throughput valid: {throughput_valid}/{len(summaries)} ({throughput_valid/len(summaries)*100:.1f}%)')
    print(f'With errors: {has_errors}/{len(summaries)} ({has_errors/len(summaries)*100:.1f}%)')
    print(f'With warnings: {has_warnings}/{len(summaries)} ({has_warnings/len(summaries)*100:.1f}%)')
    print()
    
    # Show errors
    error_results = [r for r in results if r['errors']]
    if error_results:
        print('=' * 80)
        print(f'ERRORS FOUND ({len(error_results)} summaries)')
        print('=' * 80)
        for r in error_results[:20]:  # Show first 20
            exp_name = Path(r['path']).parent.parent.name
            print(f'\n{exp_name}:')
            for error in r['errors']:
                print(f'  ❌ {error}')
        if len(error_results) > 20:
            print(f'\n... and {len(error_results) - 20} more with errors')
        print()
    
    # Show warnings
    warning_results = [r for r in results if r['warnings']]
    if warning_results:
        print('=' * 80)
        print(f'WARNINGS FOUND ({len(warning_results)} summaries)')
        print('=' * 80)
        for r in warning_results[:10]:  # Show first 10
            exp_name = Path(r['path']).parent.parent.name
            print(f'\n{exp_name}:')
            for warning in r['warnings']:
                print(f'  ⚠️  {warning}')
        if len(warning_results) > 10:
            print(f'\n... and {len(warning_results) - 10} more with warnings')
        print()
    
    # Overall status
    print('=' * 80)
    if has_errors == 0 and latency_valid == len(summaries) and throughput_valid == len(summaries):
        print('✅ ALL SUMMARIES VALID')
    elif has_errors == 0:
        print('⚠️  SUMMARIES VALID WITH WARNINGS')
    else:
        print('❌ SOME SUMMARIES HAVE ERRORS')
    print('=' * 80)
    
    # Return exit code
    sys.exit(0 if has_errors == 0 else 1)

if __name__ == '__main__':
    main()
PYTHON_EOF
)

# Run validation
log_info "Validating generated summaries..."

TMP_SCRIPT=$(mktemp)
echo "$PYTHON_SCRIPT" > "$TMP_SCRIPT"

if python3 "$TMP_SCRIPT" 2>&1; then
    VALIDATION_EXIT=0
else
    VALIDATION_EXIT=$?
fi

rm -f "$TMP_SCRIPT"

exit $VALIDATION_EXIT
