#!/usr/bin/env bash
# Quick validation script to verify experiment counting logic

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Validating Experiment Counting Logic ==="
echo ""

# Test 1: Check scenario generation
echo "1. Checking scenario generation..."
if [[ ! -f "generated-scenarios/manifest.json" ]]; then
    echo "   Generating scenarios in smoke test mode..."
    python3 orchestration/generate_scenarios.py \
        --matrix orchestration/experiment_matrix.yaml \
        --output generated-scenarios \
        --smoke-test
fi

# Test 2: Verify counting logic
echo ""
echo "2. Verifying experiment counts..."

python3 << 'EOF'
import json
from pathlib import Path

manifest_file = Path('generated-scenarios/manifest.json')
with open(manifest_file) as f:
    manifest = json.load(f)

scenarios = manifest.get('scenarios', [])
print(f"   Total scenarios: {len(scenarios)}")

# Count unique configurations (run_index=1 only)
seen_configs = set()
run1_scenarios = [s for s in scenarios if s.get('run_index', 1) == 1]

for s in run1_scenarios:
    config_key = (s['algorithm'], s['payload_size'], s['rate'], s.get('scaling_experiment', False))
    seen_configs.add(config_key)

baseline = sum(1 for s in run1_scenarios if not s.get('scaling_experiment', False))
scaling = sum(1 for s in run1_scenarios if s.get('scaling_experiment', False))

print(f"   Run-1 scenarios: {len(run1_scenarios)}")
print(f"   Unique experiment configurations: {len(seen_configs)}")
print(f"   - Baseline: {baseline}")
print(f"   - Scaling: {scaling}")
print()

# Test counting logic for different environments
replicas_smoke = [1]  # Smoke test uses replica=1 only
replicas_full = [1, 2, 4, 8]  # Full scale uses all replicas

print("   Smoke test mode (replicas=1):")
print(f"     Native: {baseline} experiments")
print(f"     Minikube/GCP: {baseline + scaling * len(replicas_smoke)} experiments")
print()

print("   Full scale mode (replicas=1,2,4,8):")
print(f"     Native: {baseline} experiments")
print(f"     Minikube/GCP: {baseline + scaling * len(replicas_full)} experiments")
EOF

# Test 3: Verify base experiment ID extraction
echo ""
echo "3. Testing base experiment ID extraction..."

python3 << 'EOF'
import re

def extract_base_experiment_id(scenario_id):
    """Match the bash function logic"""
    # Remove _run<N> pattern (where N is 1-9 or 10+)
    # Pattern: _run followed by digits, then _hash
    result = re.sub(r'_run\d+(_[a-f0-9]{8})$', r'\1', scenario_id)
    return result if result != scenario_id else scenario_id

# Test cases
test_ids = [
    "kyber512_p256_r100_run1_a1b2c3d4",
    "kyber512_p256_r100_run5_a1b2c3d4",
    "dilithium2_p1024_r500_scaling_run1_e5f6g7h8",
    "dilithium2_p1024_r500_scaling_run3_e5f6g7h8",
]

print("   Test cases:")
for test_id in test_ids:
    base_id = extract_base_experiment_id(test_id)
    print(f"     {test_id}")
    print(f"     -> {base_id}")
    # Verify run<N> was removed
    if "_run" in base_id:
        print(f"     ⚠️  WARNING: _run pattern still present!")
    else:
        print(f"     ✓ OK")
    print()
EOF

# Test 4: Check progress script counts
echo ""
echo "4. Checking progress script expected counts..."

# Run check_progress in dry-run mode to see expected counts
if [[ -f "scripts/check_progress.sh" ]]; then
    echo "   Running check_progress.sh to verify counts..."
    ./scripts/check_progress.sh 2>&1 | grep -E "Expected|Total Expected|Native|Minikube|GCP" | head -10
fi

echo ""
echo "=== Validation Complete ==="

