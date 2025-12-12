# Experiment Matrix Coverage Summary

**Date**: 2025-12-12  
**Status**: Complete Coverage Achieved  
**Purpose**: Document complete experiment matrix coverage for dissertation claims

---

## Executive Summary

The experiment matrix has been updated to ensure **complete coverage** across all algorithms for all experiment types. This enables fair comparison and supports all dissertation claims.

**Key Changes**:
- ✅ Added sustained load (5-minute) experiments for RSA-2048 and ECDSA P-256
- ✅ Added scaling experiments for RSA-2048 and ECDSA P-256
- ✅ Updated scaling configuration to include all algorithms

**Result**: Complete coverage across all 5 algorithms for all experiment types.

---

## Coverage Analysis

### Before Updates

**Missing Combinations**:
- ❌ Sustained load (5-minute): Missing RSA-2048 and ECDSA P-256 (2 algorithms)
- ❌ Scaling experiments: Missing RSA-2048 and ECDSA P-256 (2 algorithms)

**Impact**:
- Could not compare sustained load behavior across all algorithms
- Could not make claims like "Algorithm X handles sustained load better than baseline Z" for all algorithms
- Scaling comparison incomplete (only PQC algorithms)

### After Updates

**Complete Coverage**:
- ✅ Baseline (constant, 30s): All 5 algorithms × 4 payloads × 3 rates × 5 runs = **300 scenarios**
- ✅ Burst pattern: All 5 algorithms × 2 payloads × 1 rate × 5 runs = **50 scenarios**
- ✅ 10K msg/s rate: All 5 algorithms × 4 payloads × 1 rate × 5 runs = **100 scenarios**
- ✅ Sustained load (5-min): All 5 algorithms × 1 payload × 1 rate × 3 runs = **15 scenarios**
- ✅ Scaling experiments: All 5 algorithms × 1 payload × 1 rate × 3 runs = **15 scenarios**

**Total Scenarios per Environment**: **480 scenarios** (up from 468)

---

## Added Experiments

### 1. Sustained Load Experiments (CRITICAL)

**Added**:
- RSA-2048 sustained load (5-minute duration)
- ECDSA P-256 sustained load (5-minute duration)

**Configuration**:
- Payload: 1024 bytes
- Rate: 2000 msg/s
- Duration: 300 seconds (5 minutes)
- Runs: 3

**Rationale**:
- Required for fair comparison across all algorithms
- Enables claims about sustained load behavior
- Supports dissertation claims like:
  - "Algorithm X handles sustained load better than baseline Z"
  - "Sustained load performance comparison across algorithms"

### 2. Scaling Experiments (MEDIUM Priority)

**Added**:
- RSA-2048 horizontal scaling (replicas 1,2,4,8)
- ECDSA P-256 horizontal scaling (replicas 1,2,4,8)

**Configuration**:
- Payload: 1024 bytes
- Rate: 500 msg/s
- Runs: 3 per replica count
- Replicas: 1, 2, 4, 8 (Minikube + GCP only)

**Rationale**:
- Enables complete scaling comparison across all algorithms
- Less critical than sustained load (scaling primarily for PQC deployment)
- But provides comprehensive scaling analysis
- Supports claims about scaling behavior differences between classical and PQC

---

## Updated Configuration

### Scaling Configuration

**Before**:
```yaml
scaling_algorithms:
  - kyber512
  - dilithium2
  - hybrid_kyber_dilithium
```

**After**:
```yaml
scaling_algorithms:
  - rsa2048
  - ecdsa_p256
  - kyber512
  - dilithium2
  - hybrid_kyber_dilithium
```

**Note**: All algorithms now included for complete scaling comparison.

---

## Impact on Dissertation Claims

### Now Supported Claims

**Sustained Load**:
- ✅ "RSA-2048 sustained load performance at 2000 msg/s"
- ✅ "ECDSA P-256 sustained load performance at 2000 msg/s"
- ✅ "Sustained load comparison: PQC vs Classical"
- ✅ "Algorithm X handles sustained load better than baseline Z" (all algorithms)

**Scaling**:
- ✅ "RSA-2048 horizontal scaling efficiency"
- ✅ "ECDSA P-256 horizontal scaling efficiency"
- ✅ "Scaling comparison: Classical vs PQC algorithms"
- ✅ "Classical algorithms show [scaling characteristic] vs PQC"

**Complete Coverage**:
- ✅ All algorithms tested under same conditions
- ✅ Fair comparison enabled across all experiment types
- ✅ No gaps in coverage for dissertation claims

---

## Scenario Count Breakdown

### Per Environment

| Experiment Type | Scenarios | Algorithms | Payloads | Rates | Runs |
|----------------|-----------|------------|----------|-------|------|
| Baseline | 300 | 5 | 4 | 3 | 5 |
| Burst | 50 | 5 | 2 | 1 | 5 |
| 10K msg/s | 100 | 5 | 4 | 1 | 5 |
| Sustained (5-min) | 15 | 5 | 1 | 1 | 3 |
| Scaling | 15 | 5 | 1 | 1 | 3 |
| **Total** | **480** | | | | |

### Across All Environments

- **Native**: 480 scenarios (no scaling replicas > 1)
- **Minikube**: 480 baseline + 60 scaling (15 × 4 replicas) = **540 scenarios**
- **GCP**: 480 baseline + 60 scaling (15 × 4 replicas) = **540 scenarios**

**Total Experiments**: 480 + 540 + 540 = **1,560 experiments**

---

## Validation

### Coverage Verification

✅ **All algorithms covered**: RSA-2048, ECDSA P-256, Kyber-512, Dilithium-2, Hybrid  
✅ **All experiment types covered**: Baseline, Burst, 10K, Sustained, Scaling  
✅ **All payload sizes covered**: 256B, 1KB, 4KB, 16KB (where applicable)  
✅ **All rates covered**: 100, 500, 2000, 10000 msg/s (where applicable)  
✅ **All patterns covered**: Constant, Burst  
✅ **All durations covered**: 30s (baseline), 300s (sustained)  

### Gap Analysis

✅ **No critical gaps**: All required combinations present  
✅ **Complete comparison enabled**: All algorithms tested under same conditions  
✅ **Dissertation claims supported**: All claims can be made with complete data  

---

## Next Steps

1. ✅ **Matrix Updated**: All missing experiments added
2. ✅ **Configuration Updated**: Scaling configuration includes all algorithms
3. ⏭️ **Regenerate Scenarios**: Run `python orchestration/generate_scenarios.py` to generate new scenarios
4. ⏭️ **Run Experiments**: Execute full-scale data collection with updated matrix
5. ⏭️ **Update Documentation**: Update any documentation referencing scenario counts

---

## Files Modified

1. `orchestration/experiment_matrix.yaml`
   - Added sustained load experiments for RSA-2048 and ECDSA P-256
   - Added scaling experiments for RSA-2048 and ECDSA P-256
   - Updated scaling configuration to include all algorithms

2. `orchestration/analyze_matrix_gaps.py` (new)
   - Gap analysis script for identifying missing combinations

---

## References

- **Requirements**: `docs/REQUIREMENTS_SPECIFICATION.md`
- **Experimental Design**: `docs/analysis/experimental-design.md`
- **Matrix File**: `orchestration/experiment_matrix.yaml`
- **Gap Analysis**: `orchestration/analyze_matrix_gaps.py`

---

**Last Updated**: 2025-12-12  
**Status**: ✅ Complete Coverage Achieved

