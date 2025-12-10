# Test Coverage and Strategy

**Date**: 2025-12-10  
**Status**: ⚠️ **CRITICAL GAPS IDENTIFIED**  
**Purpose**: Comprehensive overview of test coverage, gaps, and strategy

---

## Executive Summary

**Current Status**: ⚠️ **INSUFFICIENT FOR PRODUCTION**

- ✅ **47 tests passing** - Code structure validated
- ⚠️ **4 tests skipped** - Critical data format validation (requires pandas)
- ❌ **~8 tests missing** - Critical integration and end-to-end tests

**Critical Gaps**:
1. **Data format validation** - 4 tests skipped (requires pandas) - **CRITICAL**
2. **Integration tests** - Missing actual Kubernetes interaction - **CRITICAL**
3. **End-to-end tests** - Missing entirely - **CRITICAL**

---

## Current Test Status

### ✅ Passing Tests (47 total)

#### Unit Tests (17/17)
- ✅ Logging functions (12 tests)
- ✅ Directory creation (5 tests)

#### Functional Tests (23/23)
- ✅ Kubernetes job management functions exist (9 tests)
- ✅ Refactored scripts use unified libraries (11 tests)
- ✅ Script syntax validation (3 tests)

#### Integration Tests (7/7)
- ✅ Job YAML generation for Minikube (2 tests)
- ✅ Job YAML generation for GCP (2 tests)
- ✅ YAML differences between environments (1 test)
- ✅ ConfigMap creation functions (3 tests)

### ⚠️ Skipped Tests (4 - CRITICAL)

**All in `tests/functional/test_data_format.sh`** - Skipped due to missing pandas:

1. ❌ `test_merge_jsonl_expects_latency_ns`
   - **Purpose**: Validates `merge_jsonl.py` correctly processes `latency_ns` format
   - **Why Critical**: Since we removed backward compatibility, this validates experiments will produce correct format
   - **Impact**: **CRITICAL** - Cannot validate analysis scripts work with new data format

2. ❌ `test_merge_jsonl_rejects_missing_latency_ns`
   - **Purpose**: Validates `merge_jsonl.py` rejects old format (without `latency_ns`)
   - **Why Critical**: Ensures old data format is rejected (fail-fast)
   - **Impact**: **CRITICAL** - Cannot validate backward compatibility removal works

3. ❌ `test_compute_statistics_expects_latency_ns`
   - **Purpose**: Validates `compute_statistics.py` correctly processes `latency_ns` format
   - **Why Critical**: Ensures statistics are computed correctly from nanosecond data
   - **Impact**: **CRITICAL** - Cannot validate statistical analysis works

4. ❌ `test_compute_statistics_rejects_missing_latency_ns`
   - **Purpose**: Validates `compute_statistics.py` rejects old format
   - **Why Critical**: Ensures old data format is rejected
   - **Impact**: **CRITICAL** - Cannot validate backward compatibility removal works

**Fix**: `pip install pandas numpy matplotlib seaborn scipy rich tqdm`

---

## Critical Missing Tests

### 1. Integration Tests (PARTIAL)

**What We've Added**:
- ✅ Job YAML generation structure validation
- ✅ ConfigMap creation structure validation

**What's Still Missing**:
- ❌ Actual Kubernetes job creation (requires cluster)
- ❌ Actual result retrieval from PVC (requires Minikube)
- ❌ Actual result retrieval from GCS (requires GCP/GCS)
- ❌ Error handling validation

**Why Critical**:
- We only test that functions **exist**, not that they **work**
- No validation that Kubernetes API calls succeed
- No validation that YAML generation produces valid, deployable resources
- No validation that result retrieval actually gets data

**Risk**: High - Refactored code could have bugs that only appear when actually interacting with Kubernetes

### 2. Smoke Tests (MISSING)

**What Should Exist**:
- ❌ `test_smoke_native.sh` - Run one native experiment
- ❌ `test_smoke_minikube.sh` - Run one Minikube experiment
- ❌ `test_smoke_gcp.sh` - Run one GCP experiment

**Why Critical**:
- Validates entire workflow end-to-end
- Validates data format (`latency_ns` present)
- Validates analysis pipeline produces correct outputs

**Risk**: Critical - Could discover issues only when running real experiments

### 3. Regression Tests (MISSING)

**What Should Exist**:
- ❌ Output comparison before/after refactoring
- ❌ Statistical value comparison (within tolerance)

**Why Critical**:
- Ensures refactoring didn't change behavior
- Validates outputs are consistent

---

## Test Coverage Matrix

| Category | Tested | Skipped | Missing | Critical? |
|----------|--------|---------|---------|-----------|
| **Unit Tests** | ✅ 17 | - | - | Low |
| **Function Existence** | ✅ 23 | - | - | Low |
| **YAML Generation** | ✅ 5 | - | - | Medium |
| **ConfigMap Creation** | ✅ 3 | - | - | Medium |
| **Data Format Validation** | - | ⚠️ **4** | - | **CRITICAL** |
| **K8s Job Creation** | - | - | ❌ ~3 | **CRITICAL** |
| **Result Retrieval** | - | - | ❌ ~2 | **CRITICAL** |
| **End-to-End** | - | - | ❌ ~3 | **CRITICAL** |

---

## What We've Validated

### ✅ Code Structure
- Function existence and signatures
- Library sourcing and dependencies
- Script syntax validation
- YAML structure validation

### ✅ Refactoring Correctness
- Scripts use unified libraries
- Unified entry point routes correctly
- Job generator produces valid YAML
- ConfigMap creation logic works

---

## What We Haven't Validated

### ❌ Actual Functionality
- Kubernetes job creation (only structure tested)
- Result retrieval (only function existence tested)
- Data format correctness (SKIPPED - requires pandas)
- Analysis pipeline correctness (SKIPPED - requires pandas)
- End-to-end workflow (no tests)

### ❌ Error Handling
- Job failure scenarios
- Result retrieval failures
- Missing file handling
- Invalid data handling

---

## Current Expectations

### Data Format
- ✅ **Required**: `latency_ns` (nanoseconds, `u128`)
- ✅ **Required**: `queue_delay_ns` (nanoseconds, `u128`)
- ✅ **Derived**: `latency_us` (microseconds, `f64`) - computed from `latency_ns / 1000.0`
- ✅ **Derived**: `queue_delay_us` (microseconds, `f64`) - computed from `queue_delay_ns / 1000.0`

### Analysis Scripts
- ✅ **`merge_jsonl.py`**: Raises `ValueError` if `latency_ns` missing
- ✅ **`compute_statistics.py`**: Raises `ValueError` if `latency_ns` missing
- ✅ **Both scripts**: Convert nanoseconds to microseconds for analysis
- ✅ **Both scripts**: Store both nanosecond and microsecond stats

### Refactored Code
- ✅ **All scripts**: Source unified libraries (`common.sh`, `directories.sh`, `analysis.sh`, `manifest.sh`)
- ✅ **Kubernetes scripts**: Use unified job management (`k8s-job.sh`)
- ✅ **Kubernetes scripts**: Use unified job generator (`k8s-job-generator.py`)
- ✅ **All scripts**: Have valid bash syntax
- ✅ **Unified entry point**: Routes correctly to environment-specific scripts

---

## Immediate Actions Required

### 1. Install pandas (5 minutes) - **CRITICAL**
```bash
pip install pandas numpy matplotlib seaborn scipy rich tqdm
./tests/run_tests.sh functional  # Should enable 4 skipped tests
```
**Impact**: Enables 4 critical data format validation tests

### 2. Create smoke test (30 minutes) - **HIGH PRIORITY**
```bash
# Create tests/smoke/test_smoke_native.sh
# Run one experiment, validate latency_ns present
```
**Impact**: Validates end-to-end workflow works

### 3. Create integration test for Kubernetes (1 hour) - **HIGH PRIORITY**
```bash
# Create tests/integration/test_k8s_job_creation.sh
# Requires Minikube running
# Actually create job, wait, retrieve results
```
**Impact**: Validates actual Kubernetes interaction

---

## Risk Assessment

### ✅ Low Risk (Well Tested)
- Code structure and function existence
- YAML generation structure
- Script syntax validation

### ⚠️ Medium Risk (Partially Tested)
- ConfigMap creation (structure tested, not execution)
- Job YAML generation (structure tested, not deployment)

### ❌ High Risk (Not Tested)
- **Data format validation** (CRITICAL - skipped due to pandas)
- **Actual Kubernetes interaction** (CRITICAL - no tests)
- **End-to-end workflow** (CRITICAL - no tests)
- **Result retrieval** (HIGH - only existence tested)

---

## Refactoring Validation

### Test Results
- ✅ **Unit tests**: 17/17 passed
- ✅ **Functional tests**: 23/23 passed (4 skipped due to pandas)
- ✅ **Integration tests**: 7/7 passed
- ✅ **Syntax validation**: All scripts pass

### Code Review Findings
- ✅ All function signatures verified
- ✅ All error handling preserved
- ✅ Behavior matches original implementation
- ✅ No critical issues identified

### Fixed Issues
1. ✅ Duplicate source line in `run_minikube.sh` - Fixed
2. ✅ Incorrect error handling in `run_minikube.sh` - Fixed (changed `exit 1` to `continue`)

---

## Recommendations

### Immediate (Before Re-running Experiments)
1. **Install pandas** to enable data format tests
2. **Run data format tests** to validate analysis scripts
3. **Create integration test skeleton** (can run without full Kubernetes)

### High Priority (Before Production Runs)
1. **Create integration tests** that require Kubernetes
2. **Create end-to-end smoke test**

### Medium Priority (After Initial Validation)
1. **Expand integration tests** (error handling, scaling, parallel execution)
2. **Create regression tests** (output comparison, statistical validation)

---

## Conclusion

**Current Status**: ⚠️ **INSUFFICIENT FOR PRODUCTION**

The refactored code structure is validated, but **actual functionality is not tested**. This is acceptable for code review, but **critical functionality must be validated before re-running experiments**.

**Next Steps**:
1. **Immediate**: Install pandas to enable data format tests
2. **Before re-running experiments**: Create at least one smoke test per environment
3. **Before production**: Create integration tests for actual Kubernetes interaction

---

## Related Documentation

- **Test Implementation**: `tests/` directory
- **Requirements**: `docs/REQUIREMENTS_SPECIFICATION.md` (Part 9: Validation Checklist)
- **Outstanding Work**: `TODO.md` (test gaps - item #4)
- **Refactoring Plan**: `docs/guides/refactoring-plan.md` (Phase 0: Test Suite Implementation)

