# Containerization Opportunities: End-to-End Flow Analysis

**Date**: 2025-12-10  
**Purpose**: Identify all opportunities for containerization to ensure consistency across experiment suites

---

## Principle

**✅ Containerize**: Everything that doesn't involve actual benchmarking and telemetry code  
**❌ Do NOT Containerize**: Benchmark binary execution (must run in native/minikube/GCP for accurate measurements)

---

## Current State

### ✅ Already Containerized

1. **Analysis Pipeline Scripts** (via `scripts/lib/run-python-container.sh`)
   - `analysis/scripts/merge_jsonl.py`
   - `analysis/scripts/compute_statistics.py`
   - `analysis/scripts/plot_*.py`
   - Used in: `scripts/lib/analysis.sh`, `scripts/generate_missing_summaries.sh`

2. **Jupyter Environment**
   - `analysis/Dockerfile.jupyter`
   - Used via: `scripts/start-jupyter.sh`

---

## ❌ NOT Containerized (Direct Python Calls)

### High Priority - Consistency Critical

#### 1. **Scenario Generation** (`orchestration/generate_scenarios.py`)
- **Location**: `run_all_experiments.sh` lines 536, 542
- **Current**: `python3 orchestration/generate_scenarios.py`
- **Dependencies**: `yaml` (PyYAML)
- **Impact**: 
  - Python version differences affect YAML parsing
  - First step in pipeline - affects all experiments
  - Inconsistent scenario generation across machines
- **Benefit**: Ensures identical scenario generation regardless of host Python version

#### 2. **Final Analysis Phase** (`run_all_experiments.sh` Phase 5-9)
- **Location**: `run_all_experiments.sh` lines 1745, 1761, 1766, 1788, 1814, 1848
- **Current**: Direct `python3` calls for:
  - `analysis/aggregate_results.py` (line 1745)
  - `analysis/plot_combined_cdfs.py` (line 1761)
  - `analysis/plot_scaling_curves.py` (line 1766)
  - `analysis/hypothesis_tests.py` (line 1788)
  - `analysis/build_final_report.py` (line 1814)
  - `analysis/plot_replica_scaling.py` (line 1848)
- **Impact**: 
  - Produces dissertation outputs - must be consistent
  - Different Python versions = different statistical results
  - Different dependency versions = different plot outputs
- **Benefit**: Guaranteed identical analysis results across all machines

#### 3. **Index Generation** (`scripts/regenerate_index_from_results.sh`)
- **Location**: Inline Python script (lines 63+)
- **Current**: `python3 <<EOF` (inline Python)
- **Dependencies**: Standard library + `yaml` (PyYAML)
- **Impact**: 
  - Used for tracking and validation
  - Python version differences affect YAML parsing
- **Benefit**: Consistent index generation

#### 4. **Inline Python in `run_all_experiments.sh`**
- **Location**: Lines 553, 620, 633
- **Current**: `python3 -c "import json; ..."`
- **Impact**: Python version differences for JSON parsing (minor but still inconsistent)
- **Benefit**: Complete consistency

### Medium Priority - Utility Scripts

#### 5. **Python Utility Scripts**
- **Files**:
  - `scripts/lib/k8s-job-generator.py`
  - `scripts/lib/scenario-patch.py`
- **Current**: Direct `python3` calls
- **Dependencies**: `yaml` (PyYAML), standard library
- **Impact**: Inconsistent YAML generation/parsing
- **Benefit**: Consistent Kubernetes job generation

#### 6. **Validation Scripts**
- **Files**:
  - `scripts/check_data_sufficiency.py`
  - `scripts/complete_incomplete_experiments.sh` (uses Python)
- **Current**: Direct `python3` calls
- **Impact**: Inconsistent validation results
- **Benefit**: Consistent data quality checks

### Low Priority - Already Has Fallback

#### 7. **GCS Fetch Analysis** (`fetch_and_analyse_from_gcs.sh`)
- **Location**: Lines 345, 355, 366, 371
- **Current**: Direct `python3` calls for analysis scripts
- **Impact**: Inconsistent analysis (but script is less critical)
- **Note**: Already has container wrapper available, just not used

---

## Implementation Strategy

### Phase 1: High Priority (Critical for Consistency)

1. **Update `run_all_experiments.sh`**:
   - Replace `python3 orchestration/generate_scenarios.py` → use `run-python-container.sh`
   - Replace all final analysis `python3` calls → use `run-python-container.sh`
   - Replace inline `python3 -c` → use containerized Python

2. **Update `scripts/regenerate_index_from_results.sh`**:
   - Extract inline Python to separate script OR
   - Use containerized Python for inline script execution

3. **Update Analysis Container**:
   - Ensure `orchestration/` directory is included in container
   - Verify all dependencies are in `analysis/requirements.txt`

### Phase 2: Medium Priority (Utility Scripts)

4. **Update utility scripts**:
   - `scripts/lib/k8s-job-generator.py` → use container wrapper
   - `scripts/lib/scenario-patch.py` → use container wrapper

5. **Update validation scripts**:
   - `scripts/check_data_sufficiency.py` → use container wrapper
   - `scripts/complete_incomplete_experiments.sh` → use container wrapper

### Phase 3: Low Priority (Cleanup)

6. **Update remaining scripts**:
   - `fetch_and_analyse_from_gcs.sh` → use container wrapper

---

## Container Requirements

### Current Container (`analysis/Dockerfile`)
- ✅ Python 3.11
- ✅ All analysis dependencies (pandas, matplotlib, scipy, etc.)
- ✅ Analysis scripts and notebooks

### Additional Requirements for Full Containerization
- ✅ Add `orchestration/` directory to container
- ✅ Ensure `pyyaml` is in `requirements.txt` (for scenario generation)
- ✅ Verify all utility scripts work in container

---

## Benefits Summary

### Consistency
- ✅ Same Python version (3.11) across all machines
- ✅ Same dependency versions (locked in container)
- ✅ Identical scenario generation
- ✅ Identical analysis results

### Reproducibility
- ✅ Results are identical regardless of host environment
- ✅ No "works on my machine" issues
- ✅ Dissertation outputs are reproducible

### Isolation
- ✅ No host Python pollution
- ✅ No dependency conflicts
- ✅ Clean environment for each run

### Developer Experience
- ✅ No manual Python environment setup
- ✅ Easier onboarding
- ✅ Consistent behavior across team

---

## What Stays Native

**❌ DO NOT Containerize**:
- `run_local.sh` - Executes native benchmark binary
- `run_minikube.sh` - Executes in Minikube environment
- `deploy_gcp.sh` - Executes in GCP environment
- Benchmark binary (`pqc-bench`) - Must run in target environment for accurate measurements
- Telemetry collection - Part of benchmark execution

**Rationale**: Benchmark execution must run in the target environment (native/minikube/GCP) to measure actual performance. Containerizing would add overhead and invalidate measurements.

---

## Verification Checklist

After implementation, verify:
- [ ] Scenario generation produces identical YAML files across machines
- [ ] Analysis scripts produce identical outputs (JSON, CSV, PNG)
- [ ] Statistical tests produce identical results
- [ ] All scripts work with `QR_USE_CONTAINER=false` fallback
- [ ] Container builds successfully on clean machine
- [ ] All dependencies are in `requirements.txt`

---

## Disadvantages & Trade-offs

### 1. Performance Overhead
- **Container startup**: ~1-2 seconds per script invocation
- **Impact**: 
  - Scenario generation: +1-2s (acceptable - runs once per experiment suite)
  - Analysis scripts: +1-2s each (acceptable - analysis is not time-critical)
  - Inline Python snippets: +1-2s each (could add up if called frequently)
- **Severity**: **LOW** - Analysis phase is not performance-critical

### 2. Dependency on Container Runtime
- **Requires**: Podman or Docker installed
- **Impact**: 
  - Additional system dependency
  - May not be available in all environments (rare)
- **Mitigation**: ✅ Fallback to host Python (`QR_USE_CONTAINER=false`)
- **Severity**: **LOW** - Fallback mechanism available

### 3. Debugging Complexity
- **Issue**: Scripts run inside container, harder to debug
- **Impact**: 
  - Can't easily attach Python debugger (pdb)
  - Error messages may reference container paths
  - Stack traces point to `/workspace/` instead of actual paths
- **Mitigation**: 
  - ✅ Fallback to host Python for debugging (`QR_USE_CONTAINER=false`)
  - ✅ Volume mounts preserve file access
  - ✅ Code changes reflected immediately (mounted as volume)
- **Severity**: **MEDIUM** - Mitigated by fallback option

### 4. Development Workflow
- **Issue**: Slower iteration during active development
- **Impact**: 
  - Need to rebuild container if dependencies change (~30s-2min)
  - Can't easily modify code and test immediately (though volume mounts help)
- **Mitigation**: 
  - ✅ Code mounted as volume (changes reflected immediately)
  - ✅ Use `QR_USE_CONTAINER=false` during development
  - ✅ Container rebuilds automatically if image missing
- **Severity**: **LOW** - Development can use host Python

### 5. Build Time
- **Issue**: First-time container build: ~30s-2min
- **Impact**: One-time delay on first use
- **Mitigation**: 
  - ✅ One-time cost
  - ✅ Cached layers speed up rebuilds
  - ✅ Automatic build (no manual step)
- **Severity**: **LOW** - One-time cost

### 6. Resource Usage
- **Issue**: Container images take disk space
- **Impact**: 
  - Analysis container: ~500MB-1GB
  - Multiple containers if needed
- **Mitigation**: 
  - ✅ Single container for all Python scripts
  - ✅ Images can be shared/cleaned
- **Severity**: **LOW** - Modern systems have ample storage

### 7. Path and Volume Mounting Issues
- **Issue**: Path resolution, SELinux contexts (Podman :Z flag)
- **Impact**: 
  - File permission issues (rare)
  - Path resolution differences (rare)
  - SELinux context problems on Fedora (handled by :Z flag)
- **Mitigation**: 
  - ✅ Wrapper handles :Z flag automatically
  - ✅ Mounts entire project root
  - ✅ Preserves relative paths
- **Severity**: **LOW** - Already handled in wrapper

### 8. CI/CD Considerations
- **Issue**: CI environments may not have container runtime
- **Impact**: 
  - CI scripts need container runtime
  - Additional setup steps
- **Mitigation**: 
  - ✅ Most CI systems have Docker/Podman
  - ✅ Fallback to host Python available
- **Severity**: **LOW** - Most CI systems support containers

## Trade-offs Summary

| Aspect | Containerized | Native Python |
|--------|--------------|---------------|
| **Consistency** | ✅ High (Python 3.11, locked deps) | ❌ Low (varies by machine) |
| **Reproducibility** | ✅ High (identical results) | ❌ Low (environment-dependent) |
| **Performance** | ⚠️ +1-2s overhead per script | ✅ Fastest |
| **Debugging** | ⚠️ More complex (can use fallback) | ✅ Easier |
| **Dependencies** | ⚠️ Requires Podman/Docker | ✅ Just Python |
| **Development Speed** | ⚠️ Slower iteration (can use fallback) | ✅ Fast iteration |
| **Onboarding** | ✅ Easy (no setup) | ❌ Manual setup required |
| **Dissertation Quality** | ✅ Identical outputs | ❌ May vary by machine |

## Recommendation

**Containerize by default, allow fallback for development:**

1. **Default**: Use containerized execution (`QR_USE_CONTAINER=true`)
   - Ensures consistency for production runs
   - Guarantees identical dissertation outputs
   - No manual Python setup needed

2. **Development**: Allow `QR_USE_CONTAINER=false` for faster iteration
   - Faster debugging
   - Immediate code changes
   - Easier to attach debugger

3. **CI/CD**: Use containerized for consistency
   - Reproducible builds
   - Consistent test results

4. **Documentation**: Clear instructions for both modes
   - When to use containerized (production)
   - When to use host Python (development)

## Mitigation Strategies (Already Implemented)

1. ✅ **Fallback mechanism**: `QR_USE_CONTAINER=false` always available
2. ✅ **Good error messages**: Clear when container fails
3. ✅ **Volume mounts**: Code changes reflected immediately
4. ✅ **Automatic detection**: Podman if available, Docker as fallback
5. ✅ **Automatic build**: Container builds on first use if missing

## Overall Assessment

**✅ Benefits Outweigh Disadvantages**

- **Consistency and reproducibility** (critical for dissertation) >> Performance overhead
- **Easier onboarding** >> Slight debugging complexity
- **Fallback mechanism** mitigates all major concerns
- **Performance overhead** is minimal and acceptable for non-time-critical operations

**Verdict**: Containerization is recommended for all Python scripts that don't involve benchmark execution.

---

**Last Updated**: 2025-12-10
