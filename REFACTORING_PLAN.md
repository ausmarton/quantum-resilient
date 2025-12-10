# Code Refactoring Plan: Eliminate Duplication Across Environments

**Date**: 2025-12-10  
**Goal**: Consolidate duplicate code between native, Minikube, and GCP execution paths

---

## Executive Summary

This document outlines a plan to refactor the codebase to eliminate duplication between `run_local.sh`, `run_minikube.sh`, and `deploy_gcp.sh`, with a focus on reusing Kubernetes-related code between Minikube and GCP (both use Kubernetes).

---

## Current Duplication Analysis

### 1. **Logging Functions** (100% duplicate)
**Location**: All three scripts  
**Duplication**: Identical functions in each script
- `log_info()`, `log_success()`, `log_warn()`, `log_error()`, `log_step()`, `log_run()`
- Same color codes, same formatting

**Impact**: Low (cosmetic), but easy win

---

### 2. **Output Directory Creation** (90% duplicate)
**Location**: All three scripts  
**Duplication**: Similar directory structure creation
```bash
mkdir -p "$OUT_DIR/raw"
mkdir -p "$OUT_DIR/merged"
mkdir -p "$OUT_DIR/stats"
mkdir -p "$OUT_DIR/figures"
```

**Impact**: Low, but consolidatable

---

### 3. **Analysis Pipeline Invocation** (95% duplicate)
**Location**: All three scripts  
**Duplication**: Identical Python script calls
- `analysis/scripts/merge_jsonl.py`
- `analysis/scripts/compute_statistics.py`
- `analysis/scripts/plot_latency.py`

**Impact**: Medium - ensures consistency

---

### 4. **Manifest Generation** (80% duplicate)
**Location**: All three scripts  
**Duplication**: Similar JSON manifest creation with environment-specific metadata

**Impact**: Medium - important for provenance

---

### 5. **Kubernetes Job YAML** (70% duplicate)
**Location**: `k8s/worker-job.yaml` vs `k8s/gcp/worker-job.yaml`  
**Duplication**: 
- Same container spec (image, args, env vars, security context)
- Same volume mounts structure
- Same resource requests/limits
- **Differences**:
  - Minikube: Uses PVC, busybox init container
  - GCP: Uses emptyDir, cloud-sdk init container, sidecar for GCS upload

**Impact**: **HIGH** - This is the biggest opportunity for consolidation

---

### 6. **ConfigMap Creation** (85% duplicate)
**Location**: `run_minikube.sh` and `scripts/submit_gcp_job_parallel.sh`  
**Duplication**:
- Same scenario YAML patching logic (Python script)
- Same ConfigMap creation via `kubectl`
- Same name sanitization logic

**Impact**: **HIGH** - Both use Kubernetes, should be identical

---

### 7. **Scenario YAML Patching** (90% duplicate)
**Location**: Multiple scripts  
**Duplication**: Python scripts that:
- Set `metrics.jsonl_out` to `/results/raw/run.jsonl`
- Handle smoke-test mode (reduce duration)
- Set RNG seed

**Impact**: Medium - ensures consistency

---

### 8. **Image Building** (80% duplicate)
**Location**: `run_minikube.sh` and `deploy_gcp.sh`  
**Duplication**:
- Same Podman build commands
- Same Dockerfile
- Same image tagging logic
- **Differences**:
  - Minikube: `minikube image load`
  - GCP: `podman push` to Artifact Registry

**Impact**: Medium - can share build logic

---

### 9. **Job Waiting Logic** (90% duplicate)
**Location**: `run_minikube.sh` and `deploy_gcp.sh`  
**Duplication**: Similar `kubectl wait` commands and pod status checking

**Impact**: Medium - ensures consistent behavior

---

### 10. **Data Retrieval** (Different methods, similar goals)
**Location**: `run_minikube.sh` (PVC copy) vs `deploy_gcp.sh` (GCS download)  
**Duplication**: Conceptual (both retrieve results), but implementation differs

**Impact**: Low - methods are environment-specific

---

## Refactoring Strategy

### Phase 1: Extract Common Utilities (Low Risk)
**Goal**: Create shared library of common functions

**Tasks**:
1. Create `scripts/lib/common.sh` with logging functions
2. Create `scripts/lib/directories.sh` for directory creation
3. Create `scripts/lib/analysis.sh` for analysis pipeline invocation
4. Create `scripts/lib/manifest.sh` for manifest generation

**Benefits**:
- Immediate code reduction
- Consistent behavior across environments
- Easy to test

---

### Phase 2: Consolidate Kubernetes Logic (High Impact)
**Goal**: Unify Minikube and GCP Kubernetes deployment

**Tasks**:
1. Create unified Kubernetes Job YAML template
2. Create `scripts/lib/k8s.sh` with common Kubernetes functions:
   - `create_scenario_configmap()` - Unified ConfigMap creation
   - `patch_scenario_yaml()` - Unified scenario patching
   - `wait_for_job()` - Unified job waiting
   - `get_job_results()` - Unified result retrieval (abstracts PVC vs GCS)
3. Create `scripts/lib/k8s-job-generator.py` - Unified Job YAML generation
   - Takes environment type (minikube/gcp) as parameter
   - Generates appropriate YAML with environment-specific overrides

**Benefits**:
- **Major code reduction** (eliminates ~500 lines of duplication)
- Ensures Minikube and GCP use identical Kubernetes patterns
- Easier to maintain and test

---

### Phase 3: Unified Experiment Runner (Medium Risk)
**Goal**: Create single entry point that delegates to environment-specific handlers

**Tasks**:
1. Create `scripts/run_experiment.sh` - Unified entry point
   - Parses arguments
   - Detects environment
   - Delegates to environment-specific handlers
2. Refactor existing scripts to be thin wrappers or handlers:
   - `run_local.sh` → `scripts/handlers/native.sh`
   - `run_minikube.sh` → `scripts/handlers/minikube.sh`
   - `deploy_gcp.sh` → `scripts/handlers/gcp.sh`

**Benefits**:
- Single entry point for all environments
- Consistent argument parsing
- Easier to add new environments

---

### Phase 4: Image Building Consolidation (Low Risk)
**Goal**: Share image building logic

**Tasks**:
1. Create `scripts/lib/image-build.sh` with:
   - `build_image()` - Unified Podman build
   - `tag_image()` - Unified tagging
   - `load_image_minikube()` - Minikube-specific loading
   - `push_image_gcp()` - GCP-specific pushing

**Benefits**:
- Consistent image building
- Easier to add new registries

---

## Detailed Task Breakdown

### Task 1: Extract Common Utilities
**Effort**: 2-3 hours  
**Risk**: Low  
**Dependencies**: None

**Files to create**:
- `scripts/lib/common.sh` - Logging, colors, usage helpers
- `scripts/lib/directories.sh` - Directory creation
- `scripts/lib/analysis.sh` - Analysis pipeline
- `scripts/lib/manifest.sh` - Manifest generation

**Files to modify**:
- `run_local.sh` - Source common.sh
- `run_minikube.sh` - Source common.sh
- `deploy_gcp.sh` - Source common.sh

---

### Task 2: Consolidate Kubernetes ConfigMap Logic
**Effort**: 3-4 hours  
**Risk**: Medium  
**Dependencies**: Task 1

**Files to create**:
- `scripts/lib/k8s-configmap.sh` - Unified ConfigMap creation
- `scripts/lib/scenario-patch.py` - Unified scenario YAML patching

**Files to modify**:
- `run_minikube.sh` - Use k8s-configmap.sh
- `scripts/submit_gcp_job_parallel.sh` - Use k8s-configmap.sh

**Key consolidation**:
```bash
# Unified function
create_scenario_configmap() {
    local scenario_path="$1"
    local exp_id="$2"
    local namespace="${3:-default}"
    local smoke_test="${4:-false}"
    
    # Sanitize name
    local cm_name=$(sanitize_k8s_name "$exp_id")
    
    # Patch scenario
    local temp_scenario=$(mktemp)
    python3 scripts/lib/scenario-patch.py \
        --input "$scenario_path" \
        --output "$temp_scenario" \
        --smoke-test "$smoke_test"
    
    # Create ConfigMap
    kubectl create configmap "$cm_name" \
        --from-file=scenario.yaml="$temp_scenario" \
        --namespace="$namespace" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    rm -f "$temp_scenario"
}
```

---

### Task 3: Unified Kubernetes Job YAML Generator
**Effort**: 4-5 hours  
**Risk**: Medium  
**Dependencies**: Task 2

**Files to create**:
- `scripts/lib/k8s-job-generator.py` - Unified Job YAML generator
- `k8s/base/worker-job-base.yaml` - Base template (common parts)

**Files to modify**:
- `k8s/worker-job.yaml` - Reference base template
- `k8s/gcp/worker-job.yaml` - Reference base template

**Key consolidation**:
```python
# k8s-job-generator.py
def generate_job_yaml(
    environment: str,  # "minikube" or "gcp"
    job_name: str,
    namespace: str,
    image: str,
    scenario_configmap: str,
    replicas: int = 1,
    gcp_config: dict = None,  # For GCP-specific config
) -> dict:
    # Load base template
    base = load_yaml("k8s/base/worker-job-base.yaml")
    
    # Set common fields
    base["metadata"]["name"] = job_name
    base["metadata"]["namespace"] = namespace
    
    # Set container image
    base["spec"]["template"]["spec"]["containers"][0]["image"] = image
    
    # Environment-specific overrides
    if environment == "minikube":
        # Use PVC volume
        base["spec"]["template"]["spec"]["volumes"][1] = {
            "name": "results",
            "persistentVolumeClaim": {"claimName": "pqc-bench-results"}
        }
        # Use busybox init container
        base["spec"]["template"]["spec"]["initContainers"][0]["image"] = "busybox:1.36"
    elif environment == "gcp":
        # Use emptyDir volume
        base["spec"]["template"]["spec"]["volumes"][1] = {
            "name": "results",
            "emptyDir": {"sizeLimit": "2Gi"}
        }
        # Use cloud-sdk init container
        base["spec"]["template"]["spec"]["initContainers"][0]["image"] = "gcr.io/google.com/cloudsdktool/cloud-sdk:alpine"
        # Add sidecar container
        base["spec"]["template"]["spec"]["containers"].append(create_gcs_upload_sidecar(gcp_config))
        # Add service account
        base["spec"]["template"]["spec"]["serviceAccountName"] = "pqc-bench-sa"
    
    return base
```

---

### Task 4: Unified Job Waiting and Result Retrieval
**Effort**: 2-3 hours  
**Risk**: Low  
**Dependencies**: Task 3

**Files to create**:
- `scripts/lib/k8s-job.sh` - Unified job management

**Key functions**:
```bash
wait_for_job() {
    local job_name="$1"
    local namespace="${2:-default}"
    local timeout="${3:-600s}"
    
    kubectl wait --for=condition=complete \
        --timeout="$timeout" \
        job/"$job_name" -n "$namespace"
}

get_job_results() {
    local job_name="$1"
    local output_dir="$2"
    local environment="$3"  # "minikube" or "gcp"
    local namespace="${4:-default}"
    
    case "$environment" in
        minikube)
            # Use PVC copy logic
            copy_from_pvc "$job_name" "$output_dir" "$namespace"
            ;;
        gcp)
            # Use GCS download logic
            download_from_gcs "$job_name" "$output_dir" "$namespace"
            ;;
    esac
}
```

---

### Task 5: Refactor Main Scripts to Use Libraries
**Effort**: 4-5 hours  
**Risk**: Medium  
**Dependencies**: Tasks 1-4

**Files to modify**:
- `run_local.sh` - Source libraries, use common functions
- `run_minikube.sh` - Source libraries, use k8s functions
- `deploy_gcp.sh` - Source libraries, use k8s functions

**Expected reduction**:
- `run_minikube.sh`: ~1156 lines → ~600 lines (48% reduction)
- `deploy_gcp.sh`: ~1409 lines → ~700 lines (50% reduction)
- `run_local.sh`: ~478 lines → ~300 lines (37% reduction)

---

### Task 6: Create Unified Entry Point (Optional)
**Effort**: 3-4 hours  
**Risk**: Medium  
**Dependencies**: Task 5

**Files to create**:
- `scripts/run_experiment.sh` - Unified entry point

**Usage**:
```bash
# Unified interface
./scripts/run_experiment.sh \
    --env native \
    --scenario scenarios/kyber512.yaml \
    --out results/exp1

./scripts/run_experiment.sh \
    --env minikube \
    --scenario scenarios/kyber512.yaml \
    --out results/exp2 \
    --replicas 4

./scripts/run_experiment.sh \
    --env gcp \
    --scenario scenarios/kyber512.yaml \
    --out results/exp3 \
    --project my-project \
    --bucket my-bucket
```

---

## Implementation Order

### Phase 0: Test Suite Implementation (BEFORE Refactoring)
**Duration**: 4-6 hours  
**Risk**: Low  
**Impact**: **CRITICAL** - Enables safe refactoring

**MUST COMPLETE BEFORE PHASE 1**

**Tasks**:
1. Create test infrastructure (`tests/` directory structure)
2. Implement baseline tests (unit, integration, functional, regression)
3. Run baseline tests and capture expected outputs
4. Document test execution and success criteria

**Deliverables**:
- ✅ All tests passing with current code
- ✅ Baseline outputs captured
- ✅ Test execution framework ready
- ✅ Test documentation complete

**Success Criteria**:
- ✅ 100% test pass rate
- ✅ All requirements (FR1-FR15, NFR1-NFR8) covered by tests
- ✅ Critical paths validated

---

### Phase 1: Quick Wins (Tasks 1-2)
**Duration**: 5-7 hours  
**Risk**: Low  
**Impact**: Medium

**Prerequisites**: Phase 0 complete (all tests passing)

Start here - immediate code reduction with low risk.

**Testing**:
- After Task 1: Run unit tests for extracted functions
- After Task 2: Run integration tests for ConfigMap logic
- Compare outputs: Verify identical to baseline

---

### Phase 2: Kubernetes Consolidation (Tasks 3-4)
**Duration**: 6-8 hours  
**Risk**: Medium  
**Impact**: **HIGH**

**Prerequisites**: Phase 1 complete (all tests passing)

Biggest opportunity - eliminates most duplication between Minikube and GCP.

**Testing**:
- After Task 3: Run unit tests for Job YAML generator
- After Task 4: Run integration tests for job management
- Compare outputs: Verify identical to baseline
- Verify: Minikube and GCP produce identical Kubernetes resources (where applicable)

---

### Phase 3: Full Integration (Tasks 5-6)
**Duration**: 7-9 hours  
**Risk**: Medium  
**Impact**: High

**Prerequisites**: Phase 2 complete (all tests passing)

Complete the refactoring and optionally add unified entry point.

**Testing**:
- After Task 5: Run full test suite (all tests)
- After Task 6: Run integration tests for unified entry point
- Regression comparison: Compare all outputs against baseline
- Performance test: Ensure no slowdown

---

## Testing Strategy

**CRITICAL**: All tests must be implemented and passing **BEFORE** starting refactoring. This ensures we have a baseline to compare against and can detect regressions immediately.

### Testing Philosophy

Tests are organized by **requirements** from `docs/REQUIREMENTS_SPECIFICATION.md` to ensure:
- All functional requirements (FR1-FR15) are validated
- All non-functional requirements (NFR1-NFR8) are validated
- Critical paths for dissertation objectives are protected
- Regression detection is comprehensive

---

### Phase 0: Baseline Test Suite (BEFORE Refactoring)

**Goal**: Establish comprehensive test suite that validates current functionality

**Duration**: 4-6 hours  
**Priority**: **CRITICAL** - Must complete before Phase 1

#### Test Categories

##### 1. Unit Tests for Library Functions

**Location**: `tests/unit/`  
**Purpose**: Test individual functions in isolation before extraction

**Tests to Create**:

**`tests/unit/test_logging.sh`**:
```bash
#!/usr/bin/env bash
# Test logging functions from all three scripts
# Verify identical output across environments

test_log_info() {
    # Capture output from each script's log_info
    # Compare formatting, colors, content
}

test_log_success() { ... }
test_log_warn() { ... }
test_log_error() { ... }
test_log_step() { ... }
```

**`tests/unit/test_directories.sh`**:
```bash
# Test directory creation logic
test_create_output_directories() {
    # Verify all required directories created
    # Verify permissions
    # Verify structure matches requirements
}
```

**`tests/unit/test_scenario_patch.py`**:
```python
# Test scenario YAML patching logic
def test_patch_jsonl_output():
    # Verify metrics.jsonl_out is set correctly
    
def test_patch_smoke_test():
    # Verify duration reduction in smoke-test mode
    
def test_patch_rng_seed():
    # Verify RNG seed is set correctly
```

**`tests/unit/test_k8s_configmap.sh`**:
```bash
# Test ConfigMap creation logic
test_sanitize_k8s_name() {
    # Test name sanitization (RFC 1123 compliance)
    # Test edge cases (long names, special chars)
}

test_create_configmap() {
    # Test ConfigMap creation
    # Verify content matches scenario YAML
    # Verify namespace handling
}
```

**`tests/unit/test_k8s_job_generator.py`**:
```python
# Test Job YAML generation
def test_generate_minikube_job():
    # Verify Minikube-specific overrides (PVC, busybox)
    
def test_generate_gcp_job():
    # Verify GCP-specific overrides (emptyDir, cloud-sdk, sidecar)
    
def test_job_common_fields():
    # Verify common fields (security context, resources, etc.)
```

---

##### 2. Integration Tests for Critical Paths

**Location**: `tests/integration/`  
**Purpose**: Test end-to-end workflows aligned with requirements

**Tests to Create**:

**`tests/integration/test_native_experiment.sh`**:
```bash
#!/usr/bin/env bash
# Test complete native experiment workflow
# Validates: FR1 (Latency), FR2 (Throughput), FR3 (Multi-Environment)

test_native_single_run() {
    # Run one experiment
    # Verify: raw/run.jsonl exists and is valid
    # Verify: merged/merged.jsonl exists
    # Verify: stats/summary.json exists with required fields
    # Verify: manifest.json exists
}

test_native_multiple_runs() {
    # Run 5 runs
    # Verify: aggregated_stats.json exists
    # Verify: All runs have identical structure
}
```

**`tests/integration/test_minikube_experiment.sh`**:
```bash
# Test complete Minikube experiment workflow
# Validates: FR3 (Multi-Environment), FR5 (Horizontal Scaling), FR14 (Isolation)

test_minikube_single_run() {
    # Run one experiment
    # Verify: Job completes successfully
    # Verify: Results copied from PVC
    # Verify: Data integrity matches native
}

test_minikube_scaling() {
    # Run with replicas 2, 4, 8
    # Verify: Scaling experiments complete
    # Verify: Results structure matches requirements
}
```

**`tests/integration/test_gcp_experiment.sh`**:
```bash
# Test complete GCP experiment workflow
# Validates: FR3 (Multi-Environment), FR5 (Horizontal Scaling), FR14 (Isolation)

test_gcp_single_run() {
    # Run one experiment (ephemeral mode)
    # Verify: Cluster created
    # Verify: Job completes successfully
    # Verify: Results uploaded to GCS
    # Verify: Results downloaded locally
    # Verify: Data integrity matches native
}

test_gcp_scaling() {
    # Run with replicas 2, 4, 8
    # Verify: Scaling experiments complete
    # Verify: Cluster scales appropriately
}
```

**`tests/integration/test_analysis_pipeline.sh`**:
```bash
# Test analysis pipeline invocation
# Validates: FR4 (Statistical Analysis), FR9 (Queue Delay), FR10 (Payload Impact)

test_merge_jsonl() {
    # Verify merge_jsonl.py produces valid output
    # Verify nanosecond precision preserved
}

test_compute_statistics() {
    # Verify compute_statistics.py produces summary.json
    # Verify percentiles (p50, p95, p99) calculated
    # Verify latency_ns and latency_us both present
}

test_plot_generation() {
    # Verify plots generated
    # Verify CDF plots exist
    # Verify scaling curves exist (if applicable)
}
```

---

##### 3. Functional Tests for Requirements

**Location**: `tests/functional/`  
**Purpose**: Validate requirements from REQUIREMENTS_SPECIFICATION.md

**Tests to Create**:

**`tests/functional/test_fr1_latency.sh`**:
```bash
# FR1: Latency Measurement
# Requirement: Nanosecond precision, sub-microsecond capture

test_nanosecond_precision() {
    # Run fast algorithm (RSA-2048, small payload)
    # Verify latency_ns field exists
    # Verify latency_ns < 1000 for sub-microsecond operations
    # Verify latency_us = latency_ns / 1000
}

test_latency_percentiles() {
    # Verify p50, p95, p99 calculated
    # Verify percentiles are non-zero (or zero is valid)
}
```

**`tests/functional/test_fr2_throughput.sh`**:
```bash
# FR2: Throughput Measurement
# Requirement: Accurate ops/sec calculation

test_throughput_calculation() {
    # Run experiment with known rate (e.g., 100 msg/s)
    # Verify throughput matches expected rate (±5%)
    # Verify throughput scales with rate
}
```

**`tests/functional/test_fr3_multi_environment.sh`**:
```bash
# FR3: Multi-Environment Support
# Requirement: Identical experiments across environments

test_environment_consistency() {
    # Run same scenario in native, minikube, gcp
    # Verify: Same number of events
    # Verify: Latency distributions similar (within expected variance)
    # Verify: Output structure identical
}
```

**`tests/functional/test_fr4_statistical_analysis.sh`**:
```bash
# FR4: Statistical Analysis
# Requirement: Multiple runs, hypothesis tests, effect sizes

test_multiple_runs() {
    # Verify 5 runs per configuration
    # Verify aggregated_stats.json exists
    # Verify confidence intervals calculated
}

test_hypothesis_tests() {
    # Verify hypothesis_tests.json exists
    # Verify p-values calculated
    # Verify effect sizes calculated
}
```

**`tests/functional/test_fr5_horizontal_scaling.sh`**:
```bash
# FR5: Horizontal Scaling Support
# Requirement: Replica scaling in Minikube and GCP

test_minikube_scaling() {
    # Run with replicas 1, 2, 4, 8
    # Verify: All complete successfully
    # Verify: Scaling plots generated
}

test_gcp_scaling() {
    # Run with replicas 1, 2, 4, 8
    # Verify: Cluster scales appropriately
    # Verify: Results collected correctly
}
```

**`tests/functional/test_fr6_resource_utilization.sh`**:
```bash
# FR6: Resource Utilization Measurement
# Requirement: CPU and memory data captured

test_memory_capture() {
    # Verify memory_rss_bytes field exists
    # Verify values are non-zero and reasonable (6-10MB)
}

test_cpu_capture() {
    # Verify cpu_user_seconds field exists
    # Note: May be 0.0 (known issue), but field must exist
}
```

**`tests/functional/test_fr7_data_completeness.sh`**:
```bash
# FR7: Data Completeness
# Requirement: All experiments have raw, merged, summary

test_data_completeness() {
    # Check all experiments in index.json
    # Verify: raw/run.jsonl exists and non-empty
    # Verify: merged/merged.jsonl exists
    # Verify: stats/summary.json exists
}
```

**`tests/functional/test_fr8_data_validation.sh`**:
```bash
# FR8: Data Validation
# Requirement: Validate data integrity and completeness

test_data_validation() {
    # Run validate_data_integrity.sh
    # Verify: All experiments pass validation
    # Verify: File sizes non-zero
    # Verify: JSONL format valid
}
```

**`tests/functional/test_fr9_queue_delay.sh`**:
```bash
# FR9: Queue Delay Analysis
# Requirement: Queue delay captured and analyzed

test_queue_delay_capture() {
    # Verify queue_delay_ns and queue_delay_us fields exist
    # Verify crypto_latency calculated (latency - queue_delay)
}
```

**`tests/functional/test_fr14_experiment_isolation.sh`**:
```bash
# FR14: Experiment Isolation
# Requirement: No interference between experiments

test_isolation() {
    # Run two experiments concurrently
    # Verify: Results are independent
    # Verify: No shared resources
    # Verify: Latency distributions don't affect each other
}
```

**`tests/functional/test_fr15_analysis_robustness.sh`**:
```bash
# FR15: Analysis Pipeline Robustness
# Requirement: Handle missing dependencies gracefully

test_missing_dependencies() {
    # Simulate missing pandas
    # Verify: Clear error message
    # Verify: Script doesn't crash silently
}
```

---

##### 4. Regression Tests

**Location**: `tests/regression/`  
**Purpose**: Capture current behavior for comparison after refactoring

**Tests to Create**:

**`tests/regression/capture_baseline.sh`**:
```bash
#!/usr/bin/env bash
# Capture baseline outputs before refactoring

capture_baseline() {
    local scenario="$1"
    local env="$2"
    local output_dir="tests/regression/baselines/${env}/$(basename $scenario .yaml)"
    
    # Run experiment
    # Capture: raw/run.jsonl hash
    # Capture: summary.json content
    # Capture: manifest.json content
    # Capture: Directory structure
    # Capture: File sizes
}

# Run for one scenario per environment
capture_baseline "scenarios/kyber512_p1024_r100.yaml" "native"
capture_baseline "scenarios/kyber512_p1024_r100.yaml" "minikube"
capture_baseline "scenarios/kyber512_p1024_r100.yaml" "gcp"
```

**`tests/regression/compare_after_refactor.sh`**:
```bash
#!/usr/bin/env bash
# Compare outputs after refactoring against baseline

compare_outputs() {
    local baseline_dir="$1"
    local new_output_dir="$2"
    
    # Compare: File hashes
    # Compare: JSON structure
    # Compare: Statistical values (within tolerance)
    # Compare: Directory structure
}
```

---

##### 5. Smoke Tests

**Location**: `tests/smoke/`  
**Purpose**: Quick validation of critical paths

**Tests to Create**:

**`tests/smoke/test_all_environments.sh`**:
```bash
#!/usr/bin/env bash
# Quick smoke test: Run one experiment per environment

test_native_smoke() {
    ./run_local.sh \
        --scenario scenarios/kyber512_p1024_r100.yaml \
        --out tests/smoke/native \
        --smoke-test
    # Verify: Experiment completes
    # Verify: Output files exist
}

test_minikube_smoke() {
    ./run_minikube.sh \
        --scenario scenarios/kyber512_p1024_r100.yaml \
        --out tests/smoke/minikube \
        --exp-id smoke-test \
        --smoke-test
    # Verify: Experiment completes
    # Verify: Results retrieved
}

test_gcp_smoke() {
    ./deploy_gcp.sh \
        --scenario scenarios/kyber512_p1024_r100.yaml \
        --exp-id smoke-test \
        --project "$PROJECT" \
        --bucket "$BUCKET" \
        --smoke-test \
        --ephemeral
    # Verify: Experiment completes
    # Verify: Results downloaded
}
```

---

#### Test Execution Framework

**Create**: `tests/run_tests.sh`

```bash
#!/usr/bin/env bash
# Unified test runner

run_unit_tests() {
    # Run all unit tests
    # Exit on first failure
}

run_integration_tests() {
    # Run integration tests
    # Skip if prerequisites not met (e.g., Minikube not running)
}

run_functional_tests() {
    # Run functional tests
    # Group by requirement
}

run_smoke_tests() {
    # Run smoke tests
    # Quick validation
}

# Main
case "${1:-all}" in
    unit) run_unit_tests ;;
    integration) run_integration_tests ;;
    functional) run_functional_tests ;;
    smoke) run_smoke_tests ;;
    all) run_unit_tests && run_integration_tests && run_functional_tests ;;
esac
```

---

### Testing During Refactoring

#### Before Each Task:

1. **Run baseline tests**: Ensure all tests pass with current code
2. **Capture current outputs**: Store expected outputs for comparison

#### During Each Task:

1. **Run unit tests**: After extracting each library function
2. **Run integration tests**: After modifying main scripts
3. **Compare outputs**: Verify outputs match baseline

#### After Each Task:

1. **Full test suite**: Run all tests
2. **Regression comparison**: Compare against baseline
3. **Smoke test**: Quick validation of critical paths

---

### Testing After Refactoring

#### Phase Completion Tests:

**After Phase 1**:
- ✅ All unit tests pass
- ✅ Integration tests pass for extracted functions
- ✅ Smoke tests pass for all environments
- ✅ Outputs match baseline (file hashes identical)

**After Phase 2**:
- ✅ Kubernetes tests pass (ConfigMap, Job generation)
- ✅ Minikube and GCP produce identical Kubernetes resources (where applicable)
- ✅ Integration tests pass for Kubernetes workflows
- ✅ Outputs match baseline (statistical values within tolerance)

**After Phase 3**:
- ✅ All functional tests pass
- ✅ All requirements validated (FR1-FR15, NFR1-NFR8)
- ✅ Full test suite passes
- ✅ Performance regression tests pass (no slowdown)

---

### Test Requirements Mapping

| Requirement | Test Coverage | Test File |
|------------|---------------|-----------|
| FR1: Latency Measurement | ✅ Unit + Functional | `test_fr1_latency.sh`, `test_nanosecond_precision.sh` |
| FR2: Throughput Measurement | ✅ Unit + Functional | `test_fr2_throughput.sh` |
| FR3: Multi-Environment | ✅ Integration + Functional | `test_fr3_multi_environment.sh`, `test_*_experiment.sh` |
| FR4: Statistical Analysis | ✅ Integration + Functional | `test_fr4_statistical_analysis.sh`, `test_analysis_pipeline.sh` |
| FR5: Horizontal Scaling | ✅ Integration + Functional | `test_fr5_horizontal_scaling.sh` |
| FR6: Resource Utilization | ✅ Functional | `test_fr6_resource_utilization.sh` |
| FR7: Data Completeness | ✅ Functional | `test_fr7_data_completeness.sh` |
| FR8: Data Validation | ✅ Functional | `test_fr8_data_validation.sh` |
| FR9: Queue Delay Analysis | ✅ Functional | `test_fr9_queue_delay.sh` |
| FR10: Payload Size Impact | ✅ Integration | `test_analysis_pipeline.sh` |
| FR11: Workload Pattern Impact | ✅ Integration | `test_analysis_pipeline.sh` |
| FR12: Error Rate Tracking | ✅ Functional | `test_fr12_error_rates.sh` |
| FR13: Cost Efficiency | ⚠️ Optional | `test_fr13_cost_efficiency.sh` |
| FR14: Experiment Isolation | ✅ Functional | `test_fr14_experiment_isolation.sh` |
| FR15: Analysis Robustness | ✅ Functional | `test_fr15_analysis_robustness.sh` |
| NFR1: Precision | ✅ Functional | `test_fr1_latency.sh` |
| NFR2: Statistical Rigor | ✅ Functional | `test_fr4_statistical_analysis.sh` |
| NFR3: Reproducibility | ✅ Regression | `test_reproducibility.sh` |
| NFR4: Scalability | ✅ Integration | `test_fr5_horizontal_scaling.sh` |
| NFR5: Dependency Consistency | ✅ Functional | `test_fr15_analysis_robustness.sh` |
| NFR6: Visualization Quality | ✅ Integration | `test_plot_generation.sh` |
| NFR7: Data Export Formats | ✅ Integration | `test_analysis_pipeline.sh` |
| NFR8: Report Generation | ⚠️ Optional | `test_fr8_report_generation.sh` |

---

### Test Data Requirements

**Baseline Test Data**:
- One complete experiment per environment (native, minikube, gcp)
- Same scenario used across all environments for consistency
- Expected outputs captured (file hashes, JSON structure, statistical values)

**Test Scenarios**:
- **Fast algorithm**: RSA-2048, small payload (tests sub-microsecond precision)
- **Slow algorithm**: Dilithium-2, large payload (tests normal precision)
- **Scaling scenario**: Kyber-512, replicas 1,2,4,8 (tests horizontal scaling)

---

### Test Execution Strategy

#### Pre-Refactoring (Phase 0):

1. **Create test infrastructure** (2-3 hours)
   - Set up test directory structure
   - Create test runner framework
   - Document test execution

2. **Implement baseline tests** (2-3 hours)
   - Unit tests for functions to be extracted
   - Integration tests for critical paths
   - Functional tests for requirements
   - Regression baseline capture

3. **Run baseline tests** (1 hour)
   - Ensure all tests pass
   - Capture baseline outputs
   - Document expected behavior

#### During Refactoring:

**After Task 1** (Extract Common Utilities):
- ✅ Unit tests for `common.sh`, `directories.sh`, `analysis.sh`, `manifest.sh`
- ✅ Integration test: Run one experiment per environment
- ✅ Compare outputs: Verify identical to baseline

**After Task 2** (Consolidate ConfigMap Logic):
- ✅ Unit tests for `k8s-configmap.sh`, `scenario-patch.py`
- ✅ Integration test: Create ConfigMap in Minikube and GCP
- ✅ Compare ConfigMaps: Verify identical content

**After Task 3** (Unified Job YAML Generator):
- ✅ Unit tests for `k8s-job-generator.py`
- ✅ Integration test: Generate Job YAML for Minikube and GCP
- ✅ Compare YAML: Verify environment-specific overrides correct

**After Task 4** (Unified Job Management):
- ✅ Unit tests for `k8s-job.sh`
- ✅ Integration test: Wait for job, retrieve results
- ✅ Compare results: Verify identical to baseline

**After Task 5** (Refactor Main Scripts):
- ✅ Full test suite: All unit, integration, functional tests
- ✅ Regression comparison: Compare all outputs against baseline
- ✅ Performance test: Ensure no slowdown

**After Task 6** (Unified Entry Point):
- ✅ Integration test: Use unified entry point
- ✅ Compare: Verify same outputs as individual scripts

---

### Test Success Criteria

#### Unit Tests:
- ✅ **100% pass rate** before refactoring
- ✅ **100% pass rate** after each task
- ✅ **Coverage**: All extracted functions tested

#### Integration Tests:
- ✅ **All environments**: Native, Minikube, GCP complete successfully
- ✅ **Output consistency**: Same scenario produces identical structure
- ✅ **Data integrity**: File hashes match baseline (or statistical values within tolerance)

#### Functional Tests:
- ✅ **All requirements**: FR1-FR15 validated
- ✅ **All NFRs**: NFR1-NFR8 validated
- ✅ **Dissertation objectives**: All 7 objectives supported

#### Regression Tests:
- ✅ **Output comparison**: File hashes identical (or within tolerance)
- ✅ **Statistical comparison**: Percentiles within 0.1% tolerance
- ✅ **Structure comparison**: Directory structure identical

---

### Test Failure Handling

**If tests fail during refactoring**:
1. **Stop refactoring** immediately
2. **Revert changes** to last passing state
3. **Investigate failure**:
   - Compare outputs (before vs after)
   - Check logs for errors
   - Verify test expectations are correct
4. **Fix issue** before continuing
5. **Re-run tests** to verify fix

**Test Tolerance**:
- **File hashes**: Must match exactly (no changes to data)
- **Statistical values**: Within 0.1% tolerance (floating-point precision)
- **Timing**: May vary (not critical for correctness)
- **Log messages**: Format may change, but content must be equivalent

---

### Test Maintenance

**After Refactoring Complete**:
- ✅ All tests become part of CI/CD pipeline
- ✅ Tests run before any code changes
- ✅ Tests updated when requirements change
- ✅ Baseline updated when expected behavior changes

---

## Risk Mitigation

### Risks:
1. **Breaking existing workflows**: Mitigated by comprehensive testing before refactoring
2. **Kubernetes differences**: Mitigated by environment-specific overrides
3. **Testing complexity**: Mitigated by incremental changes

### Backward Compatibility:
- **NOT REQUIRED**: All experiments will be re-run, so backward compatibility with old data/scripts is not needed
- Can make breaking changes to improve code quality
- Focus on correctness and maintainability over compatibility

---

## Success Metrics

### Code Reduction:
- **Target**: 40-50% reduction in total lines
- **Current**: ~3000 lines across three scripts
- **Target**: ~1500-1800 lines after refactoring

### Duplication Elimination:
- **Target**: Zero duplicate Kubernetes logic between Minikube and GCP
- **Target**: Single source of truth for common functions

### Maintainability:
- **Target**: Changes to Kubernetes logic only need to be made once
- **Target**: New environments easier to add

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Implement Phase 0** (test suite) - **CRITICAL FIRST STEP**
3. **Run baseline tests** and capture expected outputs
4. **Start with Phase 1** (quick wins) - only after Phase 0 complete
5. **Test incrementally** after each task
6. **Document** new library functions
7. **Update** `run_all_experiments.sh` to use new libraries

---

## Test Implementation Checklist

### Phase 0: Test Suite (BEFORE Refactoring)

- [ ] Create `tests/` directory structure
  - [ ] `tests/unit/` - Unit tests for library functions
  - [ ] `tests/integration/` - Integration tests for workflows
  - [ ] `tests/functional/` - Functional tests for requirements
  - [ ] `tests/regression/` - Regression tests and baselines
  - [ ] `tests/smoke/` - Quick smoke tests

- [ ] Implement test runner (`tests/run_tests.sh`)
  - [ ] Support running individual test suites
  - [ ] Support running all tests
  - [ ] Clear output formatting
  - [ ] Exit codes for CI/CD

- [ ] Implement unit tests
  - [ ] `test_logging.sh` - Test logging functions
  - [ ] `test_directories.sh` - Test directory creation
  - [ ] `test_scenario_patch.py` - Test YAML patching
  - [ ] `test_k8s_configmap.sh` - Test ConfigMap creation
  - [ ] `test_k8s_job_generator.py` - Test Job YAML generation

- [ ] Implement integration tests
  - [ ] `test_native_experiment.sh` - Complete native workflow
  - [ ] `test_minikube_experiment.sh` - Complete Minikube workflow
  - [ ] `test_gcp_experiment.sh` - Complete GCP workflow
  - [ ] `test_analysis_pipeline.sh` - Analysis pipeline workflow

- [ ] Implement functional tests (aligned with REQUIREMENTS_SPECIFICATION.md)
  - [ ] `test_fr1_latency.sh` - Latency measurement (nanosecond precision)
  - [ ] `test_fr2_throughput.sh` - Throughput calculation
  - [ ] `test_fr3_multi_environment.sh` - Multi-environment consistency
  - [ ] `test_fr4_statistical_analysis.sh` - Statistical analysis
  - [ ] `test_fr5_horizontal_scaling.sh` - Horizontal scaling
  - [ ] `test_fr6_resource_utilization.sh` - Resource utilization
  - [ ] `test_fr7_data_completeness.sh` - Data completeness
  - [ ] `test_fr8_data_validation.sh` - Data validation
  - [ ] `test_fr9_queue_delay.sh` - Queue delay analysis
  - [ ] `test_fr14_experiment_isolation.sh` - Experiment isolation
  - [ ] `test_fr15_analysis_robustness.sh` - Analysis robustness

- [ ] Implement regression tests
  - [ ] `capture_baseline.sh` - Capture expected outputs
  - [ ] `compare_after_refactor.sh` - Compare after refactoring

- [ ] Implement smoke tests
  - [ ] `test_all_environments.sh` - Quick validation per environment

- [ ] Run baseline tests
  - [ ] All unit tests pass
  - [ ] All integration tests pass
  - [ ] All functional tests pass
  - [ ] Baseline outputs captured

- [ ] Document test execution
  - [ ] Test execution guide
  - [ ] Test requirements (prerequisites)
  - [ ] Test success criteria
  - [ ] Troubleshooting guide

---

**Last Updated**: 2025-12-10  
**Status**: Planning Phase - **TESTING STRATEGY ADDED**

