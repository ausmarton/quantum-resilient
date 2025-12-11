# Parallelism Strategy for Experiment Execution

## Current Implementation

### GCP/GKE Parallelism

**Current Behavior**:
- `PARALLEL_JOBS > 1`: Submits jobs in **batches** of `PARALLEL_JOBS`
- Jobs are tracked in `JOB_TRACKING_FILE`
- After all submissions, waits for all jobs to complete
- **NOT submitting all jobs at once** - batches are submitted sequentially

**Example with PARALLEL_JOBS=20 and 100 experiments**:
1. Submit jobs 1-20 → track them
2. Submit jobs 21-40 → track them
3. Submit jobs 41-60 → track them
4. Submit jobs 61-80 → track them
5. Submit jobs 81-100 → track them
6. Wait for all 100 jobs to complete

**Issue**: This is still batching - not truly letting Kubernetes schedule all jobs at once.

### Minikube Parallelism

**Current Behavior**:
- **NO parallelism** - always runs sequentially
- Each experiment waits for completion before starting next
- `PARALLEL_JOBS` parameter is **ignored** for Minikube

**Issue**: Minikube could benefit from parallelism, but needs resource limits to avoid overutilization.

### Native Parallelism

**Current Behavior**:
- **NO parallelism** - always runs sequentially
- Each experiment waits for completion before starting next
- `PARALLEL_JOBS` parameter is **ignored** for Native

**Rationale**: Native should be baseline - no noisy neighbors, no resource contention.

---

## Proposed Improvements

### 1. GCP/GKE: Submit All Jobs at Once

**Better Approach**: Submit ALL jobs immediately, let Kubernetes scheduler handle parallelism.

**Benefits**:
- Kubernetes scheduler optimizes based on available nodes
- No artificial batching limits
- Better resource utilization
- Simpler code (no batch tracking)

**Implementation**:
```bash
# For GCP with persistent cluster:
# 1. Submit ALL jobs immediately (non-blocking)
# 2. Track all job names
# 3. Wait for all to complete
# 4. Download results
```

**Resource Management**:
- Kubernetes will schedule based on:
  - Available nodes
  - Resource requests/limits
  - Pod anti-affinity rules
- No need to pre-calculate node count based on PARALLEL_JOBS

### 2. Minikube: Conditional Parallelism

**Requirements**:
1. Minikube should be closer to native (avoid noisy neighbors)
2. But should support parallelism for efficiency
3. Must not overutilize local machine resources

**Proposed Strategy**:
- **Default**: Sequential (like native) - `PARALLEL_JOBS=1` or unset
- **Optional**: Limited parallelism with resource constraints
  - `PARALLEL_JOBS=2-4` max for Minikube
  - Use Kubernetes resource requests/limits to prevent overutilization
  - Monitor system load and warn if high

**Implementation**:
```bash
# For Minikube:
if [[ "$env" == "minikube" ]]; then
    if [[ $PARALLEL_JOBS -gt 1 ]]; then
        # Limit parallelism for Minikube
        MAX_MINIKUBE_PARALLEL=4
        if [[ $PARALLEL_JOBS -gt $MAX_MINIKUBE_PARALLEL ]]; then
            log_warn "Limiting Minikube parallelism to $MAX_MINIKUBE_PARALLEL (requested: $PARALLEL_JOBS)"
            PARALLEL_JOBS=$MAX_MINIKUBE_PARALLEL
        fi
        
        # Check system load before enabling parallelism
        if ! check_system_load --warn-threshold 1.0; then
            log_warn "System load high, using sequential mode for Minikube"
            PARALLEL_JOBS=1
        fi
    fi
fi
```

### 3. Native: Always Sequential

**Keep as-is**: Native should always run sequentially to maintain baseline measurements without resource contention.

---

## Implementation Plan

### Phase 1: GCP - Submit All Jobs at Once

1. Remove batch logic from GCP execution
2. Submit all jobs immediately (non-blocking)
3. Track all job names
4. Wait for all to complete
5. Download results

**Code Changes**:
- Remove `PARALLEL_JOBS` batching logic
- Submit all jobs in loop (non-blocking)
- Collect all job names
- Wait for all at end

### Phase 2: Minikube - Add Conditional Parallelism

1. Add `PARALLEL_JOBS` support for Minikube
2. Add resource limits to prevent overutilization
3. Add system load check
4. Limit max parallelism (e.g., 4)

**Code Changes**:
- Check `PARALLEL_JOBS` for Minikube
- If > 1, submit multiple jobs (with limits)
- Use Kubernetes resource requests/limits
- Check system load before enabling

### Phase 3: Documentation

1. Document parallelism strategy
2. Update usage examples
3. Add warnings about resource usage

---

## Resource Management

### GCP/GKE

**Current**: Pre-calculates node count based on `PARALLEL_JOBS`
**Proposed**: Let Kubernetes handle it - submit all jobs, scheduler decides

**Pod Anti-Affinity**: Already configured to spread pods across nodes
**Resource Requests**: Already configured (CPU: 500m, Memory: 512Mi)

### Minikube

**Proposed**: 
- Resource requests: CPU: 1, Memory: 1Gi (per job)
- Limit total: Max 4 parallel jobs
- System load check before enabling

**Example**:
- Minikube with 4 CPUs, 8GB RAM
- Each job requests: 1 CPU, 1GB RAM
- Max 4 parallel jobs = 4 CPUs, 4GB RAM
- Leaves resources for system

### Native

**Keep as-is**: No parallelism, no resource management needed

---

## Migration Path

### Step 1: GCP - Submit All Jobs

Change from:
```bash
# Current: Batch submission
if [[ $PARALLEL_JOBS -gt 1 ]]; then
    # Submit in batches
    # Track jobs
    # Wait at end
fi
```

To:
```bash
# Proposed: Submit all at once
# Submit all jobs immediately (non-blocking)
# Track all job names
# Wait for all to complete
```

### Step 2: Minikube - Add Parallelism Support

Add:
```bash
# For Minikube with PARALLEL_JOBS > 1:
# 1. Check system load
# 2. Limit to max 4
# 3. Submit multiple jobs
# 4. Wait for all
```

### Step 3: Update Documentation

- Update parallel execution guide
- Add Minikube parallelism section
- Document resource limits

---

## Testing Plan

1. **GCP**: Test submitting 100 jobs at once
   - Verify Kubernetes schedules appropriately
   - Verify no resource exhaustion
   - Verify all jobs complete

2. **Minikube**: Test with PARALLEL_JOBS=2, 4
   - Verify system load check works
   - Verify resource limits prevent overutilization
   - Verify jobs complete successfully

3. **Native**: Verify still sequential (no changes)

---

## Benefits

1. **GCP**: Better resource utilization, simpler code, faster execution
2. **Minikube**: Can parallelize safely with resource limits
3. **Native**: Remains baseline (no changes)
4. **Consistency**: All environments use same pattern (submit all, wait all)
