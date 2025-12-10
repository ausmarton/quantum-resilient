# GCP Run Optimization Analysis

## Current Inefficiencies

### 1. **Cluster Creation/Destruction Per Experiment (Ephemeral Mode)**

**Current Behavior:**
- Each experiment in ephemeral mode:
  1. Runs `terraform init` (~5-10s)
  2. Imports existing resources (bucket, SA, AR) (~10-20s)
  3. Runs `terraform apply` to create cluster (~3-5 minutes)
  4. Builds container image (~1-2 minutes)
  5. Pushes image to registry (~30s-1min)
  6. Runs experiment (~30s-2min)
  7. Runs `terraform destroy` (~2-5 minutes)
  8. Cleanup verification (~10s)

**Total overhead per experiment: ~8-15 minutes** (vs ~2-3 minutes for actual benchmark)

**For 468 experiments:** ~62-117 hours of overhead vs ~15-23 hours of actual work

### 2. **Terraform Operations Redundancy**

**Wasteful operations:**
- `terraform init` runs every time (downloads providers, initializes backend)
- Resource imports happen every time (bucket, service account, AR repository)
- State cleanup operations (removing Kubernetes resources from state)
- Terraform plan/apply even when cluster already exists

**Optimization opportunity:** Cache Terraform state, skip init if already initialized

### 3. **Container Image Building**

**Current behavior:**
- Image is built fresh for every experiment
- Image is pushed to registry every time
- No caching or reuse

**Minikube comparison:**
- Minikube has `--skip-build` flag to reuse existing images
- Image is built once, loaded into minikube, reused for multiple experiments

**Optimization opportunity:** 
- Build image once per batch
- Tag with git commit or timestamp
- Reuse for all experiments in batch
- Only rebuild if code changes

### 4. **Unnecessary Resource Imports**

**Current behavior:**
- Every experiment tries to import:
  - GCS bucket (already exists)
  - Service account (already exists)
  - Artifact Registry repository (already exists)

**These are persistent resources** - they don't need to be imported every time.

## Optimization Recommendations

### Option 1: Persistent Cluster Mode (Recommended for Batch Runs)

**For running 468 experiments, use persistent cluster:**

```bash
# Create cluster once (outside experiment loop)
./deploy_gcp.sh --create-cluster \
  --project <project> \
  --bucket <bucket> \
  --region <region>

# Run all experiments with --skip-terraform
for scenario in scenarios/*.yaml; do
  ./deploy_gcp.sh \
    --scenario "$scenario" \
    --exp-id "$(basename $scenario .yaml)" \
    --project <project> \
    --bucket <bucket> \
    --skip-terraform \
    --skip-build  # Reuse image from first run
done

# Destroy cluster when done
./deploy_gcp.sh --destroy-cluster \
  --project <project> \
  --bucket <bucket> \
  --region <region>
```

**Time savings:**
- Cluster creation: 1x (~5 min) vs 468x (~2340 min)
- Terraform init: 1x vs 468x
- Image build: 1x vs 468x
- **Total savings: ~35-40 hours for 468 experiments**

### Option 2: Optimize Ephemeral Mode

If ephemeral mode is required (for cost isolation), optimize:

1. **Cache Terraform state:**
   - Keep `.terraform/` directory between runs
   - Skip `terraform init` if already initialized
   - Use Terraform backend (GCS) for state

2. **Skip resource imports:**
   - Check if resources exist in state before importing
   - Only import if missing

3. **Reuse container images:**
   - Build once per batch
   - Tag with batch ID
   - Use `--skip-build` for subsequent experiments

4. **Parallel cluster operations:**
   - Create next cluster while current experiment runs
   - Use cluster pools (advanced)

### Option 3: Hybrid Approach

**Best of both worlds:**

1. **Create persistent cluster** for batch runs
2. **Build image once** at start of batch
3. **Run all experiments** with `--skip-terraform --skip-build`
4. **Destroy cluster** at end of batch

**Implementation in `run_all_experiments.sh`:**

```bash
# For GCP environment, check if cluster exists
if [[ "$env" == "gcp" ]]; then
    # Check if cluster exists
    if ! gcloud container clusters describe "$CLUSTER_NAME" \
        --region "$REGION" \
        --project "$PROJECT" &>/dev/null; then
        # Create cluster once
        log_info "Creating persistent cluster for batch run..."
        "$SCRIPT_DIR/deploy_gcp.sh" --create-cluster \
            --project "$PROJECT" \
            --bucket "$BUCKET" \
            --region "$REGION"
        
        # Build image once
        log_info "Building container image for batch run..."
        "$SCRIPT_DIR/deploy_gcp.sh" \
            --scenario "$FIRST_SCENARIO" \
            --exp-id "batch-setup" \
            --project "$PROJECT" \
            --bucket "$BUCKET" \
            --region "$REGION" \
            --skip-terraform \
            --skip-aggregation
    fi
    
    # Run all experiments with --skip-terraform --skip-build
    SKIP_TERRAFORM=true
    SKIP_BUILD=true
fi

# After all experiments complete
if [[ "$env" == "gcp" ]]; then
    log_info "Destroying cluster after batch completion..."
    "$SCRIPT_DIR/deploy_gcp.sh" --destroy-cluster \
        --project "$PROJECT" \
        --bucket "$BUCKET" \
        --region "$REGION"
fi
```

## Container Image Optimization

### Current Issues

1. **Full rebuild every time:**
   - No layer caching between experiments
   - No incremental builds
   - No image reuse

2. **No build optimization:**
   - Could use multi-stage builds more efficiently
   - Could cache dependencies

### Recommendations

1. **Add `--skip-build` flag support:**
   ```bash
   if [[ "$SKIP_BUILD" != "true" ]]; then
       # Build image
   else
       log_info "Skipping build, using existing image: $IMAGE_NAME"
   fi
   ```

2. **Image tagging strategy:**
   - Tag with git commit: `pqc-bench:$(git rev-parse --short HEAD)`
   - Tag with timestamp: `pqc-bench:batch-$(date +%Y%m%d-%H%M%S)`
   - Check if tagged image exists before building

3. **Build cache:**
   - Use Podman build cache
   - Keep intermediate layers
   - Only rebuild if dependencies change

## Terraform Optimization

### Current Issues

1. **`terraform init` every time:**
   - Downloads providers (~10-20s)
   - Initializes backend
   - No caching

2. **Redundant imports:**
   - Imports bucket/SA/AR every time
   - These are persistent resources

3. **State operations:**
   - Removes Kubernetes resources from state every time
   - Could be cached

### Recommendations

1. **Cache Terraform state:**
   ```bash
   # Check if .terraform/ exists
   if [[ ! -d "$TERRAFORM_DIR/.terraform" ]]; then
       terraform init -input=false
   else
       log_info "Terraform already initialized, skipping init"
   fi
   ```

2. **Skip imports if in state:**
   ```bash
   # Check if resource is in state before importing
   if ! terraform state show google_storage_bucket.results &>/dev/null; then
       terraform import google_storage_bucket.results "$BUCKET"
   fi
   ```

3. **Use remote state backend:**
   - Store state in GCS
   - Share state between runs
   - Avoid local state issues

## Comparison: Minikube vs GCP

### Minikube Approach (Efficient)
1. **Cluster:** Started once manually, reused for all experiments
2. **Image:** Built once, loaded into minikube, reused
3. **Terraform:** Not used (local cluster)
4. **Overhead per experiment:** ~10-30s (just job creation/execution)

### GCP Approach (Current - Inefficient)
1. **Cluster:** Created/destroyed per experiment (ephemeral) or manually managed
2. **Image:** Built and pushed per experiment
3. **Terraform:** Runs init/apply/destroy per experiment
4. **Overhead per experiment:** ~8-15 minutes

### GCP Approach (Optimized)
1. **Cluster:** Created once per batch, reused, destroyed at end
2. **Image:** Built once per batch, reused
3. **Terraform:** Runs once per batch (create), once at end (destroy)
4. **Overhead per experiment:** ~10-30s (similar to minikube)

## Implementation Priority

### High Priority (Immediate Impact)
1. ✅ Add `--skip-build` flag support
2. ✅ Add persistent cluster mode for batch runs
3. ✅ Cache Terraform init
4. ✅ Skip redundant resource imports

### Medium Priority (Significant Savings)
1. Image tagging and reuse
2. Terraform state caching
3. Parallel operations where possible

### Low Priority (Nice to Have)
1. Advanced build caching
2. Cluster pools
3. Pre-warming strategies

## Expected Time Savings

**For 468 experiments:**

| Optimization | Current Time | Optimized Time | Savings |
|-------------|--------------|----------------|---------|
| Cluster ops | ~2340 min | ~5 min | ~2335 min |
| Terraform init | ~2340 min | ~5 min | ~2335 min |
| Image builds | ~468-936 min | ~2-4 min | ~464-932 min |
| **Total overhead** | **~5148-5616 min** | **~12-14 min** | **~5134-5602 min** |
| **Total overhead** | **~86-94 hours** | **~0.2 hours** | **~85-93 hours** |

**Actual benchmark time:** ~15-23 hours (unchanged)

**Total time:** ~101-117 hours → **~15-23 hours** (5-6x faster)

## Cost Implications

**Current (ephemeral):**
- Cluster creation/destruction: ~8-15 min overhead per experiment
- Compute cost: Only during actual benchmark (~30s-2min)
- **Waste:** Paying for cluster setup/teardown time

**Optimized (persistent):**
- Cluster exists for batch duration
- Compute cost: Only during actual benchmarks
- **Savings:** No repeated setup/teardown costs
- **Trade-off:** Cluster runs between experiments (minimal cost for idle time)

**For 468 experiments:**
- Idle time between experiments: ~1-2 seconds
- Total idle time: ~8-16 minutes
- Cost: Negligible (cluster idle cost is minimal)
- **Net savings:** Eliminate 85-93 hours of setup/teardown overhead

## Recommendations Summary

1. **For batch runs (468 experiments):** Use persistent cluster mode
2. **For single experiments:** Ephemeral mode is fine (cost isolation)
3. **Always:** Reuse container images within a batch
4. **Always:** Cache Terraform operations
5. **Consider:** Pre-building images for common scenarios

## Next Steps

1. Implement `--skip-build` flag in `deploy_gcp.sh`
2. Add persistent cluster mode to `run_all_experiments.sh`
3. Optimize Terraform operations (cache init, skip redundant imports)
4. Add image tagging and reuse logic
5. Update documentation with optimization guidance

