# Documentation Verification Checklist

Use this checklist to verify all documentation is accurate and up-to-date.

## Experiment Counts

✅ **Verified**: Current experiment counts are correct (updated with ECDHE)
- Native: 576 experiments (120 unique configs)
- Minikube: 648 experiments (576 baseline + 72 scaling)
- GCP: 648 experiments (576 baseline + 72 scaling)
- Total: 1,872 experiments

**Breakdown**:
- Core baseline: 360 (6 algorithms × 4 payloads × 3 rates × 5 runs)
- Burst pattern: 60 (6 algorithms × 2 payloads × 1 rate × 5 runs)
- 10K msg/s: 120 (6 algorithms × 4 payloads × 1 rate × 5 runs)
- 5-minute duration: 18 (6 algorithms × 1 payload × 1 rate × 3 runs)
- Scaling baseline: 18 (6 algorithms × 1 payload × 1 rate × 3 runs)
- **Total baseline**: 576
- **Scaling additional**: 72 (18 scenarios × 4 replica counts: 1,2,4,8, but native only uses replica 1)

## Script References

✅ **Verified**: All script references are correct
- `run_all_experiments.sh` - ✅ Exists
- `run_full_scale_data_collection.sh` - ✅ Exists
- `run_minikube.sh` - ✅ Exists
- `run_local.sh` - ✅ Exists
- `deploy_gcp.sh` - ✅ Exists
- `scripts/submit_gcp_job_parallel.sh` - ✅ Exists
- `scripts/validate_data_quality.sh` - ✅ Exists

## File Paths

✅ **Verified**: All file paths referenced are correct
- `orchestration/experiment_matrix.yaml` - ✅ Exists
- `results/<env>/<scenario-id>/` - ✅ Correct structure
- `final-results/` - ✅ Exists
- `generated-scenarios/` - ✅ Exists

## Claims Verification

### System Load
✅ **Accurate**: Native and Minikube are affected by system load, GCP is not

### Scaling Experiments
✅ **Accurate**: Native doesn't support replicas > 1, Minikube and GCP do

### Data Validation
✅ **Accurate**: 1,456 experiments validated, 1,455 valid (99.9%)

### Cluster Sizing
✅ **Accurate**: Formula is `PARALLEL_JOBS + 1` nodes

### GCP Deployment
✅ **Accurate**: Ephemeral mode for single experiments, persistent cluster for batches

## Outdated Information Found

### Fixed
- ✅ Updated "300 baseline experiments" references to clarify it's the core subset
- ✅ Verified experiment counts match actual matrix

### Needs Review
- ⚠️ Some documents may reference old script behavior (check during review)
- ⚠️ Cost estimates may need updating if GCP pricing changed

## Documentation Structure

✅ **Complete**: All documents organized in `docs/` structure
✅ **Index**: `docs/README.md` provides navigation
✅ **Main README**: Updated to reference new structure

## Next Steps

1. Review consolidated documents for any remaining redundancies
2. Verify all internal links work
3. Archive or remove old files from root directory (after user approval)
4. Create getting-started guide from RESEARCHER_GUIDE.md (optional)

