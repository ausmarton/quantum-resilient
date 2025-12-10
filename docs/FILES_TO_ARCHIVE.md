# Files to Archive or Remove

After documentation reorganization, these files in the root directory have been consolidated or moved to `docs/`.

## Recommendation: Archive These Files

These files have been consolidated into new documents in `docs/`. They can be safely archived or removed after verification.

### Consolidated (Content Merged)

1. **GCP Documentation** (→ `docs/reference/gcp-deployment.md`):
   - `GCP_ISOLATION_AND_SIZING.md`
   - `GCP_PARALLEL_EXECUTION.md`
   - `PARALLEL_EXECUTION_GUIDE.md`
   - `UNIFIED_GCP_IMPLEMENTATION.md`
   - `GCP_OPTIMIZATION_ANALYSIS.md`
   - `OPTIMIZATION_IMPLEMENTATION.md`

2. **Scaling Documentation** (→ `docs/reference/scaling-experiments.md`):
   - `HORIZONTAL_SCALING_ANALYSIS.md`
   - `HORIZONTAL_SCALING_DISSERTATION_GUIDE.md`

3. **Data Validation** (→ `docs/reference/data-validation.md`):
   - `DATA_VALIDATION_REPORT.md` (older report, key points merged)
   - `DATA_SUFFICIENCY_CHECK.md` (key points merged)
   - `DATA_SUFFICIENCY_SUMMARY.md` (key points merged)

4. **Cluster Sizing** (→ `docs/reference/cluster-sizing.md`):
   - `CLUSTER_SIZING_ANALYSIS.md`

## Recommendation: Keep These Files (For Now)

These files are still referenced or may be useful for historical context:

- `RESEARCHER_GUIDE.md` - May want to create `docs/guides/getting-started.md` from this
- `README.md` - Main project README, keep but updated

## Files Already Moved

These files have been moved to `docs/` and can be removed from root:

- `FULL_SCALE_DATA_COLLECTION_GUIDE.md` → `docs/guides/data-collection.md`
- `STORAGE_AND_OUTPUT_GUIDE.md` → `docs/guides/storage-and-output.md`
- `STOP_AND_RESUME_GUIDE.md` → `docs/guides/stop-and-resume.md`
- `RE_RUN_EXPERIMENTS_GUIDE.md` → `docs/guides/re-running-experiments.md`
- `SYSTEM_LOAD_AND_VARIABILITY.md` → `docs/reference/system-requirements.md`
- `DATA_VALIDATION_SUMMARY.md` → `docs/reference/data-validation.md`
- `EXPERIMENTAL_DESIGN_ANALYSIS.md` → `docs/analysis/experimental-design.md`
- `HARDWARE_CONSISTENCY_ANALYSIS.md` → `docs/analysis/hardware-consistency.md`
- `COST_AND_TIME_ANALYSIS.md` → `docs/analysis/cost-analysis.md`
- `ENTERPRISE_REPRESENTATIVENESS_ANALYSIS.md` → `docs/analysis/enterprise-representativeness.md`
- `GIT_PUSH_FIX.md` → `docs/troubleshooting/git-push-fix.md`
- `SCALING_EXPERIMENTS_FIX.md` → `docs/troubleshooting/scaling-fix.md`

## Archive Command

If you want to archive these files:

```bash
# Create archive directory
mkdir -p archive/documentation-2025-12-10

# Move consolidated files
mv GCP_ISOLATION_AND_SIZING.md \
   GCP_PARALLEL_EXECUTION.md \
   PARALLEL_EXECUTION_GUIDE.md \
   UNIFIED_GCP_IMPLEMENTATION.md \
   GCP_OPTIMIZATION_ANALYSIS.md \
   OPTIMIZATION_IMPLEMENTATION.md \
   HORIZONTAL_SCALING_ANALYSIS.md \
   HORIZONTAL_SCALING_DISSERTATION_GUIDE.md \
   CLUSTER_SIZING_ANALYSIS.md \
   DATA_VALIDATION_REPORT.md \
   DATA_SUFFICIENCY_CHECK.md \
   DATA_SUFFICIENCY_SUMMARY.md \
   archive/documentation-2025-12-10/

# Move files that were relocated (optional - keep for reference)
mv FULL_SCALE_DATA_COLLECTION_GUIDE.md \
   STORAGE_AND_OUTPUT_GUIDE.md \
   STOP_AND_RESUME_GUIDE.md \
   RE_RUN_EXPERIMENTS_GUIDE.md \
   SYSTEM_LOAD_AND_VARIABILITY.md \
   DATA_VALIDATION_SUMMARY.md \
   EXPERIMENTAL_DESIGN_ANALYSIS.md \
   HARDWARE_CONSISTENCY_ANALYSIS.md \
   COST_AND_TIME_ANALYSIS.md \
   ENTERPRISE_REPRESENTATIVENESS_ANALYSIS.md \
   GIT_PUSH_FIX.md \
   SCALING_EXPERIMENTS_FIX.md \
   archive/documentation-2025-12-10/
```

## Verification Before Archiving

Before archiving, verify:
1. ✅ All new documents in `docs/` contain the information you need
2. ✅ No important details were lost in consolidation
3. ✅ All links in new documents work correctly
4. ✅ Main README.md points to correct locations

