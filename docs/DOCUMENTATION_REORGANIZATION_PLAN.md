# Documentation Reorganization Plan

## Current Issues

1. **Redundancies**: Multiple documents covering the same topics
2. **Scattered**: 25+ markdown files in root directory
3. **Outdated**: Some documents reference old experiment counts or script names
4. **No clear structure**: Hard to find relevant documentation

## Proposed Structure

```
docs/
├── README.md                          # Documentation index
├── guides/                            # User guides (how to use)
│   ├── getting-started.md            # Quick start guide
│   ├── running-experiments.md         # Consolidated from RESEARCHER_GUIDE.md
│   ├── data-collection.md             # Consolidated from FULL_SCALE_DATA_COLLECTION_GUIDE.md
│   ├── storage-and-output.md          # From STORAGE_AND_OUTPUT_GUIDE.md
│   ├── stop-and-resume.md             # From STOP_AND_RESUME_GUIDE.md
│   └── re-running-experiments.md      # From RE_RUN_EXPERIMENTS_GUIDE.md
├── reference/                         # Technical reference
│   ├── system-requirements.md         # From SYSTEM_LOAD_AND_VARIABILITY.md
│   ├── gcp-deployment.md              # Consolidated GCP docs
│   ├── scaling-experiments.md         # Consolidated scaling docs
│   └── data-validation.md             # Consolidated validation docs
├── analysis/                          # Research analysis documents
│   ├── experimental-design.md         # From EXPERIMENTAL_DESIGN_ANALYSIS.md
│   ├── hardware-consistency.md        # From HARDWARE_CONSISTENCY_ANALYSIS.md
│   ├── cluster-sizing.md              # Consolidated sizing docs
│   └── cost-analysis.md                # From COST_AND_TIME_ANALYSIS.md
└── troubleshooting/                   # Historical fixes and troubleshooting
    ├── git-push-fix.md                # From GIT_PUSH_FIX.md
    └── scaling-fix.md                 # From SCALING_EXPERIMENTS_FIX.md
```

## Consolidation Plan

### 1. User Guides (guides/)

**running-experiments.md** (NEW - consolidates):
- RESEARCHER_GUIDE.md (main content)
- Key sections from README.md (Quick Start, Local, Minikube, GCP)

**data-collection.md** (RENAME):
- FULL_SCALE_DATA_COLLECTION_GUIDE.md → guides/data-collection.md

**storage-and-output.md** (RENAME):
- STORAGE_AND_OUTPUT_GUIDE.md → guides/storage-and-output.md

**stop-and-resume.md** (RENAME):
- STOP_AND_RESUME_GUIDE.md → guides/stop-and-resume.md

**re-running-experiments.md** (RENAME):
- RE_RUN_EXPERIMENTS_GUIDE.md → guides/re-running-experiments.md

### 2. Reference (reference/)

**gcp-deployment.md** (NEW - consolidates):
- GCP_ISOLATION_AND_SIZING.md
- GCP_PARALLEL_EXECUTION.md
- PARALLEL_EXECUTION_GUIDE.md
- UNIFIED_GCP_IMPLEMENTATION.md
- GCP_OPTIMIZATION_ANALYSIS.md
- OPTIMIZATION_IMPLEMENTATION.md

**scaling-experiments.md** (NEW - consolidates):
- HORIZONTAL_SCALING_ANALYSIS.md
- HORIZONTAL_SCALING_DISSERTATION_GUIDE.md

**data-validation.md** (NEW - consolidates):
- DATA_VALIDATION_SUMMARY.md (keep as primary)
- DATA_VALIDATION_REPORT.md (archive or merge key points)
- DATA_SUFFICIENCY_SUMMARY.md (merge key points)
- DATA_SUFFICIENCY_CHECK.md (archive or merge key points)

**system-requirements.md** (RENAME):
- SYSTEM_LOAD_AND_VARIABILITY.md → reference/system-requirements.md

**cluster-sizing.md** (NEW - consolidates):
- CLUSTER_SIZING_ANALYSIS.md
- Key sizing info from GCP_ISOLATION_AND_SIZING.md

### 3. Analysis (analysis/)

**experimental-design.md** (RENAME):
- EXPERIMENTAL_DESIGN_ANALYSIS.md → analysis/experimental-design.md

**hardware-consistency.md** (RENAME):
- HARDWARE_CONSISTENCY_ANALYSIS.md → analysis/hardware-consistency.md

**cost-analysis.md** (RENAME):
- COST_AND_TIME_ANALYSIS.md → analysis/cost-analysis.md

**enterprise-representativeness.md** (RENAME):
- ENTERPRISE_REPRESENTATIVENESS_ANALYSIS.md → analysis/enterprise-representativeness.md

### 4. Troubleshooting (troubleshooting/)

**git-push-fix.md** (RENAME):
- GIT_PUSH_FIX.md → troubleshooting/git-push-fix.md

**scaling-fix.md** (RENAME):
- SCALING_EXPERIMENTS_FIX.md → troubleshooting/scaling-fix.md

## Files to Archive/Remove

These files will be consolidated into new documents:
- DATA_VALIDATION_REPORT.md (merge into data-validation.md)
- DATA_SUFFICIENCY_CHECK.md (merge into data-validation.md)
- CLUSTER_SIZING_ANALYSIS.md (merge into cluster-sizing.md)
- PARALLEL_EXECUTION_GUIDE.md (merge into gcp-deployment.md)
- GCP_PARALLEL_EXECUTION.md (merge into gcp-deployment.md)
- UNIFIED_GCP_IMPLEMENTATION.md (merge into gcp-deployment.md)
- GCP_OPTIMIZATION_ANALYSIS.md (merge into gcp-deployment.md)
- OPTIMIZATION_IMPLEMENTATION.md (merge into gcp-deployment.md)
- HORIZONTAL_SCALING_ANALYSIS.md (merge into scaling-experiments.md)
- HORIZONTAL_SCALING_DISSERTATION_GUIDE.md (merge into scaling-experiments.md)

## Update README.md

Update README.md to reference new documentation structure:
- Replace inline documentation sections with links to docs/
- Keep only essential quick start info
- Point to docs/guides/ for detailed guides

## Verification Checklist

- [ ] All documents moved to new structure
- [ ] Redundant content merged
- [ ] Outdated information updated
- [ ] All internal links updated
- [ ] README.md updated with new structure
- [ ] Old files archived or removed
- [ ] Documentation index created

