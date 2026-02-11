# Documentation Consolidation Plan

**Date**: 2025-12-15  
**Purpose**: Plan for consolidating root-level documentation files into `docs/` directory

---

## Files to Consolidate

### Move to `docs/reference/`

- `ALGORITHM_NAMING_STANDARD.md` → `docs/reference/algorithm-naming-standard.md`
  - **Reason**: Technical reference documentation

### Move to `docs/`

- `DEVELOPMENT_GUIDELINES.md` → `docs/development-guidelines.md`
  - **Reason**: Development documentation (keep at root level for visibility, but also link from docs/)
- `WRITING_GUIDELINES.md` → `docs/writing-guidelines.md`
  - **Reason**: Writing/documentation guidelines
- `DISSERTATION_READINESS_CHECKLIST.md` → `docs/dissertation-readiness-checklist.md`
  - **Reason**: Dissertation-related documentation

### Move to `docs/archive/` or `archive/`

- `DISSERTATION_READY.md` → `archive/dissertation-ready.md`
  - **Reason**: Status document, likely historical
- `DISSERTATION_READY_ECDHE.md` → `archive/dissertation-ready-ecdhe.md`
  - **Reason**: Status document, likely historical
- `REDUNDANCY_ANALYSIS.md` → `archive/redundancy-analysis.md`
  - **Reason**: Analysis document, likely historical
- `REDUNDANCY_LOG.md` → `archive/redundancy-log.md`
  - **Reason**: Log document, likely historical
- `REDUNDANCY_REMOVAL_SUMMARY.md` → `archive/redundancy-removal-summary.md`
  - **Reason**: Summary document, likely historical

### Keep at Root Level

- `README.md` - Main project README (should stay at root)
- `ARCHIVE.md` - Archive index (should stay at root)
- `TODO.md` - Active work tracking (should stay at root)
- `DEVELOPMENT_GUIDELINES.md` - Keep at root for visibility, but also link from docs/

---

## Update References

After moving files, update all references:
1. Update `docs/README.md` to link to moved files
2. Update `README.md` to link to consolidated docs
3. Search codebase for references to old paths
4. Update any scripts that reference these files

---

## Execution Order

1. Create target directories if needed
2. Move files to new locations
3. Update all references
4. Verify no broken links
5. Update documentation index

---

**Status**: Ready to execute

