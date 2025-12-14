# SELinux Container Permission Fix

**Date**: 2025-12-14  
**Issue**: Container cannot read files created by host process  
**Solution**: Use `:z` flag (shared SELinux context) instead of `:Z`

---

## Problem

When running Python analysis scripts in containers on SELinux-enabled systems (Fedora/RHEL), the container cannot read files created by the host process, resulting in:

```
PermissionError: [Errno 13] Permission denied: 'results/.../merged.jsonl'
```

This occurs even when files have correct permissions (`chmod 644`) because SELinux enforces additional security contexts.

---

## Root Cause

Podman/Docker volume mounts support SELinux context flags:
- **`:Z`** = Private SELinux context (creates unique context)
- **`:z`** = Shared SELinux context (allows multiple containers to share)

The original implementation used `:Z`, which creates a private context that may not be accessible to the container process.

---

## Solution

**Changed in**: `scripts/lib/run-python-container.sh`

**Before**:
```bash
VOLUME_FLAGS="-v $SCRIPT_DIR:/workspace:rw,Z"
```

**After**:
```bash
VOLUME_FLAGS="-v $SCRIPT_DIR:/workspace:rw,z"
```

The `:z` flag tells SELinux to use a shared content label, allowing the container to access files created by the host process.

---

## Verification

After the fix:
- ✅ Container can read merged.jsonl files
- ✅ Container can write summary.json files
- ✅ No permission errors
- ✅ Summary generation works perfectly

**Test**:
```bash
./scripts/lib/run-python-container.sh analysis/scripts/compute_statistics.py \
  --input results/native/test/merged/merged.jsonl \
  --output results/native/test/stats \
  --experiment-id test
```

---

## Alternative Solutions (Not Used)

1. **Set SELinux to permissive** (not recommended for security)
2. **Use `chcon` to set context** (requires sudo, complex)
3. **Run without containerization** (requires host Python dependencies)

---

## Related Files

- `scripts/lib/run-python-container.sh` - Container wrapper script
- `scripts/generate_experiment_summaries.sh` - Summary generation script

---

**Status**: ✅ **FIXED** - Using `:z` flag resolves all permission issues
