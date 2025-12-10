# Git Push Fix

## Problem
Git push was failing with HTTP 500 error when trying to push 2.87 GiB of data. The issue was caused by large result files (384MB each) being tracked in git.

## Solution Applied

1. **Added `results/` to `.gitignore`**
   - Result files are now ignored by git
   - These files should be stored externally (GCS for GCP, local storage for native/minikube)

2. **Removed result files from git tracking**
   - Removed 4,414 result files from git index using `git rm -r --cached results/`
   - Files remain on disk, just no longer tracked by git

3. **Increased git buffer sizes**
   - Set `http.postBuffer` to 500MB
   - Set `http.maxRequestBuffer` to 100MB

## Next Steps

### Try pushing again:
```bash
git push
```

### If push still fails:

The git history still contains the large files (3.6GB in `.git/objects`). You have a few options:

#### Option 1: Force push (if you're the only contributor)
```bash
git push --force
```

#### Option 2: Clean git history (removes large files from history)
```bash
# Install git-filter-repo if needed
# pip install git-filter-repo

# Remove results/ from entire git history
git filter-repo --path results/ --invert-paths

# Force push (WARNING: rewrites history)
git push --force --all
```

#### Option 3: Use Git LFS for large files (if you need to track some result files)
```bash
git lfs install
git lfs track "*.jsonl"
git add .gitattributes
git commit -m "Add Git LFS tracking for large files"
```

#### Option 4: Create a fresh repository (nuclear option)
If the above don't work and you're okay losing history:
```bash
# Backup current state
cp -r . ../quantum-resilient-backup

# Remove .git and reinitialize
rm -rf .git
git init
git add .
git commit -m "Initial commit (after removing large files)"
git remote add origin <your-remote-url>
git push -u origin main --force
```

## Prevention

- ✅ `results/` is now in `.gitignore`
- ✅ `data-collection-*/` is already in `.gitignore`
- ✅ `generated-scenarios/` is already in `.gitignore`

Result files should be:
- **GCP**: Stored in GCS bucket
- **Native/Minikube**: Stored locally, backed up separately if needed
- **Not in git**: Too large and change frequently

## Current Status

- ✅ 4,414 result files removed from git tracking
- ✅ `.gitignore` updated
- ✅ Git buffer sizes increased
- ⚠️ Git history still contains old large files (3.6GB)

Try `git push` first. If it fails, use one of the options above.

