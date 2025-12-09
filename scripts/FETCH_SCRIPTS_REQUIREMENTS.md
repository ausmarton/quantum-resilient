# Fetch Scripts Requirements

## Quick Answer

**For downloading and validation only: NO venv needed** ✅  
**For full analysis (plots, stats): YES venv needed** ⚠️

---

## Script Requirements

### 1. `fetch_all_gcp_results.sh` (Download Only)

**Requirements:**
- ✅ `gsutil` (Google Cloud SDK)
- ✅ `bash`
- ✅ `python3` (standard library only - for JSON validation)

**No venv needed** - Uses only standard library (`python3 -m json.tool`)

### 2. `fetch_and_analyse_from_gcs.sh` (Download + Analysis)

**For downloading only:**
- ✅ `gsutil` (Google Cloud SDK)
- ✅ `bash`
- ✅ `python3` (standard library)

**For analysis (optional):**
- ⚠️ Python venv with dependencies (pandas, numpy, matplotlib, etc.)
- Use `--skip-analysis` to skip analysis and avoid venv requirement

### 3. `validate_gcp_downloads.sh` (Validation Only)

**Requirements:**
- ✅ `bash`
- ✅ `python3` (standard library only)

**No venv needed** - Uses only `python3 -m json.tool` for JSON validation

---

## Usage Without Venv

### Download Only (No Analysis)

```bash
# Fetch all experiments (downloads only, no analysis)
./scripts/fetch_all_gcp_results.sh \
    --bucket YOUR_BUCKET \
    --skip-analysis

# This will:
# - Download raw JSONL files
# - Download metadata files
# - Validate JSON format (using standard library)
# - Skip analysis pipeline (no plots/stats)
```

### Validate Downloads

```bash
# Validate downloaded data (no venv needed)
./scripts/validate_gcp_downloads.sh

# This uses only standard library Python
```

---

## Usage With Venv (Full Analysis)

### Setup Venv (if you want analysis)

```bash
# Create venv
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r analysis/requirements.txt
```

### Fetch With Analysis

```bash
# Fetch and analyze (requires venv)
source venv/bin/activate  # Activate venv first
./scripts/fetch_all_gcp_results.sh --bucket YOUR_BUCKET

# This will:
# - Download data
# - Run analysis pipeline
# - Generate plots and statistics
```

---

## What Each Component Needs

| Component | Venv Needed? | Dependencies |
|-----------|--------------|--------------|
| **Download from GCS** | ❌ No | `gsutil` only |
| **JSON Validation** | ❌ No | Python standard library |
| **Data Validation** | ❌ No | Python standard library |
| **Merge JSONL** | ⚠️ Maybe | Depends on script (may need pandas) |
| **Compute Statistics** | ✅ Yes | pandas, numpy, scipy |
| **Generate Plots** | ✅ Yes | matplotlib, seaborn |

---

## Recommended Workflow

### Option 1: Download Only (No Venv)

```bash
# 1. List what's available
./scripts/list_gcp_experiments.sh --bucket YOUR_BUCKET

# 2. Fetch all (download only, no analysis)
./scripts/fetch_all_gcp_results.sh \
    --bucket YOUR_BUCKET \
    --skip-analysis

# 3. Validate downloads
./scripts/validate_gcp_downloads.sh
```

**Benefits:**
- ✅ No setup needed
- ✅ Fast (no analysis overhead)
- ✅ Can run analysis later when needed

### Option 2: Download + Analysis (With Venv)

```bash
# 1. Setup venv (one time)
python3 -m venv venv
source venv/bin/activate
pip install -r analysis/requirements.txt

# 2. Fetch with analysis
./scripts/fetch_all_gcp_results.sh --bucket YOUR_BUCKET

# Analysis happens automatically
```

**Benefits:**
- ✅ Complete pipeline in one step
- ✅ Plots and stats generated immediately

---

## Summary

**You can use the fetch scripts without a venv** if you:
- Use `--skip-analysis` flag
- Only need to download and validate data
- Will run analysis separately later

**You need a venv** if you:
- Want automatic analysis (plots, stats)
- Don't use `--skip-analysis` flag
- Need to run analysis scripts directly

The validation scripts (`validate_gcp_downloads.sh`) **never need a venv** - they only use Python standard library.

