# Environment Terminology Proposal: Academic Rigor

## Current Terminology Issues

**Current labels**: `native`, `minikube`, `GCP`

**Problems**:
1. **"native"** - Vague, doesn't specify what it means (native to what?)
2. **"minikube"** - Implementation detail (specific tool), not a scientific concept
3. **"GCP"** - Cloud provider name, not a deployment model

These terms lack academic rigor and focus on implementation details rather than scientific distinctions.

---

## Scientific Distinctions

Based on the experimental setup:

| Current | Execution Model | Infrastructure | Orchestration | Hardware |
|---------|----------------|----------------|---------------|----------|
| **native** | Direct (host OS) | Local | None | AMD Ryzen, 94GB |
| **minikube** | Containerized | Local | Kubernetes | Same as native |
| **GCP** | Containerized | Cloud | Kubernetes | Intel Xeon, 8GB |

**Key Scientific Dimensions**:
1. **Execution Model**: Direct vs Containerized
2. **Infrastructure Location**: Local vs Cloud
3. **Orchestration**: None vs Orchestrated (Kubernetes)

---

## Proposed Terminology Options

### Option 1: Execution Model + Infrastructure (Recommended)

**Rationale**: Focuses on what matters scientifically - how code executes and where it runs.

- **native** → **"Direct Execution (Local)"**
- **minikube** → **"Containerized Execution (Local)"**
- **GCP** → **"Containerized Execution (Cloud)"**

**Short forms for tables/figures**:
- "Direct (Local)"
- "Containerized (Local)"
- "Containerized (Cloud)"

**Pros**:
- ✅ Scientifically precise
- ✅ Clear distinction between execution models
- ✅ Infrastructure location is explicit
- ✅ Avoids implementation details
- ✅ Works well in tables/figures

**Cons**:
- Slightly longer than current terms

---

### Option 2: Deployment Context Focus

**Rationale**: Emphasizes deployment scenarios relevant to real-world systems.

- **native** → **"Baseline Execution"**
- **minikube** → **"Local Deployment"**
- **GCP** → **"Cloud Deployment"**

**Short forms**:
- "Baseline"
- "Local Deployment"
- "Cloud Deployment"

**Pros**:
- ✅ Emphasizes deployment context
- ✅ "Baseline" clearly indicates reference point
- ✅ Shorter than Option 1

**Cons**:
- "Local Deployment" is less precise (could mean many things)
- Doesn't explicitly distinguish execution models

---

### Option 3: Infrastructure + Execution Model (Inverted)

**Rationale**: Infrastructure first, then execution detail.

- **native** → **"Local Direct Execution"**
- **minikube** → **"Local Containerized Execution"**
- **GCP** → **"Cloud Containerized Execution"**

**Short forms**:
- "Local Direct"
- "Local Containerized"
- "Cloud Containerized"

**Pros**:
- ✅ Infrastructure location is primary distinction
- ✅ Execution model is secondary
- ✅ Clear and systematic

**Cons**:
- Similar to Option 1, just reordered

---

### Option 4: Minimal Descriptive Labels

**Rationale**: Short, clear labels that avoid implementation details.

- **native** → **"Host OS"**
- **minikube** → **"Local Containerized"**
- **GCP** → **"Cloud Containerized"**

**Short forms**:
- "Host OS"
- "Local Containerized"
- "Cloud Containerized"

**Pros**:
- ✅ Very concise
- ✅ Clear distinctions
- ✅ "Host OS" is technically accurate

**Cons**:
- "Host OS" might be less clear to some readers
- Less explicit about being a baseline

---

## Recommendation: **Option 1**

**Full labels**:
- **"Direct Execution (Local)"** (replaces "native")
- **"Containerized Execution (Local)"** (replaces "minikube")
- **"Containerized Execution (Cloud)"** (replaces "GCP")

**Short forms for tables/figures**:
- **"Direct (Local)"**
- **"Containerized (Local)"**
- **"Containerized (Cloud)"**

**Rationale**:
1. **Scientifically precise**: Clearly distinguishes execution models (direct vs containerized)
2. **Infrastructure explicit**: Location (local vs cloud) is clear
3. **Avoids implementation details**: No mention of Minikube, Podman, GCP, GKE
4. **Systematic**: Follows a clear pattern (Execution Model + Infrastructure)
5. **Academic rigor**: Uses standard terminology (direct execution, containerized execution)
6. **Table/Figure friendly**: Short forms are clear and concise

---

## Impact Assessment

### Dissertation Document (Chapter 4)

**Files to modify**: `FERNANDES_Dissertation.md`

**Estimated occurrences**:
- ~30-40 instances of "native", "Minikube", "GCP" in Chapter 4
- Table headers (4.1, 4.2, 4.4, 4.4a)
- Figure captions (4.1, 4.1a, 4.2a, 4.5a)
- Text references throughout sections 4.1-4.5

**Changes required**:
1. Replace "native" with "Direct Execution (Local)" (or "Direct (Local)" in tables)
2. Replace "Minikube" with "Containerized Execution (Local)" (or "Containerized (Local)" in tables)
3. Replace "GCP" with "Containerized Execution (Cloud)" (or "Containerized (Cloud)" in tables)
4. Update figure captions
5. Update table headers and notes
6. Update text references

**Effort**: Medium (systematic find/replace, but needs careful review)

---

### Figures

**Files affected**:
- `final-results/figures/combined_ecdf_native.png` → Regenerate with new labels
- `final-results/figures/pqc_vs_classical_distribution_native.png` → Regenerate
- `final-results/figures/effect_size_forest_native.png` → Regenerate
- `final-results/figures/payload_scaling_loglog_native.png` → Regenerate
- Any other figures with environment labels

**Scripts to modify**:
- `analysis/plot_pqc_vs_classical_distribution.py`
- `analysis/plot_effect_size_forest.py`
- `analysis/plot_payload_scaling_loglog.py`
- `analysis/plot_combined_cdfs.py`
- Any other plotting scripts

**Changes required**:
1. Update environment label mappings in plotting scripts
2. Regenerate all affected figures
3. Update figure file names if needed (or keep as-is, just update labels)

**Effort**: Medium (script updates + regeneration)

---

### Tables

**Tables affected**:
- Table 4.1: Environment column
- Table 4.2: Title mentions "Native Environment"
- Table 4.4: Environment column
- Table 4.4a: Environment column
- Any other tables with environment references

**Changes required**:
1. Update table headers
2. Update table row labels
3. Update table notes/captions

**Effort**: Low (direct text edits)

---

### Data Files (No Changes Needed)

**Important**: The user explicitly stated they **don't want to change code/data files**. The following remain unchanged:
- `final-results/index.json` (keeps "native", "minikube", "gcp")
- `final-results/aggregated_stats.json` (keeps original keys)
- `final-results/hypothesis_tests.json` (keeps original comparison IDs)
- Analysis scripts (data loading logic unchanged)
- Experiment results directories (`results/native/`, etc.)

**Mapping strategy**: Create a label mapping in plotting/presentation scripts:
```python
ENV_DISPLAY_NAMES = {
    'native': 'Direct Execution (Local)',
    'minikube': 'Containerized Execution (Local)',
    'gcp': 'Containerized Execution (Cloud)'
}

ENV_SHORT_NAMES = {
    'native': 'Direct (Local)',
    'minikube': 'Containerized (Local)',
    'gcp': 'Containerized (Cloud)'
}
```

---

## Implementation Plan

### Phase 1: Update Dissertation Document
1. Create environment label mapping
2. Systematic find/replace in `FERNANDES_Dissertation.md`
3. Review all instances for context-appropriate usage
4. Update table headers and captions
5. Update figure captions

### Phase 2: Update Figure Generation Scripts
1. Add environment label mapping to plotting scripts
2. Update figure generation to use new labels
3. Regenerate all affected figures
4. Verify figure labels match dissertation text

### Phase 3: Verification
1. Check all table references
2. Check all figure references
3. Verify consistency across document
4. Ensure no implementation details leak through

---

## Alternative: Gradual Introduction

If full replacement is too disruptive, consider:

1. **First mention**: Use full academic terminology
   - "Three deployment environments were evaluated: Direct Execution (Local), Containerized Execution (Local), and Containerized Execution (Cloud)."

2. **Subsequent mentions**: Use short forms
   - "Direct (Local)", "Containerized (Local)", "Containerized (Cloud)"

3. **Tables/Figures**: Use short forms consistently

This provides academic rigor at first mention while keeping text readable.

---

## Summary

**Recommended terminology**:
- **native** → **"Direct Execution (Local)"** / **"Direct (Local)"**
- **minikube** → **"Containerized Execution (Local)"** / **"Containerized (Local)"**
- **GCP** → **"Containerized Execution (Cloud)"** / **"Containerized (Cloud)"**

**Impact**:
- **Dissertation**: ~30-40 replacements (Medium effort)
- **Figures**: Script updates + regeneration (Medium effort)
- **Tables**: Direct edits (Low effort)
- **Data files**: No changes (as requested)

**Total estimated effort**: Medium (2-3 hours for careful implementation and verification)
