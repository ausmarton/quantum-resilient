# Environment Terminology Implementation Plan

## Terminology Mapping

| Label | Full Term | Implementation | Hardware Context |
|-------------|-----------|---------------|-----------------|
| **Bare-metal** | Bare-metal (non-containerised) | Local execution | AMD Ryzen AI MAX+ PRO 395, 94 GB memory, multiple cores |
| **Local-K8s** | Containerised local Kubernetes | Minikube | Same physical host (AMD Ryzen AI MAX+ PRO 395, 94 GB memory) |
| **Cloud-K8s** | Cloud-managed Kubernetes | GKE on GCP (n2-standard-2) | Intel Xeon @ 2.80GHz, 2 vCPUs, ~8 GB memory per node |

**Usage Guidelines**:
- **First mention in the chapter**: Use full term with short label in parentheses
  - Example: "Bare-metal (non-containerised) (Bare-metal)"
- **Subsequent mentions**: Use short label
  - Example: "Bare-metal", "Local-K8s", "Cloud-K8s"
- **Tables/Figures**: Use short labels for brevity
- **Reference table**: Include full table in Section 4.1 (Data Collection Summary)

---

## Task List

### Phase 1: Reference Table Creation

- [ ] **Task 1.1**: Create comprehensive environment reference table
  - Location: Section 4.1 (after Table 4.1 or in data collection summary)
  - Include: Short label, Full term, Environment Class, Implementation, Hardware Context
  - Format: Publication-quality table with proper notes

- [ ] **Task 1.2**: Add terminology note/definition
  - Location: Early in Section 4.1 (first mention of environments)
  - Content: Define the three environments with reference to the table
  - Example: "Three deployment environments were evaluated: Bare-metal (non-containerised) execution, Containerised local Kubernetes execution, and Cloud-managed Kubernetes execution (see Table 4.X for full specifications)."

---

### Phase 2: Dissertation Text Updates

- [ ] **Task 2.1**: Update Section 4.1 (Data Collection Summary)
  - Replace "native" → "Bare-metal" (after first mention with full term)
  - Replace "Minikube" → "Local-K8s" (after first mention with full term)
  - Replace "GCP" → "Cloud-K8s" (after first mention with full term)
  - Update Table 4.1 environment column
  - Add first mention with full terms

- [ ] **Task 2.2**: Update Section 4.2.1 (Algorithm Performance - Native)
  - Update section title/subtitle references
  - Replace "native" → "Bare-metal" throughout
  - Update Table 4.2 title and notes
  - Update Figure 4.1 and 4.1a captions

- [ ] **Task 2.3**: Update Section 4.2.2 (Statistical Hypothesis Testing)
  - Replace environment references
  - Update text mentioning "native, Minikube, GCP"
  - Update Figure 4.2a caption

- [ ] **Task 2.4**: Update Section 4.2.4 (Environment Comparison)
  - Replace all three environment names
  - Update Table 4.4 environment column and notes
  - Update Table 4.4a environment column
  - Update text references to environment comparisons

- [ ] **Task 2.5**: Update Section 4.2.5 (Payload Size and Workload Rate Impact)
  - Replace "native" → "Bare-metal"
  - Update Figure 4.5a caption

- [ ] **Task 2.6**: Update Section 4.3 (Interpretation in Relation to Objectives)
  - Replace all environment references
  - Ensure consistency with new terminology

- [ ] **Task 2.7**: Update Section 4.4 (Interpretation in Relation to Research Aim)
  - Replace all environment references
  - Update any environment comparison discussions

- [ ] **Task 2.8**: Update Section 4.5 (Summary of Chapter 4)
  - Replace all environment references
  - Ensure summary uses consistent terminology

- [ ] **Task 2.9**: Update Hardware Characteristics subsection (Section 4.2.4)
  - Replace environment names in hardware descriptions
  - Update numbered list (1), (2), (3) with new terminology

---

### Phase 3: Table Updates

- [ ] **Task 3.1**: Update Table 4.1 (Experiment Distribution by Environment)
  - Replace "Native", "Minikube", "GCP" in environment column
  - Use short labels: "Bare-metal", "Local-K8s", "Cloud-K8s"

- [ ] **Task 3.2**: Update Table 4.2 (Algorithm Latency Performance)
  - Update title: "Bare-metal Environment" (or just "Bare-metal" if space is tight)
  - Update table notes

- [ ] **Task 3.3**: Update Table 4.4 (Environment Comparison Summary)
  - Replace environment column values
  - Update table notes
  - Update "Overhead vs Native" → "Overhead vs Bare-metal"

- [ ] **Task 3.4**: Update Table 4.4a (Normalised Environment Comparison)
  - Replace environment column values
  - Update any notes

---

### Phase 4: Figure Updates

- [ ] **Task 4.1**: Update Figure Captions
  - Figure 4.1: "Bare-metal Environment" (or "Bare-metal")
  - Figure 4.1a: "Bare-metal Environment" (or "Bare-metal")
  - Figure 4.2a: "Bare-metal Environment" (or "Bare-metal")
  - Figure 4.5a: "Bare-metal Environment" (or "Bare-metal")

- [ ] **Task 4.2**: Update Plotting Scripts - Add Environment Label Mapping
  - File: `analysis/plot_pqc_vs_classical_distribution.py`
    - Add `ENV_DISPLAY_NAMES` mapping
    - Update environment label usage in plots
  - File: `analysis/plot_effect_size_forest.py`
    - Add `ENV_DISPLAY_NAMES` mapping
    - Update environment label usage
  - File: `analysis/plot_payload_scaling_loglog.py`
    - Add `ENV_DISPLAY_NAMES` mapping
    - Update environment label usage
  - File: `analysis/plot_combined_cdfs.py` (if exists)
    - Add `ENV_DISPLAY_NAMES` mapping
    - Update environment label usage

- [ ] **Task 4.3**: Regenerate Affected Figures
  - Regenerate all figures that display environment labels
  - Verify labels match new terminology
  - Check figure quality and readability

---

### Phase 5: Verification and Consistency Check

- [ ] **Task 5.1**: Comprehensive Terminology Check
  - Search for any remaining "native", "minikube", "GCP" (case-insensitive)
  - Verify all instances are updated
  - Check for consistency (first mention vs subsequent mentions)

- [ ] **Task 5.2**: Table Consistency Check
  - Verify all table headers use new terminology
  - Check table notes for consistency
  - Verify table column values match

- [ ] **Task 5.3**: Figure Consistency Check
  - Verify all figure captions use new terminology
  - Check figure labels match captions
  - Verify figure file names (if changed) or note that labels are updated

- [ ] **Task 5.4**: Cross-Reference Check
  - Verify references between sections are consistent
  - Check that "Bare-metal", "Local-K8s", "Cloud-K8s" are used consistently
  - Verify first mentions include full terms where appropriate

- [ ] **Task 5.5**: Academic Rigor Check
  - Verify terminology is academically appropriate
  - Check that implementation details (Minikube, GCP, GKE) are only in reference table
  - Verify prose uses abstract terminology

---

## Implementation Details

### Label Mapping Dictionary (for scripts)

```python
ENV_DISPLAY_NAMES = {
    'native': 'Bare-metal',
    'minikube': 'Local-K8s',
    'gcp': 'Cloud-K8s'
}

ENV_FULL_NAMES = {
    'native': 'Bare-metal (non-containerised) execution',
    'minikube': 'Containerised local Kubernetes execution',
    'gcp': 'Cloud-managed Kubernetes execution'
}
```

### First Mention Pattern

**Pattern**: Full term with short label in parentheses on first mention in each major section.

**Example**:
> "Performance was measured across three deployment environments: Bare-metal (non-containerised) execution (hereafter Bare-metal), Containerised local Kubernetes execution (hereafter Local-K8s), and Cloud-managed Kubernetes execution (hereafter Cloud-K8s)."

**Subsequent mentions**: Use short labels only.

---

## Files to Modify

### Dissertation Document
- `FERNANDES_Dissertation.md` - All Chapter 4 sections

### Plotting Scripts
- `analysis/plot_pqc_vs_classical_distribution.py`
- `analysis/plot_effect_size_forest.py`
- `analysis/plot_payload_scaling_loglog.py`
- `analysis/plot_combined_cdfs.py` (if exists)

### Figures to Regenerate
- `final-results/figures/combined_ecdf_native.png` (if labels visible)
- `final-results/figures/pqc_vs_classical_distribution_native.png`
- `final-results/figures/effect_size_forest_native.png`
- `final-results/figures/payload_scaling_loglog_native.png`

---

## Estimated Effort

- **Phase 1** (Reference Table): 30 minutes
- **Phase 2** (Text Updates): 2-3 hours
- **Phase 3** (Table Updates): 30 minutes
- **Phase 4** (Figure Updates): 1-2 hours
- **Phase 5** (Verification): 1 hour

**Total**: ~5-7 hours for careful, systematic implementation

---

## Notes

1. **Data files unchanged**: As requested, all data files (`index.json`, `aggregated_stats.json`, etc.) keep original keys. Only presentation layer changes.

2. **Figure file names**: Consider keeping file names as-is (e.g., `combined_ecdf_native.png`) since they're internal references. Only the labels within figures change.

3. **Consistency**: Use short labels in tables/figures for brevity, full terms in first mentions in prose.

4. **Reference table**: Include comprehensive table early in Section 4.1 to establish terminology for entire chapter.
