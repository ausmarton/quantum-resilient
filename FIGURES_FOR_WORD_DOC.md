# Figures for Word Document Insertion

All generated figures are stored in: **`final-results/figures/`**

## Figure Mapping: Dissertation → File Location

### Chapter 3 Figures

**Figure 3.1** (High-level framework architecture) - `[image1]`
- **Source**: Embedded in dissertation as base64
- **File**: Not in figures directory (likely from diagrams/)
- **Note**: Check `diagrams/` folder for original source

**Figure 3.2** (Framework architecture) - `[image2]`
- **Source**: Embedded in dissertation as base64
- **File**: Not in figures directory (likely from diagrams/)
- **Note**: Check `diagrams/` folder for original source

**Figure 3.3** (Framework representation of live production system) - `[image3]`
- **Source**: Embedded in dissertation as base64
- **File**: Not in figures directory (likely from diagrams/)
- **Note**: Check `diagrams/` folder for original source

**Figure 3.4** (Detailed research system implementation) - `[image4]`
- **Source**: Embedded in dissertation as base64
- **File**: Not in figures directory (likely from diagrams/)
- **Note**: Check `diagrams/` folder for original source

### Chapter 4 Figures

**Figure 4.1** (CDFs showing latency distributions) - `[image5]`
- **File**: `final-results/figures/combined_ecdf_bare-metal.png`
- **Alternative**: `combined_ecdf.png` (all environments)
- **Size**: ~280KB
- **Description**: Cumulative distribution functions for all algorithms in Bare-metal environment

**Figure 4.1a** (Violin and box plots) - `[image6]`
- **File**: `final-results/figures/pqc_vs_classical_distribution_bare-metal.png`
- **Size**: ~280KB
- **Description**: Violin and box plots comparing latency distributions

**Figure 4.2** (Performance comparison across deployment environments) - `[image7]`
- **File**: `final-results/figures/bare-metal_vs_local-k8s_vs_cloud-k8s.png`
- **Size**: ~427KB
- **Description**: Environment comparison showing latency overhead

**Figure 4.2a** (Effect size forest plot) - `[image12]` ⭐ **NEW**
- **File**: `final-results/figures/effect_size_forest_bare-metal.png`
- **Size**: ~300KB
- **Description**: Forest plot showing Cohen's d values with 95% confidence intervals for 59 comparisons with large effect sizes

**Figure 4.3** (Latency distribution across payload sizes) - `[image9]`
- **File**: `final-results/figures/ecdf_by_payload.png`
- **Size**: ~539KB
- **Description**: Latency distribution analysis across different payload sizes

**Figure 4.4** (Latency performance as function of workload rate) - `[image10]`
- **File**: `final-results/figures/latency_vs_rate_bare-metal.png`
- **Size**: ~179KB
- **Description**: Latency performance as a function of workload rate

**Figure 4.5** (Throughput characteristics across payload sizes) - `[image11]`
- **File**: `final-results/figures/throughput_vs_payload_bare-metal.png`
- **Size**: ~230KB
- **Description**: Throughput characteristics across payload sizes

**Figure 4.5a** (Log-log plot showing p95 latency vs payload size) - `[image8]`
- **File**: `final-results/figures/payload_scaling_loglog_bare-metal.png`
- **Size**: ~267KB
- **Description**: Log-log plot showing sub-linear scaling patterns

## Quick Reference: All Available Figures

### Main Figures (Bare-metal environment - recommended for dissertation)
- `combined_ecdf_bare-metal.png` - Figure 4.1
- `pqc_vs_classical_distribution_bare-metal.png` - Figure 4.1a
- `bare-metal_vs_local-k8s_vs_cloud-k8s.png` - Figure 4.2
- `effect_size_forest_bare-metal.png` - Figure 4.2a ⭐
- `ecdf_by_payload.png` - Figure 4.3
- `latency_vs_rate_bare-metal.png` - Figure 4.4
- `throughput_vs_payload_bare-metal.png` - Figure 4.5
- `payload_scaling_loglog_bare-metal.png` - Figure 4.5a

### Additional Figures (for reference)
- `classical_vs_pqc.png` - PQC vs classical comparison
- `scaling_curves.png` - Scaling analysis
- `throughput_vs_payload_all.png` - All environments throughput comparison

### Per-Algorithm ECDFs (if needed)
- `ecdf_kyber512.png`
- `ecdf_dilithium2.png`
- `ecdf_rsa2048.png`
- `ecdf_ecdsa.png`
- `ecdf_ecdhe.png`
- `ecdf_hybrid.png`

## Instructions for Word Document

1. **Navigate to**: `/home/ausmarton/scratchpad/quantum-resilient/final-results/figures/`

2. **Insert figures**:
   - In Word: Insert → Pictures → This Device
   - Select the PNG files listed above
   - All figures are 300 DPI (high resolution, suitable for printing)

3. **Figure numbering**: Match the figure numbers in the dissertation document

4. **Captions**: Copy captions from the dissertation document (they're already formatted)

## File Sizes

All figures are PNG format, optimized for publication:
- Smallest: ~133KB (individual ECDFs)
- Largest: ~555KB (throughput_vs_payload_all.png)
- Average: ~250-300KB

All figures are saved at 300 DPI for publication quality.

## Note on Embedded Images

The dissertation markdown file has images embedded as base64 data URIs. For Word document insertion, use the PNG files from `final-results/figures/` instead - they're the same images but in a format Word can handle directly.
