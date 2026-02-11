# Detailed Analysis Documentation

**Last Updated**: 2025-12-15  
**Purpose**: Comprehensive analysis documentation for the research (statistical methods, visualizations, interpretation).

---

## Table of Contents

1. [Analysis Overview](#analysis-overview)
2. [Statistical Analysis Methods](#statistical-analysis-methods)
3. [Visualization Methods](#visualization-methods)
4. [Hypothesis Testing](#hypothesis-testing)
5. [Effect Size Computation](#effect-size-computation)
6. [Cross-Environment Comparison](#cross-environment-comparison)
7. [Scaling Analysis](#scaling-analysis)
8. [Interpretation Framework](#interpretation-framework)

---

## Analysis Overview

### Analysis Pipeline

The analysis pipeline processes raw experiment data through multiple stages:

1. **Data Collection**: Raw JSONL event logs from experiments
2. **Data Merging**: Merge JSONL files from multiple runs
3. **Statistics Computation**: Compute summary statistics
4. **Aggregation**: Aggregate statistics across experiments
5. **Hypothesis Testing**: Statistical significance tests
6. **Effect Size Computation**: Effect size metrics
7. **Visualization**: Publication-quality figures
8. **Reporting**: Report-ready reports and tables

### Data Flow

```
Raw JSONL Files
    ↓
Merge JSONL (merge_jsonl.py)
    ↓
Compute Statistics (compute_statistics.py)
    ↓
Aggregate Results (aggregate_results.py)
    ↓
Hypothesis Tests (hypothesis_tests.py)
    ↓
Visualization (plot_*.py)
    ↓
Final Report (build_final_report.py)
```

---

## Statistical Analysis Methods

### Summary Statistics

For each experiment, the following statistics are computed:

#### Latency Statistics

- **Percentiles**: p50, p90, p95, p99, p99.9
- **Central Tendency**: Mean, median
- **Dispersion**: Standard deviation, variance, coefficient of variation
- **Range**: Min, max, interquartile range (IQR)

#### Throughput Statistics

- **Mean Throughput**: Average operations per second
- **Peak Throughput**: Maximum operations per second
- **Throughput Stability**: Coefficient of variation

#### Queue Delay Statistics

- **Percentiles**: p50, p95, p99
- **Mean Queue Delay**: Average queue delay
- **Queue Delay Distribution**: Full distribution analysis

#### Resource Utilization Statistics

- **CPU Usage**: Mean, peak CPU usage
- **Memory Usage**: Mean, peak memory usage

### Aggregation Across Runs

For each experiment configuration (5 runs):

1. **Mean Statistics**: Average across runs
2. **Standard Deviation**: Variability across runs
3. **Confidence Intervals**: 95% CI for key metrics (using t-distribution)
4. **Coefficient of Variation**: CV = std/mean (stability metric)

**Stability Interpretation**:
- **CV < 5%**: Excellent stability
- **CV 5-10%**: Good stability
- **CV 10-15%**: Acceptable (warning issued)
- **CV > 15%**: Unstable (results may not be reliable)

### Statistical Software

- **Python 3.10+**: Primary analysis language
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scipy**: Statistical tests and distributions
- **matplotlib/seaborn**: Visualization

---

## Visualization Methods

### Cumulative Distribution Functions (CDFs)

**Purpose**: Compare latency distributions across algorithms and environments

**Types**:
1. **Combined CDFs**: All algorithms on one plot
2. **Environment Comparison**: Native vs Minikube vs GCP
3. **Algorithm Comparison**: PQC vs Classical

**Implementation**: `plot_combined_cdfs.py`

**Output**: High-resolution PNG (300 DPI) and vector PDF/EPS

### Scaling Curves

**Purpose**: Analyze horizontal scaling behavior

**Metrics**:
- Throughput vs replica count
- Latency vs replica count
- Efficiency (throughput per replica)

**Implementation**: `plot_scaling_curves.py`

### Effect Size Forest Plots

**Purpose**: Visualize effect sizes with confidence intervals

**Metrics**: Cohen's d with 95% CI

**Implementation**: `plot_effect_size_forest.py`

### Payload Scaling Analysis

**Purpose**: Analyze impact of payload size on performance

**Visualization**: Log-log plots showing latency vs payload size

**Implementation**: `plot_payload_scaling_loglog.py`

### Distribution Comparisons

**Purpose**: Compare PQC vs classical distributions

**Visualization**: Overlaid histograms, box plots, violin plots

**Implementation**: `plot_pqc_vs_classical_distribution.py`

### Replica Scaling Analysis

**Purpose**: Analyze horizontal scaling behavior

**Visualization**: Throughput and latency vs replica count

**Implementation**: `plot_replica_scaling.py`

---

## Hypothesis Testing

### Tests Performed

#### 1. Mann-Whitney U Test

**Purpose**: Non-parametric test for distribution differences

**Use Case**: Compare latency distributions between algorithms

**Interpretation**:
- **p < 0.05**: Significant difference (reject null hypothesis)
- **p ≥ 0.05**: No significant difference (fail to reject null hypothesis)

**Implementation**: `scipy.stats.mannwhitneyu`

#### 2. Kolmogorov-Smirnov Test

**Purpose**: Test distribution shape similarity

**Use Case**: Compare distribution shapes between algorithms

**Interpretation**:
- **p < 0.05**: Distributions are significantly different
- **p ≥ 0.05**: Distributions are not significantly different

**Implementation**: `scipy.stats.ks_2samp`

#### 3. Welch's t-test

**Purpose**: Test mean differences (unequal variances)

**Use Case**: Compare mean latency between algorithms

**Assumptions**: 
- Independent samples
- Normal distribution (approximately)
- Unequal variances (Welch's correction)

**Interpretation**:
- **p < 0.05**: Significant mean difference
- **p ≥ 0.05**: No significant mean difference

**Implementation**: `scipy.stats.ttest_ind` with `equal_var=False`

#### 4. Holm-Bonferroni Correction

**Purpose**: Control family-wise error rate (FWER)

**Use Case**: Multiple comparisons (multiple algorithms, multiple metrics)

**Method**: Adjust p-values to control FWER at α = 0.05

**Implementation**: `statsmodels.stats.multitest.multipletests` with method='holm'

### Comparison Groups

**PQC vs Classical Signatures**:
- Baseline: ECDSA P-256
- Treatment: Dilithium-2

**PQC vs Classical Encryption/KEM**:
- Baseline: RSA-2048 (for signatures) or ECDH P-256 (for KEM)
- Treatment: Kyber-512

**Hybrid vs Pure PQC**:
- Baseline: Kyber-512
- Treatment: Hybrid Kyber-Dilithium

**Environment Comparisons**:
- Native vs Minikube
- Native vs GCP
- Minikube vs GCP

**Scaling Comparisons**:
- 1 vs 2 replicas
- 1 vs 4 replicas
- 1 vs 8 replicas

---

## Effect Size Computation

### Metrics Computed

#### 1. Cohen's d

**Formula**: d = (μ₁ - μ₂) / σ_pooled

**Interpretation**:
- |d| < 0.2: Negligible effect
- |d| < 0.5: Small effect
- |d| < 0.8: Medium effect
- |d| ≥ 0.8: Large effect

**Implementation**: `effect_sizes.py`

#### 2. Hedge's g

**Formula**: Bias-corrected Cohen's d

**Use Case**: Small sample sizes

**Interpretation**: Same as Cohen's d

#### 3. Glass's Δ

**Formula**: Uses control group standard deviation

**Use Case**: When treatment group variance differs significantly

**Interpretation**: Same as Cohen's d

#### 4. Cliff's δ

**Formula**: Non-parametric effect size

**Interpretation**:
- |δ| < 0.147: Negligible
- |δ| < 0.33: Small
- |δ| < 0.474: Medium
- |δ| ≥ 0.474: Large

#### 5. Wasserstein Distance

**Formula**: Earth mover's distance

**Use Case**: Distribution-level comparison

**Units**: Original units (e.g., microseconds)

#### 6. Kolmogorov-Smirnov Statistic

**Formula**: Maximum difference between CDFs

**Range**: 0-1

**Interpretation**: Higher values indicate larger distribution differences

### Confidence Intervals

**Method**: Bootstrap confidence intervals (BCa method)

**Confidence Level**: 95%

**Implementation**: `scipy.stats.bootstrap`

---

## Cross-Environment Comparison

### Comparison Metrics

1. **Latency Comparison**: p50, p95, p99 latency across environments
2. **Throughput Comparison**: Mean throughput across environments
3. **Variability Comparison**: Coefficient of variation across environments
4. **Distribution Comparison**: CDF comparison across environments

### Comparison Methods

#### 1. Side-by-Side CDFs

**Visualization**: Overlaid CDFs for each environment

**Purpose**: Visual comparison of latency distributions

#### 2. Box/Violin Plots

**Visualization**: Box plots or violin plots for each environment

**Purpose**: Compare distributions and identify outliers

#### 3. Environment Impact Summary

**Metrics**:
- Percentage difference from native baseline
- Statistical significance (p-value)
- Effect size (Cohen's d)

#### 4. Report Paragraph Generation

**Output**: Pre-formatted paragraphs describing environment comparisons

**Format**: Ready for inclusion in results and analysis

**Implementation**: `compare_all_environments.py`

---

## Scaling Analysis

### Horizontal Scaling Metrics

1. **Throughput Scaling**: Throughput vs replica count
2. **Latency Scaling**: Latency vs replica count
3. **Efficiency**: Throughput per replica
4. **Scaling Efficiency**: Linear scaling vs actual scaling

### Scaling Analysis Methods

#### 1. Scaling Curves

**Visualization**: Throughput and latency vs replica count

**Purpose**: Identify scaling bottlenecks

**Implementation**: `plot_scaling_curves.py`

#### 2. Replica Scaling Analysis

**Visualization**: Detailed analysis of scaling behavior

**Metrics**: 
- Throughput per replica
- Latency degradation
- Queue delay impact

**Implementation**: `plot_replica_scaling.py`

#### 3. Scaling Efficiency

**Formula**: Actual throughput / (Baseline throughput × Replicas)

**Interpretation**:
- **Efficiency = 1.0**: Perfect linear scaling
- **Efficiency < 1.0**: Sub-linear scaling (bottlenecks)
- **Efficiency > 1.0**: Super-linear scaling (rare, may indicate measurement error)

---

## Interpretation Framework

### Performance Comparison Framework

#### Step 1: Descriptive Statistics

1. Compute summary statistics for each algorithm
2. Compare percentiles (p50, p95, p99)
3. Compare mean and median
4. Compare variability (std, CV)

#### Step 2: Statistical Significance

1. Perform hypothesis tests (Mann-Whitney U, KS test, t-test)
2. Apply multiple comparison correction (Holm-Bonferroni)
3. Identify statistically significant differences

#### Step 3: Effect Size

1. Compute effect sizes (Cohen's d, etc.)
2. Interpret effect sizes (negligible, small, medium, large)
3. Compute confidence intervals

#### Step 4: Practical Significance

1. Consider effect sizes in context of application requirements
2. Consider latency differences in context of acceptable thresholds
3. Consider throughput differences in context of workload requirements

#### Step 5: Visualization

1. Generate CDF plots for visual comparison
2. Generate effect size forest plots
3. Generate comparison tables

#### Step 6: Interpretation

1. Synthesize statistical and practical significance
2. Identify key findings
3. Generate dissertation-ready text

### Key Findings Framework

For each comparison, document:

1. **Statistical Significance**: p-value, test used
2. **Effect Size**: Cohen's d, interpretation
3. **Practical Significance**: Real-world impact
4. **Confidence**: Confidence intervals
5. **Limitations**: Any limitations or caveats

---

## Analysis Scripts Reference

### Core Analysis Scripts

- `compute_statistics.py`: Compute summary statistics from JSONL
- `merge_jsonl.py`: Merge multiple JSONL files
- `aggregate_results.py`: Aggregate statistics across experiments
- `hypothesis_tests.py`: Perform hypothesis tests
- `effect_sizes.py`: Compute effect sizes

### Visualization Scripts

- `plot_combined_cdfs.py`: Combined CDF plots
- `plot_scaling_curves.py`: Scaling curves
- `plot_effect_size_forest.py`: Effect size forest plots
- `plot_payload_scaling_loglog.py`: Payload scaling analysis
- `plot_pqc_vs_classical_distribution.py`: Distribution comparisons
- `plot_replica_scaling.py`: Replica scaling analysis

### Comparison Scripts

- `compare_all_environments.py`: Cross-environment comparison
- `compare_native_vs_minikube.py`: Native vs Minikube comparison

### Reporting Scripts

- `build_final_report.py`: Generate final report
- `extract_analysis_tables.py`: Extract LaTeX tables

---

## Analysis Output Structure

```
final-results/
├── index.json                    # Experiment index
├── aggregated_stats.json         # Aggregated statistics
├── aggregated_stats.csv          # CSV format
├── hypothesis_tests.json         # Hypothesis test results
├── effect_sizes.json            # Effect size metrics
├── figures/
│   ├── combined_ecdf.png        # Combined CDFs
│   ├── scaling_curves.png        # Scaling curves
│   ├── effect_size_forest.png   # Effect size forest plot
│   └── ...
├── tables/
│   ├── latency_summary.csv      # Latency summary table
│   ├── hypothesis_summary.json  # Hypothesis test summary
│   └── ...
└── report.md                     # Final report
```

---

## Best Practices

### Statistical Analysis

1. **Always report effect sizes**: Statistical significance alone is insufficient
2. **Use appropriate tests**: Non-parametric tests for non-normal distributions
3. **Control for multiple comparisons**: Use Holm-Bonferroni correction
4. **Report confidence intervals**: Provide uncertainty estimates
5. **Check assumptions**: Verify test assumptions are met

### Visualization

1. **Use appropriate scales**: Log scales for wide ranges
2. **Include error bars**: Show confidence intervals
3. **Use consistent colors**: Same algorithm = same color across plots
4. **High resolution**: 300 DPI for publication
5. **Vector formats**: PDF/EPS for scalability

### Interpretation

1. **Consider both statistical and practical significance**
2. **Acknowledge limitations**: Be transparent about caveats
3. **Context matters**: Interpret results in context of application requirements
4. **Avoid over-interpretation**: Don't claim more than the data supports

---

**Last Updated**: 2025-12-15

