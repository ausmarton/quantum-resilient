# Dissertation Analysis Guide
## From Raw Data to Publication-Ready Visualizations

**Complete workflow to analyze all collected experiment data and generate figures/tables for your dissertation.**

---

## 🎯 Overview

You have **raw data** in `results/` directory. This guide shows you how to:
1. **Automated Analysis** (Scripts) - Generate all standard figures and statistics
2. **Interactive Analysis** (Jupyter Notebooks) - Custom exploration and deep dives
3. **Dissertation Outputs** - Tables, figures, and statistical summaries

---

## 📊 Two Analysis Approaches

### Approach 1: Automated Scripts (Recommended First)
**Use for:** Standard analysis, batch processing, dissertation-ready outputs
- Fast, reproducible, generates all standard figures
- Produces CSV/JSON files for tables
- Best for initial analysis and getting all standard outputs

### Approach 2: Jupyter Notebooks (For Deep Dives)
**Use for:** Custom analysis, exploration, hypothesis testing, publication-quality figure tweaking
- Interactive, allows experimentation
- Best for understanding data, custom visualizations, and fine-tuning figures

**Recommendation:** Start with scripts, then use notebooks for specific deep dives.

---

## 🚀 Quick Start: Automated Analysis Pipeline

### Step 1: Setup Python Environment

```bash
cd analysis
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
cd ..
```

### Step 2: Verify Your Data

```bash
# Check data exists
echo "Native experiments: $(ls -d results/native/*/ 2>/dev/null | wc -l)"
echo "Minikube experiments: $(ls -d results/minikube/*/ 2>/dev/null | wc -l)"
echo "GCP experiments: $(ls -d results/gcp/*/ 2>/dev/null | wc -l)"

# Check index file
python3 -c "import json; idx=json.load(open('final-results/index.json')); print(f\"Indexed: {len(idx['experiments'])} experiments\")"
```

### Step 3: Regenerate Index (if needed)

If your index is outdated or missing experiments:

```bash
./scripts/regenerate_index_from_results.sh \
  --matrix orchestration/experiment_matrix.yaml \
  --output final-results/
```

### Step 4: Run Complete Analysis Pipeline

**Option A: Run All Steps Manually (More Control)**

```bash
# Activate environment
cd analysis
source venv/bin/activate
cd ..

# Set paths
INDEX_FILE="final-results/index.json"
FINAL_RESULTS_DIR="final-results"

# 1. Aggregate Statistics (mean, std, CI across all runs)
echo "Step 1: Aggregating statistics..."
python3 analysis/aggregate_results.py \
  --index "$INDEX_FILE" \
  --output "$FINAL_RESULTS_DIR"

# 2. Generate Combined CDF Plots (latency distributions)
echo "Step 2: Generating CDF plots..."
python3 analysis/plot_combined_cdfs.py \
  --index "$INDEX_FILE" \
  --output "$FINAL_RESULTS_DIR/figures"

# 3. Generate Scaling Curves (if you have horizontal scaling data)
echo "Step 3: Generating scaling curves..."
python3 analysis/plot_scaling_curves.py \
  --index "$INDEX_FILE" \
  --output "$FINAL_RESULTS_DIR/figures" || echo "No scaling data found (skipping)"

# 4. Run Statistical Hypothesis Tests
echo "Step 4: Running hypothesis tests..."
python3 analysis/hypothesis_tests.py \
  --index "$INDEX_FILE" \
  --matrix orchestration/experiment_matrix.yaml \
  --output "$FINAL_RESULTS_DIR"

# 5. Generate Replica Scaling Plots (if applicable)
echo "Step 5: Generating replica scaling plots..."
python3 analysis/plot_replica_scaling.py \
  --index "$INDEX_FILE" \
  --output "$FINAL_RESULTS_DIR/figures/scaling" || echo "No replica scaling data (skipping)"
```

**Option B: Use Main Script (Automated)**

```bash
# This runs all analysis steps automatically
./run_all_experiments.sh \
  --skip-generation \
  --skip-native \
  --skip-minikube \
  --skip-gcp \
  --matrix orchestration/experiment_matrix.yaml
```

### Step 5: Cross-Environment Comparison

Compare performance across native, Minikube, and GCP:

```bash
python3 analysis/compare_all_environments.py \
  --native final-results/aggregated_stats.json \
  --minikube final-results/aggregated_stats.json \
  --gcp final-results/aggregated_stats.json \
  --output final-results/comparisons/
```

### Step 6: Generate Final Report

```bash
python3 analysis/build_final_report.py \
  --results-dir final-results \
  --output final-results/report.pdf
```

---

## 📁 What You'll Get

After running the analysis, check `final-results/`:

### Key Output Files

| File | Purpose | For Dissertation |
|------|---------|------------------|
| `aggregated_stats.json` | Summary statistics (p50, p95, p99, mean, std, CI) | **Extract for tables** |
| `aggregated_stats.csv` | Same data in CSV format | **Import into LaTeX/Excel** |
| `hypothesis_tests.json` | Statistical test results (p-values, effect sizes) | **Statistical claims** |
| `hypothesis_table.csv` | Test results in table format | **Results chapter tables** |
| `figures/*.png` | All generated plots | **Use directly in dissertation** |
| `figures/scaling/*.png` | Scaling analysis plots | **Horizontal scaling section** |
| `report.pdf` | Complete analysis report | Reference document |

### Generated Figures

- **CDF Plots**: `*_cdf.png` - Latency distributions (compare algorithms)
- **Combined Plots**: `*_combined.png` - Multi-algorithm comparisons
- **Scaling Curves**: `scaling/*.png` - Throughput/latency vs replicas
- **Environment Comparisons**: `*_env_comparison.png` - Native vs Minikube vs GCP

---

## 🔬 Jupyter Notebooks: When and Why

### Purpose of Notebooks

Jupyter notebooks are for **interactive exploration** and **custom analysis**:

1. **Deep Data Exploration** - Understand patterns before automated analysis
2. **Custom Visualizations** - Create publication-quality figures with fine control
3. **Hypothesis Testing** - Test specific research questions interactively
4. **Data Validation** - Verify data quality and identify anomalies
5. **Iterative Analysis** - Experiment with different statistical approaches

### When to Use Notebooks vs Scripts

| Use Scripts When | Use Notebooks When |
|------------------|---------------------|
| ✅ First-time analysis | ✅ Need custom visualizations |
| ✅ Batch processing all data | ✅ Exploring specific hypotheses |
| ✅ Generating standard outputs | ✅ Fine-tuning figure aesthetics |
| ✅ Reproducible pipeline | ✅ Interactive data exploration |
| ✅ Quick overview | ✅ Understanding data patterns |

### Available Notebooks

Located in `analysis/notebooks/`:

1. **`00_setup.ipynb`** - Environment setup and data loading
2. **`01_load_results.ipynb`** - Load and preview all collected data
3. **`02_latency_analysis.ipynb`** - Deep dive into latency distributions
4. **`03_throughput_analysis.ipynb`** - Throughput over time analysis
5. **`04_queue_delay_analysis.ipynb`** - Queue delay patterns
6. **`05_adapter_comparison.ipynb`** - RSA vs ECDSA vs Kyber comparison
7. **`06_effect_size.ipynb`** - Statistical significance and effect sizes
8. **`07_cluster_scaling_behavior.ipynb`** - Horizontal scaling analysis
9. **`99_generate_figures.ipynb`** - Custom publication-quality figure generation

### How to Use Notebooks

```bash
# 1. Activate environment
cd analysis
source venv/bin/activate

# 2. Start JupyterLab
jupyter lab

# 3. Open notebooks in browser
# Navigate to: analysis/notebooks/
# Start with: 00_setup.ipynb → 01_load_results.ipynb
```

**Recommended Workflow:**
1. Run automated scripts first (get standard outputs)
2. Review outputs in `final-results/figures/`
3. Use notebooks for specific deep dives or custom figures
4. Export final figures from notebooks to `final-results/figures/`

---

## 📈 Understanding Your Outputs

### Statistical Summary (`aggregated_stats.json`)

```json
{
  "experiment_id": "rsa2048_p256_r100",
  "environment": "native",
  "n_runs": 5,
  "p50": {
    "mean": 498.0,
    "std": 12.5,
    "ci_low": 485.0,
    "ci_high": 511.0
  },
  "p95": {
    "mean": 845.0,
    "std": 25.3,
    "ci_low": 820.0,
    "ci_high": 870.0
  },
  "throughput": {
    "mean": 8595.0,
    "std": 150.2
  }
}
```

**For Dissertation:**
- Use `mean` values for main results
- Use `ci_low` and `ci_high` for confidence intervals
- Use `std` to discuss variability

### Hypothesis Tests (`hypothesis_tests.json`)

```json
{
  "comparison_id": "kyber512_vs_rsa2048_native",
  "tests": {
    "welch_t": {
      "p_value": 0.0001,
      "significant": true
    }
  },
  "effect_size": {
    "cohens_d": 0.85,
    "interpretation": "large effect"
  }
}
```

**For Dissertation:**
- `p_value < 0.05` = Statistically significant difference
- `cohens_d > 0.8` = Large practical effect
- Use these to support claims about algorithm differences

---

## 🎨 Customizing Figures for Dissertation

### Using Scripts (Standard Figures)

Scripts generate standard figures automatically. To customize:

1. **Modify scripts** in `analysis/scripts/` (e.g., `plot_combined_cdfs.py`)
2. **Adjust parameters** (colors, labels, sizes) in script source
3. **Re-run** the script

### Using Notebooks (Custom Figures)

For publication-quality customizations:

1. **Open** `99_generate_figures.ipynb`
2. **Load your data** from `final-results/aggregated_stats.json`
3. **Customize** plot aesthetics (colors, fonts, sizes, labels)
4. **Export** high-resolution PNG/PDF for dissertation

**Example customization:**
```python
# In notebook
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-paper')  # Publication style
plt.rcParams['figure.dpi'] = 300     # High resolution
plt.rcParams['font.size'] = 12       # Readable font size
# ... create custom plot
plt.savefig('final-results/figures/custom_figure.png', dpi=300, bbox_inches='tight')
```

---

## 📋 Complete Workflow Checklist

### Phase 1: Automated Analysis (30-60 minutes)

- [ ] Setup Python environment
- [ ] Verify data exists in `results/`
- [ ] Regenerate index if needed
- [ ] Run `aggregate_results.py`
- [ ] Run `plot_combined_cdfs.py`
- [ ] Run `plot_scaling_curves.py` (if applicable)
- [ ] Run `hypothesis_tests.py`
- [ ] Run `compare_all_environments.py`
- [ ] Generate final report

### Phase 2: Review Outputs (15-30 minutes)

- [ ] Check `final-results/figures/` for all plots
- [ ] Review `aggregated_stats.csv` for table data
- [ ] Review `hypothesis_table.csv` for statistical tests
- [ ] Identify which figures to use in dissertation

### Phase 3: Custom Analysis (Optional, 1-3 hours)

- [ ] Open Jupyter notebooks for specific deep dives
- [ ] Create custom visualizations if needed
- [ ] Validate statistical findings
- [ ] Generate publication-quality figures

### Phase 4: Dissertation Integration

- [ ] Extract tables from CSV files
- [ ] Select best figures from `final-results/figures/`
- [ ] Write Results chapter using statistical summaries
- [ ] Cite figures and tables in dissertation

---

## 🎯 Quick Command Reference

```bash
# Complete analysis in one go
cd analysis && source venv/bin/activate && cd ..
INDEX_FILE="final-results/index.json"
FINAL_RESULTS_DIR="final-results"

python3 analysis/aggregate_results.py --index "$INDEX_FILE" --output "$FINAL_RESULTS_DIR"
python3 analysis/plot_combined_cdfs.py --index "$INDEX_FILE" --output "$FINAL_RESULTS_DIR/figures"
python3 analysis/hypothesis_tests.py --index "$INDEX_FILE" --matrix orchestration/experiment_matrix.yaml --output "$FINAL_RESULTS_DIR"
python3 analysis/compare_all_environments.py --native "$FINAL_RESULTS_DIR/aggregated_stats.json" --minikube "$FINAL_RESULTS_DIR/aggregated_stats.json" --gcp "$FINAL_RESULTS_DIR/aggregated_stats.json" --output "$FINAL_RESULTS_DIR/comparisons/"

# Start Jupyter for custom analysis
cd analysis && source venv/bin/activate && jupyter lab
```

---

## 🆘 Troubleshooting

### Scripts Fail with "ModuleNotFoundError"

```bash
cd analysis
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### No Figures Generated

Check that:
1. Raw data exists: `find results -name "run.jsonl" | wc -l`
2. Index file is valid: `python3 -c "import json; json.load(open('final-results/index.json'))"`
3. Python environment is activated

### Notebooks Don't Start

```bash
cd analysis
source venv/bin/activate
pip install jupyterlab
jupyter lab
```

---

## 📚 Next Steps

1. **Run automated analysis** (Phase 1 above)
2. **Review outputs** in `final-results/`
3. **Use notebooks** for any custom analysis needed
4. **Extract tables/figures** for dissertation
5. **Write Results chapter** using the generated statistics

**All dissertation-ready outputs will be in `final-results/` directory!** 🎓

