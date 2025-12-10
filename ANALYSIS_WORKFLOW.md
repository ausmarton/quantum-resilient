# Analysis Workflow for Dissertation

**Complete guide to analyzing collected experiment data and generating dissertation-ready outputs.**

---

## 📋 Prerequisites

### 1. Install Python Dependencies

```bash
cd analysis
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
cd ..
```

### 2. Verify Data Collection

Check that you have data in all three environments:

```bash
# Count experiments per environment
echo "Native: $(ls -d results/native/*/ 2>/dev/null | wc -l)"
echo "Minikube: $(ls -d results/minikube/*/ 2>/dev/null | wc -l)"
echo "GCP: $(ls -d results/gcp/*/ 2>/dev/null | wc -l)"
```

---

## 🚀 Step-by-Step Analysis Workflow

### Step 1: Regenerate Index (if needed)

If you collected data separately or the index is outdated:

```bash
./scripts/regenerate_index_from_results.sh \
  --matrix orchestration/experiment_matrix.yaml \
  --output final-results/
```

This creates/updates `final-results/index.json` with all your collected experiments.

**Verify the index:**
```bash
python3 -c "import json; idx=json.load(open('final-results/index.json')); print(f\"Total: {len(idx['experiments'])} experiments, Completed: {idx['completed_scenarios']}, Failed: {idx['failed_scenarios']}\")"
```

---

### Step 2: Run Complete Analysis Pipeline

This will:
- Aggregate statistics across all experiments
- Generate combined CDF plots
- Run statistical hypothesis tests
- Create scaling curves (if you have horizontal scaling data)
- Generate dissertation-ready figures

```bash
# Set the index file location
INDEX_FILE="final-results/index.json"
FINAL_RESULTS_DIR="final-results"

# Step 2a: Aggregate Results
python3 analysis/aggregate_results.py \
  --index "$INDEX_FILE" \
  --output "$FINAL_RESULTS_DIR"

# Step 2b: Generate Combined CDF Plots
python3 analysis/plot_combined_cdfs.py \
  --index "$INDEX_FILE" \
  --output "$FINAL_RESULTS_DIR/figures"

# Step 2c: Generate Scaling Curves (if you have scaling experiments)
python3 analysis/plot_scaling_curves.py \
  --index "$INDEX_FILE" \
  --output "$FINAL_RESULTS_DIR/figures"

# Step 2d: Run Statistical Hypothesis Tests
python3 analysis/hypothesis_tests.py \
  --index "$INDEX_FILE" \
  --matrix orchestration/experiment_matrix.yaml \
  --output "$FINAL_RESULTS_DIR"

# Step 2e: Generate Replica Scaling Plots (if applicable)
python3 analysis/plot_replica_scaling.py \
  --index "$INDEX_FILE" \
  --output "$FINAL_RESULTS_DIR/figures/scaling"
```

**Or run all at once using the main script (if you have the full pipeline):**

```bash
# This runs all analysis steps automatically
./run_all_experiments.sh \
  --skip-generation \
  --skip-native \
  --skip-minikube \
  --skip-gcp \
  --matrix orchestration/experiment_matrix.yaml
```

---

### Step 3: Cross-Environment Comparison

Compare performance across native, Minikube, and GCP:

```bash
# For a specific algorithm/payload/rate combination
python3 analysis/compare_all_environments.py \
  --native results/native/<scenario-id>/stats/summary.json \
  --minikube results/minikube/<scenario-id>/stats/summary.json \
  --gcp results/gcp/<scenario-id>/stats/summary.json \
  --output final-results/comparisons/
```

**For aggregated comparison across all experiments:**

```bash
# Compare using aggregated stats
python3 analysis/compare_all_environments.py \
  --native final-results/aggregated_stats.json \
  --minikube final-results/aggregated_stats.json \
  --gcp final-results/aggregated_stats.json \
  --output final-results/comparisons/
```

---

### Step 4: Generate Dissertation-Ready Outputs

#### 4a. Generate Final Report (PDF)

```bash
python3 analysis/build_final_report.py \
  --results-dir final-results \
  --output final-results/report.pdf
```

#### 4b. Export Tables (LaTeX/CSV)

If you have table generation scripts:

```bash
# Generate LaTeX tables from aggregated stats
python3 -c "
import json
import csv

# Load aggregated stats
with open('final-results/aggregated_stats.json') as f:
    stats = json.load(f)

# Export to CSV for easy import into LaTeX
# (You may need to customize this based on your table needs)
print('Tables can be generated from aggregated_stats.json')
"
```

#### 4c. Prepare Figures for Dissertation

All figures are in `final-results/figures/`. Use these directly in your dissertation:

- **CDF plots**: `final-results/figures/*_cdf.png` - Latency distributions
- **Scaling plots**: `final-results/figures/scaling/*.png` - Throughput/latency vs replicas
- **Combined plots**: `final-results/figures/*_combined.png` - Multi-algorithm comparisons

---

### Step 5: Interactive Analysis (Optional)

For deeper exploration, use Jupyter notebooks:

```bash
cd analysis
source venv/bin/activate
jupyter lab
```

**Recommended notebooks:**
- `01_load_results.ipynb` - Load and preview your data
- `02_latency_analysis.ipynb` - Deep dive into latency distributions
- `06_effect_size.ipynb` - Statistical significance analysis
- `99_generate_figures.ipynb` - Custom figure generation

---

## 📊 Understanding Your Outputs

### Key Files Generated

| File | Purpose | Dissertation Use |
|------|---------|------------------|
| `final-results/index.json` | Master index of all experiments | Reference |
| `final-results/aggregated_stats.json` | Summary statistics across all runs | **Results chapter tables** |
| `final-results/aggregated_stats.csv` | Same data in CSV format | Import into Excel/LaTeX |
| `final-results/hypothesis_tests.json` | Statistical test results | **Statistical significance claims** |
| `final-results/hypothesis_table.csv` | Test results in table format | **Results chapter tables** |
| `final-results/figures/*.png` | All generated plots | **Figures for dissertation** |
| `final-results/report.pdf` | Complete analysis report | Reference document |

### Statistical Outputs Explained

**From `hypothesis_tests.json`:**
- **p-value**: Statistical significance (p < 0.05 = significant)
- **Cohen's d**: Effect size (|d| > 0.8 = large effect)
- **Interpretation**: Practical significance of differences

**From `aggregated_stats.json`:**
- **p50/p95/p99**: Percentile latencies (median, 95th, 99th percentile)
- **CI (confidence intervals)**: Uncertainty bounds (95% CI)
- **mean/std**: Central tendency and variability

---

## 🔍 Quick Verification

After running analysis, verify outputs:

```bash
# Check that key files exist
ls -lh final-results/aggregated_stats.*
ls -lh final-results/hypothesis_tests.*
ls -lh final-results/figures/*.png | wc -l  # Count figures

# Quick stats summary
python3 <<EOF
import json
with open('final-results/aggregated_stats.json') as f:
    stats = json.load(f)
print(f"Total aggregated experiments: {len(stats.get('experiments', []))}")
EOF
```

---

## 🎯 Next Steps for Dissertation

1. **Review figures** in `final-results/figures/` - select best ones for dissertation
2. **Extract tables** from `aggregated_stats.csv` and `hypothesis_table.csv`
3. **Write Results chapter** using:
   - Statistical summaries from `aggregated_stats.json`
   - Significance claims from `hypothesis_tests.json`
   - Figures from `final-results/figures/`
4. **Cross-environment analysis** using outputs from `compare_all_environments.py`
5. **Scaling analysis** (if applicable) using figures from `figures/scaling/`

---

## 🆘 Troubleshooting

### Missing Dependencies

```bash
cd analysis
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### Index File Issues

If experiments are missing from index:

```bash
./scripts/regenerate_index_from_results.sh \
  --matrix orchestration/experiment_matrix.yaml \
  --output final-results/
```

### Analysis Script Errors

Check Python version:
```bash
python3 --version  # Should be 3.10+
```

Check data files exist:
```bash
# Verify raw data exists
find results -name "run.jsonl" | wc -l

# Verify stats exist (if analysis was run before)
find results -name "summary.json" | wc -l
```

---

## 📚 Additional Resources

- **RESEARCHER_GUIDE.md** - Complete usage guide
- **analysis/README.md** - Analysis suite documentation
- **analysis/notebooks/** - Interactive Jupyter notebooks

---

**Ready to generate your dissertation figures and tables!** 🎓

