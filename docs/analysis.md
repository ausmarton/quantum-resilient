## Analysis and Visualization Guide

This document explains how to analyze and visualize benchmark data.

### Load Results
- Results are written to the `results/` directory by the benchmarking runs.
- Primary files:
  - `benchmark_results.json` — list of detailed benchmark rows
  - `benchmark_results.csv` — same in CSV form
  - `benchmark_results.xlsx` — Excel with summary sheets

### Jupyter Notebook
Open the prepared notebook and run cells top to bottom:
```bash
jupyter notebook notebooks/benchmark_analysis.ipynb
```
The notebook provides:
- Summary tables by algorithm and data size
- Interactive latency distribution plots (Plotly)
- Throughput vs data size plots (log-log)
- PQC vs Classical overhead summary
- Security-level vs performance trade-off chart
- Simple data-driven recommendations

### Programmatic Analysis (Python)
You can also load results in Python:
```python
import pandas as pd

df = pd.read_csv('results/benchmark_results.csv')
print(df.head())
```

### Visualizations Generated Automatically
- `latency_comparison.png`
- `throughput_analysis.png`
- `performance_trends.png`
- `algorithm_comparison_heatmap.png`
- `interactive_dashboard.html` (interactive dashboard)

### Customizing Visualizations
- Modify `generate_visualizations()` in `src/python_orchestrator/benchmarking.py` to add/remove plots.
- Use `plotly` for interactive figures and `matplotlib/seaborn` for static charts.

### Exporting for Reports
- Use the Excel file for multi-sheet summaries.
- Embed static PNGs or the HTML dashboard as supplemental material.
