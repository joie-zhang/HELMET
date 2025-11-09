# Figure-to-Script Mapping

## Consolidated Notebook

**All figures can be generated from a single notebook:**
- `scripts/all_figures.ipynb` - Contains all figure scripts organized by figure number

Run this notebook to generate all paper figures in one place!

---

## Paper Figure Organization

### **Figure 1: Main Comparison Plot** (Memory vs Performance)
- **Script**: `scripts/plot_from_averaged_csv.py`
- **Output**: `results/plots/averages_comparison_1x1_all_techniques_with_connections_incl_duo.png/pdf`
- **Data Source**: `results/plot_data_averaged_across_cache_sizes.csv`
- **Support Scripts**:
  - `extract_plot_data_with_both_cache_sizes.py` - Generates averaged CSV data
  - `extract_plot_data.py` - Extracts plot data summary
  - `recalculate_with_averaged_cache_sizes.py` - Recalculates averages
  - `calculate_stats.py` - Calculates statistics

### **Figure 2: Quadrant Comparison Plot** (1x1 Grouped Bar Plot)
- **Script**: `scripts/plot_quadrant_comparison_1x1.py`
- **Output**: `results/plots/quadrant_comparison_1x1_grouped.png/pdf` (main version)
- **Variants**:
  - `results/plots/quadrant_comparison_1x1_grouped_notitle.png/pdf`
  - `results/plots/quadrant_comparison_1x1_grouped_title_nobox.png/pdf`
- **Data Source**: `results/quadrant_plot_data.csv`
- **Support Scripts**:
  - `extract_quadrant_plot_data.py` - Extracts quadrant data
  - `analyze_quadrant_findings.py` - Analyzes findings
  - `quadrant_plot_key_findings.py` - Generates key findings
  - `quadrant_plot_latex_paragraph.py` - Generates LaTeX text
- **Note**: The 2x2 version (`scripts/plot_quadrant_comparison.py`) exists but 1x1 is the one used

### **Figure 3: Task Performance Deltas**
- **Script**: `scripts/plot_task_deltas_averaged_configs.py` (main version)
- **Output**: `results/plots/task_deltas_averaged_configs.png/pdf`
- **Alternative**: `scripts/plot_task_deltas_separate_configs.py` → `results/plots/task_deltas_separate_configs.png/pdf`
- **Data Source**: `results/task_deltas_data.csv`
- **Support Scripts**:
  - `extract_task_deltas_data.py` - Extracts task delta data
  - `generate_task_deltas_findings.py` - Generates findings text

### **Figure 4: Task Correlation Heatmaps**
- **Scripts**:
  - `analyze_task_pairs_r2_reorganized.py` - **Main version** (reorganized axes)
  - `analyze_task_pairs_spearman.py` - Spearman correlation version
  - `analyze_all_task_pairs_r2.py` - Original R² version
- **Outputs**:
  - `results/plots/task_pairs_r2_reorganized.png/pdf` - **Primary**
  - `results/plots/task_pairs_spearman.png/pdf`
  - `results/plots/task_pairs_r2_heatmap.png/pdf`

### **Figure 6: ICL Memory-Only Analysis**
- **Script**: `scripts/plot_icl_memory_only.py`
- **Outputs**:
  - `results/plots/icl_average_memory_only.png/pdf` - **Main ICL plot**
  - `results/plots/multitask_average_memory_only.png/pdf` - Multitask version
  - `results/plots/icl_average_memory_only_small_cache.png/pdf` - Small cache variant
  - `results/plots/multitask_average_memory_only_small_cache.png/pdf` - Multitask small cache
  - `results/plots/icl_average_16k_32k_combined.png/pdf` - Combined context lengths
- **Data Source**: `results/icl_plot_data_16k.csv`
- **Support Scripts**:
  - `extract_icl_plot_data.py` - Extracts ICL plot data
  - `analyze_icl_findings.py` - Generates findings text

---

## Summary Table

| Figure | Script | Main Output File |
|--------|--------|------------------|
| **Figure 1** | `scripts/plot_from_averaged_csv.py` | `results/plots/averages_comparison_1x1_all_techniques_with_connections_incl_duo.png` |
| **Figure 2** | `scripts/plot_quadrant_comparison_1x1.py` | `results/plots/quadrant_comparison_1x1_grouped.png` |
| **Figure 3** | `scripts/plot_task_deltas_averaged_configs.py` | `results/plots/task_deltas_averaged_configs.png` |
| **Figure 4** | `analyze_task_pairs_r2_reorganized.py` | `results/plots/task_pairs_r2_reorganized.png` |
| **Figure 6** | `scripts/plot_icl_memory_only.py` | `results/plots/icl_average_memory_only.png` |

---

## Additional Analysis Scripts (Not Direct Figures)

- **Task Vulnerability Analysis**: `analyze_task_vulnerabilities.py`, `generate_task_vulnerability_paragraph.py`
- **Regression Analysis**: `analyze_niah_recall_regression.py` → `results/plots/niah_vs_recall_jsonkv_regression.png/pdf`

