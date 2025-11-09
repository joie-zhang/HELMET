# Script Output Files - Clickable File Paths

## **Most Recent Scripts (Oct 28, 2025)**

### 1. **ICL Analysis & Plotting** (Oct 28, 07:16-07:18)

#### `scripts/plot_icl_memory_only.py` - **PLOT SCRIPT**
**Generates:**
- `results/plots/icl_average_memory_only.png` - ICL Memory-Only Plot (Main)
- `results/plots/icl_average_memory_only.pdf`
- `results/plots/multitask_average_memory_only.png` - Multitask Memory-Only Plot
- `results/plots/multitask_average_memory_only.pdf`
- `results/plots/icl_average_memory_only_small_cache.png` - ICL Small Cache Variant
- `results/plots/icl_average_memory_only_small_cache.pdf`
- `results/plots/multitask_average_memory_only_small_cache.png` - Multitask Small Cache Variant
- `results/plots/multitask_average_memory_only_small_cache.pdf`
- `results/plots/icl_average_16k_32k_combined.png` - ICL Combined Context Lengths
- `results/plots/icl_average_16k_32k_combined.pdf`

#### `extract_icl_plot_data.py` - **Data Extraction Script**
**Generates:**
- `results/icl_plot_data_16k.csv` - Extracted ICL plot data

#### `analyze_icl_findings.py` - **Analysis Script** (Text Findings)
**Generates:**
- `results/icl_findings.txt` - Text findings comparing reasoning vs instruction-tuned models
- `results/icl_findings_summary.csv` - Summary statistics CSV

---

### 2. **Task Vulnerability Analysis** (Oct 28, 06:23-06:24)

#### `generate_task_vulnerability_paragraph.py` - **Text Generation Script**
**Generates:**
- `results/task_vulnerability_paragraph.txt` - LaTeX paragraph text
- `results/task_vulnerability_paragraph_stats.csv` - Statistics CSV

#### `analyze_task_vulnerabilities.py` - **Analysis Script**
**Generates:**
- `results/task_vulnerability_rankings.csv` - Task vulnerability rankings
- `results/technique_selectivity.csv` - Technique selectivity analysis

---

### 3. **Task Deltas Analysis** (Oct 28, 05:57-05:59)

#### `scripts/plot_task_deltas_averaged_configs.py` - **PLOT SCRIPT**
**Generates:**
- `results/plots/task_deltas_averaged_configs.png` - Task Performance Deltas (Averaged Cache Configs)
- `results/plots/task_deltas_averaged_configs.pdf`

#### `scripts/plot_task_deltas_separate_configs.py` - **PLOT SCRIPT**
**Generates:**
- `results/plots/task_deltas_separate_configs.png` - Task Performance Deltas (Separate Configs)
- `results/plots/task_deltas_separate_configs.pdf`

#### `extract_task_deltas_data.py` - **Data Extraction Script**
**Generates:**
- `results/task_deltas_data.csv` - Task delta data (grouped)
- `results/task_deltas_data_detailed.csv` - Task delta data (ungrouped, detailed)

#### `generate_task_deltas_findings.py` - **Text Generation Script**
**Generates:**
- `results/task_deltas_findings.txt` - LaTeX paragraph text
- `results/task_deltas_summary_statistics.csv` - Summary statistics CSV

---

### 4. **Quadrant Plot Analysis** (Oct 28, 05:32-05:47)

#### `scripts/plot_quadrant_comparison_1x1.py` - **PLOT SCRIPT**
**Generates:**
- `results/plots/quadrant_comparison_1x1_grouped.png` - 1x1 Grouped Bar Plot (Main)
- `results/plots/quadrant_comparison_1x1_grouped.pdf`
- `results/plots/quadrant_comparison_1x1_grouped_notitle.png` - No title variant
- `results/plots/quadrant_comparison_1x1_grouped_notitle.pdf`
- `results/plots/quadrant_comparison_1x1_grouped_title_nobox.png` - Title, no box variant
- `results/plots/quadrant_comparison_1x1_grouped_title_nobox.pdf`

#### `scripts/plot_quadrant_comparison.py` - **PLOT SCRIPT**
**Generates:**
- `results/plots/quadrant_comparison.png` - 2x2 Quadrant Plot
- `results/plots/quadrant_comparison.pdf`

#### `extract_quadrant_plot_data.py` - **Data Extraction Script**
**Generates:**
- `results/quadrant_plot_data.csv` - Extracted quadrant plot data

#### `analyze_quadrant_findings.py` - **Analysis Script**
**Generates:**
- `results/quadrant_plot_summary_statistics.csv` - Summary statistics CSV

#### `quadrant_plot_key_findings.py` - **Analysis Script**
**Generates:**
- `results/quadrant_plot_summary_statistics.csv` - Key findings statistics

#### `quadrant_plot_latex_paragraph.py` - **Text Generation Script**
**Generates:**
- `results/quadrant_plot_latex_paragraph.txt` - LaTeX paragraph text

---

### 5. **Main Plot Data Extraction** (Oct 28, 05:18-05:36)

#### `scripts/plot_from_averaged_csv.py` - **PLOT SCRIPT**
**Generates:**
- `results/plots/averages_comparison_1x1_all_techniques_with_connections_incl_duo.png` - Main Comparison Plot (Figure 1)
- `results/plots/averages_comparison_1x1_all_techniques_with_connections_incl_duo.pdf`

#### `extract_plot_data_with_both_cache_sizes.py` - **Data Extraction Script**
**Generates:**
- `results/plot_data_all_cache_sizes.csv` - Full plot data with all cache sizes
- `results/plot_data_averaged_across_cache_sizes.csv` - Averaged plot data (used by plot_from_averaged_csv.py)

#### `extract_plot_data.py` - **Data Extraction Script**
**Generates:**
- `results/plot_data_summary_for_connections.csv` - Plot data summary for connection lines

#### `recalculate_with_averaged_cache_sizes.py` - **Data Processing Script**
**Generates:** (No direct output files - processes data)

#### `recalculate_paragraph_stats.py` - **Data Processing Script**
**Generates:** (No direct output files - processes data)

#### `calculate_stats.py` - **Statistics Calculation Script**
**Generates:** (No direct output files - calculates statistics)

---

### 6. **Task Pairs Correlation Analysis** (Oct 27-28)

#### `analyze_task_pairs_r2_reorganized.py` - **PLOT SCRIPT**
**Generates:**
- `results/plots/task_pairs_r2_reorganized.png` - Task Pairs R² Heatmap (Reorganized Axes)
- `results/plots/task_pairs_r2_reorganized.pdf`

#### `analyze_task_pairs_spearman.py` - **PLOT SCRIPT**
**Generates:**
- `results/plots/task_pairs_spearman.png` - Task Pairs Spearman Correlation Heatmap
- `results/plots/task_pairs_spearman.pdf`

#### `analyze_all_task_pairs_r2.py` - **PLOT SCRIPT**
**Generates:**
- `results/plots/task_pairs_r2_heatmap.png` - Task Pairs R² Heatmap
- `results/plots/task_pairs_r2_heatmap.pdf`

---

### 7. **Regression Analysis** (Oct 26)

#### `analyze_niah_recall_regression.py` - **PLOT SCRIPT**
**Generates:**
- `results/plots/niah_vs_recall_jsonkv_regression.png` - NIAH vs RECALL Scatter Plot with Regression
- `results/plots/niah_vs_recall_jsonkv_regression.pdf`

---

## **Older Scripts (Utility & Config Generation)**

### 8. **Config & Utility Scripts**

#### `scripts/generate_multilex_priority_configs.py` - **Config Generation Script**
**Generates:** Config files in `scripts/configs/multilex_priority/` directory

#### `analyze_missing_multilex.py` - **Analysis Script**
**Generates:** (No output files - prints to console)

#### `analyze_missing_experiments.py` - **Analysis Script**
**Generates:** (No output files - prints to console)

#### `scripts/plot_averages_comparison_old.py` - **PLOT SCRIPT (OLD VERSION)**
**Generates:** (Old version - likely superseded by plot_averages_comparison.py)

#### `scripts/eval_gpt4_summ.py` - **Evaluation Script**
**Generates:** (Evaluation outputs, not plots)

#### `filter_priority_jobs.py` - **Utility Script**
**Generates:** (Filters job priorities, no output files)

---

## **Summary: Files That Exist**

### **Plot Files (PNG/PDF):**
✅ `results/plots/icl_average_memory_only.png/pdf`
✅ `results/plots/multitask_average_memory_only.png/pdf`
✅ `results/plots/icl_average_memory_only_small_cache.png/pdf`
✅ `results/plots/multitask_average_memory_only_small_cache.png/pdf`
✅ `results/plots/icl_average_16k_32k_combined.png/pdf`
✅ `results/plots/quadrant_comparison_1x1_grouped.png/pdf`
✅ `results/plots/quadrant_comparison_1x1_grouped_notitle.png/pdf`
✅ `results/plots/quadrant_comparison_1x1_grouped_title_nobox.png/pdf`
✅ `results/plots/quadrant_comparison.png/pdf`
✅ `results/plots/task_deltas_averaged_configs.png/pdf`
✅ `results/plots/task_deltas_separate_configs.png/pdf`
✅ `results/plots/task_pairs_r2_reorganized.png/pdf`
✅ `results/plots/task_pairs_spearman.png/pdf`
✅ `results/plots/task_pairs_r2_heatmap.png/pdf`
✅ `results/plots/niah_vs_recall_jsonkv_regression.png/pdf`
✅ `results/plots/averages_comparison_1x1_all_techniques_with_connections_incl_duo.png/pdf`

### **Data Files (CSV):**
✅ `results/icl_plot_data_16k.csv`
✅ `results/icl_findings_summary.csv`
✅ `results/task_deltas_data.csv`
✅ `results/task_deltas_data_detailed.csv`
✅ `results/task_deltas_summary_statistics.csv`
✅ `results/quadrant_plot_data.csv`
✅ `results/quadrant_plot_summary_statistics.csv`
✅ `results/plot_data_all_cache_sizes.csv`
✅ `results/plot_data_averaged_across_cache_sizes.csv`
✅ `results/plot_data_summary_for_connections.csv`
✅ `results/task_vulnerability_paragraph_stats.csv`

### **Text Files:**
✅ `results/icl_findings.txt`
✅ `results/task_vulnerability_paragraph.txt`
✅ `results/task_deltas_findings.txt`
✅ `results/quadrant_plot_latex_paragraph.txt`

---

## **Quick Decision Guide**

### **Keep These (Generate Important Figures):**
- ✅ `scripts/plot_icl_memory_only.py` - ICL plots
- ✅ `scripts/plot_quadrant_comparison_1x1.py` - Quadrant 1x1 plot
- ✅ `scripts/plot_quadrant_comparison.py` - Quadrant 2x2 plot
- ✅ `scripts/plot_task_deltas_averaged_configs.py` - Task deltas (averaged)
- ✅ `scripts/plot_task_deltas_separate_configs.py` - Task deltas (separate)
- ✅ `scripts/plot_from_averaged_csv.py` - Main comparison plot (Figure 1)
- ✅ `analyze_task_pairs_r2_reorganized.py` - Correlation heatmap
- ✅ `analyze_task_pairs_spearman.py` - Spearman correlation
- ✅ `analyze_all_task_pairs_r2.py` - R² heatmap
- ✅ `analyze_niah_recall_regression.py` - Regression plot

### **Keep These (Support Data Extraction):**
- ✅ `extract_icl_plot_data.py` - Extracts ICL data
- ✅ `extract_task_deltas_data.py` - Extracts task delta data
- ✅ `extract_quadrant_plot_data.py` - Extracts quadrant data
- ✅ `extract_plot_data_with_both_cache_sizes.py` - Extracts main plot data
- ✅ `extract_plot_data.py` - Extracts plot data

### **Keep These (Generate Text Findings):**
- ✅ `analyze_icl_findings.py` - ICL findings text
- ✅ `generate_task_vulnerability_paragraph.py` - Task vulnerability text
- ✅ `generate_task_deltas_findings.py` - Task delta findings text
- ✅ `quadrant_plot_latex_paragraph.py` - Quadrant plot text
- ✅ `analyze_quadrant_findings.py` - Quadrant findings
- ✅ `quadrant_plot_key_findings.py` - Quadrant key findings
- ✅ `analyze_task_vulnerabilities.py` - Task vulnerability analysis

### **Consider Removing (Utility/Old Versions):**
- ⚠️ `scripts/plot_averages_comparison_old.py` - Old version (superseded)
- ⚠️ `analyze_missing_multilex.py` - One-time analysis script
- ⚠️ `analyze_missing_experiments.py` - One-time analysis script
- ⚠️ `filter_priority_jobs.py` - Utility script (may be useful)
- ⚠️ `scripts/generate_multilex_priority_configs.py` - Config generation (may be useful)
- ⚠️ `scripts/eval_gpt4_summ.py` - Evaluation script (may be useful)
- ⚠️ `recalculate_with_averaged_cache_sizes.py` - Data processing (may be useful)
- ⚠️ `recalculate_paragraph_stats.py` - Data processing (may be useful)
- ⚠️ `calculate_stats.py` - Statistics calculation (may be useful)

