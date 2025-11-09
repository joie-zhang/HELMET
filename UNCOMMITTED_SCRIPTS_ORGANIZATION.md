# Uncommitted Scripts Organization - Reverse Chronological Order

## Scripts Organized by Most Recent Modification Date

### **Most Recent (Oct 28, 2025)**

#### 1. **ICL Analysis & Plotting** (Oct 28, 07:16-07:18)
- `analyze_icl_findings.py` (Oct 28, 07:18) - **Analysis script** (generates text findings, not plots)
  - Outputs: `results/icl_findings.txt`, `results/icl_findings_summary.csv`
  - **Figure**: Supports ICL memory-only plots analysis
  
- `extract_icl_plot_data.py` (Oct 28, 07:16) - **Data extraction script**
  - Outputs: `results/icl_plot_data_16k.csv`
  - **Figure**: Supports ICL memory-only plots (Figure for ICL analysis)

- `scripts/plot_icl_memory_only.py` (Oct 28, 00:39) - **PLOT SCRIPT**
  - Outputs: 
    - `icl_average_memory_only.png/pdf` - **ICL Memory-Only Plot (Main)**
    - `multitask_average_memory_only.png/pdf` - **Multitask Memory-Only Plot**
    - `icl_average_memory_only_small_cache.png/pdf` - **ICL Small Cache Variant**
    - `multitask_average_memory_only_small_cache.png/pdf` - **Multitask Small Cache Variant**
    - `icl_average_16k_32k_combined.png/pdf` - **ICL Combined Context Lengths**
  - **Figure**: ICL-specific memory vs performance analysis

#### 2. **Task Vulnerability Analysis** (Oct 28, 06:23-06:24)
- `generate_task_vulnerability_paragraph.py` (Oct 28, 06:24) - **Text generation script**
  - Outputs: `results/task_vulnerability_paragraph.txt`, `results/task_vulnerability_paragraph_stats.csv`
  - **Figure**: Supports task vulnerability analysis (text findings)

- `analyze_task_vulnerabilities.py` (Oct 28, 06:23) - **Analysis script**
  - Outputs: `results/task_vulnerability_rankings.csv`, `results/technique_selectivity.csv`
  - **Figure**: Supports task vulnerability analysis

#### 3. **Task Deltas Analysis** (Oct 28, 05:57-05:59)
- `generate_task_deltas_findings.py` (Oct 28, 05:59) - **Text generation script**
  - Outputs: `results/task_deltas_findings.txt`, `results/task_deltas_summary_statistics.csv`
  - **Figure**: Supports task delta plots (text findings)

- `extract_task_deltas_data.py` (Oct 28, 05:57) - **Data extraction script**
  - Outputs: `results/task_deltas_data.csv`, `results/task_deltas_data_detailed.csv`
  - **Figure**: Supports task delta plots

- `scripts/plot_task_deltas_averaged_configs.py` (Oct 28, 02:25) - **PLOT SCRIPT**
  - Outputs: `task_deltas_averaged_configs.png/pdf` - **Task Performance Deltas (Averaged)**
  - **Figure**: Task-wise performance differences (technique - baseline), averaged cache configs

- `scripts/plot_task_deltas_separate_configs.py` (Oct 26, 15:02) - **PLOT SCRIPT**
  - Outputs: `task_deltas_separate_configs.png/pdf` - **Task Performance Deltas (Separate Configs)**
  - **Figure**: Task-wise performance differences with separate cache configurations

#### 4. **Quadrant Plot Analysis** (Oct 28, 05:32-05:47)
- `quadrant_plot_latex_paragraph.py` (Oct 28, 05:47) - **Text generation script**
  - Outputs: `results/quadrant_plot_latex_paragraph.txt`
  - **Figure**: Supports quadrant plots (text findings)

- `quadrant_plot_key_findings.py` (Oct 28, 05:40) - **Analysis script**
  - Outputs: `results/quadrant_plot_summary_statistics.csv`
  - **Figure**: Supports quadrant plots

- `scripts/plot_quadrant_comparison_1x1.py` (Oct 28, 02:03) - **PLOT SCRIPT**
  - Outputs: `quadrant_comparison_1x1_grouped.png/pdf` (and variants: `_notitle`, `_title_nobox`)
  - **Figure**: 1x1 Quadrant Plot - Task Performance by Difficulty (grouped bars)

- `scripts/plot_quadrant_comparison.py` (Oct 26, 17:02) - **PLOT SCRIPT**
  - Outputs: `quadrant_comparison.png/pdf` - **2x2 Quadrant Plot**
  - **Figure**: 2x2 Quadrant Plot - Task Performance by Output Length and Dispersion

- `analyze_quadrant_findings.py` (Oct 28, 05:33) - **Analysis script**
  - Outputs: `results/quadrant_plot_data.csv`, `results/quadrant_plot_summary_statistics.csv`
  - **Figure**: Supports quadrant plots

- `extract_quadrant_plot_data.py` (Oct 28, 05:32) - **Data extraction script**
  - Outputs: `results/quadrant_plot_data.csv`
  - **Figure**: Supports quadrant plots

#### 5. **Main Plot Data Extraction** (Oct 28, 05:18-05:36)
- `recalculate_with_averaged_cache_sizes.py` (Oct 28, 05:36) - **Data processing script**
  - **Figure**: Supports main comparison plots (recalculates with averaged cache sizes)

- `extract_plot_data_with_both_cache_sizes.py` (Oct 28, 05:34) - **Data extraction script**
  - Outputs: 
    - `results/plot_data_all_cache_sizes.csv`
    - `results/plot_data_averaged_across_cache_sizes.csv`
  - **Figure**: Supports main comparison plots (Figure 1)

- `recalculate_paragraph_stats.py` (Oct 28, 05:28) - **Data processing script**
  - **Figure**: Supports main comparison plots (recalculates statistics)

- `extract_plot_data.py` (Oct 28, 05:25) - **Data extraction script**
  - Outputs: `results/plot_data_summary_for_connections.csv`
  - **Figure**: Supports main comparison plots (Figure 1)

- `calculate_stats.py` (Oct 28, 05:18) - **Statistics calculation script**
  - **Figure**: Supports various plots (general statistics)

- `scripts/plot_from_averaged_csv.py` (Oct 28, 05:45) - **PLOT SCRIPT**
  - Outputs: `averages_comparison_1x1_all_techniques_with_connections_incl_duo.png/pdf`
  - **Figure**: Main comparison plot (Figure 1) - generated from pre-computed CSV

#### 6. **Task Pairs Correlation Analysis** (Oct 27-28)
- `analyze_task_pairs_r2_reorganized.py` (Oct 28, 03:26) - **PLOT SCRIPT**
  - Outputs: `task_pairs_r2_reorganized.png/pdf` - **Task Pairs R² Heatmap (Reorganized)**
  - **Figure**: Correlation heatmap between task pairs (R² coefficients)

- `analyze_task_pairs_spearman.py` (Oct 27, 10:09) - **PLOT SCRIPT**
  - Outputs: `task_pairs_spearman.png/pdf` - **Task Pairs Spearman Correlation Heatmap**
  - **Figure**: Spearman correlation heatmap between task pairs

- `analyze_all_task_pairs_r2.py` (Oct 27, 10:08) - **PLOT SCRIPT**
  - Outputs: `task_pairs_r2_heatmap.png/pdf` - **Task Pairs R² Heatmap**
  - **Figure**: R² correlation heatmap between task pairs

#### 7. **Regression Analysis** (Oct 26)
- `analyze_niah_recall_regression.py` (Oct 26, 18:26) - **PLOT SCRIPT**
  - Outputs: `niah_vs_recall_jsonkv_regression.png/pdf` - **NIAH vs RECALL Regression**
  - **Figure**: Scatter plot showing relationship between NIAH and RECALL_JSONKV tasks

#### 8. **Other Scripts** (Older)
- `scripts/generate_multilex_priority_configs.py` (Oct 26, 16:06) - **Config generation script**
  - **Figure**: Not a plot script - generates config files for experiments

- `analyze_missing_multilex.py` (Oct 26, 15:49) - **Analysis script**
  - **Figure**: Not a plot script - analyzes missing experiments

- `scripts/plot_averages_comparison_old.py` (Oct 26, 12:01) - **PLOT SCRIPT (OLD VERSION)**
  - **Figure**: Old version of main comparison plot (likely superseded)

- `scripts/eval_gpt4_summ.py` (Oct 24, 11:08) - **Evaluation script**
  - **Figure**: Not a plot script - evaluates GPT-4 summaries

- `filter_priority_jobs.py` (Oct 13, 10:01) - **Utility script**
  - **Figure**: Not a plot script - filters job priorities

- `analyze_missing_experiments.py` (Oct 13, 09:39) - **Analysis script**
  - **Figure**: Not a plot script - analyzes missing experiments

---

## Summary by Figure Type

### **Main Paper Figures:**

1. **Figure 1: Main Comparison Plot** (Memory vs Performance)
   - `scripts/plot_from_averaged_csv.py` - Generates from CSV
   - `extract_plot_data.py` - Extracts data
   - `extract_plot_data_with_both_cache_sizes.py` - Extracts with cache sizes
   - `recalculate_with_averaged_cache_sizes.py` - Recalculates averages
   - `calculate_stats.py` - Calculates statistics

2. **ICL Memory-Only Plots** (ICL-specific analysis)
   - `scripts/plot_icl_memory_only.py` - **Main plot script**
   - `extract_icl_plot_data.py` - Extracts ICL data
   - `analyze_icl_findings.py` - Generates findings text

3. **Quadrant Plots** (Task difficulty analysis)
   - `scripts/plot_quadrant_comparison_1x1.py` - **1x1 grouped bar plot**
   - `scripts/plot_quadrant_comparison.py` - **2x2 quadrant plot**
   - `extract_quadrant_plot_data.py` - Extracts quadrant data
   - `analyze_quadrant_findings.py` - Analyzes quadrant findings
   - `quadrant_plot_key_findings.py` - Generates key findings
   - `quadrant_plot_latex_paragraph.py` - Generates LaTeX paragraph

4. **Task Delta Plots** (Performance differences)
   - `scripts/plot_task_deltas_averaged_configs.py` - **Averaged configs**
   - `scripts/plot_task_deltas_separate_configs.py` - **Separate configs**
   - `extract_task_deltas_data.py` - Extracts task delta data
   - `generate_task_deltas_findings.py` - Generates findings text

5. **Task Correlation Heatmaps**
   - `analyze_task_pairs_r2_reorganized.py` - **R² heatmap (reorganized)**
   - `analyze_task_pairs_spearman.py` - **Spearman correlation heatmap**
   - `analyze_all_task_pairs_r2.py` - **R² heatmap**

6. **Regression Analysis**
   - `analyze_niah_recall_regression.py` - **NIAH vs RECALL scatter plot**

### **Support Scripts (Not Direct Plot Generators):**
- Analysis scripts: `analyze_task_vulnerabilities.py`, `analyze_missing_multilex.py`, `analyze_missing_experiments.py`
- Text generation: `generate_task_vulnerability_paragraph.py`, `recalculate_paragraph_stats.py`
- Config generation: `scripts/generate_multilex_priority_configs.py`
- Utility: `filter_priority_jobs.py`, `scripts/eval_gpt4_summ.py`
- Old versions: `scripts/plot_averages_comparison_old.py`

---

## Recommended Git Commit Organization

### **Commit Group 1: ICL Analysis & Plots**
- `scripts/plot_icl_memory_only.py`
- `extract_icl_plot_data.py`
- `analyze_icl_findings.py`
- **Message**: "Added ICL memory-only analysis and plotting scripts"

### **Commit Group 2: Quadrant Analysis & Plots**
- `scripts/plot_quadrant_comparison_1x1.py`
- `scripts/plot_quadrant_comparison.py`
- `extract_quadrant_plot_data.py`
- `analyze_quadrant_findings.py`
- `quadrant_plot_key_findings.py`
- `quadrant_plot_latex_paragraph.py`
- **Message**: "Added quadrant comparison plots and analysis scripts"

### **Commit Group 3: Task Delta Analysis & Plots**
- `scripts/plot_task_deltas_averaged_configs.py`
- `scripts/plot_task_deltas_separate_configs.py`
- `extract_task_deltas_data.py`
- `generate_task_deltas_findings.py`
- **Message**: "Added task delta plots and analysis scripts"

### **Commit Group 4: Task Correlation Analysis**
- `analyze_task_pairs_r2_reorganized.py`
- `analyze_task_pairs_spearman.py`
- `analyze_all_task_pairs_r2.py`
- **Message**: "Added task pairs correlation analysis and heatmaps"

### **Commit Group 5: Main Plot Data Extraction**
- `scripts/plot_from_averaged_csv.py`
- `extract_plot_data.py`
- `extract_plot_data_with_both_cache_sizes.py`
- `recalculate_with_averaged_cache_sizes.py`
- `recalculate_paragraph_stats.py`
- `calculate_stats.py`
- **Message**: "Added plot data extraction and processing scripts for main comparison plots"

### **Commit Group 6: Regression & Other Analysis**
- `analyze_niah_recall_regression.py`
- `analyze_task_vulnerabilities.py`
- `generate_task_vulnerability_paragraph.py`
- **Message**: "Added regression analysis and task vulnerability scripts"

### **Commit Group 7: Utility & Config Scripts**
- `scripts/generate_multilex_priority_configs.py`
- `analyze_missing_multilex.py`
- `analyze_missing_experiments.py`
- `filter_priority_jobs.py`
- `scripts/eval_gpt4_summ.py`
- `scripts/plot_averages_comparison_old.py` (if keeping as reference)
- **Message**: "Added utility scripts for experiment management and config generation"

