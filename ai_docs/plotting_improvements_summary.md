# HELMET Plotting Script Improvements Summary

This document summarizes all the improvements and changes made to the plotting scripts for the HELMET benchmark results.

## Overview

Two main plotting scripts were created and improved:
1. `scripts/plot_icl_banking_task.py` - ICL Banking Task analysis
2. `scripts/plot_rerank_task.py` - Rerank Task analysis

## Major Changes and Improvements

### 1. Data Filtering

#### Techniques Filtered Out
- **Removed MInference**: Filtered from all plots and legends
- **Removed Quest**: Already filtered out
- **Removed streamingllm_original**: Already filtered out

#### Model-Specific Filtering

**Reasoning Models (DeepSeek-R1-Distill-Llama-8B):**
- For SnapKV and PyramidKV: Only keep specific configurations:
  - `w256_c2048_k7_avgpool`
  - `w2048_c8192_k7_avgpool`
  - `w256_c2048_k7_maxpool`
  - `w2048_c8192_k7_maxpool`
  - `default`
- All other cache size configurations are filtered out

**Baseline Models (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct):**
- For SnapKV and PyramidKV: Filtered out cache sizes with window sizes:
  - `w32_*` (any cache size with window=32)
  - `w1024_*` (any cache size with window=1024)

**Models Excluded from Legends:**
- DeepSeek-R1-Distill-Qwen-7B
- Qwen3-8B
- Yarn-Qwen3-8B (excluded from ICL banking legend but included in Rerank plot data)

#### Cache Size Filtering (Applied to All Plots)

**IMPORTANT: All plotting scripts should apply these cache size filters to maintain consistency:**

**For Reasoning Models** (DeepSeek-R1-Distill-Llama-8B, Yarn-Qwen3-8B):
- SnapKV and PyramidKV: Keep **only** these configurations:
  - `w256_c2048_k7_avgpool`
  - `w2048_c8192_k7_avgpool`
  - `w256_c2048_k7_maxpool`
  - `w2048_c8192_k7_maxpool`
  - `default`
- All other cache size configurations are filtered out

**For Baseline Models** (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct):
- SnapKV and PyramidKV: Filter out:
  - Any configurations with `w32_*` (window size 32)
  - Any configurations with `w1024_*` (window size 1024)

**For StreamingLLM** (All Models):
- Keep **only** the `n_local_4096_n_init_4` configuration
- Filter out all other cache size configurations

**Implementation Example:**
```python
# Filter SnapKV and PyramidKV for reasoning models
allowed_reasoning_configs = ['w256_c2048_k7_avgpool', 'w2048_c8192_k7_avgpool',
                              'w256_c2048_k7_maxpool', 'w2048_c8192_k7_maxpool', 'default']
reasoning_models = ['DeepSeek-R1-Distill-Llama-8B', 'Yarn-Qwen3-8B']

for df in [memory_df, throughput_df, performance_df]:
    condition = (
        (df['model'].isin(reasoning_models)) &
        (df['technique'].isin(['snapkv', 'pyramidkv'])) &
        (~df['cache_size'].isin(allowed_reasoning_configs))
    )
    df.drop(df[condition].index, inplace=True)

# Filter out w32 and w1024 for baseline models
baseline_models = ['Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct']
for df in [memory_df, throughput_df, performance_df]:
    condition = (
        (df['model'].isin(baseline_models)) &
        (df['technique'].isin(['snapkv', 'pyramidkv'])) &
        (df['cache_size'].str.contains('w32_|w1024_', na=False))
    )
    df.drop(df[condition].index, inplace=True)

# Filter StreamingLLM to keep only n_local=4096, n_init=4
allowed_streamingllm_config = 'n_local_4096_n_init_4'
for df in [memory_df, throughput_df, performance_df]:
    condition = (
        (df['technique'] == 'streamingllm') &
        (df['cache_size'] != allowed_streamingllm_config)
    )
    df.drop(df[condition].index, inplace=True)
```

### 2. Visual Style Improvements

#### Marker Styling
- **Removed black bold outlines** from all scatter plot markers
- Set `edgecolor` removed and `linewidth` removed from scatter plots
- Markers now appear cleaner without heavy borders

#### Legend Improvements
- **Removed black outlines** from legend markers (`markeredgewidth=0`)
- **Removed shadow effects** for cleaner appearance
- **Legend font size increased to 11pt minimum** for better readability
- **Consistent legend layout across all plots:**
  - **Combined plots (2 subplots)**: Single horizontal row at bottom with all models and techniques
  - **Standalone plots**: Vertical column on the right side with sections for "Models:" and "Techniques:"
  - **Rerank plot**: Changed from horizontal bottom layout to vertical right-side layout (matching ICL plots)

### 3. Annotation Improvements

#### Cache Size Label Format
Changed from `C=X,K=Y` format to `W=X,C=Y` format:
- **Before**: `C=8192,K=7` (showed cache size and K parameter)
- **After**: `W=2048,C=8192` (shows window size and cache size)
- K parameter removed as it's redundant

#### Font Size
- **Increased annotation font size** from 7pt to 10pt
- Ensures readability when printed in LaTeX papers (minimum 8pt requirement)
- **All fonts meet minimum 10pt requirement** (11pt for legends and y-tick labels)

### 4. Plot Layout and Sizing

#### ICL Banking Task (`plot_icl_banking_task.py`)
- **Six versions created (3 for Banking77, 3 for Clinic150):**

  **Banking77:**
  1. **Combined**: 16x6 inches, 2 subplots (Memory + Latency), horizontal legend
  2. **Memory only**: 14x6 inches, 1 subplot, vertical legend on right
  3. **Latency only**: 14x6 inches, 1 subplot, vertical legend on right

  **Clinic150:**
  4. **Combined**: 16x6 inches, 2 subplots (Memory + Latency), horizontal legend
  5. **Memory only**: 14x6 inches, 1 subplot, vertical legend on right
  6. **Latency only**: 14x6 inches, 1 subplot, vertical legend on right

- **Wider standalone plots** (14" instead of 12") to prevent legend from covering annotations

#### Rerank Task (`plot_rerank_task.py`)
- **Single plot**: 14x8 inches with vertical legend on right (changed from 12x8 with horizontal legend below)
- Models included: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, DeepSeek-R1-Distill-Llama-8B, Yarn-Qwen3-8B

### 5. Plot Titles

#### ICL Banking Task
**Title**: "Reasoning Models Recover Performance Degradation Compared to Non-Reasoning Models on In-Context Learning Tasks"

**Subtitles Added:**
- Banking77: "HELMET In-Context Learning Task: Banking77 (16K Context)"
- Clinic150: "HELMET In-Context Learning Task: Clinic150 (16K Context)"
- Applied to all 6 plot variations

#### Rerank Task
**Title**: "Token Eviction Methods Struggle on the Pareto Frontier of Performance vs Efficiency"

**Subtitle**: "HELMET Rerank Task: Memory vs NDCG@10 (16K Context)"

#### Title Spacing
- **Reduced title-to-plot distance** by changing y-coordinate from 1.02 to 0.995 (ICL Banking) and 0.98 (Rerank)
- Tighter spacing creates more cohesive visual layout

### 6. Jupyter Notebook Integration

**IMPORTANT: This code block should ALWAYS be added at the end of every plotting script:**

```python
# Display plot inline (useful for Jupyter notebooks)
try:
    from IPython.display import display, Image as IPImage

    print("\nDisplaying plot:")
    display(IPImage(filename=output_path))
except ImportError:
    # Not in a Jupyter environment, skip display
    print("\nNote: Run in Jupyter notebook to see inline plot preview.")
```

This allows plots to be displayed inline when running in Jupyter notebooks while still working as standalone scripts. This pattern should be used consistently across all plotting scripts to provide a better user experience in both Jupyter and terminal environments.

### 7. Font Sizes (All >= 10pt minimum, 11pt for legends)

| Element | Size |
|---------|------|
| Annotations | 10pt |
| Y-tick labels | 11pt |
| Legend | 11pt |
| X-tick labels | 10pt |
| Base font | 11pt |
| Axis labels | 12pt |
| Subplot titles | 14pt |
| Main title | 16-18pt |

All text now meets publication standards with minimum 10pt font size (11pt for legends and y-tick labels).

## Output Files

### ICL Banking Task - Banking77
- `results/plots/icl_banking_task_analysis_combined.png` (and .pdf)
- `results/plots/icl_banking_task_analysis_memory.png` (and .pdf)
- `results/plots/icl_banking_task_analysis_latency.png` (and .pdf)

### ICL Banking Task - Clinic150
- `results/plots/icl_clinic150_task_analysis_combined.png` (and .pdf)
- `results/plots/icl_clinic150_task_analysis_memory.png` (and .pdf)
- `results/plots/icl_clinic150_task_analysis_latency.png` (and .pdf)

### Rerank Task
- `results/plots/rerank_task_analysis.png` (and .pdf)

All plots are saved at 300 DPI with white backgrounds.

## Techniques Shown

Both plots now show these 7 techniques:
1. baseline
2. INT8
3. INT4
4. pyramidkv
5. snapkv
6. streamingllm
7. duoattn

## Color Palette

Models use distinct colors for easy identification:
- **Llama-3.1-8B-Instruct**: Dark orange (#FF8C00)
- **Qwen2.5-7B-Instruct**: Dodger blue (#1E90FF)
- **DeepSeek-R1-Distill-Llama-8B**: Crimson (#DC143C)
- **Yarn-Qwen3-8B**: Saddle brown (#8B4513)

## Code Structure

Both scripts follow a similar structure:
1. Data loading and filtering
2. Style configuration
3. Helper functions (cache size formatting)
4. Plotting functions
5. Legend creation
6. Output saving
7. Jupyter display support

## Key Technical Details

- **Data sources**: CSV files from `results/helmet_results/`
  - `helmet_memory_usage.csv`
  - `helmet_throughput_csv`
  - `helmet_performance.csv`

- **Context length**: All plots show 16K context results

- **Metrics**:
  - ICL Banking: Exact Match Score
  - Rerank: NDCG@10 Score

- **X-axes**:
  - Memory Usage (GB)
  - Latency (s/token) - converted from throughput

## Data Availability Notes

### Yarn-Qwen3-8B Rerank Results
Yarn-Qwen3-8B has limited rerank data available:
- **Baseline**: 49.72 NDCG@10
- **INT4**: 53.41 NDCG@10
- **INT8**: 48.33 NDCG@10
- **Other techniques** (SnapKV, PyramidKV, StreamingLLM, DuoAttn): No rerank results (0.0 or empty)

The model shows reasonable performance on baseline and quantization techniques but lacks results for advanced KV cache compression methods.

## Future Improvements

Potential areas for enhancement:
- Add 32K context length plots
- Support for additional tasks from HELMET benchmark
- Interactive plots with Plotly
- Automated comparison across multiple context lengths
- Statistical significance indicators
- Complete evaluation of Yarn-Qwen3-8B on all techniques for rerank task
