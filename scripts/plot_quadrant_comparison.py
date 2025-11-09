#!/usr/bin/env python3
"""
2x2 Quadrant Plot: Task Performance by Output Length and Dispersion
Groups tasks into quadrants based on output length (short/long) and dispersion (low/high).
Averages performance across all tasks in each quadrant for each efficient inference technique.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style for professional appearance
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Define paths
results_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/results'
output_dir = os.path.join(results_dir, 'plots')
os.makedirs(output_dir, exist_ok=True)

# Load data
helmet_performance_df = pd.read_csv(os.path.join(results_dir, 'helmet_results', 'helmet_performance.csv'))
longproc_performance_df = pd.read_csv(os.path.join(results_dir, 'longproc_results', 'longproc_performance.csv'))

# Filter out unwanted techniques
unwanted_techniques = ['quest', 'streamingllm_original', 'minference']
helmet_performance_df = helmet_performance_df[~helmet_performance_df['technique'].isin(unwanted_techniques)].copy()
longproc_performance_df = longproc_performance_df[~longproc_performance_df['technique'].isin(unwanted_techniques)].copy()

# Filter out Qwen3 models
unwanted_models = ['Qwen3-8B', 'Yarn-Qwen3-8B']
helmet_performance_df = helmet_performance_df[~helmet_performance_df['model'].isin(unwanted_models)].copy()
longproc_performance_df = longproc_performance_df[~longproc_performance_df['model'].isin(unwanted_models)].copy()

# Filter out specific unwanted cache size configurations
unwanted_configs = [
    ('pyramidkv', 'w32_c4096_k5_avgpool'),
    ('snapkv', 'w32_c4096_k5_avgpool'),
    ('streamingllm', 'n_local_3968_n_init_128'),
    ('duoattn', 'sp0.7_pf32768')
]

for technique, cache_size in unwanted_configs:
    helmet_performance_df = helmet_performance_df[~((helmet_performance_df['technique'] == technique) & (helmet_performance_df['cache_size'] == cache_size))].copy()
    longproc_performance_df = longproc_performance_df[~((longproc_performance_df['technique'] == technique) & (longproc_performance_df['cache_size'] == cache_size))].copy()

# Filter SnapKV and PyramidKV configurations for reasoning models
allowed_reasoning_configs = ['w256_c2048_k7_avgpool', 'w2048_c8192_k7_avgpool', 'w256_c2048_k7_maxpool', 'w2048_c8192_k7_maxpool', 'default']
reasoning_models = ['DeepSeek-R1-Distill-Llama-8B', 'DeepSeek-R1-Distill-Qwen-7B']

helmet_performance_df = helmet_performance_df[~(
    (helmet_performance_df['model'].isin(reasoning_models)) &
    (helmet_performance_df['technique'].isin(['snapkv', 'pyramidkv'])) &
    (~helmet_performance_df['cache_size'].isin(allowed_reasoning_configs))
)].copy()

longproc_performance_df = longproc_performance_df[~(
    (longproc_performance_df['model'].isin(reasoning_models)) &
    (longproc_performance_df['technique'].isin(['snapkv', 'pyramidkv'])) &
    (~longproc_performance_df['cache_size'].isin(allowed_reasoning_configs))
)].copy()

# Filter out w32 and w1024 cache sizes for baseline models
baseline_models = ['Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct']
helmet_performance_df = helmet_performance_df[~(
    (helmet_performance_df['model'].isin(baseline_models)) &
    (helmet_performance_df['technique'].isin(['snapkv', 'pyramidkv'])) &
    (helmet_performance_df['cache_size'].str.contains('w32_|w1024_', na=False))
)].copy()

longproc_performance_df = longproc_performance_df[~(
    (longproc_performance_df['model'].isin(baseline_models)) &
    (longproc_performance_df['technique'].isin(['snapkv', 'pyramidkv'])) &
    (longproc_performance_df['cache_size'].str.contains('w32_|w1024_', na=False))
)].copy()

# Filter StreamingLLM configurations
allowed_streamingllm_configs = ['n_local_4092_n_init_4', 'n_local_4096_n_init_4']
helmet_performance_df = helmet_performance_df[~(
    (helmet_performance_df['technique'] == 'streamingllm') &
    (~helmet_performance_df['cache_size'].isin(allowed_streamingllm_configs))
)].copy()

longproc_performance_df = longproc_performance_df[~(
    (longproc_performance_df['technique'] == 'streamingllm') &
    (~longproc_performance_df['cache_size'].isin(allowed_streamingllm_configs))
)].copy()

# Filter to include only desired models
filtered_models = ['Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct',
                   'DeepSeek-R1-Distill-Llama-8B', 'DeepSeek-R1-Distill-Qwen-7B']

helmet_performance_df = helmet_performance_df[helmet_performance_df['model'].isin(filtered_models)].copy()
longproc_performance_df = longproc_performance_df[longproc_performance_df['model'].isin(filtered_models)].copy()

# Define task groupings for each quadrant
quadrants = {
    'Short Output, Low Dispersion': {
        'tasks': ['niah', 'recall_jsonkv', 'rag_hotpotqa', 'rag_nq'],
        'dataset': 'helmet'
    },
    'Short Output, High Dispersion': {
        'tasks': ['cite_str_em', 'cite_citation_rec', 'cite_citation_prec', 'rerank',
                  'icl_clinic', 'icl_banking', 'summ_multilex'],
        'dataset': 'helmet'
    },
    'Long Output, High Dispersion': {
        'tasks': ['pseudo_to_code', 'html_to_tsv', 'travel_planning'],
        'dataset': 'longproc'
    }
}

# Define techniques and their display labels
techniques_order = ['baseline', 'INT4', 'INT8', 'streamingllm', 'snapkv', 'pyramidkv', 'duoattn']
technique_labels = {
    'baseline': 'Baseline',
    'INT4': 'NF4',
    'INT8': 'Int8',
    'streamingllm': 'StreamingLLM',
    'snapkv': 'SnapKV',
    'pyramidkv': 'PyramidKV',
    'duoattn': 'DuoAttn'
}

# Define color palette
technique_colors = {
    'baseline': '#636EFA',
    'INT4': '#EF553B',
    'INT8': '#00CC96',
    'streamingllm': '#19D3F3',
    'snapkv': '#FFA15A',
    'pyramidkv': '#AB63FA',
    'duoattn': '#FF6692'
}

def calculate_quadrant_average(quadrant_info, helmet_df, longproc_df, technique, models, all_context_lengths=True):
    """
    Calculate average performance and standard error for a technique across all tasks in a quadrant.
    Averages across all models and all context lengths, and all cache configurations.
    For SnapKV/PyramidKV, averages across both cache size configurations.
    Returns: (mean, std_error) tuple
    """
    tasks = quadrant_info['tasks']
    dataset = quadrant_info['dataset']

    if dataset == 'helmet':
        df = helmet_df
        context_lengths = ['16k', '32k'] if all_context_lengths else ['16k']
    else:  # longproc
        df = longproc_df
        context_lengths = ['2k', '5k'] if all_context_lengths else ['2k']

    all_perf_values = []

    # Iterate over all models and context lengths
    for model in models:
        for context in context_lengths:
            # Get all rows for this technique/model/context combination
            technique_rows = df[
                (df['technique'] == technique) &
                (df['model'] == model) &
                (df['context_length'] == context)
            ]

            # For SnapKV/PyramidKV, we want to average across both cache configs
            # For StreamingLLM, we want to average across both configs
            # For other techniques, there's typically only one config
            for _, row in technique_rows.iterrows():
                for task in tasks:
                    val = row[task]
                    if not pd.isna(val) and val != 0:
                        all_perf_values.append(val)

    if len(all_perf_values) == 0:
        return None, None

    mean_val = np.mean(all_perf_values)
    # Calculate standard error of the mean
    std_error = np.std(all_perf_values, ddof=1) / np.sqrt(len(all_perf_values)) if len(all_perf_values) > 1 else 0

    return mean_val, std_error

# Create 2x2 subplot figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Task Performance by Output Length and Dispersion\nAveraged Across All Models and Context Lengths',
             fontsize=18, fontweight='bold', y=0.995)

# Map quadrants to subplot positions
quadrant_positions = {
    'Short Output, Low Dispersion': (1, 0),  # Bottom-left
    'Short Output, High Dispersion': (0, 0),  # Top-left
    'Long Output, High Dispersion': (0, 1),   # Top-right
}

# First pass: collect all data and find global y-axis range
all_quadrant_data = {}
global_max = 0

for quadrant_name, quadrant_info in quadrants.items():
    technique_averages = []
    technique_errors = []
    technique_names = []
    colors = []

    for technique in techniques_order:
        avg_perf, std_error = calculate_quadrant_average(
            quadrant_info,
            helmet_performance_df,
            longproc_performance_df,
            technique,
            filtered_models
        )

        if avg_perf is not None:
            technique_averages.append(avg_perf)
            technique_errors.append(std_error)
            technique_names.append(technique_labels[technique])
            colors.append(technique_colors[technique])
            global_max = max(global_max, avg_perf + std_error)

    all_quadrant_data[quadrant_name] = {
        'averages': technique_averages,
        'errors': technique_errors,
        'names': technique_names,
        'colors': colors
    }

# Set uniform y-axis limit (with 10% headroom)
uniform_ylim = (0, global_max * 1.1)

# Second pass: create plots with uniform y-axis
for quadrant_name, quadrant_info in quadrants.items():
    pos = quadrant_positions[quadrant_name]
    ax = axes[pos]

    data = all_quadrant_data[quadrant_name]
    technique_averages = data['averages']
    technique_errors = data['errors']
    technique_names = data['names']
    colors = data['colors']

    # Create bar plot with error bars
    if len(technique_averages) > 0:
        bars = ax.bar(range(len(technique_averages)), technique_averages,
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1.2,
                      yerr=technique_errors,
                      error_kw={'elinewidth': 2, 'capsize': 5, 'capthick': 2, 'ecolor': 'black'})

        # Add value labels on bars (1.75x BIGGER = 17.5pt)
        for i, (bar, val, err) in enumerate(zip(bars, technique_averages, technique_errors)):
            height = bar.get_height()
            # Position label above error bar
            label_y = height + err + 1
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{val:.1f}',
                   ha='center', va='bottom', fontsize=17, fontweight='bold')

        # Set labels and title - BIGGER subplot titles
        ax.set_title(quadrant_name, fontweight='bold', fontsize=18, pad=10)

        # Set x-axis labels with BIGGER font (1.5x bigger = 16.5pt)
        ax.set_xticks(range(len(technique_names)))
        ax.set_xticklabels(technique_names, rotation=45, ha='right', fontsize=16)

        ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # Set uniform y-axis limits
        ax.set_ylim(uniform_ylim)

# Hide the bottom-right subplot (Long Output, Low Dispersion - empty)
axes[1, 1].axis('off')
axes[1, 1].text(0.5, 0.5, 'No tasks in this category',
                ha='center', va='center', fontsize=14, style='italic',
                transform=axes[1, 1].transAxes)

# Add unified y-axis label for the left column (Short Output subplots)
fig.text(0.02, 0.5, 'Average Performance', va='center', rotation='vertical',
         fontweight='bold', fontsize=16)

plt.tight_layout(rect=[0.03, 0, 1, 0.99])

# Save plot
output_path_png = os.path.join(output_dir, 'quadrant_comparison.png')
output_path_pdf = os.path.join(output_dir, 'quadrant_comparison.pdf')

plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path_png}")

plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path_pdf}")

# Display plot inline (useful for Jupyter notebooks)
try:
    from IPython.display import display, Image as IPImage

    print("\nDisplaying quadrant comparison plot:")
    display(IPImage(filename=output_path_png))
except ImportError:
    # Not in a Jupyter environment, skip display
    print("\nNote: Run in Jupyter notebook to see inline plot preview.")

plt.close()

print("\nQuadrant comparison plot created successfully!")
