#!/usr/bin/env python3
"""
1x1 Grouped Bar Plot: Task Performance by Difficulty
Shows three grouped bars for each technique, representing increasing difficulty:
1. Short Output, Low Dispersion (Easiest)
2. Short Output, High Dispersion (Medium)
3. Long Output, High Dispersion (Hardest)
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
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

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

# Define difficulty levels (ordered from easiest to hardest)
difficulty_levels = {
    'Easy': {
        'label': 'Short Output, Low Dispersion',
        'tasks': ['niah', 'recall_jsonkv', 'rag_hotpotqa', 'rag_nq'],
        'dataset': 'helmet',
        'color': '#90EE90'  # Light green
    },
    'Medium': {
        'label': 'Short Output, High Dispersion',
        'tasks': ['cite_str_em', 'cite_citation_rec', 'cite_citation_prec', 'rerank',
                  'icl_clinic', 'icl_banking', 'summ_multilex'],
        'dataset': 'helmet',
        'color': '#FFD700'  # Gold
    },
    'Hard': {
        'label': 'Long Output, High Dispersion',
        'tasks': ['pseudo_to_code', 'html_to_tsv', 'travel_planning'],
        'dataset': 'longproc',
        'color': '#FF6B6B'  # Coral red
    }
}

# Define techniques and their display labels (ordered: baseline -> NF4 -> Int8 -> SnapKV -> PyramidKV -> StreamingLLM -> DuoAttention)
techniques_order = ['baseline', 'INT4', 'INT8', 'snapkv', 'pyramidkv', 'streamingllm', 'duoattn']
technique_labels = {
    'baseline': 'Baseline',
    'INT4': 'NF4',
    'INT8': 'Int8',
    'snapkv': 'SnapKV',
    'pyramidkv': 'PyramidKV',
    'streamingllm': 'StreamingLLM',
    'duoattn': 'DuoAttention'
}

def calculate_difficulty_average(difficulty_info, helmet_df, longproc_df, technique, models):
    """
    Calculate average performance and standard error for a technique across all tasks in a difficulty level.
    Averages across all models, context lengths, and cache configurations.
    Returns: (mean, std_error) tuple
    """
    tasks = difficulty_info['tasks']
    dataset = difficulty_info['dataset']

    if dataset == 'helmet':
        df = helmet_df
        context_lengths = ['16k', '32k']
    else:  # longproc
        df = longproc_df
        context_lengths = ['2k', '5k']

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

            # Average across all cache configurations
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

# Collect data for all techniques and difficulty levels
print("Calculating performance averages...")
technique_data = {}
technique_errors = {}

# Reorganize data structure: difficulty -> technique -> value
difficulty_data = {}
difficulty_errors = {}

for technique in techniques_order:
    for difficulty_name, difficulty_info in difficulty_levels.items():
        avg_perf, std_error = calculate_difficulty_average(
            difficulty_info,
            helmet_performance_df,
            longproc_performance_df,
            technique,
            filtered_models
        )
        
        # Initialize if needed
        if difficulty_name not in difficulty_data:
            difficulty_data[difficulty_name] = {}
            difficulty_errors[difficulty_name] = {}
        
        difficulty_data[difficulty_name][technique] = avg_perf
        difficulty_errors[difficulty_name][technique] = std_error
        
        # Keep old structure for printing
        if technique not in technique_data:
            technique_data[technique] = {}
            technique_errors[technique] = {}
        technique_data[technique][difficulty_name] = avg_perf
        technique_errors[technique][difficulty_name] = std_error
    
    easy_str = f"{technique_data[technique]['Easy']:.1f}" if technique_data[technique]['Easy'] is not None else 'N/A'
    medium_str = f"{technique_data[technique]['Medium']:.1f}" if technique_data[technique]['Medium'] is not None else 'N/A'
    hard_str = f"{technique_data[technique]['Hard']:.1f}" if technique_data[technique]['Hard'] is not None else 'N/A'
    print(f"  {technique_labels[technique]}: Easy={easy_str}, Medium={medium_str}, Hard={hard_str}")

# Define a function to create the plot with configurable options
def create_plot(show_title=True, show_legend_frame=True, suffix=''):
    """Create a grouped bar plot with configurable title and legend frame."""
    # Adjust figure height based on whether title is shown
    if show_title:
        fig, ax = plt.subplots(figsize=(18, 10))
    else:
        # Scale down the y-dimension when no title is shown
        fig, ax = plt.subplots(figsize=(18, 8))

    # Define bar parameters - NOW GROUP BY DIFFICULTY
    n_techniques = len(techniques_order)
    n_difficulties = len(difficulty_levels)
    bar_width = 0.10  # Smaller since we have 7 techniques per group
    group_gap = 0.5
    group_width = n_techniques * bar_width

    # X positions for difficulty groups (3 groups: Easy, Medium, Hard)
    x_positions = np.arange(n_difficulties) * (group_width + group_gap)

    # Plot bars for each technique within each difficulty group
    difficulty_names = ['Easy', 'Medium', 'Hard']
    bars_by_technique = {}

    # Define colors for techniques (ordered: blue -> purple -> cyan -> green -> orange -> red -> pink)
    technique_colors = {
        'baseline': '#636EFA',      # Blue
        'INT4': '#AB63FA',          # Purple
        'INT8': '#19D3F3',          # Cyan
        'snapkv': '#00CC96',        # Green
        'pyramidkv': '#FFA15A',     # Orange
        'streamingllm': '#EF553B',  # Red
        'duoattn': '#FF6692'        # Pink
    }

    for i, technique in enumerate(techniques_order):
        values = [difficulty_data[diff][technique] if difficulty_data[diff][technique] is not None else 0
                  for diff in difficulty_names]
        errors = [difficulty_errors[diff][technique] if difficulty_errors[diff][technique] is not None else 0
                  for diff in difficulty_names]

        # Calculate x positions for this technique within each difficulty group
        x_offsets = x_positions + i * bar_width

        bars = ax.bar(x_offsets, values,
                      width=bar_width,
                      label=technique_labels[technique],
                      color=technique_colors[technique],
                      alpha=0.85,
                      edgecolor='black',
                      linewidth=1.2,
                      yerr=errors,
                      error_kw={'elinewidth': 2, 'capsize': 3, 'capthick': 2, 'ecolor': 'black'})

        bars_by_technique[technique] = bars

        # Add value labels on bars
        for bar, val, err in zip(bars, values, errors):
            if val > 0:
                height = bar.get_height()
                # Position label above error bar
                label_y = height + err + 1
                # Shift text right slightly to avoid overlap
                label_x = bar.get_x() + bar.get_width()/2.
                ax.text(label_x, label_y,
                       f'{val:.1f}',
                       ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Customize plot
    # ax.set_xlabel('Task Difficulty Level', fontweight='bold', fontsize=24)  # Changed: removed x-axis title
    ax.set_ylabel('Average Performance', fontweight='bold', fontsize=24)
    if show_title:
        ax.set_title('Performance by Task Difficulty Level\nAveraged Across All Models, Context Lengths, and Cache Configs',
                    fontweight='bold', fontsize=28, pad=10)

    # Set x-axis ticks and labels at the center of each difficulty group
    ax.set_xticks(x_positions + (n_techniques - 1) * bar_width / 2)
    ax.set_xticklabels([difficulty_levels[diff]['label'] for diff in difficulty_names],
                    fontsize=24, fontweight='bold')  # Changed: from 20 to 24

    # Make y-axis tick labels bigger
    ax.tick_params(axis='y', labelsize=16)

    # Add legend for techniques - horizontal at the bottom, touching x-axis
    # Increased fontsize from 16 to 18 for legend text
    if show_legend_frame:
        legend = ax.legend(title='Technique', title_fontsize=18, fontsize=18,
                        loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=7,  # Changed: from -0.15 to -0.08
                        framealpha=0.95, edgecolor='black', fancybox=True)
    else:
        legend = ax.legend(title='Technique', title_fontsize=18, fontsize=18,
                        loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=7,
                        frameon=False)
    legend.get_title().set_fontweight('bold')

    # Styling
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Set y-axis limits with some headroom
    max_val = 0
    for diff_name in difficulty_names:
        for tech in techniques_order:
            if difficulty_data[diff_name][tech] is not None:
                max_val = max(max_val, difficulty_data[diff_name][tech])
    ax.set_ylim(0, max_val * 1.15)

    plt.tight_layout(rect=[0, 0.15, 1, 0.99])  # Changed: add bottom padding for legend

# Create the plot with title and legend frame
create_plot(show_title=True, show_legend_frame=True)

# Save plot
output_path_png = os.path.join(output_dir, 'quadrant_comparison_1x1_grouped.png')
output_path_pdf = os.path.join(output_dir, 'quadrant_comparison_1x1_grouped.pdf')

plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {output_path_png}")

plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path_pdf}")

plt.close()

# Create the plot with title and without legend frame
create_plot(show_title=True, show_legend_frame=False)

# Save plot with different filename
output_path_png_title_nobox = os.path.join(output_dir, 'quadrant_comparison_1x1_grouped_title_nobox.png')
output_path_pdf_title_nobox = os.path.join(output_dir, 'quadrant_comparison_1x1_grouped_title_nobox.pdf')

plt.savefig(output_path_png_title_nobox, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {output_path_png_title_nobox}")

plt.savefig(output_path_pdf_title_nobox, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path_pdf_title_nobox}")

plt.close()

# Create the plot without title and without legend frame
create_plot(show_title=False, show_legend_frame=False)

# Save plot with different filename
output_path_png_notitle = os.path.join(output_dir, 'quadrant_comparison_1x1_grouped_notitle.png')
output_path_pdf_notitle = os.path.join(output_dir, 'quadrant_comparison_1x1_grouped_notitle.pdf')

plt.savefig(output_path_png_notitle, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {output_path_png_notitle}")

plt.savefig(output_path_pdf_notitle, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path_pdf_notitle}")

plt.close()

# Display plot inline (useful for Jupyter notebooks)
try:
    from IPython.display import display, Image as IPImage

    print("\nDisplaying 1x1 grouped bar plot (with title):")
    display(IPImage(filename=output_path_png))
    print("\nDisplaying 1x1 grouped bar plot (with title, no legend box):")
    display(IPImage(filename=output_path_png_title_nobox))
    print("\nDisplaying 1x1 grouped bar plot (no title, no legend frame):")
    display(IPImage(filename=output_path_png_notitle))
except ImportError:
    # Not in a Jupyter environment, skip display
    print("\nNote: Run in Jupyter notebook to see inline plot preview.")

print("\n1x1 Grouped bar plot created successfully!")
print("\nPlot now groups by difficulty level (3 groups) with 7 technique bars each.")
