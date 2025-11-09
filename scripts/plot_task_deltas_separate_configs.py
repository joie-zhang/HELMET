#!/usr/bin/env python3
"""
PLOT 1: Task-wise Performance Deltas (Separate Cache Configs)
Shows performance difference (technique - baseline) for each task across efficient inference techniques.
Each SnapKV/PyramidKV cache configuration gets its own row (8 rows total).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Define paths
results_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/results'
output_dir = os.path.join(results_dir, 'plots')
os.makedirs(output_dir, exist_ok=True)

# Load data
helmet_performance_df = pd.read_csv(os.path.join(results_dir, 'helmet_results', 'helmet_performance.csv'))
longproc_performance_df = pd.read_csv(os.path.join(results_dir, 'longproc_results', 'longproc_performance.csv'))

# Define constants
HELMET_CONTEXT_16K = '16k'
HELMET_CONTEXT_32K = '32k'
LONGPROC_CONTEXT_2K = '2k'
LONGPROC_CONTEXT_5K = '5k'

# Models to include
all_models = ['Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct',
              'DeepSeek-R1-Distill-Llama-8B', 'DeepSeek-R1-Distill-Qwen-7B']
duoattn_models = ['Llama-3.1-8B-Instruct']  # Only Llama for DuoAttn

# Task definitions with display names
helmet_tasks = {
    'niah': 'niah',
    'recall_jsonkv': 'recall_jsonkv',
    'rag_hotpotqa': 'rag_hotpotqa',
    'rag_nq': 'rag_nq',
    'cite_str_em': 'cite_str_em',
    'cite_citation_rec': 'cite_rec',
    'cite_citation_prec': 'cite_prec',
    'rerank': 'rerank',
    'icl_clinic': 'icl_clinc',
    'icl_banking': 'icl_banking',
    'summ_multilex': 'multi_lexsum'
}

longproc_tasks = {
    'pseudo_to_code': 'pseudo_to_code',
    'html_to_tsv': 'html_to_tsv',
    'travel_planning': 'travel_planning'
}

# Ordered task list for x-axis
task_order = list(helmet_tasks.values()) + list(longproc_tasks.values())

# Technique configurations
techniques_config = [
    {'name': 'DuoAttention (retrieval head sparsity = 50%)', 'technique': 'duoattn', 'cache_sizes': None, 'models': duoattn_models},
    {'name': 'SnapKV (cache size setting of w=256, c=2048, k=7, maxpool)', 'technique': 'snapkv', 'cache_sizes': ['w256_c2048_k7_maxpool'], 'models': all_models},
    {'name': 'SnapKV (cache size setting of w=2048, c=8192, k=7, maxpool)', 'technique': 'snapkv', 'cache_sizes': ['w2048_c8192_k7_maxpool'], 'models': all_models},
    {'name': 'PyramidKV (cache size setting of w=256, c=2048, k=7, avgpool)', 'technique': 'pyramidkv', 'cache_sizes': ['w256_c2048_k7_avgpool'], 'models': all_models},
    {'name': 'PyramidKV (cache size setting of w=2048, c=8192, k=7, avgpool)', 'technique': 'pyramidkv', 'cache_sizes': ['w2048_c8192_k7_avgpool'], 'models': all_models},
    {'name': 'StreamingLLM (n_local = 4092, n_init = 4)', 'technique': 'streamingllm', 'cache_sizes': ['n_local_4092_n_init_4', 'n_local_4096_n_init_4'], 'models': all_models},
    {'name': 'NF4', 'technique': 'INT4', 'cache_sizes': None, 'models': all_models},
    {'name': 'Int8', 'technique': 'INT8', 'cache_sizes': None, 'models': all_models},
]

def get_performance_values(df, technique, model, cache_sizes, context_length, task_mapping):
    """Get performance values for a technique-model-context combination"""
    values = []

    if cache_sizes is None:
        # No cache size (INT4, INT8, baseline)
        rows = df[
            (df['technique'] == technique) &
            (df['model'] == model) &
            (df['context_length'] == context_length)
        ]
    else:
        # With cache sizes
        rows = df[
            (df['technique'] == technique) &
            (df['model'] == model) &
            (df['context_length'] == context_length) &
            (df['cache_size'].isin(cache_sizes))
        ]

    if rows.empty:
        return {}

    # Extract task values
    task_values = {}
    for original_task, display_task in task_mapping.items():
        if original_task in rows.columns:
            vals = rows[original_task].values
            vals = vals[~pd.isna(vals) & (vals != 0)]
            if len(vals) > 0:
                task_values[display_task] = np.mean(vals)

    return task_values

def compute_deltas_for_technique(tech_config):
    """Compute performance deltas (technique - baseline) for all tasks"""
    technique = tech_config['technique']
    cache_sizes = tech_config['cache_sizes']
    models = tech_config['models']

    # Aggregate across models and contexts
    all_task_deltas = {task: [] for task in task_order}

    for model in models:
        # Get baseline performance
        baseline_helmet_16k = get_performance_values(helmet_performance_df, 'baseline', model, None, HELMET_CONTEXT_16K, helmet_tasks)
        baseline_helmet_32k = get_performance_values(helmet_performance_df, 'baseline', model, None, HELMET_CONTEXT_32K, helmet_tasks)
        baseline_longproc_2k = get_performance_values(longproc_performance_df, 'baseline', model, None, LONGPROC_CONTEXT_2K, longproc_tasks)
        baseline_longproc_5k = get_performance_values(longproc_performance_df, 'baseline', model, None, LONGPROC_CONTEXT_5K, longproc_tasks)

        # Get technique performance
        tech_helmet_16k = get_performance_values(helmet_performance_df, technique, model, cache_sizes, HELMET_CONTEXT_16K, helmet_tasks)
        tech_helmet_32k = get_performance_values(helmet_performance_df, technique, model, cache_sizes, HELMET_CONTEXT_32K, helmet_tasks)
        tech_longproc_2k = get_performance_values(longproc_performance_df, technique, model, cache_sizes, LONGPROC_CONTEXT_2K, longproc_tasks)
        tech_longproc_5k = get_performance_values(longproc_performance_df, technique, model, cache_sizes, LONGPROC_CONTEXT_5K, longproc_tasks)

        # Compute deltas for each task
        for task in task_order:
            deltas = []

            # HELMET 16K
            if task in tech_helmet_16k and task in baseline_helmet_16k:
                deltas.append(tech_helmet_16k[task] - baseline_helmet_16k[task])

            # HELMET 32K
            if task in tech_helmet_32k and task in baseline_helmet_32k:
                deltas.append(tech_helmet_32k[task] - baseline_helmet_32k[task])

            # LongProc 2K
            if task in tech_longproc_2k and task in baseline_longproc_2k:
                deltas.append(tech_longproc_2k[task] - baseline_longproc_2k[task])

            # LongProc 5K
            if task in tech_longproc_5k and task in baseline_longproc_5k:
                deltas.append(tech_longproc_5k[task] - baseline_longproc_5k[task])

            if deltas:
                all_task_deltas[task].extend(deltas)

    # Average deltas across all models and contexts
    averaged_deltas = {}
    for task, deltas in all_task_deltas.items():
        if deltas:
            averaged_deltas[task] = np.mean(deltas)
        else:
            averaged_deltas[task] = np.nan

    return averaged_deltas

# Compute deltas for all techniques
print("Computing performance deltas for all techniques...")
all_deltas = []
technique_names = []

for tech_config in techniques_config:
    print(f"  Processing {tech_config['name']}...")
    deltas = compute_deltas_for_technique(tech_config)
    all_deltas.append([deltas.get(task, np.nan) for task in task_order])
    technique_names.append(tech_config['name'])

# Create the plot
print("Creating plot...")
fig, axes = plt.subplots(8, 1, figsize=(22, 24), sharex=True)
fig.suptitle('Task-wise Performance Delta: Efficient Techniques vs Baseline\nAveraged Across Models and Context Lengths',
             fontsize=28, fontweight='bold', y=0.998)

# Color palette
colors = ['#636EFA', '#EF553B', '#AB63FA', '#00CC96', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']

# Plot each technique as a separate row
x_positions = np.arange(len(task_order)) * 0.7  # Squish columns closer together
for idx, (ax, deltas, tech_name, color) in enumerate(zip(axes, all_deltas, technique_names, colors)):
    bars = ax.bar(x_positions, deltas, width=0.4, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Color bars by positive/negative
    for bar, delta in zip(bars, deltas):
        if not np.isnan(delta):
            if delta < 0:
                bar.set_color('#EF553B')  # Red for negative
            else:
                bar.set_color('#00CC96')  # Green for positive

    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)

    # Labels and styling
    # Only add y-axis label to middle subplot
    if idx == len(axes) // 2:
        ax.set_ylabel('Δ Performance = Efficient Technique Performance - Baseline Performance',
                     fontsize=22, fontweight='bold')
    ax.set_title(tech_name, fontsize=22, fontweight='bold', loc='left', pad=12)

    # White background with horizontal grid lines only (except for Int8)
    ax.set_facecolor('white')
    if tech_name != 'Int8':
        ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5, color='gray')

    # Set y-axis limits to normalized range
    ax.set_ylim(-50, 50)

    # Move x-axis to y=0 and add tick marks there
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.5)

    # Remove top and hide the lines at y=50 and y=-50
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_color('black')
    ax.spines['right'].set_linewidth(1.5)

    # Add tick marks on x-axis at y=0
    ax.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, color='black', bottom=True, top=False)
    ax.tick_params(axis='y', labelsize=16)

    # Adjust x-axis limits to fit squished bars
    ax.set_xlim(-0.5, x_positions[-1] + 0.5)

# Set x-axis labels on bottom plot only
x_positions = np.arange(len(task_order)) * 0.7  # Squish columns closer together
axes[-1].set_xticks(x_positions)
axes[-1].set_xticklabels(task_order, rotation=45, ha='right', fontsize=22, fontweight='bold')
axes[-1].set_xlabel('Tasks from HELMET (16K/32K) and LongProc (500/2K)', fontsize=24, fontweight='bold')
axes[-1].set_xlim(-0.5, x_positions[-1] + 0.5)  # Adjust x-axis limits to fit squished bars

plt.tight_layout(rect=[0, 0.005, 1, 0.995])

# Save outputs
output_path_png = os.path.join(output_dir, 'task_deltas_separate_configs.png')
plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path_png}")

output_path_pdf = os.path.join(output_dir, 'task_deltas_separate_configs.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path_pdf}")

# Display plot inline (useful for Jupyter notebooks)
try:
    from IPython.display import display, Image as IPImage

    print("\nDisplaying task deltas plot (separate configs):")
    display(IPImage(filename=output_path_png))
except ImportError:
    # Not in a Jupyter environment, skip display
    print("\nNote: Run in Jupyter notebook to see inline plot preview.")

plt.close()

print("\nPlot created successfully!")
