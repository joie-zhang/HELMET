#!/usr/bin/env python3
"""
PLOT 2: Task-wise Performance Deltas (Averaged Cache Configs)
Shows performance difference (technique - baseline) for each task across efficient inference techniques.
SnapKV/PyramidKV cache configurations are averaged (6 rows total).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Set font to Arial
plt.rcParams['font.family'] = 'Arial'

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

# Technique configurations - averaging both cache configs for SnapKV/PyramidKV
# Reordered: NF4, Int8, SnapKV, PyramidKV, StreamingLLM, DuoAttention
techniques_config = [
    {'name': 'NF4', 'technique': 'INT4', 'cache_sizes': None, 'models': all_models},
    {'name': 'Int8', 'technique': 'INT8', 'cache_sizes': None, 'models': all_models},
    {'name': 'SnapKV', 'technique': 'snapkv', 'cache_sizes': ['w256_c2048_k7_maxpool', 'w2048_c8192_k7_maxpool'], 'models': all_models},
    {'name': 'PyramidKV', 'technique': 'pyramidkv', 'cache_sizes': ['w256_c2048_k7_avgpool', 'w2048_c8192_k7_avgpool'], 'models': all_models},
    {'name': 'StreamingLLM', 'technique': 'streamingllm', 'cache_sizes': ['n_local_4092_n_init_4', 'n_local_4096_n_init_4'], 'models': all_models},
    {'name': 'DuoAttention', 'technique': 'duoattn', 'cache_sizes': None, 'models': duoattn_models},
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
fig, axes = plt.subplots(6, 1, figsize=(16, 14), sharex=True)

# Color palette
colors = ['#636EFA', '#EF553B', '#AB63FA', '#00CC96', '#19D3F3', '#FF6692']

# Grouped task labels
grouped_task_labels = ['NIAH', 'Recall', 'RAG', 'ICL', 'Cite', 'Re-rank', 'Summ', 'Pseudo', 'HTML', 'Travel']

# Map original tasks to grouped labels
task_grouping = {
    'niah': 'NIAH',
    'recall_jsonkv': 'Recall',
    'rag_hotpotqa': 'RAG',
    'rag_nq': 'RAG',
    'icl_clinc': 'ICL',
    'icl_banking': 'ICL',
    'cite_str_em': 'Cite',
    'cite_rec': 'Cite',
    'cite_prec': 'Cite',
    'rerank': 'Re-rank',
    'multi_lexsum': 'Summ',
    'pseudo_to_code': 'Pseudo',
    'html_to_tsv': 'HTML',
    'travel_planning': 'Travel'
}

# Compute grouped deltas
grouped_deltas = []
for tech_deltas in all_deltas:
    grouped_tech_deltas = {}
    for task, delta in zip(task_order, tech_deltas):
        group = task_grouping.get(task, task)
        if group not in grouped_tech_deltas:
            grouped_tech_deltas[group] = []
        if not np.isnan(delta):
            grouped_tech_deltas[group].append(delta)
    
    # Average deltas within each group
    averaged_grouped = []
    for label in grouped_task_labels:
        if label in grouped_tech_deltas and grouped_tech_deltas[label]:
            averaged_grouped.append(np.mean(grouped_tech_deltas[label]))
        else:
            averaged_grouped.append(np.nan)
    grouped_deltas.append(averaged_grouped)

# Plot each technique as a separate row
x_positions = np.arange(len(grouped_task_labels)) * 0.62
for idx, (ax, deltas, tech_name, color) in enumerate(zip(axes, grouped_deltas, technique_names, colors)):
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
    # Position title at y=40, slightly nested on the top gray line
    ax.text(0.02, 40, tech_name, fontsize=28,
            transform=ax.get_yaxis_transform(), verticalalignment='center',
            fontweight='bold')

    # White background with no grid lines
    ax.set_facecolor('white')
    ax.grid(False)

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
axes[-1].set_xticks(x_positions)
axes[-1].set_xticklabels(grouped_task_labels, rotation=45, ha='right', fontsize=28)
axes[-1].set_xlim(-0.5, x_positions[-1] + 0.5)

# Add shared y-axis label to middle subplot (SnapKV, index 2) with minimal padding
# This will span properly and be flush with the y-axis
axes[2].set_ylabel('Performance Degradation vs Baseline',
                   fontsize=36, fontweight='bold', labelpad=2)

plt.tight_layout(h_pad=0.5, rect=[0, 0.005, 1, 0.995])

# Save outputs
output_path_png = os.path.join(output_dir, 'task_deltas_averaged_configs.png')
plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path_png}")

output_path_pdf = os.path.join(output_dir, 'task_deltas_averaged_configs.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path_pdf}")

# Display plot inline (useful for Jupyter notebooks)
try:
    from IPython.display import display, Image as IPImage

    print("\nDisplaying task deltas plot (averaged configs):")
    display(IPImage(filename=output_path_png))
except ImportError:
    # Not in a Jupyter environment, skip display
    print("\nNote: Run in Jupyter notebook to see inline plot preview.")

plt.close()

print("\nPlot created successfully!")
