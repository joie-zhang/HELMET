import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from tqdm import tqdm

# Load the LongProc data
longproc_memory_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/longproc_results/longproc_memory_usage.csv')
longproc_throughput_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/longproc_results/longproc_throughput.csv')
longproc_performance_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/longproc_results/longproc_performance.csv')

# Filter out the 'quest' technique
longproc_memory_df = longproc_memory_df[longproc_memory_df['technique'] != 'quest']
longproc_throughput_df = longproc_throughput_df[longproc_throughput_df['technique'] != 'quest']
longproc_performance_df = longproc_performance_df[longproc_performance_df['technique'] != 'quest']

# Create output directory for plots
plots_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/results/plots'
os.makedirs(plots_dir, exist_ok=True)

# 1) Seaborn style
sns.set(style='whitegrid')

# 2) Define color palette for models and marker shapes for techniques
model_palette = {
    'Llama-3.1-8B-Instruct': 'tab:orange',
    'Qwen2.5-7B-Instruct':    'tab:blue'
}
marker_dict = {
    'baseline':     'o',
    'INT8':         's',
    'INT4':         '^',
    'minference':   'D',
    'pyramidkv':    'P',
    'snapkv':       'X',
    'streamingllm': '*',
    'streamingllm_original': '8',
}

# Add marker size dictionary to compensate for visual differences
marker_size_dict = {
    'o': 100,  # baseline
    's': 100,  # INT8
    '^': 100,  # INT4
    'D': 100,  # minference
    'P': 120,  # pyramidkv - slightly larger
    'X': 120,  # snapkv - slightly larger
    '*': 200,  # streamingllm - make stars bigger
    '8': 85,   # streamingllm_original - make octagons smaller
}

# --- Unified 3×4 grid for LongProc Memory/Throughput vs Performance ---
perf_tasks = ['html_to_tsv', 'pseudo_to_code', 'travel_planning']
contexts   = ['5k', '2k']

fig, axes = plt.subplots(
    nrows=len(perf_tasks),
    ncols=4,
    figsize=(20, 12),
)

for i, task in tqdm(enumerate(perf_tasks), desc='Processing tasks', total=len(perf_tasks)):
    for j in tqdm(range(4), desc=f'Processing plots for {task}', leave=False):
        ax = axes[i, j]
        if j < 2:
            df = longproc_memory_df
            context = contexts[j]
            x_label = 'Memory (GB)'
        else:
            df = longproc_throughput_df
            context = contexts[j - 2]
            # Calculate latency as 1/throughput with zero handling
            df = df.copy()  # Create a copy to avoid modifying the original
            for task in perf_tasks:
                # Replace 0 with NaN to avoid division by zero
                df[task] = df[task].replace(0, float('nan'))
                # Convert non-zero throughput to latency
                df[task] = 1 / df[task]
            x_label = 'Latency (s/sample)'

        subset = df[df['context_length'] == context]
        for _, row in tqdm(subset.iterrows(), desc=f'Processing datapoints for {task} ({context})', leave=False):
            x = row[task]
            perf_row = longproc_performance_df[
                (longproc_performance_df['technique'] == row['technique']) &
                (longproc_performance_df['context_length'] == context) &
                (longproc_performance_df['model'] == row['model'])
            ]
            if perf_row.empty or pd.isna(x):
                continue
            y = perf_row.iloc[0][task]

            ax.scatter(
                x, y,
                color=model_palette[row['model']],
                marker=marker_dict[row['technique']],
                s=marker_size_dict[marker_dict[row['technique']]],
                edgecolor='k',
                linewidth=0.5,
            )

        if j == 0:
            ax.set_ylabel(task.replace('_', ' ').title(), fontsize=8)
        if i == len(perf_tasks) - 1:
            ax.set_xlabel(x_label, fontsize=8)
        if i == 0:
            col_title = (f"Memory {context.replace('5k', '0.5K').replace('2k', '2K')}" if j < 2 else 
                        f"Latency {context.replace('5k', '0.5K').replace('2k', '2K')}")
            ax.set_title(col_title, fontsize=10)

# --- shared legend across all subplots, placed at bottom ---
model_handles = [
    Line2D([0], [0],
           marker='o', color='w',
           markerfacecolor=color,
           markersize=10,
           label=model)
    for model, color in model_palette.items()
]
tech_handles = [
    Line2D([0], [0],
           marker=marker, color='k',
           linestyle='None',
           markersize=10,
           label=tech)
    for tech, marker in marker_dict.items()
]
all_handles = model_handles + tech_handles
all_labels  = list(model_palette.keys()) + list(marker_dict.keys())

fig.legend(
    all_handles,
    all_labels,
    loc='center',
    ncol=len(all_handles),
    bbox_to_anchor=(0.5, 0.05),
    bbox_transform=fig.transFigure,
    title='Legend',
    fontsize=10,
)

plt.tight_layout(rect=[0, 0.06, 1, 0.98])
fig.savefig(
    os.path.join(plots_dir, 'longproc_overall_plot.png'),
    bbox_inches='tight',
    dpi=300,
)