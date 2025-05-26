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

# Add helper function to format cache size for display
def format_cache_size(cache_size: str) -> str:
    if cache_size == "default":
        return "default"
    elif cache_size.startswith("n_local_"):
        # For streamingllm, format as "n_local=X, n_init=Y"
        parts = cache_size.split('_')
        n_local = parts[2]
        n_init = parts[5]
        return f"n_local={n_local}, n_init={n_init}"
    else:
        # For other techniques, format as "cache=X"
        return f"cache={cache_size.replace('cache_', '')}"

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
contexts = ['5k', '2k', '8k']  # Reordered to 0.5K, 2K, 8K

fig, axes = plt.subplots(
    nrows=len(perf_tasks),
    ncols=6,  # Changed from 4 to 6 to accommodate 8k
    figsize=(30, 12),  # Increased figure width
)

for i, task in tqdm(enumerate(perf_tasks), desc='Processing tasks', total=len(perf_tasks)):
    for j in tqdm(range(6), desc=f'Processing plots for {task}', leave=False):  # Changed from 4 to 6
        ax = axes[i, j]
        if j < 3:  # Changed from 2 to 3
            df = longproc_memory_df
            context = contexts[j]
            x_label = 'Memory (GB)'
        else:
            df = longproc_throughput_df
            context = contexts[j - 3]  # Changed from j - 2 to j - 3
            # Calculate latency as 1/throughput with zero handling
            df = df.copy()  # Create a copy to avoid modifying the original
            for task in perf_tasks:
                # Replace 0 with NaN to avoid division by zero
                df[task] = df[task].replace(0, float('nan'))
                # Convert non-zero throughput to latency
                df[task] = 1 / df[task]
            x_label = 'Latency (s/sample)'

        # Group data by technique and model
        subset = df[df['context_length'] == context]
        for (technique, model), group in subset.groupby(['technique', 'model']):
            # Sort by cache size if available
            if 'cache_size' in group.columns:
                # Custom sorting for streamingllm
                if technique == "streamingllm":
                    # Extract n_local for sorting
                    group['sort_key'] = group['cache_size'].apply(
                        lambda x: int(x.split('_')[2]) if x.startswith('n_local_') else 0
                    )
                    group = group.sort_values('sort_key')
                    group = group.drop('sort_key', axis=1)
                else:
                    # For other techniques, sort by cache size number
                    group['sort_key'] = group['cache_size'].apply(
                        lambda x: int(x.replace('cache_', '')) if x.startswith('cache_') else 0
                    )
                    group = group.sort_values('sort_key')
                    group = group.drop('sort_key', axis=1)
            
            x_values = []
            y_values = []
            
            for _, row in group.iterrows():
                x = row[task]
                perf_row = longproc_performance_df[
                    (longproc_performance_df['technique'] == row['technique']) &
                    (longproc_performance_df['context_length'] == context) &
                    (longproc_performance_df['model'] == row['model']) &
                    (longproc_performance_df['cache_size'] == row['cache_size'])
                ]
                if perf_row.empty or pd.isna(x):
                    continue
                y = perf_row.iloc[0][task]
                
                x_values.append(x)
                y_values.append(y)
                
                # Plot individual points
                ax.scatter(
                    x, y,
                    color=model_palette[model],
                    marker=marker_dict[technique],
                    s=marker_size_dict[marker_dict[technique]],
                    edgecolor='k',
                    linewidth=0.5,
                )
                
                # Add cache size annotation for the last point in each group
                if len(x_values) > 1 and row.name == group.index[-1]:
                    # Format the cache size for display
                    cache_label = format_cache_size(row['cache_size'])
                    # Add annotation with offset
                    ax.annotate(
                        cache_label,
                        (x, y),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=6,
                        alpha=0.7
                    )
            
            # Connect points with lines if there are multiple cache sizes
            if len(x_values) > 1:
                ax.plot(
                    x_values, y_values,
                    color=model_palette[model],
                    linestyle='--',
                    alpha=0.5,
                    linewidth=1
                )

        if j == 0:
            ax.set_ylabel(task.replace('_', ' ').title(), fontsize=8)
        if i == len(perf_tasks) - 1:
            ax.set_xlabel(x_label, fontsize=8)
        if i == 0:
            col_title = (f"Memory {context.replace('5k', '0.5K').replace('2k', '2K').replace('8k', '8K')}" if j < 3 else 
                        f"Latency {context.replace('5k', '0.5K').replace('2k', '2K').replace('8k', '8K')}")
            ax.set_title(col_title, fontsize=10)

# --- shared legend across all subplots, placed at bottom ---
legend_handles = []
legend_labels = []

# Add model handles
for model, color in model_palette.items():
    legend_handles.append(Line2D([0], [0],
                                marker='o', color='w',
                                markerfacecolor=color,
                                markersize=10,
                                label=model))
    legend_labels.append(model)

# Add technique handles with cache size info
for tech, marker in marker_dict.items():
    if tech in ["snapkv", "pyramidkv", "streamingllm"]:
        # Add a line to show connection between cache sizes
        legend_handles.append(Line2D([0], [0],
                                   marker=marker, color='k',
                                   linestyle='--',
                                   markersize=10,
                                   label=f"{tech} (connected cache sizes)"))
        legend_labels.append(f"{tech} (connected cache sizes)")
    else:
        legend_handles.append(Line2D([0], [0],
                                   marker=marker, color='k',
                                   linestyle='None',
                                   markersize=10,
                                   label=tech))
        legend_labels.append(tech)

fig.legend(
    legend_handles,
    legend_labels,
    loc='center',
    ncol=len(legend_handles),
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