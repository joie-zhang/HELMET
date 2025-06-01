import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib.lines import Line2D
from tqdm import tqdm
import numpy as np

# Load the HELMET data
helmet_memory_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_memory_usage.csv')
helmet_throughput_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_throughput.csv')
helmet_performance_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_performance.csv')

# Filter out the 'quest' and 'streamingllm_original' techniques
helmet_memory_df = helmet_memory_df[~helmet_memory_df['technique'].isin(['quest', 'streamingllm_original'])]
helmet_throughput_df = helmet_throughput_df[~helmet_throughput_df['technique'].isin(['quest', 'streamingllm_original'])]
helmet_performance_df = helmet_performance_df[~helmet_performance_df['technique'].isin(['quest', 'streamingllm_original'])]

# Filter out specific unwanted cache size configurations
unwanted_configs = [
    ('pyramidkv', 'w32_c4096_k5_avgpool'),
    ('snapkv', 'w32_c4096_k5_avgpool'), 
    ('streamingllm', 'n_local_3968_n_init_128')
]

# Apply filters to all dataframes
for technique, cache_size in unwanted_configs:
    condition = (helmet_memory_df['technique'] == technique) & (helmet_memory_df['cache_size'] == cache_size)
    helmet_memory_df = helmet_memory_df[~condition]
    
    condition = (helmet_throughput_df['technique'] == technique) & (helmet_throughput_df['cache_size'] == cache_size)
    helmet_throughput_df = helmet_throughput_df[~condition]
    
    condition = (helmet_performance_df['technique'] == technique) & (helmet_performance_df['cache_size'] == cache_size)
    helmet_performance_df = helmet_performance_df[~condition]

# Create output directory for plots
plots_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/results/plots'
os.makedirs(plots_dir, exist_ok=True)

# 1) Seaborn style
sns.set(style='whitegrid')

# 2) Define the 8 performance tasks (including the three cite metrics) and the two contexts
perf_tasks = [
    'recall_jsonkv', 'rag_nq', 'rag_hotpotqa', 'rerank',
    'cite_str_em', 'cite_citation_rec', 'cite_citation_prec', 'niah'
]
contexts = ['16k', '32k']

# 3) Color palette for models, marker shapes for techniques
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
#     'streamingllm_original': '8',
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
#     '8': 85,  # streamingllm_original - make octagons smaller
}

# 4) Create a 9×4 grid (changed from 8×4 to add average row)
fig, axes = plt.subplots(
    nrows=len(perf_tasks) + 1,  # Add 1 for the average row
    ncols=4,
    figsize=(20, 33),  # Increase height slightly to accommodate new row
)

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
    elif cache_size.startswith("w32_c") and "_k" in cache_size:
        # For SnapKV and PyramidKV with format w32_c{cache}_k{k}_{pool}
        parts = cache_size.split('_')
        cache_val = parts[1][1:]  # Remove 'c' prefix
        k_val = parts[2][1:]      # Remove 'k' prefix
        pool_type = parts[3]      # maxpool or avgpool
        return f"c={cache_val}, k={k_val}, {pool_type}"
    else:
        # For other techniques, format as "cache=X"
        return f"cache={cache_size.replace('cache_', '')}"

# 5) Populate each subplot for individual tasks
for i, perf_task in tqdm(enumerate(perf_tasks), desc='Processing tasks', total=len(perf_tasks)):
    mem_thr_col = 'cite' if perf_task.startswith('cite_') else perf_task

    for j in range(4):
        ax = axes[i, j]
        if j < 2:
            df = helmet_memory_df
            context = contexts[j]
            x_label = 'Memory (GB)'
        else:
            df = helmet_throughput_df
            context = contexts[j - 2]
            df = df.copy()
            for task in ['recall_jsonkv', 'rag_nq', 'rag_hotpotqa', 'rerank', 'cite', 'niah']:
                df[task] = df[task].replace(0, float('nan'))
                df[task] = 1 / df[task]
            x_label = 'Latency (s/sample)'

        # Group data by technique and model
        subset = df[df['context_length'] == context]
        for (technique, model), group in subset.groupby(['technique', 'model']):
            # Sort by cache size if available
            if 'cache_size' in group.columns:
                # Custom sorting for different techniques
                if technique == "streamingllm":
                    # Extract n_local for sorting
                    group['sort_key'] = group['cache_size'].apply(
                        lambda x: int(x.split('_')[2]) if x.startswith('n_local_') else 0
                    )
                    group = group.sort_values('sort_key')
                    group = group.drop('sort_key', axis=1)
                elif technique in ["snapkv", "pyramidkv"] and group['cache_size'].iloc[0].startswith('w32_c'):
                    # For SnapKV/PyramidKV with format w32_c{cache}_k{k}_{pool}
                    # Sort by cache size first, then by k value
                    def extract_cache_and_k(cache_size):
                        parts = cache_size.split('_')
                        cache_val = int(parts[1][1:])  # Remove 'c' prefix and convert to int
                        k_val = int(parts[2][1:])      # Remove 'k' prefix and convert to int
                        return (cache_val, k_val)
                    
                    group['sort_key'] = group['cache_size'].apply(extract_cache_and_k)
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
            cache_sizes = []  # Store cache sizes for annotation
            
            for _, row in group.iterrows():
                x = row[mem_thr_col]
                perf_row = helmet_performance_df[
                    (helmet_performance_df['technique'] == row['technique']) &
                    (helmet_performance_df['context_length'] == context) &
                    (helmet_performance_df['model'] == row['model']) &
                    (helmet_performance_df['cache_size'] == row['cache_size'])
                ]
                if perf_row.empty or pd.isna(x):
                    continue
                y = perf_row.iloc[0][perf_task]
                
                x_values.append(x)
                y_values.append(y)
                cache_sizes.append(row['cache_size'])
                
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

        # only label the leftmost column with the task name
        if j == 0:
            ax.set_ylabel(perf_task.replace('_', ' ').title(), fontsize=8)
        # Update condition for bottom row x-axis label (now it's len(perf_tasks) instead of len(perf_tasks) - 1)
        if i == len(perf_tasks):  # This will now be the average row
            ax.set_xlabel(x_label, fontsize=8)
        # only title the top row with the column header
        if i == 0:
            col_title = (f"Memory {context.replace('k', 'K')}" if j < 2 else 
                        f"Latency {context.replace('k', 'K')}")
            ax.set_title(col_title, fontsize=10)

# 5.5) Add the average row (new section)
print("Computing and plotting averages...")
avg_row_idx = len(perf_tasks)  # This is index 8 (the 9th row)

for j in range(4):
    ax = axes[avg_row_idx, j]
    if j < 2:
        df = helmet_memory_df
        context = contexts[j]
        x_label = 'Memory (GB)'
    else:
        df = helmet_throughput_df
        context = contexts[j - 2]
        df = df.copy()
        for task in ['recall_jsonkv', 'rag_nq', 'rag_hotpotqa', 'rerank', 'cite', 'niah']:
            df[task] = df[task].replace(0, float('nan'))
            df[task] = 1 / df[task]
        x_label = 'Latency (s/sample)'

    # Group data by technique and model
    subset = df[df['context_length'] == context]
    for (technique, model), group in subset.groupby(['technique', 'model']):
        # Apply same sorting logic as above
        if 'cache_size' in group.columns:
            # Custom sorting for different techniques (same as above)
            if technique == "streamingllm":
                group['sort_key'] = group['cache_size'].apply(
                    lambda x: int(x.split('_')[2]) if x.startswith('n_local_') else 0
                )
                group = group.sort_values('sort_key')
                group = group.drop('sort_key', axis=1)
            elif technique in ["snapkv", "pyramidkv"] and group['cache_size'].iloc[0].startswith('w32_c'):
                def extract_cache_and_k(cache_size):
                    parts = cache_size.split('_')
                    cache_val = int(parts[1][1:])
                    k_val = int(parts[2][1:])
                    return (cache_val, k_val)
                
                group['sort_key'] = group['cache_size'].apply(extract_cache_and_k)
                group = group.sort_values('sort_key')
                group = group.drop('sort_key', axis=1)
            else:
                group['sort_key'] = group['cache_size'].apply(
                    lambda x: int(x.replace('cache_', '')) if x.startswith('cache_') else 0
                )
                group = group.sort_values('sort_key')
                group = group.drop('sort_key', axis=1)
        
        x_values = []
        y_values = []
        
        for _, row in group.iterrows():
            x = row['cite' if any(task.startswith('cite_') for task in perf_tasks) else 'recall_jsonkv']  # Use appropriate column
            
            # Find corresponding performance row
            perf_row = helmet_performance_df[
                (helmet_performance_df['technique'] == row['technique']) &
                (helmet_performance_df['context_length'] == context) &
                (helmet_performance_df['model'] == row['model']) &
                (helmet_performance_df['cache_size'] == row['cache_size'])
            ]
            if perf_row.empty or pd.isna(x):
                continue
            
            # Calculate average across all performance tasks
            perf_values = []
            for task in perf_tasks:
                task_val = perf_row.iloc[0][task]
                if not pd.isna(task_val):
                    perf_values.append(task_val)
            
            if len(perf_values) > 0:
                y = np.mean(perf_values)
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
                    cache_label = format_cache_size(row['cache_size'])
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

    # Label the average row
    if j == 0:
        ax.set_ylabel('Average Score', fontsize=8)
    # Label x-axis for the bottom row
    ax.set_xlabel(x_label, fontsize=8)

# 6) Build a shared legend to the right
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

# Adjust the bottom margin in tight_layout
plt.tight_layout(rect=[0, 0.06, 1, 0.98])

print("Saving plot...")
# Save with a higher DPI to ensure legend is clear
fig.savefig(os.path.join(plots_dir, 'helmet_overall_plot.png'), 
            bbox_inches='tight',  # Added to ensure legend is included
            dpi=300)  # Increased DPI for better quality
print("Done!")