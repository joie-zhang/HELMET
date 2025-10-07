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

# Filter out the 'quest' and 'streamingllm_original' techniques
longproc_memory_df = longproc_memory_df[~longproc_memory_df['technique'].isin(['quest', 'streamingllm_original'])]
longproc_throughput_df = longproc_throughput_df[~longproc_throughput_df['technique'].isin(['quest', 'streamingllm_original'])]
longproc_performance_df = longproc_performance_df[~longproc_performance_df['technique'].isin(['quest', 'streamingllm_original'])]

# Filter out specific unwanted cache size configurations
unwanted_configs = [
    ('pyramidkv', 'w32_c4096_k5_avgpool'),
    ('snapkv', 'w32_c4096_k5_avgpool'), 
    ('streamingllm', 'n_local_3968_n_init_128'),
    ('duoattn', 'sp0.7_pf32768')
]

# Apply filters to all dataframes
for technique, cache_size in unwanted_configs:
    condition = (longproc_memory_df['technique'] == technique) & (longproc_memory_df['cache_size'] == cache_size)
    longproc_memory_df = longproc_memory_df[~condition]
    
    condition = (longproc_throughput_df['technique'] == technique) & (longproc_throughput_df['cache_size'] == cache_size)
    longproc_throughput_df = longproc_throughput_df[~condition]
    
    condition = (longproc_performance_df['technique'] == technique) & (longproc_performance_df['cache_size'] == cache_size)
    longproc_performance_df = longproc_performance_df[~condition]

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
    elif cache_size.startswith("w") and "_c" in cache_size and "_k" in cache_size:
        # For SnapKV and PyramidKV with format w{window}_c{cache}_k{k}_{pool}
        parts = cache_size.split('_')
        window_val = parts[0][1:]  # Remove 'w' prefix
        cache_val = parts[1][1:]  # Remove 'c' prefix
        k_val = parts[2][1:]      # Remove 'k' prefix
        pool_type = parts[3]      # maxpool or avgpool
        return f"w={window_val}, c={cache_val}, k={k_val}, {pool_type}"
    else:
        # For other techniques, format as "cache=X"
        return f"cache={cache_size.replace('cache_', '')}"

# 2) Define color palette for models and marker shapes for techniques
model_palette = {
    'Llama-3.1-8B-Instruct': 'tab:orange',
    'Qwen2.5-7B-Instruct':    'tab:blue',
    'DeepSeek-R1-Distill-Llama-8B': 'tab:red',
    'DeepSeek-R1-Distill-Qwen-7B': 'tab:green',
    'Qwen3-8B': 'tab:purple',
    'Yarn-Qwen3-8B': 'tab:brown',
    
}
marker_dict = {
    'baseline':     'o',
    'INT8':         's',
    'INT4':         '^',
    'minference':   'D',
    'pyramidkv':    'P',
    'snapkv':       'X',
    'streamingllm': '*',
    'duoattn': 'v',
    # 'streamingllm_original': '8',
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
    'v': 100,
    # '8': 85,   # streamingllm_original - make octagons smaller
}

# --- Unified 3×4 grid for LongProc Memory/Throughput vs Performance ---
perf_tasks = ['html_to_tsv', 'pseudo_to_code', 'travel_planning']
contexts = ['5k', '2k', '8k']  # Updated to include 8k

fig, axes = plt.subplots(
    nrows=len(perf_tasks) + 1,  # Changed from 3 to 4 to add average row
    ncols=6,  # Changed from 4 to 6 to accommodate 8k
    figsize=(30, 16),  # Increased figure height for extra row
)

for i, task in tqdm(enumerate(perf_tasks + ['average']), desc='Processing tasks', total=len(perf_tasks) + 1):  # Added 'average'
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
            for task_col in perf_tasks:
                # Replace 0 with NaN to avoid division by zero
                df[task_col] = df[task_col].replace(0.0, float('nan'))
                # Convert non-zero throughput to latency
                df[task_col] = 1 / df[task_col]
            x_label = 'Latency (s/token)'

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
                elif technique in ["snapkv", "pyramidkv"] and group['cache_size'].iloc[0].startswith('w'):
                    # For SnapKV/PyramidKV with format w{window}_c{cache}_k{k}_{pool}
                    # Sort by cache size first, then by k value
                    def extract_cache_and_k(cache_size):
                        parts = cache_size.split('_')
                        window_val = int(parts[0][1:])  # Remove 'w' prefix and convert to int
                        cache_val = int(parts[1][1:])  # Remove 'c' prefix and convert to int
                        k_val = int(parts[2][1:])      # Remove 'k' prefix and convert to int
                        return (window_val, cache_val, k_val)
                    
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
            
            for _, row in group.iterrows():
                x = row[task] if task != 'average' else row[perf_tasks].mean()  # Use average for bottom row
                perf_row = longproc_performance_df[
                    (longproc_performance_df['technique'] == row['technique']) &
                    (longproc_performance_df['context_length'] == context) &
                    (longproc_performance_df['model'] == row['model']) &
                    (longproc_performance_df['cache_size'] == row['cache_size'])
                ]
                if perf_row.empty or pd.isna(x):
                    continue
                
                # Calculate y value (performance)
                if task == 'average':
                    # For average row, calculate mean performance across all tasks
                    y = perf_row.iloc[0][perf_tasks].mean()
                else:
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
            if task == 'average':
                ax.set_ylabel('Average Performance', fontsize=8)
            else:
                ax.set_ylabel(task.replace('_', ' ').title(), fontsize=8)
        if i == len(perf_tasks):  # Changed condition for bottom row
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
    if tech in ["snapkv", "pyramidkv"]:
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
plt.show()

# Directory for the split plots
split_plots_dir = os.path.join(plots_dir, 'split_plots_longproc')
os.makedirs(split_plots_dir, exist_ok=True)

# Define splits: (row_start, row_end, col_start, col_end)
splits = [
    (0, 2, 0, 3),  # top-left 2x3
    (0, 2, 3, 6),  # top-right 2x3
    (2, 4, 0, 3),  # bottom-left 2x3
    (2, 4, 3, 6),  # bottom-right 2x3
]

for idx, (row_start, row_end, col_start, col_end) in enumerate(splits):
    # Create figure with extra space at bottom for legend
    sub_fig, sub_axes = plt.subplots(
        nrows=row_end - row_start,
        ncols=col_end - col_start,
        figsize=(15, 10)  # Increased height to accommodate legend
    )
    
    # Copy over plots from the original axes
    for i, row in enumerate(range(row_start, row_end)):
        for j, col in enumerate(range(col_start, col_end)):
            orig_ax = axes[row, col]
            sub_ax = sub_axes[i, j]
            
            # Transfer content: lines, scatter, annotations, etc.
            for line in orig_ax.get_lines():
                sub_ax.plot(
                    line.get_xdata(),
                    line.get_ydata(),
                    color=line.get_color(),
                    linestyle=line.get_linestyle(),
                    marker=line.get_marker(),
                    alpha=line.get_alpha(),
                    linewidth=line.get_linewidth(),
                )
            
            for coll in orig_ax.collections:
                sub_ax.scatter(
                    *coll.get_offsets().T,
                    marker=coll.get_paths()[0],
                    color=coll.get_facecolor(),
                    edgecolors=coll.get_edgecolor(),
                    linewidths=coll.get_linewidths(),
                    s=coll.get_sizes(),
                )
            
            # Annotations
            for annotation in orig_ax.texts:
                sub_ax.annotate(
                    annotation.get_text(),
                    annotation.get_position(),
                    xytext=annotation.get_position(),
                    textcoords='data',
                    fontsize=annotation.get_fontsize(),
                    alpha=annotation.get_alpha()
                )
            
            # Handle axis labels and titles based on which split plot we're in
            if idx == 0:  # Top-left plot (Memory plots for first two tasks)
                # Keep y-axis labels for leftmost plots
                if j == 0:
                    sub_ax.set_ylabel(orig_ax.get_ylabel(), fontsize=8)
                else:
                    sub_ax.set_ylabel('')
                # Keep x-axis labels for bottom row
                if i == 1:
                    sub_ax.set_xlabel(orig_ax.get_xlabel(), fontsize=8)
                else:
                    sub_ax.set_xlabel('')
                # Keep all titles
                sub_ax.set_title(orig_ax.get_title(), fontsize=10)
                
            elif idx == 1:  # Top-right plot (Latency plots for first two tasks)
                # Keep y-axis labels for leftmost plots
                if j == 0:
                    # Get the task name from the original plot's ylabel
                    task_name = orig_ax.get_ylabel()
                    sub_ax.set_ylabel(task_name, fontsize=8)
                else:
                    sub_ax.set_ylabel('')
                # Keep x-axis labels for bottom row
                if i == 1:
                    sub_ax.set_xlabel(orig_ax.get_xlabel(), fontsize=8)
                else:
                    sub_ax.set_xlabel('')
                # Keep all titles
                sub_ax.set_title(orig_ax.get_title(), fontsize=10)
                
            elif idx == 2:  # Bottom-left plot (Memory plots for last two tasks)
                # Keep y-axis labels for leftmost plots
                if j == 0:
                    sub_ax.set_ylabel(orig_ax.get_ylabel(), fontsize=8)
                else:
                    sub_ax.set_ylabel('')
                # Remove x-axis labels
                sub_ax.set_xlabel('')
                # Move titles to top
                if i == 0:
                    sub_ax.set_title(orig_ax.get_title(), fontsize=10)
                else:
                    sub_ax.set_title('')
                
            elif idx == 3:  # Bottom-right plot (Latency plots for last two tasks)
                # Keep y-axis labels for leftmost plots
                if j == 0:
                    # Get the task name from the original plot's ylabel
                    task_name = orig_ax.get_ylabel()
                    sub_ax.set_ylabel(task_name, fontsize=8)
                else:
                    sub_ax.set_ylabel('')
                # Remove x-axis labels
                sub_ax.set_xlabel('')
                # Move titles to top
                if i == 0:
                    sub_ax.set_title(orig_ax.get_title(), fontsize=10)
                else:
                    sub_ax.set_title('')
            
            # Keep grid settings and limits
            sub_ax.grid(True, linestyle='--', alpha=0.7)
            sub_ax.set_xlim(orig_ax.get_xlim())
            sub_ax.set_ylim(orig_ax.get_ylim())
    
    # Add column titles for bottom plots (idx 2 and 3)
    if idx in [2, 3]:
        # Get the column titles from the original plot
        col_titles = []
        for col in range(col_start, col_end):
            col_titles.append(axes[0, col].get_title())
        
        # Add column titles at the top of the figure, closer to the plots
        for j, title in enumerate(col_titles):
            sub_fig.text(
                (j + 0.5) / (col_end - col_start),  # x position
                0.91,  # y position (moved closer to plots, was 0.95)
                title,
                ha='center',
                va='center',
                fontsize=10
            )
    
    # Add legend to each split plot
    # Create legend handles and labels
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
    
    # Add technique handles
    for tech, marker in marker_dict.items():
        if tech in ["snapkv", "pyramidkv"]:
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
    
    # Add legend to the figure
    sub_fig.legend(
        legend_handles,
        legend_labels,
        loc='center',
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, 0.02),  # Position at bottom
        bbox_transform=sub_fig.transFigure,
        title='Legend',
        fontsize=10,
    )
    
    sub_fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    
    sub_fig.savefig(os.path.join(split_plots_dir, f'longproc_split_{idx+1}.png'), 
                    bbox_inches='tight',
                    dpi=300)
    plt.close(sub_fig)
