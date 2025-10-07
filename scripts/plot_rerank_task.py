import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
import os

# Set style for professional appearance
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16

# Load data
helmet_memory_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_memory_usage.csv')
helmet_throughput_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_throughput.csv')
helmet_performance_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_performance.csv')

# Filter out unwanted techniques and configurations
unwanted_techniques = ['quest', 'streamingllm_original', 'minference']
helmet_memory_df = helmet_memory_df[~helmet_memory_df['technique'].isin(unwanted_techniques)]
helmet_throughput_df = helmet_throughput_df[~helmet_throughput_df['technique'].isin(unwanted_techniques)]
helmet_performance_df = helmet_performance_df[~helmet_performance_df['technique'].isin(unwanted_techniques)]

# Filter out specific unwanted cache size configurations
unwanted_configs = [
    ('pyramidkv', 'w32_c4096_k5_avgpool'),
    ('snapkv', 'w32_c4096_k5_avgpool'),
    ('streamingllm', 'n_local_3968_n_init_128'),
    ('duoattn', 'sp0.7_pf32768')
]

for technique, cache_size in unwanted_configs:
    condition = (helmet_memory_df['technique'] == technique) & (helmet_memory_df['cache_size'] == cache_size)
    helmet_memory_df = helmet_memory_df[~condition]

    condition = (helmet_throughput_df['technique'] == technique) & (helmet_throughput_df['cache_size'] == cache_size)
    helmet_throughput_df = helmet_throughput_df[~condition]

    condition = (helmet_performance_df['technique'] == technique) & (helmet_performance_df['cache_size'] == cache_size)
    helmet_performance_df = helmet_performance_df[~condition]

# Filter SnapKV and PyramidKV to keep only specific configurations for reasoning models
allowed_reasoning_configs = ['w256_c2048_k7_avgpool', 'w2048_c8192_k7_avgpool', 'w256_c2048_k7_maxpool', 'w2048_c8192_k7_maxpool', 'default']
reasoning_models = ['DeepSeek-R1-Distill-Llama-8B', 'Yarn-Qwen3-8B']

for df in [helmet_memory_df, helmet_throughput_df, helmet_performance_df]:
    condition = (
        (df['model'].isin(reasoning_models)) &
        (df['technique'].isin(['snapkv', 'pyramidkv'])) &
        (~df['cache_size'].isin(allowed_reasoning_configs))
    )
    df.drop(df[condition].index, inplace=True)

# Filter out w32 and w1024 cache sizes for SnapKV/PyramidKV on baseline models
baseline_models = ['Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct']
for df in [helmet_memory_df, helmet_throughput_df, helmet_performance_df]:
    condition = (
        (df['model'].isin(baseline_models)) &
        (df['technique'].isin(['snapkv', 'pyramidkv'])) &
        (df['cache_size'].str.contains('w32_|w1024_', na=False))
    )
    df.drop(df[condition].index, inplace=True)

# Define color palette for models
model_palette = {
    'Llama-3.1-8B-Instruct': '#AB63FA',
    'Qwen2.5-7B-Instruct': '#00CC96',
    'DeepSeek-R1-Distill-Llama-8B': '#EF553B',
    'DeepSeek-R1-Distill-Qwen-7B': '#FFA15A',
    'Qwen3-8B': '#19D3F3',
    'Yarn-Qwen3-8B': '#FF97FF',
}

# Define marker shapes for techniques (minference excluded)
marker_dict = {
    'baseline': 'o',
    'INT8': 's',
    'INT4': '^',
    'pyramidkv': 'P',
    'snapkv': 'X',
    'streamingllm': '*',
    'duoattn': 'v',
}

# Define display labels for techniques
technique_labels = {
    'baseline': 'Baseline',
    'INT8': 'Int8',
    'INT4': 'NF4',
    'pyramidkv': 'PyramidKV',
    'snapkv': 'SnapKV',
    'streamingllm': 'StreamingLLM',
    'duoattn': 'DuoAttn',
}

# Marker sizes
marker_size_dict = {
    'o': 270,
    's': 270,
    '^': 270,
    'D': 270,
    'P': 330,
    'X': 330,
    '*': 450,
    'v': 270,
}

# Create figure with single plot
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
fig.suptitle('Token Eviction Methods Struggle on the Pareto Frontier of Performance vs Efficiency\nHELMET Rerank Task: Memory vs NDCG@10 (16K Context)',
             fontsize=16, fontweight='bold', y=0.98)

contexts = ['16k']

# Helper function to format cache size for display
def format_cache_size(cache_size: str) -> str:
    if cache_size == "default":
        return ""
    elif cache_size.startswith("n_local_"):
        parts = cache_size.split('_')
        n_local = parts[2]
        n_init = parts[4]
        return f"L={n_local},I={n_init}"
    elif cache_size.startswith("w") and "_c" in cache_size and "_k" in cache_size:
        parts = cache_size.split('_')
        window_val = parts[0][1:]
        cache_val = parts[1][1:]
        return f"W={window_val},C={cache_val}"
    elif cache_size.startswith("sp"):
        parts = cache_size.split('_')
        sparsity = parts[0].replace('sp', '')
        return f"S={sparsity}"
    else:
        return cache_size

# Process single plot - back to original memory vs performance style
context = '16k'
df = helmet_memory_df
x_label = 'Memory Usage (GB)'
x_col = 'rerank'

# Filter by context - now including DeepSeek-R1-Distill-Llama-8B and Yarn-Qwen3-8B
subset = df[df['context_length'] == context].copy()
# Exclude only DeepSeek-R1-Distill-Qwen-7B and Qwen3-8B
excluded_models = ['DeepSeek-R1-Distill-Qwen-7B', 'Qwen3-8B']
subset = subset[~subset['model'].isin(excluded_models)]

# Plot each technique and model combination
for (technique, model), group in subset.groupby(['technique', 'model']):
    if model not in model_palette or technique not in marker_dict:
        continue

    # Sort by cache size if applicable
    if 'cache_size' in group.columns:
        if technique == "streamingllm":
            group['sort_key'] = group['cache_size'].apply(
                lambda x: int(x.split('_')[2]) if x.startswith('n_local_') else 0
            )
            group = group.sort_values('sort_key').drop('sort_key', axis=1)
        elif technique in ["snapkv", "pyramidkv"]:
            def extract_cache_and_k(cache_size):
                if not cache_size.startswith('w'):
                    return (0, 0, 0)
                parts = cache_size.split('_')
                cache_val = int(parts[1][1:])
                k_val = int(parts[2][1:])
                return (cache_val, k_val, 0)

            group['sort_key'] = group['cache_size'].apply(extract_cache_and_k)
            group = group.sort_values('sort_key').drop('sort_key', axis=1)

    x_values = []
    y_values = []
    labels = []

    # Collect data points
    for idx, row in group.iterrows():
        x = row[x_col]

        # Find corresponding performance value
        perf_row = helmet_performance_df[
            (helmet_performance_df['technique'] == row['technique']) &
            (helmet_performance_df['context_length'] == context) &
            (helmet_performance_df['model'] == row['model']) &
            (helmet_performance_df['cache_size'] == row['cache_size'])
        ]

        if perf_row.empty or pd.isna(x):
            continue

        y = perf_row.iloc[0]['rerank']

        if pd.isna(y) or y == 0:
            continue

        x_values.append(x)
        y_values.append(y)
        labels.append(format_cache_size(row['cache_size']))

    # Plot points
    if len(x_values) > 0:
        # Scatter plot
        ax.scatter(
            x_values, y_values,
            color=model_palette[model],
            marker=marker_dict[technique],
            s=marker_size_dict[marker_dict[technique]],
            alpha=0.8,
            zorder=3
        )

        # Connect points with lines if multiple cache sizes
        if len(x_values) > 1:
            ax.plot(
                x_values, y_values,
                color=model_palette[model],
                linestyle='--',
                alpha=0.4,
                linewidth=1.5,
                zorder=2
            )

            # Add label to the last point
            if labels[-1]:
                ax.annotate(
                    labels[-1],
                    (x_values[-1], y_values[-1]),
                    xytext=(8, 8),
                    textcoords='offset points',
                    fontsize=10,
                    alpha=0.7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='none', alpha=0.6)
                )

# Styling
ax.set_xlabel(x_label, fontweight='bold', fontsize=13)
ax.set_ylabel('NDCG@10 Score', fontweight='bold', fontsize=13)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Set reasonable axis limits
if len(ax.get_lines()) > 0 or len(ax.collections) > 0:
    ax.margins(0.1)

# Create comprehensive legend with both models and techniques
legend_elements = []

# Models to include in legend
excluded_legend_models = ['DeepSeek-R1-Distill-Qwen-7B', 'Qwen3-8B']
included_models = {k: v for k, v in model_palette.items() if k not in excluded_legend_models}

# Add model legend entries
legend_elements.append(Line2D([0], [0], marker='none', color='none',
                             label='Models:', markersize=0))
for model, color in included_models.items():
    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=color, markersize=10,
                                 markeredgewidth=0,
                                 label=model))

# Add separator
legend_elements.append(Line2D([0], [0], marker='none', color='none',
                             label='', markersize=0))

# Add technique legend entries
legend_elements.append(Line2D([0], [0], marker='none', color='none',
                             label='Techniques:', markersize=0))
for tech, marker in marker_dict.items():
    legend_elements.append(Line2D([0], [0], marker=marker, color='gray',
                                 linestyle='None', markersize=10,
                                 markeredgewidth=0,
                                 label=technique_labels[tech]))

# Position legend on the right side (vertical layout)
ax.legend(handles=legend_elements,
         loc='center left',
         bbox_to_anchor=(1.05, 0.5),
         frameon=True,
         fancybox=False,
         shadow=False,
         fontsize=11)

# Adjust layout
plt.tight_layout()

# Create output directory
output_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/results/plots'
os.makedirs(output_dir, exist_ok=True)

# Save figure
output_path = os.path.join(output_dir, 'rerank_task_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Plot saved to: {output_path}")

# Also save as PDF for publication quality
output_path_pdf = os.path.join(output_dir, 'rerank_task_analysis.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"PDF saved to: {output_path_pdf}")

plt.close()

# Display plot inline (useful for Jupyter notebooks)
try:
    from IPython.display import display, Image as IPImage

    print("\nDisplaying plot:")
    display(IPImage(filename=output_path))
except ImportError:
    # Not in a Jupyter environment, skip display
    print("\nNote: Run in Jupyter notebook to see inline plot preview.")
