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
plt.rcParams['font.size'] = 33
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['figure.titlesize'] = 25

# Load data
helmet_memory_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_memory_usage.csv')
helmet_performance_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_performance.csv')

# Filter to 16k context only
helmet_memory_df = helmet_memory_df[helmet_memory_df['context_length'] == '16k']
helmet_performance_df = helmet_performance_df[helmet_performance_df['context_length'] == '16k']

# Remove Qwen models (per requirement #4 - no R1-Distill-Qwen ICL data with token eviction)
excluded_models = ['Qwen2.5-7B-Instruct', 'DeepSeek-R1-Distill-Qwen-7B', 'Qwen3-8B', 'Yarn-Qwen3-8B']
helmet_memory_df = helmet_memory_df[~helmet_memory_df['model'].isin(excluded_models)]
helmet_performance_df = helmet_performance_df[~helmet_performance_df['model'].isin(excluded_models)]

# Remove quantization techniques (per requirement #3)
quantization_techniques = ['INT4', 'INT8']
helmet_memory_df = helmet_memory_df[~helmet_memory_df['technique'].isin(quantization_techniques)]
helmet_performance_df = helmet_performance_df[~helmet_performance_df['technique'].isin(quantization_techniques)]

# Remove unwanted techniques
unwanted_techniques = ['quest', 'streamingllm_original', 'minference']
helmet_memory_df = helmet_memory_df[~helmet_memory_df['technique'].isin(unwanted_techniques)]
helmet_performance_df = helmet_performance_df[~helmet_performance_df['technique'].isin(unwanted_techniques)]

# Filter StreamingLLM to only n_local = 4092 or 4096
streamingllm_mask = (
    (helmet_memory_df['technique'] != 'streamingllm') |
    (helmet_memory_df['cache_size'].str.contains('n_local_4092|n_local_4096', na=False))
)
helmet_memory_df = helmet_memory_df[streamingllm_mask]

streamingllm_mask_perf = (
    (helmet_performance_df['technique'] != 'streamingllm') |
    (helmet_performance_df['cache_size'].str.contains('n_local_4092|n_local_4096', na=False))
)
helmet_performance_df = helmet_performance_df[streamingllm_mask_perf]

# Filter SnapKV and PyramidKV to specific cache sizes:
# w=256, c=2048 and w=2048, c=8192
allowed_snapkv_pyramidkv_configs = ['w256_c2048_k7_avgpool', 'w256_c2048_k7_maxpool',
                                     'w2048_c8192_k7_avgpool', 'w2048_c8192_k7_maxpool']

snapkv_pyramidkv_mask = (
    (~helmet_memory_df['technique'].isin(['snapkv', 'pyramidkv'])) |
    (helmet_memory_df['cache_size'].isin(allowed_snapkv_pyramidkv_configs))
)
helmet_memory_df = helmet_memory_df[snapkv_pyramidkv_mask]

snapkv_pyramidkv_mask_perf = (
    (~helmet_performance_df['technique'].isin(['snapkv', 'pyramidkv'])) |
    (helmet_performance_df['cache_size'].isin(allowed_snapkv_pyramidkv_configs))
)
helmet_performance_df = helmet_performance_df[snapkv_pyramidkv_mask_perf]

# Define color palette for models (only Llama models now)
model_palette = {
    'Llama-3.1-8B-Instruct': '#AB63FA',
    'DeepSeek-R1-Distill-Llama-8B': '#EF553B',
}

# Define marker shapes for techniques
marker_dict = {
    'baseline': 'o',
    'pyramidkv': 'P',
    'snapkv': 'X',
    'streamingllm': '*',
    'duoattn': 'v',
}

# Define display labels for techniques
technique_labels = {
    'baseline': 'Baseline',
    'pyramidkv': 'PyramidKV',
    'snapkv': 'SnapKV',
    'streamingllm': 'StreamingLLM',
    'duoattn': 'DuoAttention',
}

# Marker sizes
marker_size_dict = {
    'o': 270,
    'P': 330,
    'X': 330,
    '*': 450,
    'v': 270,
}

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

# Function to plot memory vs performance
def plot_memory_vs_performance(ax, task_columns, task_name, use_small_cache_only=False):
    """Plot memory usage vs average performance across specified tasks

    Args:
        ax: matplotlib axis
        task_columns: list of task column names to average
        task_name: name of the task for the title
        use_small_cache_only: if True, only plot the smaller cache size for SnapKV/PyramidKV
    """

    # Calculate average performance across the specified task columns
    perf_df = helmet_performance_df.copy()

    # For each row, calculate the average of the specified task columns
    perf_df['avg_performance'] = perf_df[task_columns].mean(axis=1)

    # Plot each technique and model combination
    for (technique, model), group in helmet_memory_df.groupby(['technique', 'model']):
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

                # If use_small_cache_only, filter to only w256
                if use_small_cache_only:
                    group = group[group['cache_size'].str.contains('w256_', na=False)]

        x_values = []
        y_values = []
        labels = []

        # Collect data points - average memory across ICL tasks
        for idx, row in group.iterrows():
            # Average memory usage across icl_clinic and icl_banking
            memory_cols = [col for col in ['icl_clinic', 'icl_banking'] if col in row.index]
            if not memory_cols:
                continue

            x = row[memory_cols].mean()

            # Find corresponding performance value
            perf_row = perf_df[
                (perf_df['technique'] == row['technique']) &
                (perf_df['context_length'] == row['context_length']) &
                (perf_df['model'] == row['model']) &
                (perf_df['cache_size'] == row['cache_size'])
            ]

            if perf_row.empty or pd.isna(x):
                continue

            y = perf_row.iloc[0]['avg_performance']

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

                # Add label to the last point (only for SnapKV)
                if labels[-1] and technique == 'snapkv':
                    ax.annotate(
                        labels[-1],
                        (x_values[-1], y_values[-1]),
                        xytext=(8, 8),
                        textcoords='offset points',
                        fontsize=15,
                        alpha=0.7,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='none', alpha=0.6)
                    )

    # Styling
    ax.set_xlabel('Memory Usage (GB)', fontweight='bold')
    ax.set_ylabel('Performance', fontweight='bold')
    # No plot title - using figure title only
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Set reasonable axis limits
    if len(ax.get_lines()) > 0 or len(ax.collections) > 0:
        ax.margins(0.1)

# Function to create legend elements
def create_legend_elements():
    """Create legend elements for models and techniques"""
    model_elements = [
        Line2D([0], [0], marker='o', color='w',
              markerfacecolor=model_palette['Llama-3.1-8B-Instruct'], markersize=13,
              markeredgewidth=0,
              label='Llama-8B'),
        Line2D([0], [0], marker='o', color='w',
              markerfacecolor=model_palette['DeepSeek-R1-Distill-Llama-8B'], markersize=13,
              markeredgewidth=0,
              label='R1-Distill-Llama')
    ]

    technique_elements = []
    for tech, marker in marker_dict.items():
        # Use larger marker size for StreamingLLM
        marker_size = 15 if tech == 'streamingllm' else 13
        technique_elements.append(Line2D([0], [0], marker=marker, color='gray',
                                        linestyle='None', markersize=marker_size,
                                        markeredgewidth=0,
                                        label=technique_labels[tech]))

    return model_elements, technique_elements

# Create output directory
output_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/results/plots'
os.makedirs(output_dir, exist_ok=True)

# ============ VERSION 1: ICL TASKS AVERAGE ============
print("Creating Version 1: ICL tasks average (icl_clinic + icl_banking)...")
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
# fig.suptitle('Reasoning Models Show Superior Performance-Memory Trade-offs on In-Context Learning Tasks\n(16K Context, Averaged Across Banking77 and Clinc150)',
            #  fontsize=25, fontweight='bold', y=0.98)

plot_memory_vs_performance(ax, ['icl_clinic', 'icl_banking'], 'ICL Tasks Average')

# Create vertical legend on the right
model_elements, technique_elements = create_legend_elements()

# Add section headers
model_header = Line2D([0], [0], marker='none', color='none', label='Models:', markersize=0)
tech_header = Line2D([0], [0], marker='none', color='none', label='', markersize=0)  # Empty line as separator
tech_label = Line2D([0], [0], marker='none', color='none', label='Techniques:', markersize=0)

all_elements = [model_header] + model_elements + [tech_header, tech_label] + technique_elements

ax.legend(handles=all_elements,
         loc='center left',
         bbox_to_anchor=(1.02, 0.35),
         frameon=False,
         fancybox=False,
         shadow=False,
         fontsize=16.5)

plt.tight_layout(rect=[0, 0, 1, 0.96], pad=0.5)
output_path = os.path.join(output_dir, 'icl_average_memory_only.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  PNG saved to: {output_path}")

output_path_pdf = os.path.join(output_dir, 'icl_average_memory_only.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"  PDF saved to: {output_path_pdf}")
plt.close()

# ============ VERSION 2: ICL + CITE + RERANK + SUMM AVERAGE ============
print("\nCreating Version 2: Averaged across ICL, Cite, Rerank, and Summ...")
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
# fig.suptitle('Reasoning Models Show Superior Performance-Memory Trade-offs Across Multiple Task Types\n(16K Context, Averaged Across ICL, Citation, Rerank, and Summarization Tasks)',
            #  fontsize=25, fontweight='bold', y=0.98)

# For this version, we use cite_str_em for citation, rerank for rerank, summ_multilex for summarization
plot_memory_vs_performance(ax, ['icl_clinic', 'icl_banking', 'cite_str_em', 'rerank', 'summ_multilex'],
                          'Multi-Task Average')

# Create vertical legend on the right
model_elements, technique_elements = create_legend_elements()

# Add section headers
model_header = Line2D([0], [0], marker='none', color='none', label='Models:', markersize=0)
tech_header = Line2D([0], [0], marker='none', color='none', label='', markersize=0)  # Empty line as separator
tech_label = Line2D([0], [0], marker='none', color='none', label='Techniques:', markersize=0)

all_elements = [model_header] + model_elements + [tech_header, tech_label] + technique_elements

ax.legend(handles=all_elements,
         loc='center left',
         bbox_to_anchor=(1.02, 0.35),
         frameon=False,
         fancybox=False,
         shadow=False,
         fontsize=16.5)

plt.tight_layout(rect=[0, 0, 1, 0.96], pad=0.5)
output_path = os.path.join(output_dir, 'multitask_average_memory_only.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  PNG saved to: {output_path}")

output_path_pdf = os.path.join(output_dir, 'multitask_average_memory_only.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"  PDF saved to: {output_path_pdf}")
plt.close()

# ============ VERSION 3: ICL TASKS AVERAGE (SMALL CACHE ONLY) ============
print("\nCreating Version 3: ICL tasks average with small cache only...")
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
# fig.suptitle('Reasoning Models Show Superior Performance-Memory Trade-offs on In-Context Learning Tasks\n(16K Context, Averaged Across Banking77 and Clinc150, Small Cache Size Only)',
            #  fontsize=25, fontweight='bold', y=0.98)

plot_memory_vs_performance(ax, ['icl_clinic', 'icl_banking'], 'ICL Tasks Average', use_small_cache_only=True)

# Create vertical legend on the right
model_elements, technique_elements = create_legend_elements()

# Add section headers
model_header = Line2D([0], [0], marker='none', color='none', label='Models:', markersize=0)
tech_header = Line2D([0], [0], marker='none', color='none', label='', markersize=0)  # Empty line as separator
tech_label = Line2D([0], [0], marker='none', color='none', label='Techniques:', markersize=0)

all_elements = [model_header] + model_elements + [tech_header, tech_label] + technique_elements

ax.legend(handles=all_elements,
         loc='center left',
         bbox_to_anchor=(1.02, 0.35),
         frameon=False,
         fancybox=False,
         shadow=False,
         fontsize=16.5)

plt.tight_layout(rect=[0, 0, 1, 0.96], pad=0.5)
output_path = os.path.join(output_dir, 'icl_average_memory_only_small_cache.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  PNG saved to: {output_path}")

output_path_pdf = os.path.join(output_dir, 'icl_average_memory_only_small_cache.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"  PDF saved to: {output_path_pdf}")
plt.close()

# ============ VERSION 4: MULTITASK AVERAGE (SMALL CACHE ONLY) ============
print("\nCreating Version 4: Multi-task average with small cache only...")
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
# fig.suptitle('Reasoning Models Show Superior Performance-Memory Trade-offs Across Multiple Task Types\n(16K Context, Averaged Across ICL, Citation, Rerank, and Summarization Tasks, Small Cache Size Only)',
            #  fontsize=25, fontweight='bold', y=0.98)

plot_memory_vs_performance(ax, ['icl_clinic', 'icl_banking', 'cite_str_em', 'rerank', 'summ_multilex'],
                          'Multi-Task Average', use_small_cache_only=True)

# Create vertical legend on the right
model_elements, technique_elements = create_legend_elements()

# Add section headers
model_header = Line2D([0], [0], marker='none', color='none', label='Models:', markersize=0)
tech_header = Line2D([0], [0], marker='none', color='none', label='', markersize=0)  # Empty line as separator
tech_label = Line2D([0], [0], marker='none', color='none', label='Techniques:', markersize=0)

all_elements = [model_header] + model_elements + [tech_header, tech_label] + technique_elements

ax.legend(handles=all_elements,
         loc='center left',
         bbox_to_anchor=(1.02, 0.35),
         frameon=False,
         fancybox=False,
         shadow=False,
         fontsize=16.5)

plt.tight_layout(rect=[0, 0, 1, 0.96], pad=0.5)
output_path = os.path.join(output_dir, 'multitask_average_memory_only_small_cache.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  PNG saved to: {output_path}")

output_path_pdf = os.path.join(output_dir, 'multitask_average_memory_only_small_cache.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"  PDF saved to: {output_path_pdf}")
plt.close()

# ============ VERSION 5: ICL AVERAGE ACROSS 16K AND 32K ============
print("\nCreating Version 5: ICL average across 16K and 32K contexts...")

# Load fresh data for this version
helmet_memory_df_all = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_memory_usage.csv')
helmet_performance_df_all = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_performance.csv')

# Filter to 16k and 32k contexts
helmet_memory_df_all = helmet_memory_df_all[helmet_memory_df_all['context_length'].isin(['16k', '32k'])]
helmet_performance_df_all = helmet_performance_df_all[helmet_performance_df_all['context_length'].isin(['16k', '32k'])]

# Remove Qwen models
excluded_models = ['Qwen2.5-7B-Instruct', 'DeepSeek-R1-Distill-Qwen-7B', 'Qwen3-8B', 'Yarn-Qwen3-8B']
helmet_memory_df_all = helmet_memory_df_all[~helmet_memory_df_all['model'].isin(excluded_models)]
helmet_performance_df_all = helmet_performance_df_all[~helmet_performance_df_all['model'].isin(excluded_models)]

# Remove quantization and unwanted techniques
quantization_techniques = ['INT4', 'INT8']
unwanted_techniques = ['quest', 'streamingllm_original', 'minference']

# Check if duoattn has 32K ICL data, if not exclude it
duoattn_32k_data = helmet_performance_df_all[
    (helmet_performance_df_all['technique'] == 'duoattn') &
    (helmet_performance_df_all['context_length'] == '32k') &
    (helmet_performance_df_all['icl_clinic'].notna())
]
include_duoattn = len(duoattn_32k_data) > 0

if not include_duoattn:
    print("  Note: DuoAttention 32K ICL data not found, excluding from 16K+32K average plot")
    unwanted_techniques.append('duoattn')

helmet_memory_df_all = helmet_memory_df_all[~helmet_memory_df_all['technique'].isin(quantization_techniques + unwanted_techniques)]
helmet_performance_df_all = helmet_performance_df_all[~helmet_performance_df_all['technique'].isin(quantization_techniques + unwanted_techniques)]

# Filter StreamingLLM, SnapKV, PyramidKV
streamingllm_mask = (
    (helmet_memory_df_all['technique'] != 'streamingllm') |
    (helmet_memory_df_all['cache_size'].str.contains('n_local_4092|n_local_4096', na=False))
)
helmet_memory_df_all = helmet_memory_df_all[streamingllm_mask]

streamingllm_mask_perf = (
    (helmet_performance_df_all['technique'] != 'streamingllm') |
    (helmet_performance_df_all['cache_size'].str.contains('n_local_4092|n_local_4096', na=False))
)
helmet_performance_df_all = helmet_performance_df_all[streamingllm_mask_perf]

allowed_snapkv_pyramidkv_configs = ['w256_c2048_k7_avgpool', 'w256_c2048_k7_maxpool',
                                     'w2048_c8192_k7_avgpool', 'w2048_c8192_k7_maxpool']

snapkv_pyramidkv_mask = (
    (~helmet_memory_df_all['technique'].isin(['snapkv', 'pyramidkv'])) |
    (helmet_memory_df_all['cache_size'].isin(allowed_snapkv_pyramidkv_configs))
)
helmet_memory_df_all = helmet_memory_df_all[snapkv_pyramidkv_mask]

snapkv_pyramidkv_mask_perf = (
    (~helmet_performance_df_all['technique'].isin(['snapkv', 'pyramidkv'])) |
    (helmet_performance_df_all['cache_size'].isin(allowed_snapkv_pyramidkv_configs))
)
helmet_performance_df_all = helmet_performance_df_all[snapkv_pyramidkv_mask_perf]

# Now average across context lengths (16k and 32k)
# Group by technique, model, cache_size and average the metrics
memory_grouped = helmet_memory_df_all.groupby(['technique', 'model', 'cache_size']).agg({
    'icl_clinic': 'mean',
    'icl_banking': 'mean'
}).reset_index()

perf_grouped = helmet_performance_df_all.groupby(['technique', 'model', 'cache_size']).agg({
    'icl_clinic': 'mean',
    'icl_banking': 'mean'
}).reset_index()

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
# fig.suptitle('Reasoning Models Show Superior Performance-Memory Trade-offs on In-Context Learning Tasks\n(Averaged Across 16K and 32K Contexts, Banking77 and Clinc150)',
            #  fontsize=25, fontweight='bold', y=0.98)

# Plot using the aggregated data
for (technique, model), group in memory_grouped.groupby(['technique', 'model']):
    if model not in model_palette or technique not in marker_dict:
        continue

    # Sort by cache size if applicable
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

    for idx, row in group.iterrows():
        # Average memory across icl tasks
        x = pd.Series([row['icl_clinic'], row['icl_banking']]).mean()

        # Find corresponding performance
        perf_row = perf_grouped[
            (perf_grouped['technique'] == row['technique']) &
            (perf_grouped['model'] == row['model']) &
            (perf_grouped['cache_size'] == row['cache_size'])
        ]

        if perf_row.empty or pd.isna(x):
            continue

        y = pd.Series([perf_row.iloc[0]['icl_clinic'], perf_row.iloc[0]['icl_banking']]).mean()

        if pd.isna(y) or y == 0:
            continue

        x_values.append(x)
        y_values.append(y)
        labels.append(format_cache_size(row['cache_size']))

    # Plot points
    if len(x_values) > 0:
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

            # Add label to the last point (only for SnapKV)
            if labels[-1] and technique == 'snapkv':
                ax.annotate(
                    labels[-1],
                    (x_values[-1], y_values[-1]),
                    xytext=(8, 8),
                    textcoords='offset points',
                    fontsize=15,
                    alpha=0.7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='none', alpha=0.6)
                )

# Styling
ax.set_xlabel('Memory Usage (GB)', fontweight='bold')
ax.set_ylabel('Performance', fontweight='bold')
# No plot title - using figure title only
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)
ax.margins(0.1)

# Create vertical legend on the right
model_elements = [
    Line2D([0], [0], marker='o', color='w',
          markerfacecolor=model_palette['Llama-3.1-8B-Instruct'], markersize=20,
          markeredgewidth=0,
          label='Llama-8B'),
    Line2D([0], [0], marker='o', color='w',
          markerfacecolor=model_palette['DeepSeek-R1-Distill-Llama-8B'], markersize=20,
          markeredgewidth=0,
          label='R1-Distill-Llama')
]

# Dynamically build technique legend based on what's included
if include_duoattn:
    technique_dict_filtered = marker_dict
    technique_labels_filtered = technique_labels
else:
    technique_dict_filtered = {k: v for k, v in marker_dict.items() if k != 'duoattn'}
    technique_labels_filtered = {k: v for k, v in technique_labels.items() if k != 'duoattn'}

technique_elements = []
for tech, marker in technique_dict_filtered.items():
    technique_elements.append(Line2D([0], [0], marker=marker, color='gray',
                                    linestyle='None', markersize=20,
                                    markeredgewidth=0,
                                    label=technique_labels_filtered[tech]))

model_header = Line2D([0], [0], marker='none', color='none', label='Models:', markersize=0)
tech_header = Line2D([0], [0], marker='none', color='none', label='', markersize=0)
tech_label = Line2D([0], [0], marker='none', color='none', label='Techniques:', markersize=0)

all_elements = [model_header] + model_elements + [tech_header, tech_label] + technique_elements

ax.legend(handles=all_elements,
         loc='center left',
         bbox_to_anchor=(1.02, 0.35),
         frameon=False,
         fancybox=False,
         shadow=False,
         fontsize=16.5)

plt.tight_layout(rect=[0, 0, 1, 0.96], pad=0.5)
output_path = os.path.join(output_dir, 'icl_average_16k_32k_combined.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  PNG saved to: {output_path}")

output_path_pdf = os.path.join(output_dir, 'icl_average_16k_32k_combined.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"  PDF saved to: {output_path_pdf}")
plt.close()

print("\nAll plots created successfully!")

# Display plots inline (useful for Jupyter notebooks)
try:
    from IPython.display import display, Image as IPImage

    print("\nDisplaying plots:")
    print("\n=== Version 1: ICL Tasks Average (Both Cache Sizes) ===")
    display(IPImage(filename=os.path.join(output_dir, 'icl_average_memory_only.png')))

    print("\n=== Version 2: Multi-Task Average (Both Cache Sizes) ===")
    display(IPImage(filename=os.path.join(output_dir, 'multitask_average_memory_only.png')))

    print("\n=== Version 3: ICL Tasks Average (Small Cache Only) ===")
    display(IPImage(filename=os.path.join(output_dir, 'icl_average_memory_only_small_cache.png')))

    print("\n=== Version 4: Multi-Task Average (Small Cache Only) ===")
    display(IPImage(filename=os.path.join(output_dir, 'multitask_average_memory_only_small_cache.png')))

    print("\n=== Version 5: ICL Average Across 16K and 32K ===")
    display(IPImage(filename=os.path.join(output_dir, 'icl_average_16k_32k_combined.png')))
except ImportError:
    # Not in a Jupyter environment, skip display
    print("\nNote: Run in Jupyter notebook to see inline plot previews.")
