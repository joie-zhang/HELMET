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

# Filter out SnapKV and PyramidKV results for R1-Distill-Llama-8B except specific configurations
allowed_r1_configs = ['w256_c2048_k7_avgpool', 'w2048_c8192_k7_avgpool', 'w256_c2048_k7_maxpool', 'w2048_c8192_k7_maxpool', 'default']
for df in [helmet_memory_df, helmet_throughput_df, helmet_performance_df]:
    condition = (
        (df['model'] == 'DeepSeek-R1-Distill-Llama-8B') &
        (df['technique'].isin(['snapkv', 'pyramidkv'])) &
        (~df['cache_size'].isin(allowed_r1_configs))
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

# Function to plot a single metric
def plot_metric(ax, metric_type, context, task='icl_banking'):
    """Plot a single metric (memory or throughput/latency) vs performance"""
    # Select appropriate dataframe
    if metric_type == 'memory':
        df = helmet_memory_df
        x_label = 'Memory Usage (GB)'
        x_col = task
    else:
        df = helmet_throughput_df.copy()
        # Convert throughput to latency (tokens/second -> seconds/token)
        df[task] = df[task].replace(0, float('nan'))
        df[task] = 1 / df[task]
        x_label = 'Latency (s/token)'
        x_col = task

    # Filter by context
    subset = df[df['context_length'] == context]

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

            y = perf_row.iloc[0][task]

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
    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel('Exact Match Score', fontweight='bold')
    title_metric = 'Latency' if metric_type == 'throughput' else metric_type.capitalize()
    ax.set_title(f'Performance vs {title_metric}',
                fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Set reasonable axis limits
    if len(ax.get_lines()) > 0 or len(ax.collections) > 0:
        ax.margins(0.1)

# Function to create legend elements
def create_legend_elements():
    """Create legend elements for models and techniques"""
    # Models to exclude from legend
    excluded_models = ['DeepSeek-R1-Distill-Qwen-7B', 'Qwen3-8B', 'Yarn-Qwen3-8B']

    model_elements = []
    for model, color in model_palette.items():
        if model not in excluded_models:
            model_elements.append(Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=color, markersize=10,
                                         markeredgewidth=0,
                                         label=model))

    technique_elements = []
    for tech, marker in marker_dict.items():
        technique_elements.append(Line2D([0], [0], marker=marker, color='gray',
                                         linestyle='None', markersize=10,
                                         markeredgewidth=0,
                                         label=technique_labels[tech]))

    return model_elements, technique_elements

# Create output directory
output_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/results/plots'
os.makedirs(output_dir, exist_ok=True)

# Version 1: Combined plot with both memory and latency subplots
print("Creating Version 1: Combined plot...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Reasoning Models Recover Performance Degradation Compared to Non-Reasoning Models on In-Context Learning Tasks\nHELMET In-Context Learning Task: Banking77 (16K Context)',
             fontsize=18, fontweight='bold', y=0.995)

for j, metric_type in enumerate(['memory', 'throughput']):
    plot_metric(axes[j], metric_type, '16k')

# Create legend in single row
model_elements, technique_elements = create_legend_elements()
all_elements = model_elements + technique_elements

fig.legend(handles=all_elements,
          loc='lower center',
          ncol=len(all_elements),  # All items in one row
          bbox_to_anchor=(0.5, -0.02),
          frameon=True,
          fancybox=True,
          shadow=False,
          fontsize=11)

plt.tight_layout(rect=[0, 0.06, 1, 0.96])
output_path = os.path.join(output_dir, 'icl_banking_task_analysis_combined.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  PNG saved to: {output_path}")

output_path_pdf = os.path.join(output_dir, 'icl_banking_task_analysis_combined.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"  PDF saved to: {output_path_pdf}")
plt.close()

# Version 2: Standalone memory plot
print("Creating Version 2: Memory plot...")
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
fig.suptitle('Reasoning Models Recover Performance Degradation Compared to Non-Reasoning Models on In-Context Learning Tasks\nHELMET In-Context Learning Task: Banking77 (16K Context)',
             fontsize=18, fontweight='bold', y=0.995)

plot_metric(ax, 'memory', '16k')

# Create vertical legend on the right
model_elements, technique_elements = create_legend_elements()

# Add section headers
model_header = Line2D([0], [0], marker='none', color='none', label='Models:', markersize=0)
tech_header = Line2D([0], [0], marker='none', color='none', label='', markersize=0)  # Empty line as separator
tech_label = Line2D([0], [0], marker='none', color='none', label='Techniques:', markersize=0)

all_elements = [model_header] + model_elements + [tech_header, tech_label] + technique_elements

ax.legend(handles=all_elements,
         loc='center left',
         bbox_to_anchor=(1.05, 0.5),
         frameon=True,
         fancybox=False,
         shadow=False,
         fontsize=11)

plt.tight_layout()
output_path = os.path.join(output_dir, 'icl_banking_task_analysis_memory.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  PNG saved to: {output_path}")

output_path_pdf = os.path.join(output_dir, 'icl_banking_task_analysis_memory.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"  PDF saved to: {output_path_pdf}")
plt.close()

# Version 3: Standalone latency plot
print("Creating Version 3: Latency plot...")
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
fig.suptitle('Reasoning Models Recover Performance Degradation Compared to Non-Reasoning Models on In-Context Learning Tasks\nHELMET In-Context Learning Task: Banking77 (16K Context)',
             fontsize=18, fontweight='bold', y=0.995)

plot_metric(ax, 'throughput', '16k')

# Create vertical legend on the right
model_elements, technique_elements = create_legend_elements()

# Add section headers
model_header = Line2D([0], [0], marker='none', color='none', label='Models:', markersize=0)
tech_header = Line2D([0], [0], marker='none', color='none', label='', markersize=0)  # Empty line as separator
tech_label = Line2D([0], [0], marker='none', color='none', label='Techniques:', markersize=0)

all_elements = [model_header] + model_elements + [tech_header, tech_label] + technique_elements

ax.legend(handles=all_elements,
         loc='center left',
         bbox_to_anchor=(1.05, 0.5),
         frameon=True,
         fancybox=False,
         shadow=False,
         fontsize=11)

plt.tight_layout()
output_path = os.path.join(output_dir, 'icl_banking_task_analysis_latency.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  PNG saved to: {output_path}")

output_path_pdf = os.path.join(output_dir, 'icl_banking_task_analysis_latency.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"  PDF saved to: {output_path_pdf}")
plt.close()

# ============ CLINIC150 PLOTS ============

# Version 4: Combined plot for Clinic150
print("Creating Version 4: Clinic150 Combined plot...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Reasoning Models Recover Performance Degradation Compared to Non-Reasoning Models on In-Context Learning Tasks\nHELMET In-Context Learning Task: Clinic150 (16K Context)',
             fontsize=18, fontweight='bold', y=0.995)

for j, metric_type in enumerate(['memory', 'throughput']):
    plot_metric(axes[j], metric_type, '16k', task='icl_clinic')

# Create legend in single row
model_elements, technique_elements = create_legend_elements()
all_elements = model_elements + technique_elements

fig.legend(handles=all_elements,
          loc='lower center',
          ncol=len(all_elements),  # All items in one row
          bbox_to_anchor=(0.5, -0.02),
          frameon=True,
          fancybox=True,
          shadow=False,
          fontsize=11)

plt.tight_layout(rect=[0, 0.06, 1, 0.96])
output_path = os.path.join(output_dir, 'icl_clinic150_task_analysis_combined.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  PNG saved to: {output_path}")

output_path_pdf = os.path.join(output_dir, 'icl_clinic150_task_analysis_combined.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"  PDF saved to: {output_path_pdf}")
plt.close()

# Version 5: Standalone memory plot for Clinic150
print("Creating Version 5: Clinic150 Memory plot...")
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
fig.suptitle('Reasoning Models Recover Performance Degradation Compared to Non-Reasoning Models on In-Context Learning Tasks\nHELMET In-Context Learning Task: Clinic150 (16K Context)',
             fontsize=18, fontweight='bold', y=0.995)

plot_metric(ax, 'memory', '16k', task='icl_clinic')

# Create vertical legend on the right
model_elements, technique_elements = create_legend_elements()

# Add section headers
model_header = Line2D([0], [0], marker='none', color='none', label='Models:', markersize=0)
tech_header = Line2D([0], [0], marker='none', color='none', label='', markersize=0)  # Empty line as separator
tech_label = Line2D([0], [0], marker='none', color='none', label='Techniques:', markersize=0)

all_elements = [model_header] + model_elements + [tech_header, tech_label] + technique_elements

ax.legend(handles=all_elements,
         loc='center left',
         bbox_to_anchor=(1.05, 0.5),
         frameon=True,
         fancybox=False,
         shadow=False,
         fontsize=11)

plt.tight_layout()
output_path = os.path.join(output_dir, 'icl_clinic150_task_analysis_memory.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  PNG saved to: {output_path}")

output_path_pdf = os.path.join(output_dir, 'icl_clinic150_task_analysis_memory.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"  PDF saved to: {output_path_pdf}")
plt.close()

# Version 6: Standalone latency plot for Clinic150
print("Creating Version 6: Clinic150 Latency plot...")
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
fig.suptitle('Reasoning Models Recover Performance Degradation Compared to Non-Reasoning Models on In-Context Learning Tasks\nHELMET In-Context Learning Task: Clinic150 (16K Context)',
             fontsize=18, fontweight='bold', y=0.995)

plot_metric(ax, 'throughput', '16k', task='icl_clinic')

# Create vertical legend on the right
model_elements, technique_elements = create_legend_elements()

# Add section headers
model_header = Line2D([0], [0], marker='none', color='none', label='Models:', markersize=0)
tech_header = Line2D([0], [0], marker='none', color='none', label='', markersize=0)  # Empty line as separator
tech_label = Line2D([0], [0], marker='none', color='none', label='Techniques:', markersize=0)

all_elements = [model_header] + model_elements + [tech_header, tech_label] + technique_elements

ax.legend(handles=all_elements,
         loc='center left',
         bbox_to_anchor=(1.05, 0.5),
         frameon=True,
         fancybox=False,
         shadow=False,
         fontsize=11)

plt.tight_layout()
output_path = os.path.join(output_dir, 'icl_clinic150_task_analysis_latency.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  PNG saved to: {output_path}")

output_path_pdf = os.path.join(output_dir, 'icl_clinic150_task_analysis_latency.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"  PDF saved to: {output_path_pdf}")
plt.close()

print("\nAll plots created successfully!")

# Display plots inline (useful for Jupyter notebooks)
try:
    from IPython.display import display, Image as IPImage

    print("\nDisplaying plots:")
    print("\n=== Banking77 Plots ===")
    print("\n1. Combined plot (Memory + Latency):")
    display(IPImage(filename=os.path.join(output_dir, 'icl_banking_task_analysis_combined.png')))

    print("\n2. Memory plot:")
    display(IPImage(filename=os.path.join(output_dir, 'icl_banking_task_analysis_memory.png')))

    print("\n3. Latency plot:")
    display(IPImage(filename=os.path.join(output_dir, 'icl_banking_task_analysis_latency.png')))

    print("\n=== Clinic150 Plots ===")
    print("\n4. Combined plot (Memory + Latency):")
    display(IPImage(filename=os.path.join(output_dir, 'icl_clinic150_task_analysis_combined.png')))

    print("\n5. Memory plot:")
    display(IPImage(filename=os.path.join(output_dir, 'icl_clinic150_task_analysis_memory.png')))

    print("\n6. Latency plot:")
    display(IPImage(filename=os.path.join(output_dir, 'icl_clinic150_task_analysis_latency.png')))
except ImportError:
    # Not in a Jupyter environment, skip display
    print("\nNote: Run in Jupyter notebook to see inline plot previews.")
