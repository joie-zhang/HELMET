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
plt.rcParams['font.size'] = 13
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# Load HELMET data
helmet_memory_df = pd.read_csv('helmet_memory_usage.csv')
helmet_throughput_df = pd.read_csv('helmet_throughput.csv')
helmet_performance_df = pd.read_csv('helmet_performance.csv')

# Load LongProc data
longproc_memory_df = pd.read_csv('longproc_memory_usage.csv')
longproc_throughput_df = pd.read_csv('longproc_throughput.csv')
longproc_performance_df = pd.read_csv('longproc_performance.csv')

# Filter out unwanted techniques
unwanted_techniques = ['quest', 'streamingllm_original', 'minference']
for df in [helmet_memory_df, helmet_throughput_df, helmet_performance_df]:
    df.drop(df[df['technique'].isin(unwanted_techniques)].index, inplace=True)

for df in [longproc_memory_df, longproc_throughput_df, longproc_performance_df]:
    df.drop(df[df['technique'].isin(unwanted_techniques)].index, inplace=True)

# Filter out Qwen3 models
unwanted_models = ['Qwen3-8B', 'Yarn-Qwen3-8B']
for df in [helmet_memory_df, helmet_throughput_df, helmet_performance_df]:
    df.drop(df[df['model'].isin(unwanted_models)].index, inplace=True)

for df in [longproc_memory_df, longproc_throughput_df, longproc_performance_df]:
    df.drop(df[df['model'].isin(unwanted_models)].index, inplace=True)

# Filter out specific unwanted cache size configurations
unwanted_configs = [
    ('pyramidkv', 'w32_c4096_k5_avgpool'),
    ('snapkv', 'w32_c4096_k5_avgpool'),
    ('streamingllm', 'n_local_3968_n_init_128'),
    ('duoattn', 'sp0.7_pf32768')
]

for technique, cache_size in unwanted_configs:
    for df in [helmet_memory_df, helmet_throughput_df, helmet_performance_df]:
        condition = (df['technique'] == technique) & (df['cache_size'] == cache_size)
        df.drop(df[condition].index, inplace=True)

    for df in [longproc_memory_df, longproc_throughput_df, longproc_performance_df]:
        condition = (df['technique'] == technique) & (df['cache_size'] == cache_size)
        df.drop(df[condition].index, inplace=True)

# Filter SnapKV and PyramidKV to keep only specific configurations for reasoning models
allowed_reasoning_configs = ['w256_c2048_k7_avgpool', 'w2048_c8192_k7_avgpool', 'w256_c2048_k7_maxpool', 'w2048_c8192_k7_maxpool', 'default']
reasoning_models = ['DeepSeek-R1-Distill-Llama-8B', 'DeepSeek-R1-Distill-Qwen-7B']

for df in [helmet_memory_df, helmet_throughput_df, helmet_performance_df]:
    condition = (
        (df['model'].isin(reasoning_models)) &
        (df['technique'].isin(['snapkv', 'pyramidkv'])) &
        (~df['cache_size'].isin(allowed_reasoning_configs))
    )
    df.drop(df[condition].index, inplace=True)

for df in [longproc_memory_df, longproc_throughput_df, longproc_performance_df]:
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

for df in [longproc_memory_df, longproc_throughput_df, longproc_performance_df]:
    condition = (
        (df['model'].isin(baseline_models)) &
        (df['technique'].isin(['snapkv', 'pyramidkv'])) &
        (df['cache_size'].str.contains('w32_|w1024_', na=False))
    )
    df.drop(df[condition].index, inplace=True)

# Filter StreamingLLM to keep both n_local=4092 and n_local=4096 configurations
# 4092: Most tasks evaluated, 4096: ICL tasks evaluated
# Keeping both to show complete performance picture
allowed_streamingllm_configs = ['n_local_4092_n_init_4', 'n_local_4096_n_init_4']
for df in [helmet_memory_df, helmet_throughput_df, helmet_performance_df]:
    condition = (
        (df['technique'] == 'streamingllm') &
        (~df['cache_size'].isin(allowed_streamingllm_configs))
    )
    df.drop(df[condition].index, inplace=True)

for df in [longproc_memory_df, longproc_throughput_df, longproc_performance_df]:
    condition = (
        (df['technique'] == 'streamingllm') &
        (~df['cache_size'].isin(allowed_streamingllm_configs))
    )
    df.drop(df[condition].index, inplace=True)

# Define color palette for models
model_palette = {
    'Llama-3.1-8B-Instruct': '#AB63FA',
    'Qwen2.5-7B-Instruct': '#00CC96',
    'DeepSeek-R1-Distill-Llama-8B': '#EF553B',
    'DeepSeek-R1-Distill-Qwen-7B': '#FFA15A',
}

# Define marker shapes for techniques
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
    'P': 330,
    'X': 330,
    '*': 450,
    'v': 270,
}

# HELMET tasks for averaging (16k context)
helmet_tasks = [
    'recall_jsonkv', 'rag_nq', 'rag_hotpotqa', 'rerank',
    'cite_str_em', 'cite_citation_rec', 'cite_citation_prec', 'niah',
    'icl_clinic', 'icl_banking'
]

# LongProc tasks for averaging (2k context)
longproc_tasks = ['html_to_tsv', 'pseudo_to_code', 'travel_planning']

# Context lengths
HELMET_CONTEXT = '16k'
LONGPROC_CONTEXT = '2k'

# Create output directory
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

def calculate_average_performance(performance_df, tasks, technique, model, cache_size, context):
    """Calculate average performance across specified tasks"""
    perf_row = performance_df[
        (performance_df['technique'] == technique) &
        (performance_df['context_length'] == context) &
        (performance_df['model'] == model) &
        (performance_df['cache_size'] == cache_size)
    ]

    if perf_row.empty:
        return None

    # Get performance values for all tasks
    perf_values = []
    for task in tasks:
        val = perf_row.iloc[0][task]
        if not pd.isna(val) and val != 0:
            perf_values.append(val)

    if len(perf_values) == 0:
        return None

    return np.mean(perf_values)

def get_memory_value(memory_df, technique, model, cache_size, context, tasks):
    """Get memory value for a configuration"""
    mem_row = memory_df[
        (memory_df['technique'] == technique) &
        (memory_df['context_length'] == context) &
        (memory_df['model'] == model) &
        (memory_df['cache_size'] == cache_size)
    ]

    if mem_row.empty:
        return None

    # For HELMET, use average across task-specific memory columns
    # For LongProc, use average across task columns
    mem_values = []
    for task in tasks:
        # Map task names to memory column names if needed
        if task.startswith('cite_'):
            col = 'cite'
        else:
            col = task

        if col in mem_row.columns:
            val = mem_row.iloc[0][col]
            if not pd.isna(val) and val != 0:
                mem_values.append(val)

    if len(mem_values) == 0:
        return None

    return np.mean(mem_values)

def plot_comparison(ax, memory_df, performance_df, tasks, context, techniques, title):
    """Plot performance vs memory for given techniques"""
    subset = memory_df[memory_df['context_length'] == context]

    for (technique, model), group in subset.groupby(['technique', 'model']):
        # Skip techniques not in our list
        if technique not in techniques:
            continue

        if model not in model_palette or technique not in marker_dict:
            continue

        x_values = []
        y_values = []

        for _, row in group.iterrows():
            # Get memory value
            x = get_memory_value(memory_df, technique, model, row['cache_size'], context, tasks)

            # Get average performance
            y = calculate_average_performance(performance_df, tasks, technique, model, row['cache_size'], context)

            if x is None or y is None:
                continue

            x_values.append(x)
            y_values.append(y)

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

            # Connect points with lines if multiple configurations
            if len(x_values) > 1:
                ax.plot(
                    x_values, y_values,
                    color=model_palette[model],
                    linestyle='--',
                    alpha=0.4,
                    linewidth=1.5,
                    zorder=2
                )

    ax.set_xlabel('Memory Usage (GB)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Average Performance Score', fontweight='bold', fontsize=14)
    ax.set_title(title, fontweight='bold', fontsize=16, pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

def create_legend_elements(techniques):
    """Create legend elements for models and techniques"""
    elements = []

    # Add model legend entries
    elements.append(Line2D([0], [0], marker='none', color='none',
                          label='Models:', markersize=0))
    for model, color in model_palette.items():
        elements.append(Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color, markersize=10,
                              markeredgewidth=0,
                              label=model))

    # Add separator
    elements.append(Line2D([0], [0], marker='none', color='none',
                          label='', markersize=0))

    # Add technique legend entries
    elements.append(Line2D([0], [0], marker='none', color='none',
                          label='Techniques:', markersize=0))
    for tech in techniques:
        if tech in marker_dict:
            elements.append(Line2D([0], [0], marker=marker_dict[tech], color='gray',
                                  linestyle='None', markersize=10,
                                  markeredgewidth=0,
                                  label=technique_labels[tech]))

    return elements

# ============================================================================
# PLOT 1: 2x2 Layout (HELMET and LongProc separate)
# ============================================================================
print("Creating 2x2 comparison plot...")

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Model Comparison: Average Performance vs Memory Usage\nHELMET (16K) and LongProc (2K) Benchmarks',
             fontsize=18, fontweight='bold', y=0.995)

# Define technique sets
quantization_techniques = ['baseline', 'INT4', 'INT8']
kv_techniques = ['baseline', 'streamingllm', 'snapkv', 'pyramidkv']

# Row 1: HELMET
plot_comparison(axes[0, 0], helmet_memory_df, helmet_performance_df, helmet_tasks,
                HELMET_CONTEXT, quantization_techniques,
                'HELMET: Quantization Methods')
plot_comparison(axes[0, 1], helmet_memory_df, helmet_performance_df, helmet_tasks,
                HELMET_CONTEXT, kv_techniques,
                'HELMET: KV Cache Methods')

# Row 2: LongProc
plot_comparison(axes[1, 0], longproc_memory_df, longproc_performance_df, longproc_tasks,
                LONGPROC_CONTEXT, quantization_techniques,
                'LongProc: Quantization Methods')
plot_comparison(axes[1, 1], longproc_memory_df, longproc_performance_df, longproc_tasks,
                LONGPROC_CONTEXT, kv_techniques,
                'LongProc: KV Cache Methods')

# Get axis limits for consistent scaling
all_xlims = [ax.get_xlim() for ax in axes.flat]
all_ylims = [ax.get_ylim() for ax in axes.flat]

# Calculate global limits with some padding
x_min = min(xlim[0] for xlim in all_xlims)
x_max = max(xlim[1] for xlim in all_xlims)
y_min = min(ylim[0] for ylim in all_ylims)
y_max = max(ylim[1] for ylim in all_ylims)

# Add 5% padding
x_padding = (x_max - x_min) * 0.05
y_padding = (y_max - y_min) * 0.05

# Apply consistent limits to all subplots
for ax in axes.flat:
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

# Create legend (combine both technique sets)
all_techniques = sorted(set(quantization_techniques + kv_techniques))
legend_elements = create_legend_elements(all_techniques)

fig.legend(handles=legend_elements,
          loc='center left',
          bbox_to_anchor=(1.01, 0.5),
          frameon=True,
          fancybox=False,
          shadow=False,
          fontsize=12)

plt.tight_layout()

output_path = os.path.join(output_dir, 'averages_comparison_2x2.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  Saved: {output_path}")

output_path_pdf = os.path.join(output_dir, 'averages_comparison_2x2.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"  Saved: {output_path_pdf}")

# Display plot inline (useful for Jupyter notebooks)
try:
    from IPython.display import display, Image as IPImage

    print("\nDisplaying 2x2 plot:")
    display(IPImage(filename=output_path))
except ImportError:
    # Not in a Jupyter environment, skip display
    print("\nNote: Run in Jupyter notebook to see inline plot preview.")

plt.close()

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

# ============================================================================
# PLOT 2: 1x2 Layout (HELMET and LongProc combined)
# ============================================================================
print("Creating 1x2 comparison plot...")

fig_1x2, axes_1x2 = plt.subplots(1, 2, figsize=(24, 10))
fig_1x2.suptitle('Model Comparison: Average Performance vs Memory Usage\nCombined HELMET (16K) and LongProc (2K) Benchmarks',
             fontsize=18, fontweight='bold', y=0.995)

# Combine data for both benchmarks
def plot_combined_comparison(ax, techniques, title):
    """Plot combined HELMET + LongProc data with unified averaging"""
    # Get all unique (technique, model, cache_size) combinations from HELMET
    helmet_subset = helmet_memory_df[helmet_memory_df['context_length'] == HELMET_CONTEXT]

    # Process each technique/model combination
    for (technique, model) in helmet_subset.groupby(['technique', 'model']).groups.keys():
        if technique not in techniques or model not in model_palette or technique not in marker_dict:
            continue

        # Get cache sizes for this technique/model from HELMET (sorted)
        cache_sizes_df = helmet_subset[
            (helmet_subset['technique'] == technique) &
            (helmet_subset['model'] == model)
        ].copy()

        # Sort cache sizes if applicable
        if technique == "streamingllm":
            cache_sizes_df['sort_key'] = cache_sizes_df['cache_size'].apply(
                lambda x: int(x.split('_')[2]) if x.startswith('n_local_') else 0
            )
            cache_sizes_df = cache_sizes_df.sort_values('sort_key')
        elif technique in ["snapkv", "pyramidkv"]:
            def extract_cache_and_k(cache_size):
                if not cache_size.startswith('w'):
                    return (0, 0, 0)
                parts = cache_size.split('_')
                cache_val = int(parts[1][1:])
                k_val = int(parts[2][1:])
                return (cache_val, k_val, 0)

            cache_sizes_df['sort_key'] = cache_sizes_df['cache_size'].apply(extract_cache_and_k)
            cache_sizes_df = cache_sizes_df.sort_values('sort_key')

        cache_sizes = cache_sizes_df['cache_size'].values

        x_values = []
        y_values = []
        labels = []

        # Special handling for StreamingLLM: merge results from both configs
        if technique == 'streamingllm' and len(cache_sizes) > 1:
            # Collect all unique task results across both configs
            all_mem_values = []
            all_perf_values = []
            seen_tasks = set()

            for cache_size in cache_sizes:
                # Get memory
                mem = get_memory_value(helmet_memory_df, technique, model, cache_size, HELMET_CONTEXT, helmet_tasks)
                if mem is not None:
                    all_mem_values.append(mem)

                mem = get_memory_value(longproc_memory_df, technique, model, cache_size, LONGPROC_CONTEXT, longproc_tasks)
                if mem is not None:
                    all_mem_values.append(mem)

                # Get HELMET task performances
                helmet_row = helmet_performance_df[
                    (helmet_performance_df['technique'] == technique) &
                    (helmet_performance_df['context_length'] == HELMET_CONTEXT) &
                    (helmet_performance_df['model'] == model) &
                    (helmet_performance_df['cache_size'] == cache_size)
                ]
                if not helmet_row.empty:
                    for task in helmet_tasks:
                        if task not in seen_tasks:
                            val = helmet_row.iloc[0][task]
                            if not pd.isna(val) and val != 0:
                                all_perf_values.append(val)
                                seen_tasks.add(task)

                # Get LongProc task performances
                longproc_row = longproc_performance_df[
                    (longproc_performance_df['technique'] == technique) &
                    (longproc_performance_df['context_length'] == LONGPROC_CONTEXT) &
                    (longproc_performance_df['model'] == model) &
                    (longproc_performance_df['cache_size'] == cache_size)
                ]
                if not longproc_row.empty:
                    for task in longproc_tasks:
                        if task not in seen_tasks:
                            val = longproc_row.iloc[0][task]
                            if not pd.isna(val) and val != 0:
                                all_perf_values.append(val)
                                seen_tasks.add(task)

            if len(all_mem_values) > 0 and len(all_perf_values) > 0:
                x_values.append(np.mean(all_mem_values))
                y_values.append(np.mean(all_perf_values))
                labels.append("StreamingLLM")
        else:
            # Normal processing for non-StreamingLLM or single config
            for cache_size in cache_sizes:
                # Get HELMET metrics
                helmet_mem = get_memory_value(helmet_memory_df, technique, model, cache_size, HELMET_CONTEXT, helmet_tasks)

                # Get LongProc metrics
                longproc_mem = get_memory_value(longproc_memory_df, technique, model, cache_size, LONGPROC_CONTEXT, longproc_tasks)

                # Average memory across both benchmarks
                mem_values = [m for m in [helmet_mem, longproc_mem] if m is not None]
                if len(mem_values) > 0:
                    avg_mem = np.mean(mem_values)
                else:
                    continue

                # Calculate performance across ALL 13 tasks (10 HELMET + 3 LongProc)
                all_perf_values = []

                # Get HELMET task performances
                helmet_perf_row = helmet_performance_df[
                    (helmet_performance_df['technique'] == technique) &
                    (helmet_performance_df['context_length'] == HELMET_CONTEXT) &
                    (helmet_performance_df['model'] == model) &
                    (helmet_performance_df['cache_size'] == cache_size)
                ]
                if not helmet_perf_row.empty:
                    for task in helmet_tasks:
                        val = helmet_perf_row.iloc[0][task]
                        if not pd.isna(val) and val != 0:
                            all_perf_values.append(val)

                # Get LongProc task performances
                longproc_perf_row = longproc_performance_df[
                    (longproc_performance_df['technique'] == technique) &
                    (longproc_performance_df['context_length'] == LONGPROC_CONTEXT) &
                    (longproc_performance_df['model'] == model) &
                    (longproc_performance_df['cache_size'] == cache_size)
                ]
                if not longproc_perf_row.empty:
                    for task in longproc_tasks:
                        val = longproc_perf_row.iloc[0][task]
                        if not pd.isna(val) and val != 0:
                            all_perf_values.append(val)

                if len(all_perf_values) == 0:
                    continue

                avg_perf = np.mean(all_perf_values)

                x_values.append(avg_mem)
                y_values.append(avg_perf)
                labels.append(format_cache_size(cache_size))

        if len(x_values) > 0:
            ax.scatter(x_values, y_values, color=model_palette[model], marker=marker_dict[technique],
                      s=marker_size_dict[marker_dict[technique]], alpha=0.8, zorder=3)
            # Don't draw lines for StreamingLLM since we merge configs into single point
            if len(x_values) > 1 and technique != 'streamingllm':
                ax.plot(x_values, y_values, color=model_palette[model], linestyle='--',
                       alpha=0.4, linewidth=1.5, zorder=2)

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

    ax.set_xlabel('Memory Usage (GB)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Average Performance Score', fontweight='bold', fontsize=14)
    ax.set_title(title, fontweight='bold', fontsize=16, pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

plot_combined_comparison(axes_1x2[0], quantization_techniques, 'Quantization Methods')
plot_combined_comparison(axes_1x2[1], kv_techniques, 'KV Cache Methods')

# Get axis limits for consistent scaling
all_xlims = [ax.get_xlim() for ax in axes_1x2]
all_ylims = [ax.get_ylim() for ax in axes_1x2]

x_min = min(xlim[0] for xlim in all_xlims)
x_max = max(xlim[1] for xlim in all_xlims)
y_min = 0  # Force y-axis to start at 0
y_max = max(ylim[1] for ylim in all_ylims)

x_padding = (x_max - x_min) * 0.05
y_padding = (y_max - y_min) * 0.05

for ax in axes_1x2:
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min, y_max + y_padding)

# Create legend in horizontal layout at bottom - two separate rows
model_elements = []
for model, color in model_palette.items():
    model_elements.append(Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color, markersize=10,
                              markeredgewidth=0,
                              label=model))

technique_elements = []
for tech in all_techniques:
    if tech in marker_dict:
        technique_elements.append(Line2D([0], [0], marker=marker_dict[tech], color='gray',
                                      linestyle='None', markersize=10,
                                      markeredgewidth=0,
                                      label=technique_labels[tech]))

# Create first legend for models (top row)
leg1 = fig_1x2.legend(handles=model_elements,
          title='Models',
          title_fontsize=15,
          loc='lower center',
          bbox_to_anchor=(0.5, 0.05),
          ncol=len(model_elements),
          frameon=True,
          fancybox=False,
          shadow=False,
          fontsize=15,
          columnspacing=1.5,
          handletextpad=0.5)

# Add second legend for techniques (bottom row)
leg2 = fig_1x2.legend(handles=technique_elements,
              title='Techniques',
              title_fontsize=15,
              loc='lower center',
              bbox_to_anchor=(0.5, -0.05),
              ncol=len(technique_elements),
              frameon=True,
              fancybox=False,
              shadow=False,
              fontsize=15,
              columnspacing=1.5,
              handletextpad=0.5)

# Add first legend back as artist (since second legend call replaces it)
fig_1x2.add_artist(leg1)

plt.tight_layout(rect=[0, 0.15, 1, 1])

output_path = os.path.join(output_dir, 'averages_comparison_1x2.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  Saved: {output_path}")

output_path_pdf = os.path.join(output_dir, 'averages_comparison_1x2.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"  Saved: {output_path_pdf}")

# Display plot inline (useful for Jupyter notebooks)
try:
    from IPython.display import display, Image as IPImage

    print("\nDisplaying 1x2 plot:")
    display(IPImage(filename=output_path))
except ImportError:
    # Not in a Jupyter environment, skip display
    print("\nNote: Run in Jupyter notebook to see inline plot preview.")

plt.close()

# ============================================================================
# PLOT 3: 2x2 Layout - Filtered (Only Instruct + Reasoning Models)
# ============================================================================
print("Creating 2x2 filtered comparison plot (Instruct + Reasoning models only)...")

# Filter to include only desired models
filtered_models = ['Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct',
                   'DeepSeek-R1-Distill-Llama-8B', 'DeepSeek-R1-Distill-Qwen-7B']

# Create filtered dataframes
helmet_memory_filtered = helmet_memory_df[helmet_memory_df['model'].isin(filtered_models)].copy()
helmet_performance_filtered = helmet_performance_df[helmet_performance_df['model'].isin(filtered_models)].copy()
longproc_memory_filtered = longproc_memory_df[longproc_memory_df['model'].isin(filtered_models)].copy()
longproc_performance_filtered = longproc_performance_df[longproc_performance_df['model'].isin(filtered_models)].copy()

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Model Comparison: Average Performance vs Memory Usage\nHELMET (16K) and LongProc (2K) Benchmarks',
             fontsize=18, fontweight='bold', y=0.995)

# Row 1: HELMET
plot_comparison(axes[0, 0], helmet_memory_filtered, helmet_performance_filtered, helmet_tasks,
                HELMET_CONTEXT, quantization_techniques,
                'HELMET: Quantization Methods')
plot_comparison(axes[0, 1], helmet_memory_filtered, helmet_performance_filtered, helmet_tasks,
                HELMET_CONTEXT, kv_techniques,
                'HELMET: KV Cache Methods')

# Row 2: LongProc
plot_comparison(axes[1, 0], longproc_memory_filtered, longproc_performance_filtered, longproc_tasks,
                LONGPROC_CONTEXT, quantization_techniques,
                'LongProc: Quantization Methods')
plot_comparison(axes[1, 1], longproc_memory_filtered, longproc_performance_filtered, longproc_tasks,
                LONGPROC_CONTEXT, kv_techniques,
                'LongProc: KV Cache Methods')

# Get axis limits for consistent scaling
all_xlims = [ax.get_xlim() for ax in axes.flat]
all_ylims = [ax.get_ylim() for ax in axes.flat]

x_min = min(xlim[0] for xlim in all_xlims)
x_max = max(xlim[1] for xlim in all_xlims)
y_min = min(ylim[0] for ylim in all_ylims)
y_max = max(ylim[1] for ylim in all_ylims)

x_padding = (x_max - x_min) * 0.05
y_padding = (y_max - y_min) * 0.05

for ax in axes.flat:
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

# Create filtered legend (only show included models)
def create_legend_elements_filtered(techniques, models):
    """Create legend elements for filtered models and techniques"""
    elements = []

    # Add model legend entries
    elements.append(Line2D([0], [0], marker='none', color='none',
                          label='Models:', markersize=0))
    for model in models:
        if model in model_palette:
            elements.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=model_palette[model], markersize=10,
                                  markeredgewidth=0,
                                  label=model))

    # Add separator
    elements.append(Line2D([0], [0], marker='none', color='none',
                          label='', markersize=0))

    # Add technique legend entries
    elements.append(Line2D([0], [0], marker='none', color='none',
                          label='Techniques:', markersize=0))
    for tech in techniques:
        if tech in marker_dict:
            elements.append(Line2D([0], [0], marker=marker_dict[tech], color='gray',
                                  linestyle='None', markersize=10,
                                  markeredgewidth=0,
                                  label=technique_labels[tech]))

    return elements

legend_elements = create_legend_elements_filtered(all_techniques, filtered_models)

fig.legend(handles=legend_elements,
          loc='center left',
          bbox_to_anchor=(1.02, 0.5),
          frameon=True,
          fancybox=False,
          shadow=False,
          fontsize=11)

plt.tight_layout()

output_path = os.path.join(output_dir, 'averages_comparison_2x2_filtered.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  Saved: {output_path}")

output_path_pdf = os.path.join(output_dir, 'averages_comparison_2x2_filtered.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"  Saved: {output_path_pdf}")

# Display plot inline (useful for Jupyter notebooks)
try:
    from IPython.display import display, Image as IPImage

    print("\nDisplaying 2x2 filtered plot:")
    display(IPImage(filename=output_path))
except ImportError:
    # Not in a Jupyter environment, skip display
    print("\nNote: Run in Jupyter notebook to see inline plot preview.")

plt.close()

# ============================================================================
# PLOT 4: 1x2 Layout - Filtered (Only Instruct + Reasoning Models)
# ============================================================================
print("Creating 1x2 filtered comparison plot (Instruct + Reasoning models only)...")

fig, axes = plt.subplots(1, 2, figsize=(24, 10))
fig.suptitle('Model Comparison: Average Performance vs Memory Usage\nCombined HELMET (16K) and LongProc (2K) Benchmarks',
             fontsize=18, fontweight='bold', y=0.995)

# Combine data for both benchmarks (filtered)
def plot_combined_comparison_filtered(ax, techniques, title, memory_helmet, perf_helmet,
                                      memory_longproc, perf_longproc):
    """Plot combined HELMET + LongProc data with filtered models and unified averaging"""
    # Get all unique (technique, model, cache_size) combinations from HELMET
    helmet_subset = memory_helmet[memory_helmet['context_length'] == HELMET_CONTEXT]

    # Process each technique/model combination
    for (technique, model) in helmet_subset.groupby(['technique', 'model']).groups.keys():
        if technique not in techniques or model not in model_palette or technique not in marker_dict:
            continue

        # Get cache sizes for this technique/model from HELMET (sorted)
        cache_sizes_df = helmet_subset[
            (helmet_subset['technique'] == technique) &
            (helmet_subset['model'] == model)
        ].copy()

        # Sort cache sizes if applicable
        if technique == "streamingllm":
            cache_sizes_df['sort_key'] = cache_sizes_df['cache_size'].apply(
                lambda x: int(x.split('_')[2]) if x.startswith('n_local_') else 0
            )
            cache_sizes_df = cache_sizes_df.sort_values('sort_key')
        elif technique in ["snapkv", "pyramidkv"]:
            def extract_cache_and_k(cache_size):
                if not cache_size.startswith('w'):
                    return (0, 0, 0)
                parts = cache_size.split('_')
                cache_val = int(parts[1][1:])
                k_val = int(parts[2][1:])
                return (cache_val, k_val, 0)

            cache_sizes_df['sort_key'] = cache_sizes_df['cache_size'].apply(extract_cache_and_k)
            cache_sizes_df = cache_sizes_df.sort_values('sort_key')

        cache_sizes = cache_sizes_df['cache_size'].values

        x_values = []
        y_values = []
        labels = []

        # Special handling for StreamingLLM: merge results from both configs
        if technique == 'streamingllm' and len(cache_sizes) > 1:
            # Collect all unique task results across both configs
            all_mem_values = []
            all_perf_values = []
            seen_tasks = set()

            for cache_size in cache_sizes:
                # Get memory
                mem = get_memory_value(memory_helmet, technique, model, cache_size, HELMET_CONTEXT, helmet_tasks)
                if mem is not None:
                    all_mem_values.append(mem)

                mem = get_memory_value(memory_longproc, technique, model, cache_size, LONGPROC_CONTEXT, longproc_tasks)
                if mem is not None:
                    all_mem_values.append(mem)

                # Get HELMET task performances
                helmet_row = perf_helmet[
                    (perf_helmet['technique'] == technique) &
                    (perf_helmet['context_length'] == HELMET_CONTEXT) &
                    (perf_helmet['model'] == model) &
                    (perf_helmet['cache_size'] == cache_size)
                ]
                if not helmet_row.empty:
                    for task in helmet_tasks:
                        if task not in seen_tasks:
                            val = helmet_row.iloc[0][task]
                            if not pd.isna(val) and val != 0:
                                all_perf_values.append(val)
                                seen_tasks.add(task)

                # Get LongProc task performances
                longproc_row = perf_longproc[
                    (perf_longproc['technique'] == technique) &
                    (perf_longproc['context_length'] == LONGPROC_CONTEXT) &
                    (perf_longproc['model'] == model) &
                    (perf_longproc['cache_size'] == cache_size)
                ]
                if not longproc_row.empty:
                    for task in longproc_tasks:
                        if task not in seen_tasks:
                            val = longproc_row.iloc[0][task]
                            if not pd.isna(val) and val != 0:
                                all_perf_values.append(val)
                                seen_tasks.add(task)

            if len(all_mem_values) > 0 and len(all_perf_values) > 0:
                x_values.append(np.mean(all_mem_values))
                y_values.append(np.mean(all_perf_values))
                labels.append("StreamingLLM")
        else:
            # Normal processing for non-StreamingLLM or single config
            for cache_size in cache_sizes:
                # Get HELMET metrics
                helmet_mem = get_memory_value(memory_helmet, technique, model, cache_size, HELMET_CONTEXT, helmet_tasks)

                # Get LongProc metrics
                longproc_mem = get_memory_value(memory_longproc, technique, model, cache_size, LONGPROC_CONTEXT, longproc_tasks)

                # Average memory across both benchmarks
                mem_values = [m for m in [helmet_mem, longproc_mem] if m is not None]
                if len(mem_values) > 0:
                    avg_mem = np.mean(mem_values)
                else:
                    continue

                # Calculate performance across ALL 13 tasks (10 HELMET + 3 LongProc)
                all_perf_values = []

                # Get HELMET task performances
                helmet_perf_row = perf_helmet[
                    (perf_helmet['technique'] == technique) &
                    (perf_helmet['context_length'] == HELMET_CONTEXT) &
                    (perf_helmet['model'] == model) &
                    (perf_helmet['cache_size'] == cache_size)
                ]
                if not helmet_perf_row.empty:
                    for task in helmet_tasks:
                        val = helmet_perf_row.iloc[0][task]
                        if not pd.isna(val) and val != 0:
                            all_perf_values.append(val)

                # Get LongProc task performances
                longproc_perf_row = perf_longproc[
                    (perf_longproc['technique'] == technique) &
                    (perf_longproc['context_length'] == LONGPROC_CONTEXT) &
                    (perf_longproc['model'] == model) &
                    (perf_longproc['cache_size'] == cache_size)
                ]
                if not longproc_perf_row.empty:
                    for task in longproc_tasks:
                        val = longproc_perf_row.iloc[0][task]
                        if not pd.isna(val) and val != 0:
                            all_perf_values.append(val)

                if len(all_perf_values) == 0:
                    continue

                avg_perf = np.mean(all_perf_values)

                x_values.append(avg_mem)
                y_values.append(avg_perf)
                labels.append(format_cache_size(cache_size))

        if len(x_values) > 0:
            ax.scatter(x_values, y_values, color=model_palette[model], marker=marker_dict[technique],
                      s=marker_size_dict[marker_dict[technique]], alpha=0.8, zorder=3)
            # Don't draw lines for StreamingLLM since we merge configs into single point
            if len(x_values) > 1 and technique != 'streamingllm':
                ax.plot(x_values, y_values, color=model_palette[model], linestyle='--',
                       alpha=0.4, linewidth=1.5, zorder=2)

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

    ax.set_xlabel('Memory Usage (GB)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Average Performance Score', fontweight='bold', fontsize=14)
    ax.set_title(title, fontweight='bold', fontsize=16, pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

plot_combined_comparison_filtered(axes[0], quantization_techniques, 'Quantization Methods',
                                  helmet_memory_filtered, helmet_performance_filtered,
                                  longproc_memory_filtered, longproc_performance_filtered)
plot_combined_comparison_filtered(axes[1], kv_techniques, 'Token Eviction Methods',
                                  helmet_memory_filtered, helmet_performance_filtered,
                                  longproc_memory_filtered, longproc_performance_filtered)

# Get axis limits for consistent scaling
all_xlims = [ax.get_xlim() for ax in axes]
all_ylims = [ax.get_ylim() for ax in axes]

x_min = min(xlim[0] for xlim in all_xlims)
x_max = max(xlim[1] for xlim in all_xlims)
y_min = 0  # Force y-axis to start at 0
y_max = max(ylim[1] for ylim in all_ylims)

x_padding = (x_max - x_min) * 0.05
y_padding = (y_max - y_min) * 0.05

for ax in axes:
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min, y_max + y_padding)

# Create legend
legend_elements = create_legend_elements_filtered(all_techniques, filtered_models)

fig.legend(handles=legend_elements,
          loc='center left',
          bbox_to_anchor=(0.965, 0.5),
          frameon=True,
          fancybox=False,
          shadow=False,
          fontsize=12)

plt.tight_layout(rect=[0, 0, 0.98, 1])

output_path = os.path.join(output_dir, 'averages_comparison_1x2_filtered.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  Saved: {output_path}")

output_path_pdf = os.path.join(output_dir, 'averages_comparison_1x2_filtered.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"  Saved: {output_path_pdf}")

# Display plot inline (useful for Jupyter notebooks)
try:
    from IPython.display import display, Image as IPImage

    print("\nDisplaying 1x2 filtered plot:")
    display(IPImage(filename=output_path))
except ImportError:
    # Not in a Jupyter environment, skip display
    print("\nNote: Run in Jupyter notebook to see inline plot preview.")

plt.close()

print("\nAll comparison plots created successfully!")
