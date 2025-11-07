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
helmet_memory_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_memory_usage.csv')
helmet_throughput_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_throughput.csv')
helmet_performance_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_performance.csv')

# Load LongProc data
longproc_memory_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/longproc_results/longproc_memory_usage.csv')
longproc_throughput_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/longproc_results/longproc_throughput.csv')
longproc_performance_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/longproc_results/longproc_performance.csv')

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

# Cleaner model names for legend display
model_display_names = {
    'Llama-3.1-8B-Instruct': 'Llama-3.1-8B',
    'Qwen2.5-7B-Instruct': 'Qwen2.5-7B',
    'DeepSeek-R1-Distill-Llama-8B': 'R1-Distill-Llama-8B',
    'DeepSeek-R1-Distill-Qwen-7B': 'R1-Distill-Qwen-7B',
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

# Marker sizes - increased for better visibility
marker_size_dict = {
    'o': 450,
    's': 450,
    '^': 450,
    'P': 550,
    'X': 550,
    '*': 850,
    'v': 270,
}

# HELMET tasks for averaging (16k context)
helmet_tasks = [
    'recall_jsonkv', 'rag_nq', 'rag_hotpotqa', 'rerank',
    'cite_str_em', 'cite_citation_rec', 'cite_citation_prec', 'niah',
    'icl_clinic', 'icl_banking', 'summ_multilex'
]

# LongProc tasks for averaging (2k context)
longproc_tasks = ['html_to_tsv', 'pseudo_to_code', 'travel_planning']

# Context lengths
HELMET_CONTEXT = '16k'
HELMET_CONTEXT_32K = '32k'
LONGPROC_CONTEXT = '2k'
LONGPROC_CONTEXT_5K = '5k'

# Create output directory
output_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/results/plots'
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

# Define technique sets
quantization_techniques = ['baseline', 'INT4', 'INT8']
kv_techniques = ['baseline', 'streamingllm', 'snapkv', 'pyramidkv', 'duoattn']
all_techniques = sorted(set(quantization_techniques + kv_techniques))

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
# PLOT 5: 1x1 Layout - All Techniques Combined (Filtered Models, No w2048)
# ============================================================================
print("Creating 1x1 all-techniques comparison plot...")

# Filter to include only desired models
filtered_models_1x1 = ['Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct',
                       'DeepSeek-R1-Distill-Llama-8B', 'DeepSeek-R1-Distill-Qwen-7B']

# Create filtered dataframes
helmet_memory_1x1 = helmet_memory_df[helmet_memory_df['model'].isin(filtered_models_1x1)].copy()
helmet_performance_1x1 = helmet_performance_df[helmet_performance_df['model'].isin(filtered_models_1x1)].copy()
longproc_memory_1x1 = longproc_memory_df[longproc_memory_df['model'].isin(filtered_models_1x1)].copy()
longproc_performance_1x1 = longproc_performance_df[longproc_performance_df['model'].isin(filtered_models_1x1)].copy()

# Additional filtering: Remove w2048_c8192_k7 configurations for SnapKV and PyramidKV
w2048_configs = ['w2048_c8192_k7_avgpool', 'w2048_c8192_k7_maxpool']
for df in [helmet_memory_1x1, helmet_performance_1x1]:
    condition = (
        (df['technique'].isin(['snapkv', 'pyramidkv'])) &
        (df['cache_size'].isin(w2048_configs))
    )
    df.drop(df[condition].index, inplace=True)

for df in [longproc_memory_1x1, longproc_performance_1x1]:
    condition = (
        (df['technique'].isin(['snapkv', 'pyramidkv'])) &
        (df['cache_size'].isin(w2048_configs))
    )
    df.drop(df[condition].index, inplace=True)

# Create single subplot with larger figure for better readability
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
# fig.suptitle('Performance vs Memory Usage\nAveraged Across HELMET (16K/32K) and LongProc (500/2K) Benchmarks',
            #  fontsize=24, fontweight='bold', y=0.98)

# All techniques combined (without duoattn for plots 5 and 7)
all_techniques_combined = sorted(set(quantization_techniques + kv_techniques))
all_techniques_combined_no_duo = [t for t in all_techniques_combined if t != 'duoattn']

# Define custom legend order (with SnapKV before PyramidKV)
legend_order = ['baseline', 'INT4', 'INT8', 'snapkv', 'pyramidkv', 'streamingllm', 'duoattn']
legend_order_no_duo = [t for t in legend_order if t != 'duoattn']

# Plot combined comparison with all techniques
def plot_all_techniques_combined(ax, techniques, memory_helmet, perf_helmet,
                                  memory_longproc, perf_longproc):
    """Plot combined HELMET + LongProc data with all techniques on single subplot"""
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

        # Special handling for StreamingLLM, SnapKV, PyramidKV: average across cache configs
        if (technique == 'streamingllm' or technique in ['snapkv', 'pyramidkv']) and len(cache_sizes) > 1:
            # Collect memory across configs and contexts, and collect per-task values to average per task
            all_mem_values = []
            task_to_vals = {}

            for cache_size in cache_sizes:
                # Memory (HELMET 16k)
                mem = get_memory_value(memory_helmet, technique, model, cache_size, HELMET_CONTEXT, helmet_tasks)
                if mem is not None:
                    all_mem_values.append(mem)

                # Memory (LongProc 2k)
                mem = get_memory_value(memory_longproc, technique, model, cache_size, LONGPROC_CONTEXT, longproc_tasks)
                if mem is not None:
                    all_mem_values.append(mem)

                # HELMET task performances (16k)
                helmet_row = perf_helmet[
                    (perf_helmet['technique'] == technique) &
                    (perf_helmet['context_length'] == HELMET_CONTEXT) &
                    (perf_helmet['model'] == model) &
                    (perf_helmet['cache_size'] == cache_size)
                ]
                if not helmet_row.empty:
                    for task in helmet_tasks:
                        val = helmet_row.iloc[0][task]
                        if not pd.isna(val) and val != 0:
                            task_to_vals.setdefault(task, []).append(val)

                # LongProc task performances (2k)
                longproc_row = perf_longproc[
                    (perf_longproc['technique'] == technique) &
                    (perf_longproc['context_length'] == LONGPROC_CONTEXT) &
                    (perf_longproc['model'] == model) &
                    (perf_longproc['cache_size'] == cache_size)
                ]
                if not longproc_row.empty:
                    for task in longproc_tasks:
                        val = longproc_row.iloc[0][task]
                        if not pd.isna(val) and val != 0:
                            task_to_vals.setdefault(task, []).append(val)

            # Compute per-task means, then overall mean performance
            per_task_means = [np.mean(vs) for vs in task_to_vals.values() if len(vs) > 0]

            if len(all_mem_values) > 0 and len(per_task_means) > 0:
                x_values.append(np.mean(all_mem_values))
                y_values.append(np.mean(per_task_means))
                # Use technique-specific label
                if technique == 'streamingllm':
                    labels.append("StreamingLLM")
                elif technique == 'snapkv':
                    labels.append("SnapKV")
                elif technique == 'pyramidkv':
                    labels.append("PyramidKV")
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
            # Don't draw lines for StreamingLLM, SnapKV, PyramidKV since we merge configs into single point
            if len(x_values) > 1 and technique not in ['streamingllm', 'snapkv', 'pyramidkv']:
                ax.plot(x_values, y_values, color=model_palette[model], linestyle='--',
                       alpha=0.4, linewidth=1.5, zorder=2)

                # Add label to the last point
                if labels[-1]:
                    ax.annotate(
                        labels[-1],
                        (x_values[-1], y_values[-1]),
                        xytext=(8, 8),
                        textcoords='offset points',
                        fontsize=14,
                        alpha=0.7,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='none', alpha=0.6)
                    )

    ax.set_xlabel('Memory Usage (GB)', fontweight='bold', fontsize=24)
    ax.set_ylabel('Average Performance Score', fontweight='bold', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

plot_all_techniques_combined(ax, all_techniques_combined_no_duo,
                              helmet_memory_1x1, helmet_performance_1x1,
                              longproc_memory_1x1, longproc_performance_1x1)

# Set y-axis to start at 0
ylim = ax.get_ylim()
ax.set_ylim(0, ylim[1] * 1.05)

# Create legend in horizontal layout at bottom - two separate rows
model_elements_1x1 = []
for model in filtered_models_1x1:
    if model in model_palette:
        display_name = model_display_names.get(model, model)
        model_elements_1x1.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=model_palette[model], markersize=16,
                                  markeredgewidth=0,
                                  label=display_name))

technique_elements_1x1 = []
for tech in legend_order_no_duo:
    if tech in marker_dict and tech in all_techniques_combined_no_duo:
        # Use larger marker size for StreamingLLM star
        ms = 20 if tech == 'streamingllm' else 16
        technique_elements_1x1.append(Line2D([0], [0], marker=marker_dict[tech], color='gray',
                                      linestyle='None', markersize=ms,
                                      markeredgewidth=0,
                                      label=technique_labels[tech]))

# Create first legend for models (top row) - positioned to overlap with x-axis title
leg1_1x1 = fig.legend(handles=model_elements_1x1,
          title='Models',
          title_fontsize=22,
          loc='lower center',
          bbox_to_anchor=(0.5, 0.01),
          ncol=len(model_elements_1x1),
          frameon=False,
          fancybox=False,
          shadow=False,
          fontsize=20,
          columnspacing=1.5,
          handletextpad=0.5)

# Add second legend for techniques (bottom row)
leg2_1x1 = fig.legend(handles=technique_elements_1x1,
              title='Techniques',
              title_fontsize=22,
              loc='lower center',
              bbox_to_anchor=(0.5, -0.07),
              ncol=len(technique_elements_1x1),
              frameon=False,
              fancybox=False,
              shadow=False,
              fontsize=20,
              columnspacing=1.5,
              handletextpad=0.5)

# Add first legend back as artist (since second legend call replaces it)
fig.add_artist(leg1_1x1)

plt.tight_layout(rect=[0, 0.12, 1, 0.98])

output_path = os.path.join(output_dir, 'averages_comparison_1x1_all_techniques.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  Saved: {output_path}")

output_path_pdf = os.path.join(output_dir, 'averages_comparison_1x1_all_techniques.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"  Saved: {output_path_pdf}")

# Display plot inline (useful for Jupyter notebooks)
try:
    from IPython.display import display, Image as IPImage

    print("\nDisplaying 1x1 all-techniques plot:")
    display(IPImage(filename=output_path))
except ImportError:
    # Not in a Jupyter environment, skip display
    print("\nNote: Run in Jupyter notebook to see inline plot preview.")

plt.close()

# ============================================================================
# PLOT 6: 1x1 Layout - Model-Averaged Results (Techniques as Colors)
# ============================================================================
print("Creating 1x1 model-averaged comparison plot...")

# Define color palette for techniques (instead of models)
technique_palette = {
    'baseline': '#636EFA',      # Blue
    'INT4': '#EF553B',          # Red
    'INT8': '#00CC96',          # Green
    'pyramidkv': '#AB63FA',     # Purple
    'snapkv': '#FFA15A',        # Orange
    'streamingllm': '#19D3F3',  # Cyan
    'duoattn': '#FF6692',       # Pink
}

# Create single subplot with larger figure for better readability
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
# fig.suptitle('Performance vs Memory Usage Averaged Across HELMET (16K/32K) and LongProc (500/2K) Benchmarks',
#              fontsize=24, fontweight='bold', y=0.98)

# All techniques combined
all_techniques_averaged = sorted(set(quantization_techniques + kv_techniques))

# Plot with model-averaged data
def plot_model_averaged_techniques(ax, techniques, memory_helmet, perf_helmet,
                                    memory_longproc, perf_longproc):
    """Plot technique comparison with results averaged across all models"""

    # For each technique, aggregate across all models
    for technique in techniques:
        # Dictionary to store {cache_size: {memory: [], performance: []}}
        cache_aggregates = {}

        # Get all data for this technique across all models
        for model in filtered_models_1x1:
            helmet_subset = memory_helmet[
                (memory_helmet['context_length'] == HELMET_CONTEXT) &
                (memory_helmet['technique'] == technique) &
                (memory_helmet['model'] == model)
            ]

            for _, row in helmet_subset.iterrows():
                cache_size = row['cache_size']

                # Get memory values from all context lengths
                helmet_mem_16k = get_memory_value(memory_helmet, technique, model, cache_size, HELMET_CONTEXT, helmet_tasks)
                helmet_mem_32k = get_memory_value(memory_helmet, technique, model, cache_size, HELMET_CONTEXT_32K, helmet_tasks)
                longproc_mem_2k = get_memory_value(memory_longproc, technique, model, cache_size, LONGPROC_CONTEXT, longproc_tasks)
                longproc_mem_5k = get_memory_value(memory_longproc, technique, model, cache_size, LONGPROC_CONTEXT_5K, longproc_tasks)

                # Average memory across all available benchmarks/contexts
                mem_values = [m for m in [helmet_mem_16k, helmet_mem_32k, longproc_mem_2k, longproc_mem_5k] if m is not None]
                if len(mem_values) > 0:
                    avg_mem = np.mean(mem_values)
                else:
                    continue

                # Get performance values from all context lengths
                all_perf_values = []

                # HELMET 16K tasks
                helmet_perf_row_16k = perf_helmet[
                    (perf_helmet['technique'] == technique) &
                    (perf_helmet['context_length'] == HELMET_CONTEXT) &
                    (perf_helmet['model'] == model) &
                    (perf_helmet['cache_size'] == cache_size)
                ]
                if not helmet_perf_row_16k.empty:
                    for task in helmet_tasks:
                        val = helmet_perf_row_16k.iloc[0][task]
                        if not pd.isna(val) and val != 0:
                            all_perf_values.append(val)

                # HELMET 32K tasks
                helmet_perf_row_32k = perf_helmet[
                    (perf_helmet['technique'] == technique) &
                    (perf_helmet['context_length'] == HELMET_CONTEXT_32K) &
                    (perf_helmet['model'] == model) &
                    (perf_helmet['cache_size'] == cache_size)
                ]
                if not helmet_perf_row_32k.empty:
                    for task in helmet_tasks:
                        val = helmet_perf_row_32k.iloc[0][task]
                        if not pd.isna(val) and val != 0:
                            all_perf_values.append(val)

                # LongProc 5K tasks
                longproc_perf_row_5k = perf_longproc[
                    (perf_longproc['technique'] == technique) &
                    (perf_longproc['context_length'] == LONGPROC_CONTEXT_5K) &
                    (perf_longproc['model'] == model) &
                    (perf_longproc['cache_size'] == cache_size)
                ]
                if not longproc_perf_row_5k.empty:
                    for task in longproc_tasks:
                        val = longproc_perf_row_5k.iloc[0][task]
                        if not pd.isna(val) and val != 0:
                            all_perf_values.append(val)

                # LongProc 2K tasks
                longproc_perf_row_2k = perf_longproc[
                    (perf_longproc['technique'] == technique) &
                    (perf_longproc['context_length'] == LONGPROC_CONTEXT) &
                    (perf_longproc['model'] == model) &
                    (perf_longproc['cache_size'] == cache_size)
                ]
                if not longproc_perf_row_2k.empty:
                    for task in longproc_tasks:
                        val = longproc_perf_row_2k.iloc[0][task]
                        if not pd.isna(val) and val != 0:
                            all_perf_values.append(val)

                if len(all_perf_values) == 0:
                    continue

                avg_perf = np.mean(all_perf_values)

                # Aggregate by cache_size
                if cache_size not in cache_aggregates:
                    cache_aggregates[cache_size] = {'memory': [], 'performance': []}

                cache_aggregates[cache_size]['memory'].append(avg_mem)
                cache_aggregates[cache_size]['performance'].append(avg_perf)

        # Now compute averages across models for each cache_size
        x_values = []
        y_values = []
        labels = []

        # Sort cache sizes (same logic as before)
        sorted_cache_sizes = sorted(cache_aggregates.keys())

        if technique == "streamingllm":
            sorted_cache_sizes = sorted(cache_aggregates.keys(),
                key=lambda x: int(x.split('_')[2]) if x.startswith('n_local_') else 0)
        elif technique in ["snapkv", "pyramidkv"]:
            def extract_cache_and_k(cache_size):
                if not cache_size.startswith('w'):
                    return (0, 0, 0)
                parts = cache_size.split('_')
                cache_val = int(parts[1][1:])
                k_val = int(parts[2][1:])
                return (cache_val, k_val, 0)
            sorted_cache_sizes = sorted(cache_aggregates.keys(), key=extract_cache_and_k)

        # Special handling for StreamingLLM, SnapKV, PyramidKV: merge configs if multiple
        if (technique == 'streamingllm' or technique in ['snapkv', 'pyramidkv']) and len(sorted_cache_sizes) > 1:
            all_mems = []
            all_perfs = []
            for cache_size in sorted_cache_sizes:
                all_mems.extend(cache_aggregates[cache_size]['memory'])
                all_perfs.extend(cache_aggregates[cache_size]['performance'])

            if len(all_mems) > 0 and len(all_perfs) > 0:
                x_values.append(np.mean(all_mems))
                y_values.append(np.mean(all_perfs))
                labels.append("")
        else:
            for cache_size in sorted_cache_sizes:
                mems = cache_aggregates[cache_size]['memory']
                perfs = cache_aggregates[cache_size]['performance']

                if len(mems) > 0 and len(perfs) > 0:
                    x_values.append(np.mean(mems))
                    y_values.append(np.mean(perfs))
                    labels.append(format_cache_size(cache_size))

        # Plot the technique
        if len(x_values) > 0:
            ax.scatter(x_values, y_values,
                      color=technique_palette[technique],
                      marker=marker_dict[technique],
                      s=marker_size_dict[marker_dict[technique]],
                      alpha=0.8,
                      zorder=3,
                      label=technique_labels[technique])

            # Connect points with lines if multiple cache sizes (except StreamingLLM, SnapKV, PyramidKV)
            if len(x_values) > 1 and technique not in ['streamingllm', 'snapkv', 'pyramidkv']:
                ax.plot(x_values, y_values,
                       color=technique_palette[technique],
                       linestyle='--',
                       alpha=0.4,
                       linewidth=1.5,
                       zorder=2)

                # Add label to the last point
                if labels[-1]:
                    ax.annotate(
                        labels[-1],
                        (x_values[-1], y_values[-1]),
                        xytext=(8, 8),
                        textcoords='offset points',
                        fontsize=14,
                        alpha=0.7,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='none', alpha=0.6)
                    )

    ax.set_xlabel('Memory Usage (GB)', fontweight='bold', fontsize=24)
    ax.set_ylabel('Average Performance Score', fontweight='bold', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

plot_model_averaged_techniques(ax, all_techniques_averaged,
                                helmet_memory_1x1, helmet_performance_1x1,
                                longproc_memory_1x1, longproc_performance_1x1)

# Set y-axis to start at 25 for better visualization
ylim = ax.get_ylim()
ax.set_ylim(25, ylim[1] * 1.05)

# Create legend with only techniques (no models row)
technique_elements_avg = []
for tech in legend_order:
    if tech in marker_dict and tech in technique_palette and tech in all_techniques_averaged:
        # Use larger marker size for StreamingLLM star
        ms = 20 if tech == 'streamingllm' else 16
        technique_elements_avg.append(Line2D([0], [0],
                                      marker=marker_dict[tech],
                                      color='w',
                                      markerfacecolor=technique_palette[tech],
                                      markersize=ms,
                                      markeredgewidth=0,
                                      label=technique_labels[tech]))

fig.legend(handles=technique_elements_avg,
          title='Techniques (Averaged Across All Models)',
          title_fontsize=20,
          loc='lower center',
          bbox_to_anchor=(0.5, 0.00),
          ncol=len(technique_elements_avg),
          frameon=False,
          fancybox=False,
          shadow=False,
          fontsize=18,
          columnspacing=1.5,
          handletextpad=0.5)

plt.tight_layout(rect=[0, 0.08, 1, 0.98])

output_path = os.path.join(output_dir, 'averages_comparison_1x1_model_averaged.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  Saved: {output_path}")

output_path_pdf = os.path.join(output_dir, 'averages_comparison_1x1_model_averaged.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"  Saved: {output_path_pdf}")

# Display plot inline (useful for Jupyter notebooks)
try:
    from IPython.display import display, Image as IPImage

    print("\nDisplaying 1x1 model-averaged plot:")
    display(IPImage(filename=output_path))
except ImportError:
    # Not in a Jupyter environment, skip display
    print("\nNote: Run in Jupyter notebook to see inline plot preview.")

plt.close()

# ============================================================================
# PLOT 7: 1x1 Layout - All Techniques Combined WITH Connection Lines
# ============================================================================
print("Creating 1x1 all-techniques comparison plot with connection lines...")

# Define technique connection order
technique_connection_order = ['INT4', 'INT8', 'baseline', 'snapkv', 'pyramidkv', 'streamingllm']

# Create single subplot with larger figure for better readability
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
# fig.suptitle('Performance vs Memory Usage\nAveraged Across HELMET (16K/32K) and LongProc (500/2K) Benchmarks',
#              fontsize=24, fontweight='bold', y=0.98)

# Modified plot function with connection lines
def plot_all_techniques_with_connections(ax, techniques, memory_helmet, perf_helmet,
                                         memory_longproc, perf_longproc, technique_order):
    """Plot combined HELMET + LongProc data with all techniques and connection lines"""
    # Get all unique (technique, model, cache_size) combinations from HELMET
    helmet_subset = memory_helmet[memory_helmet['context_length'] == HELMET_CONTEXT]
    
    # Store points for each model/technique to enable connecting them
    model_points = {}  # {model: {technique: (x, y)}}

    # Process each technique/model combination
    for (technique, model) in helmet_subset.groupby(['technique', 'model']).groups.keys():
        if technique not in techniques or model not in model_palette or technique not in marker_dict:
            continue
        
        # Initialize model_points if needed
        if model not in model_points:
            model_points[model] = {}

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

        # Special handling for StreamingLLM, SnapKV, PyramidKV: average across cache configs
        if (technique == 'streamingllm' or technique in ['snapkv', 'pyramidkv']) and len(cache_sizes) > 1:
            # Collect memory across configs and contexts, and collect per-task values to average per task
            all_mem_values = []
            task_to_vals = {}

            for cache_size in cache_sizes:
                # Memory (HELMET 16k)
                mem = get_memory_value(memory_helmet, technique, model, cache_size, HELMET_CONTEXT, helmet_tasks)
                if mem is not None:
                    all_mem_values.append(mem)

                # Memory (LongProc 2k)
                mem = get_memory_value(memory_longproc, technique, model, cache_size, LONGPROC_CONTEXT, longproc_tasks)
                if mem is not None:
                    all_mem_values.append(mem)

                # HELMET task performances (16k)
                helmet_row = perf_helmet[
                    (perf_helmet['technique'] == technique) &
                    (perf_helmet['context_length'] == HELMET_CONTEXT) &
                    (perf_helmet['model'] == model) &
                    (perf_helmet['cache_size'] == cache_size)
                ]
                if not helmet_row.empty:
                    for task in helmet_tasks:
                        val = helmet_row.iloc[0][task]
                        if not pd.isna(val) and val != 0:
                            task_to_vals.setdefault(task, []).append(val)

                # LongProc task performances (2k)
                longproc_row = perf_longproc[
                    (perf_longproc['technique'] == technique) &
                    (perf_longproc['context_length'] == LONGPROC_CONTEXT) &
                    (perf_longproc['model'] == model) &
                    (perf_longproc['cache_size'] == cache_size)
                ]
                if not longproc_row.empty:
                    for task in longproc_tasks:
                        val = longproc_row.iloc[0][task]
                        if not pd.isna(val) and val != 0:
                            task_to_vals.setdefault(task, []).append(val)

            # Compute per-task means, then overall mean performance
            per_task_means = [np.mean(vs) for vs in task_to_vals.values() if len(vs) > 0]

            if len(all_mem_values) > 0 and len(per_task_means) > 0:
                x_values.append(np.mean(all_mem_values))
                y_values.append(np.mean(per_task_means))
                # Store the point for connecting
                if len(x_values) > 0:
                    model_points[model][technique] = (x_values[0], y_values[0])
                # Use technique-specific label
                if technique == 'streamingllm':
                    labels.append("StreamingLLM")
                elif technique == 'snapkv':
                    labels.append("SnapKV")
                elif technique == 'pyramidkv':
                    labels.append("PyramidKV")
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

        # Plot points (but don't connect with lines yet)
        if len(x_values) > 0:
            # Store the representative point for each technique (use first point)
            # For connection lines, we want one point per technique-model combination
            if technique not in model_points[model]:
                # Store the first point for this technique
                model_points[model][technique] = (x_values[0], y_values[0])
            
            ax.scatter(x_values, y_values, color=model_palette[model], marker=marker_dict[technique],
                      s=marker_size_dict[marker_dict[technique]], alpha=0.8, zorder=3)

    # Now connect points for each model in the specified order
    for model in model_points.keys():
        points_to_connect = []
        for tech in technique_order:
            if tech in model_points[model]:
                x, y = model_points[model][tech]
                points_to_connect.append((x, y))
        
        # Draw dashed line connecting the points
        if len(points_to_connect) > 1:
            x_coords = [p[0] for p in points_to_connect]
            y_coords = [p[1] for p in points_to_connect]
            ax.plot(x_coords, y_coords, color=model_palette[model], linestyle='--',
                   alpha=0.4, linewidth=3.0, zorder=2)

    ax.set_xlabel('Memory Usage (GB)', fontweight='bold', fontsize=24)
    ax.set_ylabel('Average Performance Score', fontweight='bold', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

plot_all_techniques_with_connections(ax, all_techniques_combined_no_duo,
                                      helmet_memory_1x1, helmet_performance_1x1,
                                      longproc_memory_1x1, longproc_performance_1x1,
                                      technique_connection_order)

# Set y-axis to start at 0
ylim = ax.get_ylim()
ax.set_ylim(0, ylim[1] * 1.05)

# Create legend in horizontal layout at bottom - two separate rows
model_elements_7 = []
for model in filtered_models_1x1:
    if model in model_palette:
        display_name = model_display_names.get(model, model)
        model_elements_7.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=model_palette[model], markersize=16,
                                  markeredgewidth=0,
                                  label=display_name))

technique_elements_7 = []
for tech in legend_order_no_duo:
    if tech in marker_dict and tech in all_techniques_combined_no_duo:
        # Use larger marker size for StreamingLLM star
        ms = 20 if tech == 'streamingllm' else 16
        technique_elements_7.append(Line2D([0], [0], marker=marker_dict[tech], color='gray',
                                      linestyle='None', markersize=ms,
                                      markeredgewidth=0,
                                      label=technique_labels[tech]))

# Create first legend for models (top row) - positioned to overlap with x-axis title
leg1_7 = fig.legend(handles=model_elements_7,
          title='Models',
          title_fontsize=22,
          loc='lower center',
          bbox_to_anchor=(0.5, 0.01),
          ncol=len(model_elements_7),
          frameon=False,
          fancybox=False,
          shadow=False,
          fontsize=20,
          columnspacing=1.5,
          handletextpad=0.5)

# Add second legend for techniques (bottom row)
leg2_7 = fig.legend(handles=technique_elements_7,
              title='Techniques',
              title_fontsize=22,
              loc='lower center',
              bbox_to_anchor=(0.5, -0.07),
              ncol=len(technique_elements_7),
              frameon=False,
              fancybox=False,
              shadow=False,
              fontsize=20,
              columnspacing=1.5,
              handletextpad=0.5)

# Add first legend back as artist (since second legend call replaces it)
fig.add_artist(leg1_7)

plt.tight_layout(rect=[0, 0.12, 1, 0.98])

output_path = os.path.join(output_dir, 'averages_comparison_1x1_all_techniques_with_connections.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  Saved: {output_path}")

output_path_pdf = os.path.join(output_dir, 'averages_comparison_1x1_all_techniques_with_connections.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"  Saved: {output_path_pdf}")

# Display plot inline (useful for Jupyter notebooks)
try:
    from IPython.display import display, Image as IPImage

    print("\nDisplaying 1x1 all-techniques with connections plot:")
    display(IPImage(filename=output_path))
except ImportError:
    # Not in a Jupyter environment, skip display
    print("\nNote: Run in Jupyter notebook to see inline plot preview.")

plt.close()

# ============================================================================
# PLOT 8: 1x1 Layout - All Techniques WITH Connection Lines (Including DuoAttn)
# Using pre-computed averaged data from CSV
# ============================================================================
print("Creating 1x1 all-techniques comparison plot with connection lines including DuoAttn...")
print("  Using pre-computed averaged cache size data from CSV...")

# Define technique connection order (with duoattn between INT8 and baseline)
technique_connection_order_with_duo = ['INT4', 'INT8', 'duoattn', 'baseline', 'snapkv', 'pyramidkv', 'streamingllm']

# Load the pre-computed averaged data
df_averaged_plot = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/plot_data_averaged_across_cache_sizes.csv')

# Create single subplot with larger figure for better readability
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
# fig.suptitle('Performance vs Memory Usage\nAveraged Across HELMET (16K/32K) and LongProc (500/2K) Benchmarks',
#              fontsize=24, fontweight='bold', y=0.98)

# Store points for each model/technique to enable connecting them
model_points_plot8 = {}  # {model: {technique: (x, y)}}

# Plot each point from the CSV
for _, row in df_averaged_plot.iterrows():
    model = row['model']
    technique = row['technique']
    x = row['avg_memory_gb']
    y = row['avg_performance_score']

    # Skip if model or technique not in our palettes
    if model not in model_palette or technique not in marker_dict:
        continue

    # Initialize model_points if needed
    if model not in model_points_plot8:
        model_points_plot8[model] = {}

    # Store the point for connecting
    model_points_plot8[model][technique] = (x, y)

    # Plot the point
    ax.scatter(
        [x], [y],
        color=model_palette[model],
        marker=marker_dict[technique],
        s=marker_size_dict[marker_dict[technique]],
        alpha=0.8,
        zorder=3
    )

# Now connect points for each model in the specified order
for model in model_points_plot8.keys():
    points_to_connect = []
    for tech in technique_connection_order_with_duo:
        if tech in model_points_plot8[model]:
            x, y = model_points_plot8[model][tech]
            points_to_connect.append((x, y))

    # Draw dashed line connecting the points
    if len(points_to_connect) > 1:
        x_coords = [p[0] for p in points_to_connect]
        y_coords = [p[1] for p in points_to_connect]
        ax.plot(x_coords, y_coords, color=model_palette[model], linestyle='--',
               alpha=0.4, linewidth=3.0, zorder=2)

ax.set_xlabel('Memory Usage (GB)', fontweight='bold', fontsize=24)
ax.set_ylabel('Average Performance Score', fontweight='bold', fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Set y-axis to start at 0
ylim = ax.get_ylim()
ax.set_ylim(0, ylim[1] * 1.05)

# Create legend in horizontal layout at bottom - two separate rows
model_elements_8 = []
for model in filtered_models_1x1:
    if model in model_palette:
        display_name = model_display_names.get(model, model)
        model_elements_8.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=model_palette[model], markersize=16,
                                  markeredgewidth=0,
                                  label=display_name))

technique_elements_8 = []
for tech in legend_order:
    if tech in marker_dict and tech in all_techniques_combined:
        # Use larger marker size for StreamingLLM star
        ms = 20 if tech == 'streamingllm' else 16
        technique_elements_8.append(Line2D([0], [0], marker=marker_dict[tech], color='gray',
                                      linestyle='None', markersize=ms,
                                      markeredgewidth=0,
                                      label=technique_labels[tech]))

# Create first legend for models (top row) - positioned to overlap with x-axis title
leg1_8 = fig.legend(handles=model_elements_8,
          title='Models',
          title_fontsize=22,
          loc='lower center',
          bbox_to_anchor=(0.5, 0.01),
          ncol=len(model_elements_8),
          frameon=False,
          fancybox=False,
          shadow=False,
          fontsize=20,
          columnspacing=1.5,
          handletextpad=0.5)

# Add second legend for techniques (bottom row)
leg2_8 = fig.legend(handles=technique_elements_8,
              title='Techniques',
              title_fontsize=22,
              loc='lower center',
              bbox_to_anchor=(0.5, -0.07),
              ncol=len(technique_elements_8),
              frameon=False,
              fancybox=False,
              shadow=False,
              fontsize=20,
              columnspacing=1.5,
              handletextpad=0.5)

# Add first legend back as artist (since second legend call replaces it)
fig.add_artist(leg1_8)

plt.tight_layout(rect=[0, 0.12, 1, 0.98])

output_path = os.path.join(output_dir, 'averages_comparison_1x1_all_techniques_with_connections_incl_duo.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  Saved: {output_path}")

output_path_pdf = os.path.join(output_dir, 'averages_comparison_1x1_all_techniques_with_connections_incl_duo.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"  Saved: {output_path_pdf}")

# Display plot inline (useful for Jupyter notebooks)
try:
    from IPython.display import display, Image as IPImage

    print("\nDisplaying 1x1 all-techniques with connections incl duo plot:")
    display(IPImage(filename=output_path))
except ImportError:
    # Not in a Jupyter environment, skip display
    print("\nNote: Run in Jupyter notebook to see inline plot preview.")

plt.close()

print("\nAll comparison plots created successfully!")
