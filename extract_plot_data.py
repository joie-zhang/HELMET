import pandas as pd
import numpy as np

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

# Filter to include only desired models
filtered_models = ['Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct',
                   'DeepSeek-R1-Distill-Llama-8B', 'DeepSeek-R1-Distill-Qwen-7B']

helmet_memory_df = helmet_memory_df[helmet_memory_df['model'].isin(filtered_models)].copy()
helmet_performance_df = helmet_performance_df[helmet_performance_df['model'].isin(filtered_models)].copy()
longproc_memory_df = longproc_memory_df[longproc_memory_df['model'].isin(filtered_models)].copy()
longproc_performance_df = longproc_performance_df[longproc_performance_df['model'].isin(filtered_models)].copy()

# Additional filtering: Remove w2048_c8192_k7 configurations for SnapKV and PyramidKV
w2048_configs = ['w2048_c8192_k7_avgpool', 'w2048_c8192_k7_maxpool']
for df in [helmet_memory_df, helmet_performance_df]:
    condition = (
        (df['technique'].isin(['snapkv', 'pyramidkv'])) &
        (df['cache_size'].isin(w2048_configs))
    )
    df.drop(df[condition].index, inplace=True)

for df in [longproc_memory_df, longproc_performance_df]:
    condition = (
        (df['technique'].isin(['snapkv', 'pyramidkv'])) &
        (df['cache_size'].isin(w2048_configs))
    )
    df.drop(df[condition].index, inplace=True)

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
LONGPROC_CONTEXT = '2k'

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

    mem_values = []
    for task in tasks:
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

    perf_values = []
    for task in tasks:
        val = perf_row.iloc[0][task]
        if not pd.isna(val) and val != 0:
            perf_values.append(val)

    if len(perf_values) == 0:
        return None

    return np.mean(perf_values)

# Store all data points that go into the plot
plot_data = []

techniques = ['baseline', 'INT4', 'INT8', 'duoattn', 'snapkv', 'pyramidkv', 'streamingllm']

for technique in techniques:
    for model in filtered_models:
        # Get all cache sizes for this technique-model combination
        helmet_subset = helmet_memory_df[
            (helmet_memory_df['technique'] == technique) &
            (helmet_memory_df['context_length'] == HELMET_CONTEXT) &
            (helmet_memory_df['model'] == model)
        ]

        for _, row in helmet_subset.iterrows():
            cache_size = row['cache_size']

            # Get HELMET metrics
            helmet_mem = get_memory_value(helmet_memory_df, technique, model, cache_size, HELMET_CONTEXT, helmet_tasks)
            helmet_perf = calculate_average_performance(helmet_performance_df, helmet_tasks, technique, model, cache_size, HELMET_CONTEXT)

            # Get LongProc metrics
            longproc_mem = get_memory_value(longproc_memory_df, technique, model, cache_size, LONGPROC_CONTEXT, longproc_tasks)
            longproc_perf = calculate_average_performance(longproc_performance_df, longproc_tasks, technique, model, cache_size, LONGPROC_CONTEXT)

            # Average memory across both benchmarks
            mem_values = [m for m in [helmet_mem, longproc_mem] if m is not None]
            if len(mem_values) > 0:
                avg_mem = np.mean(mem_values)
            else:
                avg_mem = None

            # Average performance across all tasks
            perf_values = []
            if helmet_perf is not None:
                perf_values.append(helmet_perf)
            if longproc_perf is not None:
                perf_values.append(longproc_perf)

            if len(perf_values) > 0:
                avg_perf = np.mean(perf_values)
            else:
                avg_perf = None

            # Only add if we have both memory and performance
            if avg_mem is not None and avg_perf is not None:
                plot_data.append({
                    'model': model,
                    'technique': technique,
                    'cache_size': cache_size,
                    'memory_gb': round(avg_mem, 3),
                    'performance_score': round(avg_perf, 3),
                    'helmet_memory_gb': round(helmet_mem, 3) if helmet_mem is not None else None,
                    'helmet_performance': round(helmet_perf, 3) if helmet_perf is not None else None,
                    'longproc_memory_gb': round(longproc_mem, 3) if longproc_mem is not None else None,
                    'longproc_performance': round(longproc_perf, 3) if longproc_perf is not None else None,
                })

# Create DataFrame and save to CSV
df_plot = pd.DataFrame(plot_data)

# Sort by model, then technique order
technique_order = ['INT4', 'INT8', 'duoattn', 'baseline', 'snapkv', 'pyramidkv', 'streamingllm']
df_plot['technique_order'] = df_plot['technique'].apply(lambda x: technique_order.index(x) if x in technique_order else 999)
df_plot = df_plot.sort_values(['model', 'technique_order', 'memory_gb'])
df_plot = df_plot.drop('technique_order', axis=1)

# Save to CSV
output_path = '/scratch/gpfs/DANQIC/jz4391/HELMET/results/plot_data_with_connections_incl_duo.csv'
df_plot.to_csv(output_path, index=False)

print(f"Plot data saved to: {output_path}")
print(f"\nTotal data points: {len(df_plot)}")
print(f"\nData preview:")
print(df_plot.head(20))

print(f"\n\nData points by technique:")
print(df_plot.groupby('technique').size())

print(f"\n\nData points by model:")
print(df_plot.groupby('model').size())

# Also create a summary table for the plot (one representative point per technique-model)
summary_data = []

for technique in techniques:
    for model in filtered_models:
        # Get all points for this technique-model combination
        subset = df_plot[(df_plot['technique'] == technique) & (df_plot['model'] == model)]

        if len(subset) == 0:
            continue

        # For techniques with multiple cache sizes, decide which point to use for the connection lines
        # According to the plot code, for streamingllm/snapkv/pyramidkv with multiple configs,
        # we average across all configs
        if technique in ['streamingllm', 'snapkv', 'pyramidkv'] and len(subset) > 1:
            # Average across all cache configs
            avg_mem = subset['memory_gb'].mean()
            avg_perf = subset['performance_score'].mean()
            cache_label = f"averaged_{len(subset)}_configs"
        else:
            # Use the single point or first point
            avg_mem = subset['memory_gb'].iloc[0]
            avg_perf = subset['performance_score'].iloc[0]
            cache_label = subset['cache_size'].iloc[0]

        summary_data.append({
            'model': model,
            'technique': technique,
            'cache_size_or_label': cache_label,
            'memory_gb': round(avg_mem, 3),
            'performance_score': round(avg_perf, 3),
            'num_configs_averaged': len(subset) if len(subset) > 1 else 1
        })

# Create summary DataFrame and save
df_summary = pd.DataFrame(summary_data)
df_summary['technique_order'] = df_summary['technique'].apply(lambda x: technique_order.index(x) if x in technique_order else 999)
df_summary = df_summary.sort_values(['model', 'technique_order'])
df_summary = df_summary.drop('technique_order', axis=1)

output_summary_path = '/scratch/gpfs/DANQIC/jz4391/HELMET/results/plot_data_summary_for_connections.csv'
df_summary.to_csv(output_summary_path, index=False)

print(f"\n\n{'='*80}")
print(f"Summary data (points used for connection lines) saved to: {output_summary_path}")
print(f"\nSummary data preview:")
print(df_summary)
