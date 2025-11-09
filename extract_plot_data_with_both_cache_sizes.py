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

# IMPORTANT: DO NOT FILTER OUT w2048 configs - we need both small and large cache sizes!
print("NOTE: Keeping BOTH w256 and w2048 cache configurations for SnapKV/PyramidKV")

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

# Store all data points
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
                # Classify cache size
                cache_type = 'default'
                if 'w256' in cache_size or 'c2048' in cache_size:
                    cache_type = 'small'
                elif 'w2048' in cache_size or 'c8192' in cache_size:
                    cache_type = 'large'
                elif 'n_local' in cache_size:
                    cache_type = 'streamingllm'
                elif 'sp0' in cache_size:
                    cache_type = 'duoattn'

                plot_data.append({
                    'model': model,
                    'technique': technique,
                    'cache_size': cache_size,
                    'cache_type': cache_type,
                    'memory_gb': round(avg_mem, 3),
                    'performance_score': round(avg_perf, 3),
                    'helmet_memory_gb': round(helmet_mem, 3) if helmet_mem is not None else None,
                    'helmet_performance': round(helmet_perf, 3) if helmet_perf is not None else None,
                    'longproc_memory_gb': round(longproc_mem, 3) if longproc_mem is not None else None,
                    'longproc_performance': round(longproc_perf, 3) if longproc_perf is not None else None,
                })

# Create DataFrame
df_plot = pd.DataFrame(plot_data)

# Sort by model, then technique order
technique_order = ['INT4', 'INT8', 'duoattn', 'baseline', 'snapkv', 'pyramidkv', 'streamingllm']
df_plot['technique_order'] = df_plot['technique'].apply(lambda x: technique_order.index(x) if x in technique_order else 999)
df_plot = df_plot.sort_values(['model', 'technique_order', 'cache_type', 'memory_gb'])
df_plot = df_plot.drop('technique_order', axis=1)

# Save full data
output_path = '/scratch/gpfs/DANQIC/jz4391/HELMET/results/plot_data_all_cache_sizes.csv'
df_plot.to_csv(output_path, index=False)

print(f"Full plot data saved to: {output_path}")
print(f"\nTotal data points: {len(df_plot)}")

# Show breakdown
print(f"\n{'='*80}")
print("DATA BREAKDOWN BY TECHNIQUE AND CACHE SIZE")
print(f"{'='*80}")
for tech in techniques:
    tech_data = df_plot[df_plot['technique'] == tech]
    if len(tech_data) > 0:
        print(f"\n{tech.upper()}:")
        cache_counts = tech_data.groupby('cache_type').size()
        for cache_type, count in cache_counts.items():
            print(f"  {cache_type}: {count} configurations")

# Create averaged data (average across cache sizes for each model-technique)
averaged_data = []

for technique in techniques:
    for model in filtered_models:
        model_tech_data = df_plot[(df_plot['technique'] == technique) & (df_plot['model'] == model)]

        if len(model_tech_data) == 0:
            continue

        # Average across all cache configurations for this model-technique
        avg_mem = model_tech_data['memory_gb'].mean()
        avg_perf = model_tech_data['performance_score'].mean()
        num_configs = len(model_tech_data)
        cache_sizes_list = ', '.join(model_tech_data['cache_size'].values)

        averaged_data.append({
            'model': model,
            'technique': technique,
            'num_cache_configs': num_configs,
            'cache_sizes': cache_sizes_list,
            'avg_memory_gb': round(avg_mem, 3),
            'avg_performance_score': round(avg_perf, 3),
            'min_memory_gb': round(model_tech_data['memory_gb'].min(), 3),
            'max_memory_gb': round(model_tech_data['memory_gb'].max(), 3),
            'min_performance': round(model_tech_data['performance_score'].min(), 3),
            'max_performance': round(model_tech_data['performance_score'].max(), 3),
        })

df_averaged = pd.DataFrame(averaged_data)
df_averaged['technique_order'] = df_averaged['technique'].apply(lambda x: technique_order.index(x) if x in technique_order else 999)
df_averaged = df_averaged.sort_values(['model', 'technique_order'])
df_averaged = df_averaged.drop('technique_order', axis=1)

output_averaged_path = '/scratch/gpfs/DANQIC/jz4391/HELMET/results/plot_data_averaged_across_cache_sizes.csv'
df_averaged.to_csv(output_averaged_path, index=False)

print(f"\n{'='*80}")
print(f"Averaged data (across cache sizes) saved to: {output_averaged_path}")
print(f"\n{'='*80}")
print("PREVIEW OF AVERAGED DATA:")
print(df_averaged.to_string())

# Show specific info about SnapKV and PyramidKV
print(f"\n{'='*80}")
print("SNAPKV AND PYRAMIDKV CACHE SIZE DETAILS")
print(f"{'='*80}")

for tech in ['snapkv', 'pyramidkv']:
    print(f"\n{tech.upper()}:")
    tech_data = df_plot[df_plot['technique'] == tech]
    for model in filtered_models:
        model_data = tech_data[tech_data['model'] == model]
        if len(model_data) > 0:
            print(f"\n  {model}:")
            for _, row in model_data.iterrows():
                print(f"    {row['cache_size']:30s} ({row['cache_type']:10s}): {row['memory_gb']:6.2f} GB, {row['performance_score']:6.2f} score")
