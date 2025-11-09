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

# Calculate statistics for each technique
techniques = ['baseline', 'INT4', 'INT8', 'duoattn', 'snapkv', 'pyramidkv', 'streamingllm']

results = {}

for technique in techniques:
    technique_results = {'memory': [], 'performance': []}

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
                technique_results['memory'].append(avg_mem)

            # Average performance across all tasks
            perf_values = []
            if helmet_perf is not None:
                perf_values.append(helmet_perf)
            if longproc_perf is not None:
                perf_values.append(longproc_perf)

            if len(perf_values) > 0:
                avg_perf = np.mean(perf_values)
                technique_results['performance'].append(avg_perf)

    results[technique] = technique_results

# Calculate baseline statistics
baseline_mem = np.mean(results['baseline']['memory'])
baseline_perf = np.mean(results['baseline']['performance'])

print(f"\n{'='*80}")
print(f"BASELINE STATISTICS")
print(f"{'='*80}")
print(f"Baseline Average Memory: {baseline_mem:.2f} GB")
print(f"Baseline Average Performance: {baseline_perf:.2f}")

# Calculate statistics for each technique vs baseline
print(f"\n{'='*80}")
print(f"TECHNIQUE COMPARISONS (vs Baseline)")
print(f"{'='*80}\n")

for technique in ['INT4', 'INT8', 'duoattn', 'snapkv', 'pyramidkv', 'streamingllm']:
    if technique not in results or len(results[technique]['memory']) == 0:
        continue

    tech_mems = results[technique]['memory']
    tech_perfs = results[technique]['performance']

    # Calculate memory change
    mem_changes = [(m - baseline_mem) / baseline_mem * 100 for m in tech_mems]
    avg_mem_change = np.mean(mem_changes)
    std_mem_change = np.std(mem_changes)

    # Calculate performance change
    perf_changes = [(p - baseline_perf) / baseline_perf * 100 for p in tech_perfs]
    avg_perf_change = np.mean(perf_changes)
    std_perf_change = np.std(perf_changes)

    print(f"{technique.upper()}:")
    print(f"  Memory Change: {avg_mem_change:+.2f}% ± {std_mem_change:.2f}%")
    print(f"  Performance Change: {avg_perf_change:+.2f}% ± {std_perf_change:.2f}%")
    print(f"  Avg Memory: {np.mean(tech_mems):.2f} GB (n={len(tech_mems)})")
    print(f"  Avg Performance: {np.mean(tech_perfs):.2f} (n={len(tech_perfs)})")

    # For token eviction methods, show min and max cache configurations
    if technique in ['snapkv', 'pyramidkv', 'streamingllm']:
        min_mem_idx = np.argmin(tech_mems)
        max_mem_idx = np.argmax(tech_mems)

        print(f"  Small cache config:")
        print(f"    Memory: {tech_mems[min_mem_idx]:.2f} GB ({mem_changes[min_mem_idx]:+.2f}%)")
        print(f"    Performance: {tech_perfs[min_mem_idx]:.2f} ({perf_changes[min_mem_idx]:+.2f}%)")

        print(f"  Large cache config:")
        print(f"    Memory: {tech_mems[max_mem_idx]:.2f} GB ({mem_changes[max_mem_idx]:+.2f}%)")
        print(f"    Performance: {tech_perfs[max_mem_idx]:.2f} ({perf_changes[max_mem_idx]:+.2f}%)")

        # Calculate change from small to large cache
        mem_diff_pct = (tech_mems[max_mem_idx] - tech_mems[min_mem_idx]) / tech_mems[min_mem_idx] * 100
        perf_diff_pct = (tech_perfs[max_mem_idx] - tech_perfs[min_mem_idx]) / tech_perfs[min_mem_idx] * 100

        print(f"  Small→Large cache change:")
        print(f"    Memory increase: +{mem_diff_pct:.2f}%")
        print(f"    Performance improvement: +{perf_diff_pct:.2f}%")

    print()

# Calculate aggregated stats for token eviction methods
print(f"{'='*80}")
print(f"TOKEN EVICTION METHODS (SnapKV, PyramidKV, StreamingLLM) AGGREGATED")
print(f"{'='*80}\n")

kv_mems = []
kv_perfs = []
kv_mem_changes = []
kv_perf_changes = []

for technique in ['snapkv', 'pyramidkv', 'streamingllm']:
    if technique in results:
        kv_mems.extend(results[technique]['memory'])
        kv_perfs.extend(results[technique]['performance'])
        kv_mem_changes.extend([(m - baseline_mem) / baseline_mem * 100 for m in results[technique]['memory']])
        kv_perf_changes.extend([(p - baseline_perf) / baseline_perf * 100 for p in results[technique]['performance']])

# Split into small and large cache configs
sorted_indices = np.argsort(kv_mems)
half_point = len(sorted_indices) // 2

small_cache_mems = [kv_mems[i] for i in sorted_indices[:half_point]]
small_cache_perfs = [kv_perfs[i] for i in sorted_indices[:half_point]]
small_cache_mem_changes = [kv_mem_changes[i] for i in sorted_indices[:half_point]]
small_cache_perf_changes = [kv_perf_changes[i] for i in sorted_indices[:half_point]]

large_cache_mems = [kv_mems[i] for i in sorted_indices[half_point:]]
large_cache_perfs = [kv_perfs[i] for i in sorted_indices[half_point:]]
large_cache_mem_changes = [kv_mem_changes[i] for i in sorted_indices[half_point:]]
large_cache_perf_changes = [kv_perf_changes[i] for i in sorted_indices[half_point:]]

print(f"SMALL CACHE CONFIGURATIONS:")
print(f"  Memory Change: {np.mean(small_cache_mem_changes):+.2f}% ± {np.std(small_cache_mem_changes):.2f}%")
print(f"  Performance Change: {np.mean(small_cache_perf_changes):+.2f}% ± {np.std(small_cache_perf_changes):.2f}%")

print(f"\nLARGE CACHE CONFIGURATIONS:")
print(f"  Memory Change: {np.mean(large_cache_mem_changes):+.2f}% ± {np.std(large_cache_mem_changes):.2f}%")
print(f"  Performance Change: {np.mean(large_cache_perf_changes):+.2f}% ± {np.std(large_cache_perf_changes):.2f}%")

# Calculate improvement from small to large
avg_small_mem = np.mean(small_cache_mems)
avg_large_mem = np.mean(large_cache_mems)
avg_small_perf = np.mean(small_cache_perfs)
avg_large_perf = np.mean(large_cache_perfs)

mem_increase = (avg_large_mem - avg_small_mem) / avg_small_mem * 100
perf_increase = (avg_large_perf - avg_small_perf) / avg_small_perf * 100

# Calculate vs baseline
mem_increase_vs_baseline = (avg_large_mem - baseline_mem) / baseline_mem * 100
perf_increase_vs_baseline = (avg_large_perf - baseline_perf) / baseline_perf * 100

print(f"\nSMALL→LARGE CACHE CHANGE:")
print(f"  Memory increase: +{mem_increase:.2f}%")
print(f"  Performance improvement: +{perf_increase:.2f}%")

print(f"\nLARGE CACHE vs BASELINE:")
print(f"  Memory change: {mem_increase_vs_baseline:+.2f}%")
print(f"  Performance change: {perf_increase_vs_baseline:+.2f}%")

print(f"\n{'='*80}")
