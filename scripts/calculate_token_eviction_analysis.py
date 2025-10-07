import pandas as pd
import numpy as np

# Load data
helmet_memory_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_memory_usage.csv')
helmet_performance_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_performance.csv')
longproc_memory_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/longproc_results/longproc_memory_usage.csv')
longproc_performance_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/longproc_results/longproc_performance.csv')

# Filter out unwanted techniques
unwanted_techniques = ['quest', 'streamingllm_original', 'minference']
for df in [helmet_memory_df, helmet_performance_df, longproc_memory_df, longproc_performance_df]:
    df.drop(df[df['technique'].isin(unwanted_techniques)].index, inplace=True)

# Filter out Qwen3 models
unwanted_models = ['Qwen3-8B', 'Yarn-Qwen3-8B']
for df in [helmet_memory_df, helmet_performance_df, longproc_memory_df, longproc_performance_df]:
    df.drop(df[df['model'].isin(unwanted_models)].index, inplace=True)

# Filter out specific unwanted cache size configurations
unwanted_configs = [
    ('pyramidkv', 'w32_c4096_k5_avgpool'),
    ('snapkv', 'w32_c4096_k5_avgpool'),
    ('streamingllm', 'n_local_3968_n_init_128'),
    ('duoattn', 'sp0.7_pf32768')
]

for technique, cache_size in unwanted_configs:
    for df in [helmet_memory_df, helmet_performance_df, longproc_memory_df, longproc_performance_df]:
        condition = (df['technique'] == technique) & (df['cache_size'] == cache_size)
        df.drop(df[condition].index, inplace=True)

# Filter SnapKV and PyramidKV to keep only specific configurations for reasoning models
allowed_reasoning_configs = ['w256_c2048_k7_avgpool', 'w2048_c8192_k7_avgpool', 'w256_c2048_k7_maxpool', 'w2048_c8192_k7_maxpool', 'default']
reasoning_models = ['DeepSeek-R1-Distill-Llama-8B', 'DeepSeek-R1-Distill-Qwen-7B']

for df in [helmet_memory_df, helmet_performance_df, longproc_memory_df, longproc_performance_df]:
    condition = (
        (df['model'].isin(reasoning_models)) &
        (df['technique'].isin(['snapkv', 'pyramidkv'])) &
        (~df['cache_size'].isin(allowed_reasoning_configs))
    )
    df.drop(df[condition].index, inplace=True)

# Filter out w32 and w1024 cache sizes for SnapKV/PyramidKV on baseline models
baseline_models = ['Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct']
for df in [helmet_memory_df, helmet_performance_df, longproc_memory_df, longproc_performance_df]:
    condition = (
        (df['model'].isin(baseline_models)) &
        (df['technique'].isin(['snapkv', 'pyramidkv'])) &
        (df['cache_size'].str.contains('w32_|w1024_', na=False))
    )
    df.drop(df[condition].index, inplace=True)

# Filter StreamingLLM to keep both n_local=4092 and n_local=4096 configurations
allowed_streamingllm_configs = ['n_local_4092_n_init_4', 'n_local_4096_n_init_4']
for df in [helmet_memory_df, helmet_performance_df, longproc_memory_df, longproc_performance_df]:
    condition = (
        (df['technique'] == 'streamingllm') &
        (~df['cache_size'].isin(allowed_streamingllm_configs))
    )
    df.drop(df[condition].index, inplace=True)

# Models to analyze
models = ['Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct',
          'DeepSeek-R1-Distill-Llama-8B', 'DeepSeek-R1-Distill-Qwen-7B']

# HELMET tasks
helmet_tasks = [
    'recall_jsonkv', 'rag_nq', 'rag_hotpotqa', 'rerank',
    'cite_str_em', 'cite_citation_rec', 'cite_citation_prec', 'niah',
    'icl_clinic', 'icl_banking'
]

# LongProc tasks
longproc_tasks = ['html_to_tsv', 'pseudo_to_code', 'travel_planning']

def get_average_memory(memory_df, technique, model, context, tasks):
    """Get average memory for a technique/model/context combination"""
    rows = memory_df[
        (memory_df['technique'] == technique) &
        (memory_df['context_length'] == context) &
        (memory_df['model'] == model)
    ]

    if rows.empty:
        return None

    # For each configuration, calculate average memory
    config_memories = []
    for _, row in rows.iterrows():
        mem_values = []
        for task in tasks:
            # Map task names to memory column names if needed
            if task.startswith('cite_'):
                col = 'cite'
            else:
                col = task

            if col in row.index:
                val = row[col]
                if not pd.isna(val) and val != 0:
                    mem_values.append(val)

        if len(mem_values) > 0:
            config_memories.append(np.mean(mem_values))

    if len(config_memories) == 0:
        return None

    # Return the average across all configurations
    return np.mean(config_memories)

def get_average_performance(performance_df, technique, model, context, tasks):
    """Get average performance for a technique/model/context combination"""
    rows = performance_df[
        (performance_df['technique'] == technique) &
        (performance_df['context_length'] == context) &
        (performance_df['model'] == model)
    ]

    if rows.empty:
        return None

    # For each configuration, calculate average performance
    config_perfs = []
    for _, row in rows.iterrows():
        perf_values = []
        for task in tasks:
            if task in row.index:
                val = row[task]
                if not pd.isna(val) and val != 0:
                    perf_values.append(val)

        if len(perf_values) > 0:
            config_perfs.append(np.mean(perf_values))

    if len(config_perfs) == 0:
        return None

    # Return the average across all configurations
    return np.mean(config_perfs)

print("=" * 80)
print("TOKEN EVICTION METHODS: MEMORY SAVINGS AND PERFORMANCE ANALYSIS")
print("=" * 80)
print()

# Token eviction techniques
token_eviction_techniques = ['streamingllm', 'snapkv', 'pyramidkv']

# Quantization techniques for comparison
quantization_techniques = ['INT4', 'INT8']

# Store results
results = {
    'streamingllm': {'memory_savings': [], 'perf_degradation': []},
    'snapkv': {'memory_savings': [], 'perf_degradation': []},
    'pyramidkv': {'memory_savings': [], 'perf_degradation': []},
    'INT4': {'memory_savings': [], 'perf_degradation': []},
    'INT8': {'memory_savings': [], 'perf_degradation': []},
}

for model in models:
    print(f"\n{model}:")
    print("-" * 60)

    # Get baseline memory and performance from both HELMET and LongProc
    helmet_baseline_mem = get_average_memory(helmet_memory_df, 'baseline', model, '16k', helmet_tasks)
    longproc_baseline_mem = get_average_memory(longproc_memory_df, 'baseline', model, '2k', longproc_tasks)
    helmet_baseline_perf = get_average_performance(helmet_performance_df, 'baseline', model, '16k', helmet_tasks)
    longproc_baseline_perf = get_average_performance(longproc_performance_df, 'baseline', model, '2k', longproc_tasks)

    # Average across both benchmarks
    baseline_mem_values = [m for m in [helmet_baseline_mem, longproc_baseline_mem] if m is not None]
    baseline_perf_values = [p for p in [helmet_baseline_perf, longproc_baseline_perf] if p is not None]

    if len(baseline_mem_values) == 0 or len(baseline_perf_values) == 0:
        print("  No baseline data found")
        continue

    baseline_mem = np.mean(baseline_mem_values)
    baseline_perf = np.mean(baseline_perf_values)

    print(f"  Baseline: {baseline_mem:.2f} GB, {baseline_perf:.4f} performance")
    print()

    # Analyze token eviction methods
    for technique in token_eviction_techniques:
        helmet_mem = get_average_memory(helmet_memory_df, technique, model, '16k', helmet_tasks)
        longproc_mem = get_average_memory(longproc_memory_df, technique, model, '2k', longproc_tasks)
        helmet_perf = get_average_performance(helmet_performance_df, technique, model, '16k', helmet_tasks)
        longproc_perf = get_average_performance(longproc_performance_df, technique, model, '2k', longproc_tasks)

        mem_values = [m for m in [helmet_mem, longproc_mem] if m is not None]
        perf_values = [p for p in [helmet_perf, longproc_perf] if p is not None]

        if len(mem_values) > 0 and len(perf_values) > 0:
            avg_mem = np.mean(mem_values)
            avg_perf = np.mean(perf_values)
            mem_savings = ((baseline_mem - avg_mem) / baseline_mem) * 100
            perf_degradation = ((baseline_perf - avg_perf) / baseline_perf) * 100

            results[technique]['memory_savings'].append(mem_savings)
            results[technique]['perf_degradation'].append(perf_degradation)

            print(f"  {technique:15s}: {avg_mem:.2f} GB ({mem_savings:+.2f}%), "
                  f"{avg_perf:.4f} perf ({perf_degradation:+.2f}%)")
        else:
            print(f"  {technique:15s}: No data")

    # Also get quantization for comparison
    for technique in quantization_techniques:
        helmet_mem = get_average_memory(helmet_memory_df, technique, model, '16k', helmet_tasks)
        longproc_mem = get_average_memory(longproc_memory_df, technique, model, '2k', longproc_tasks)
        helmet_perf = get_average_performance(helmet_performance_df, technique, model, '16k', helmet_tasks)
        longproc_perf = get_average_performance(longproc_performance_df, technique, model, '2k', longproc_tasks)

        mem_values = [m for m in [helmet_mem, longproc_mem] if m is not None]
        perf_values = [p for p in [helmet_perf, longproc_perf] if p is not None]

        if len(mem_values) > 0 and len(perf_values) > 0:
            avg_mem = np.mean(mem_values)
            avg_perf = np.mean(perf_values)
            mem_savings = ((baseline_mem - avg_mem) / baseline_mem) * 100
            perf_degradation = ((baseline_perf - avg_perf) / baseline_perf) * 100

            results[technique]['memory_savings'].append(mem_savings)
            results[technique]['perf_degradation'].append(perf_degradation)

            print(f"  {technique:15s}: {avg_mem:.2f} GB ({mem_savings:+.2f}%), "
                  f"{avg_perf:.4f} perf ({perf_degradation:+.2f}%)")

print()
print("=" * 80)
print("OVERALL AVERAGES ACROSS ALL MODELS")
print("=" * 80)
print()

print("TOKEN EVICTION METHODS:")
print("-" * 60)
for technique in token_eviction_techniques:
    if len(results[technique]['memory_savings']) > 0:
        avg_mem_savings = np.mean(results[technique]['memory_savings'])
        std_mem_savings = np.std(results[technique]['memory_savings'], ddof=1) if len(results[technique]['memory_savings']) > 1 else 0
        avg_perf_degradation = np.mean(results[technique]['perf_degradation'])
        std_perf_degradation = np.std(results[technique]['perf_degradation'], ddof=1) if len(results[technique]['perf_degradation']) > 1 else 0
        print(f"{technique:15s}: {avg_mem_savings:+.2f}% ± {std_mem_savings:.2f}% memory, "
              f"{avg_perf_degradation:+.2f}% ± {std_perf_degradation:.2f}% performance")
        print(f"                 (averaged across {len(results[technique]['memory_savings'])} models)")
    else:
        print(f"{technique:15s}: No data")

print()
print("QUANTIZATION METHODS (for comparison):")
print("-" * 60)
for technique in quantization_techniques:
    if len(results[technique]['memory_savings']) > 0:
        avg_mem_savings = np.mean(results[technique]['memory_savings'])
        std_mem_savings = np.std(results[technique]['memory_savings'], ddof=1) if len(results[technique]['memory_savings']) > 1 else 0
        avg_perf_degradation = np.mean(results[technique]['perf_degradation'])
        std_perf_degradation = np.std(results[technique]['perf_degradation'], ddof=1) if len(results[technique]['perf_degradation']) > 1 else 0
        print(f"{technique:15s}: {avg_mem_savings:+.2f}% ± {std_mem_savings:.2f}% memory, "
              f"{avg_perf_degradation:+.2f}% ± {std_perf_degradation:.2f}% performance")
        print(f"                 (averaged across {len(results[technique]['memory_savings'])} models)")
    else:
        print(f"{technique:15s}: No data")

print()
print("=" * 80)
print("COMPARISON: TOKEN EVICTION vs QUANTIZATION")
print("=" * 80)
print()

# Calculate average performance degradation for token eviction vs quantization
token_eviction_perf_deg = []
for technique in token_eviction_techniques:
    if len(results[technique]['perf_degradation']) > 0:
        token_eviction_perf_deg.extend(results[technique]['perf_degradation'])

quantization_perf_deg = []
for technique in quantization_techniques:
    if len(results[technique]['perf_degradation']) > 0:
        quantization_perf_deg.extend(results[technique]['perf_degradation'])

if len(token_eviction_perf_deg) > 0 and len(quantization_perf_deg) > 0:
    avg_token_eviction_perf_deg = np.mean(token_eviction_perf_deg)
    std_token_eviction_perf_deg = np.std(token_eviction_perf_deg, ddof=1) if len(token_eviction_perf_deg) > 1 else 0
    avg_quantization_perf_deg = np.mean(quantization_perf_deg)
    std_quantization_perf_deg = np.std(quantization_perf_deg, ddof=1) if len(quantization_perf_deg) > 1 else 0

    print(f"Average performance degradation:")
    print(f"  Token Eviction Methods: {avg_token_eviction_perf_deg:+.2f}% ± {std_token_eviction_perf_deg:.2f}%")
    print(f"  Quantization Methods:   {avg_quantization_perf_deg:+.2f}% ± {std_quantization_perf_deg:.2f}%")
    print()
    print(f"Token eviction methods experience {avg_token_eviction_perf_deg - avg_quantization_perf_deg:.2f}% ")
    print(f"more performance degradation than quantization methods.")

print()
print("=" * 80)
