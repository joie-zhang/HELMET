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

# Helper tasks for memory (cite tasks are grouped)
helmet_memory_tasks = [
    'recall_jsonkv', 'rag_nq', 'rag_hotpotqa', 'rerank',
    'cite', 'niah', 'icl_clinic', 'icl_banking'
]

def get_memory_and_performance(memory_df, performance_df, technique, model, cache_size, context,
                                memory_tasks, perf_tasks):
    """Get average memory and performance for a specific configuration"""
    # Memory
    mem_row = memory_df[
        (memory_df['technique'] == technique) &
        (memory_df['context_length'] == context) &
        (memory_df['model'] == model) &
        (memory_df['cache_size'] == cache_size)
    ]

    if mem_row.empty:
        return None, None

    mem_values = []
    for task in memory_tasks:
        if task in mem_row.columns:
            val = mem_row.iloc[0][task]
            if not pd.isna(val) and val != 0:
                mem_values.append(val)

    avg_mem = np.mean(mem_values) if len(mem_values) > 0 else None

    # Performance
    perf_row = performance_df[
        (performance_df['technique'] == technique) &
        (performance_df['context_length'] == context) &
        (performance_df['model'] == model) &
        (performance_df['cache_size'] == cache_size)
    ]

    if perf_row.empty:
        return avg_mem, None

    perf_values = []
    for task in perf_tasks:
        if task in perf_row.columns:
            val = perf_row.iloc[0][task]
            if not pd.isna(val) and val != 0:
                perf_values.append(val)

    avg_perf = np.mean(perf_values) if len(perf_values) > 0 else None

    return avg_mem, avg_perf

def extract_cache_size_value(technique, cache_size):
    """Extract numeric cache size for sorting and reporting"""
    if cache_size == 'default':
        return float('inf'), cache_size
    elif technique == 'streamingllm':
        # Extract n_local value
        parts = cache_size.split('_')
        return int(parts[2]), cache_size
    elif technique in ['snapkv', 'pyramidkv']:
        # Extract cache value (c parameter)
        if cache_size.startswith('w'):
            parts = cache_size.split('_')
            return int(parts[1][1:]), cache_size
        return 0, cache_size
    return 0, cache_size

print("=" * 90)
print("TOKEN EVICTION METHODS: CACHE SIZE TRADE-OFF ANALYSIS")
print("=" * 90)
print()

# Token eviction techniques
token_eviction_techniques = ['streamingllm', 'snapkv', 'pyramidkv']

# Store aggregated results by cache size category
results_by_size = {
    'small': {'memory': [], 'perf_deg': []},
    'large': {'memory': [], 'perf_deg': []}
}

for technique in token_eviction_techniques:
    print(f"\n{'='*90}")
    print(f"{technique.upper()}")
    print(f"{'='*90}")

    # Collect all configurations across all models
    all_configs = []

    for model in models:
        # Get baseline
        helmet_baseline_mem, helmet_baseline_perf = get_memory_and_performance(
            helmet_memory_df, helmet_performance_df, 'baseline', model, 'default',
            '16k', helmet_memory_tasks, helmet_tasks
        )
        longproc_baseline_mem, longproc_baseline_perf = get_memory_and_performance(
            longproc_memory_df, longproc_performance_df, 'baseline', model, 'default',
            '2k', longproc_tasks, longproc_tasks
        )

        baseline_mem_values = [m for m in [helmet_baseline_mem, longproc_baseline_mem] if m is not None]
        baseline_perf_values = [p for p in [helmet_baseline_perf, longproc_baseline_perf] if p is not None]

        if len(baseline_mem_values) == 0 or len(baseline_perf_values) == 0:
            continue

        baseline_mem = np.mean(baseline_mem_values)
        baseline_perf = np.mean(baseline_perf_values)

        # Get all cache sizes for this technique/model
        helmet_subset = helmet_memory_df[
            (helmet_memory_df['technique'] == technique) &
            (helmet_memory_df['model'] == model)
        ]

        cache_sizes = helmet_subset['cache_size'].unique()

        for cache_size in cache_sizes:
            helmet_mem, helmet_perf = get_memory_and_performance(
                helmet_memory_df, helmet_performance_df, technique, model, cache_size,
                '16k', helmet_memory_tasks, helmet_tasks
            )
            longproc_mem, longproc_perf = get_memory_and_performance(
                longproc_memory_df, longproc_performance_df, technique, model, cache_size,
                '2k', longproc_tasks, longproc_tasks
            )

            mem_values = [m for m in [helmet_mem, longproc_mem] if m is not None]
            perf_values = [p for p in [helmet_perf, longproc_perf] if p is not None]

            if len(mem_values) == 0 or len(perf_values) == 0:
                continue

            avg_mem = np.mean(mem_values)
            avg_perf = np.mean(perf_values)

            mem_change = ((avg_mem - baseline_mem) / baseline_mem) * 100
            perf_degradation = ((baseline_perf - avg_perf) / baseline_perf) * 100

            cache_val, _ = extract_cache_size_value(technique, cache_size)

            all_configs.append({
                'model': model,
                'cache_size': cache_size,
                'cache_val': cache_val,
                'memory': avg_mem,
                'memory_change': mem_change,
                'performance': avg_perf,
                'perf_degradation': perf_degradation,
                'baseline_mem': baseline_mem,
                'baseline_perf': baseline_perf
            })

    # Sort by cache size
    all_configs.sort(key=lambda x: (x['cache_val'], x['model']))

    # Split into small and large cache sizes (using median as threshold)
    cache_vals = [c['cache_val'] for c in all_configs if c['cache_val'] != float('inf')]
    if len(cache_vals) > 0:
        median_cache = np.median(cache_vals)
    else:
        median_cache = 0

    small_configs = [c for c in all_configs if c['cache_val'] <= median_cache]
    large_configs = [c for c in all_configs if c['cache_val'] > median_cache]

    print(f"\nSmaller Cache Sizes (≤ {int(median_cache) if median_cache != float('inf') else 'N/A'}):")
    print("-" * 90)
    if len(small_configs) > 0:
        for config in small_configs:
            print(f"  {config['model']:35s} | Cache: {config['cache_size']:25s} | "
                  f"Mem: {config['memory']:6.2f}GB ({config['memory_change']:+6.2f}%) | "
                  f"Perf Deg: {config['perf_degradation']:+6.2f}%")
            results_by_size['small']['memory'].append(config['memory_change'])
            results_by_size['small']['perf_deg'].append(config['perf_degradation'])

        avg_mem_small = np.mean([c['memory_change'] for c in small_configs])
        avg_perf_small = np.mean([c['perf_degradation'] for c in small_configs])
        print(f"\n  Average: Memory {avg_mem_small:+.2f}%, Performance degradation {avg_perf_small:+.2f}%")
    else:
        print("  No configurations")

    print(f"\nLarger Cache Sizes (> {int(median_cache) if median_cache != float('inf') else 'N/A'}):")
    print("-" * 90)
    if len(large_configs) > 0:
        for config in large_configs:
            print(f"  {config['model']:35s} | Cache: {config['cache_size']:25s} | "
                  f"Mem: {config['memory']:6.2f}GB ({config['memory_change']:+6.2f}%) | "
                  f"Perf Deg: {config['perf_degradation']:+6.2f}%")
            results_by_size['large']['memory'].append(config['memory_change'])
            results_by_size['large']['perf_deg'].append(config['perf_degradation'])

        avg_mem_large = np.mean([c['memory_change'] for c in large_configs])
        avg_perf_large = np.mean([c['perf_degradation'] for c in large_configs])
        print(f"\n  Average: Memory {avg_mem_large:+.2f}%, Performance degradation {avg_perf_large:+.2f}%")
    else:
        print("  No configurations")

print()
print("=" * 90)
print("OVERALL SUMMARY: CACHE SIZE TRADE-OFF")
print("=" * 90)
print()

if len(results_by_size['small']['memory']) > 0:
    small_mem_avg = np.mean(results_by_size['small']['memory'])
    small_mem_std = np.std(results_by_size['small']['memory'], ddof=1) if len(results_by_size['small']['memory']) > 1 else 0
    small_perf_avg = np.mean(results_by_size['small']['perf_deg'])
    small_perf_std = np.std(results_by_size['small']['perf_deg'], ddof=1) if len(results_by_size['small']['perf_deg']) > 1 else 0

    print("Smaller Cache Sizes:")
    print(f"  Memory change:          {small_mem_avg:+.2f}% ± {small_mem_std:.2f}%")
    print(f"  Performance degradation: {small_perf_avg:+.2f}% ± {small_perf_std:.2f}%")
    print(f"  → Less memory overhead, but worse performance")
    print()

if len(results_by_size['large']['memory']) > 0:
    large_mem_avg = np.mean(results_by_size['large']['memory'])
    large_mem_std = np.std(results_by_size['large']['memory'], ddof=1) if len(results_by_size['large']['memory']) > 1 else 0
    large_perf_avg = np.mean(results_by_size['large']['perf_deg'])
    large_perf_std = np.std(results_by_size['large']['perf_deg'], ddof=1) if len(results_by_size['large']['perf_deg']) > 1 else 0

    print("Larger Cache Sizes:")
    print(f"  Memory change:          {large_mem_avg:+.2f}% ± {large_mem_std:.2f}%")
    print(f"  Performance degradation: {large_perf_avg:+.2f}% ± {large_perf_std:.2f}%")
    print(f"  → Better performance recovery, but much higher memory")
    print()

if len(results_by_size['small']['memory']) > 0 and len(results_by_size['large']['memory']) > 0:
    print("Key Finding:")
    print(f"  Increasing cache size improves performance by {small_perf_avg - large_perf_avg:.2f} percentage points,")
    print(f"  but at the cost of {large_mem_avg - small_mem_avg:.2f} percentage points more memory overhead.")
    print(f"  → Neither configuration is Pareto optimal compared to baseline.")

print()
print("=" * 90)
