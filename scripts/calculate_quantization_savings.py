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

def get_average_memory(memory_df, technique, model, context, tasks):
    """Get average memory for a technique/model/context combination"""
    row = memory_df[
        (memory_df['technique'] == technique) &
        (memory_df['context_length'] == context) &
        (memory_df['model'] == model) &
        (memory_df['cache_size'] == 'default')
    ]

    if row.empty:
        return None

    mem_values = []
    for task in tasks:
        if task in row.columns:
            val = row.iloc[0][task]
            if not pd.isna(val) and val != 0:
                mem_values.append(val)

    if len(mem_values) == 0:
        return None

    return np.mean(mem_values)

def get_average_performance(performance_df, technique, model, context, tasks):
    """Get average performance for a technique/model/context combination"""
    row = performance_df[
        (performance_df['technique'] == technique) &
        (performance_df['context_length'] == context) &
        (performance_df['model'] == model) &
        (performance_df['cache_size'] == 'default')
    ]

    if row.empty:
        return None

    perf_values = []
    for task in tasks:
        if task in row.columns:
            val = row.iloc[0][task]
            if not pd.isna(val) and val != 0:
                perf_values.append(val)

    if len(perf_values) == 0:
        return None

    return np.mean(perf_values)

print("=" * 80)
print("QUANTIZATION MEMORY SAVINGS AND PERFORMANCE ANALYSIS")
print("=" * 80)
print()

# Calculate savings for each model
int4_memory_savings_per_model = []
int8_memory_savings_per_model = []
int4_perf_degradation_per_model = []
int8_perf_degradation_per_model = []

for model in models:
    print(f"\n{model}:")
    print("-" * 60)

    # Get baseline memory and performance from both HELMET and LongProc
    helmet_baseline_mem = get_average_memory(helmet_memory_df, 'baseline', model, '16k', helmet_memory_tasks)
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

    print(f"  Baseline:  {baseline_mem:.2f} GB, {baseline_perf:.4f} performance")

    # Get INT4 memory and performance
    helmet_int4_mem = get_average_memory(helmet_memory_df, 'INT4', model, '16k', helmet_memory_tasks)
    longproc_int4_mem = get_average_memory(longproc_memory_df, 'INT4', model, '2k', longproc_tasks)
    helmet_int4_perf = get_average_performance(helmet_performance_df, 'INT4', model, '16k', helmet_tasks)
    longproc_int4_perf = get_average_performance(longproc_performance_df, 'INT4', model, '2k', longproc_tasks)

    int4_mem_values = [m for m in [helmet_int4_mem, longproc_int4_mem] if m is not None]
    int4_perf_values = [p for p in [helmet_int4_perf, longproc_int4_perf] if p is not None]

    if len(int4_mem_values) > 0 and len(int4_perf_values) > 0:
        int4_mem = np.mean(int4_mem_values)
        int4_perf = np.mean(int4_perf_values)
        int4_mem_savings = ((baseline_mem - int4_mem) / baseline_mem) * 100
        int4_perf_degradation = ((baseline_perf - int4_perf) / baseline_perf) * 100
        int4_memory_savings_per_model.append(int4_mem_savings)
        int4_perf_degradation_per_model.append(int4_perf_degradation)
        print(f"  INT4:      {int4_mem:.2f} GB ({int4_mem_savings:+.2f}%), "
              f"{int4_perf:.4f} perf ({int4_perf_degradation:+.2f}%)")
    else:
        print(f"  INT4:      No data")

    # Get INT8 memory and performance
    helmet_int8_mem = get_average_memory(helmet_memory_df, 'INT8', model, '16k', helmet_memory_tasks)
    longproc_int8_mem = get_average_memory(longproc_memory_df, 'INT8', model, '2k', longproc_tasks)
    helmet_int8_perf = get_average_performance(helmet_performance_df, 'INT8', model, '16k', helmet_tasks)
    longproc_int8_perf = get_average_performance(longproc_performance_df, 'INT8', model, '2k', longproc_tasks)

    int8_mem_values = [m for m in [helmet_int8_mem, longproc_int8_mem] if m is not None]
    int8_perf_values = [p for p in [helmet_int8_perf, longproc_int8_perf] if p is not None]

    if len(int8_mem_values) > 0 and len(int8_perf_values) > 0:
        int8_mem = np.mean(int8_mem_values)
        int8_perf = np.mean(int8_perf_values)
        int8_mem_savings = ((baseline_mem - int8_mem) / baseline_mem) * 100
        int8_perf_degradation = ((baseline_perf - int8_perf) / baseline_perf) * 100
        int8_memory_savings_per_model.append(int8_mem_savings)
        int8_perf_degradation_per_model.append(int8_perf_degradation)
        print(f"  INT8:      {int8_mem:.2f} GB ({int8_mem_savings:+.2f}%), "
              f"{int8_perf:.4f} perf ({int8_perf_degradation:+.2f}%)")
    else:
        print(f"  INT8:      No data")

print()
print("=" * 80)
print("OVERALL AVERAGES ACROSS ALL MODELS")
print("=" * 80)
print()

print("INT4:")
print("-" * 60)
if len(int4_memory_savings_per_model) > 0:
    avg_int4_mem_savings = np.mean(int4_memory_savings_per_model)
    std_int4_mem_savings = np.std(int4_memory_savings_per_model, ddof=1) if len(int4_memory_savings_per_model) > 1 else 0
    print(f"  Memory savings:         {avg_int4_mem_savings:.2f}% ± {std_int4_mem_savings:.2f}%")
else:
    print("  Memory savings:         No data")

if len(int4_perf_degradation_per_model) > 0:
    avg_int4_perf_deg = np.mean(int4_perf_degradation_per_model)
    std_int4_perf_deg = np.std(int4_perf_degradation_per_model, ddof=1) if len(int4_perf_degradation_per_model) > 1 else 0
    print(f"  Performance degradation: {avg_int4_perf_deg:+.2f}% ± {std_int4_perf_deg:.2f}%")
else:
    print("  Performance degradation: No data")

if len(int4_memory_savings_per_model) > 0 or len(int4_perf_degradation_per_model) > 0:
    print(f"  (averaged across {max(len(int4_memory_savings_per_model), len(int4_perf_degradation_per_model))} models)")

print()
print("INT8:")
print("-" * 60)
if len(int8_memory_savings_per_model) > 0:
    avg_int8_mem_savings = np.mean(int8_memory_savings_per_model)
    std_int8_mem_savings = np.std(int8_memory_savings_per_model, ddof=1) if len(int8_memory_savings_per_model) > 1 else 0
    print(f"  Memory savings:         {avg_int8_mem_savings:.2f}% ± {std_int8_mem_savings:.2f}%")
else:
    print("  Memory savings:         No data")

if len(int8_perf_degradation_per_model) > 0:
    avg_int8_perf_deg = np.mean(int8_perf_degradation_per_model)
    std_int8_perf_deg = np.std(int8_perf_degradation_per_model, ddof=1) if len(int8_perf_degradation_per_model) > 1 else 0
    print(f"  Performance degradation: {avg_int8_perf_deg:+.2f}% ± {std_int8_perf_deg:.2f}%")
else:
    print("  Performance degradation: No data")

if len(int8_memory_savings_per_model) > 0 or len(int8_perf_degradation_per_model) > 0:
    print(f"  (averaged across {max(len(int8_memory_savings_per_model), len(int8_perf_degradation_per_model))} models)")

print()
print("=" * 80)
