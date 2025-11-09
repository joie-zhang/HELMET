#!/usr/bin/env python3
"""
Extract the exact data used in the quadrant_comparison_1x1_grouped_notitle plot
and save it to a CSV file.
"""

import pandas as pd
import numpy as np
import os

# Define paths
results_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/results'
output_dir = os.path.join(results_dir, 'plots')

# Load data
helmet_performance_df = pd.read_csv(os.path.join(results_dir, 'helmet_results', 'helmet_performance.csv'))
longproc_performance_df = pd.read_csv(os.path.join(results_dir, 'longproc_results', 'longproc_performance.csv'))

# Filter out unwanted techniques
unwanted_techniques = ['quest', 'streamingllm_original', 'minference']
helmet_performance_df = helmet_performance_df[~helmet_performance_df['technique'].isin(unwanted_techniques)].copy()
longproc_performance_df = longproc_performance_df[~longproc_performance_df['technique'].isin(unwanted_techniques)].copy()

# Filter out Qwen3 models
unwanted_models = ['Qwen3-8B', 'Yarn-Qwen3-8B']
helmet_performance_df = helmet_performance_df[~helmet_performance_df['model'].isin(unwanted_models)].copy()
longproc_performance_df = longproc_performance_df[~longproc_performance_df['model'].isin(unwanted_models)].copy()

# Filter out specific unwanted cache size configurations
unwanted_configs = [
    ('pyramidkv', 'w32_c4096_k5_avgpool'),
    ('snapkv', 'w32_c4096_k5_avgpool'),
    ('streamingllm', 'n_local_3968_n_init_128'),
    ('duoattn', 'sp0.7_pf32768')
]

for technique, cache_size in unwanted_configs:
    helmet_performance_df = helmet_performance_df[~((helmet_performance_df['technique'] == technique) & (helmet_performance_df['cache_size'] == cache_size))].copy()
    longproc_performance_df = longproc_performance_df[~((longproc_performance_df['technique'] == technique) & (longproc_performance_df['cache_size'] == cache_size))].copy()

# Filter SnapKV and PyramidKV configurations for reasoning models
allowed_reasoning_configs = ['w256_c2048_k7_avgpool', 'w2048_c8192_k7_avgpool', 'w256_c2048_k7_maxpool', 'w2048_c8192_k7_maxpool', 'default']
reasoning_models = ['DeepSeek-R1-Distill-Llama-8B', 'DeepSeek-R1-Distill-Qwen-7B']

helmet_performance_df = helmet_performance_df[~(
    (helmet_performance_df['model'].isin(reasoning_models)) &
    (helmet_performance_df['technique'].isin(['snapkv', 'pyramidkv'])) &
    (~helmet_performance_df['cache_size'].isin(allowed_reasoning_configs))
)].copy()

longproc_performance_df = longproc_performance_df[~(
    (longproc_performance_df['model'].isin(reasoning_models)) &
    (longproc_performance_df['technique'].isin(['snapkv', 'pyramidkv'])) &
    (~longproc_performance_df['cache_size'].isin(allowed_reasoning_configs))
)].copy()

# Filter out w32 and w1024 cache sizes for baseline models
baseline_models = ['Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct']
helmet_performance_df = helmet_performance_df[~(
    (helmet_performance_df['model'].isin(baseline_models)) &
    (helmet_performance_df['technique'].isin(['snapkv', 'pyramidkv'])) &
    (helmet_performance_df['cache_size'].str.contains('w32_|w1024_', na=False))
)].copy()

longproc_performance_df = longproc_performance_df[~(
    (longproc_performance_df['model'].isin(baseline_models)) &
    (longproc_performance_df['technique'].isin(['snapkv', 'pyramidkv'])) &
    (longproc_performance_df['cache_size'].str.contains('w32_|w1024_', na=False))
)].copy()

# Filter StreamingLLM configurations
allowed_streamingllm_configs = ['n_local_4092_n_init_4', 'n_local_4096_n_init_4']
helmet_performance_df = helmet_performance_df[~(
    (helmet_performance_df['technique'] == 'streamingllm') &
    (~helmet_performance_df['cache_size'].isin(allowed_streamingllm_configs))
)].copy()

longproc_performance_df = longproc_performance_df[~(
    (longproc_performance_df['technique'] == 'streamingllm') &
    (~longproc_performance_df['cache_size'].isin(allowed_streamingllm_configs))
)].copy()

# Filter to include only desired models
filtered_models = ['Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct',
                   'DeepSeek-R1-Distill-Llama-8B', 'DeepSeek-R1-Distill-Qwen-7B']

helmet_performance_df = helmet_performance_df[helmet_performance_df['model'].isin(filtered_models)].copy()
longproc_performance_df = longproc_performance_df[longproc_performance_df['model'].isin(filtered_models)].copy()

# Define difficulty levels (ordered from easiest to hardest)
difficulty_levels = {
    'Easy': {
        'label': 'Short Output, Low Dispersion',
        'tasks': ['niah', 'recall_jsonkv', 'rag_hotpotqa', 'rag_nq'],
        'dataset': 'helmet',
    },
    'Medium': {
        'label': 'Short Output, High Dispersion',
        'tasks': ['cite_str_em', 'cite_citation_rec', 'cite_citation_prec', 'rerank',
                  'icl_clinic', 'icl_banking', 'summ_multilex'],
        'dataset': 'helmet',
    },
    'Hard': {
        'label': 'Long Output, High Dispersion',
        'tasks': ['pseudo_to_code', 'html_to_tsv', 'travel_planning'],
        'dataset': 'longproc',
    }
}

# Define techniques and their display labels
techniques_order = ['baseline', 'INT4', 'INT8', 'snapkv', 'pyramidkv', 'streamingllm', 'duoattn']
technique_labels = {
    'baseline': 'Baseline',
    'INT4': 'NF4',
    'INT8': 'Int8',
    'snapkv': 'SnapKV',
    'pyramidkv': 'PyramidKV',
    'streamingllm': 'StreamingLLM',
    'duoattn': 'DuoAttention'
}

def calculate_difficulty_average(difficulty_info, helmet_df, longproc_df, technique, models):
    """
    Calculate average performance and standard error for a technique across all tasks in a difficulty level.
    Averages across all models, context lengths, and cache configurations.
    Returns: (mean, std_error, all_values) tuple
    """
    tasks = difficulty_info['tasks']
    dataset = difficulty_info['dataset']

    if dataset == 'helmet':
        df = helmet_df
        context_lengths = ['16k', '32k']
    else:  # longproc
        df = longproc_df
        context_lengths = ['2k', '5k']

    all_perf_values = []

    # Iterate over all models and context lengths
    for model in models:
        for context in context_lengths:
            # Get all rows for this technique/model/context combination
            technique_rows = df[
                (df['technique'] == technique) &
                (df['model'] == model) &
                (df['context_length'] == context)
            ]

            # Average across all cache configurations
            for _, row in technique_rows.iterrows():
                for task in tasks:
                    val = row[task]
                    if not pd.isna(val) and val != 0:
                        all_perf_values.append(val)

    if len(all_perf_values) == 0:
        return None, None, []

    mean_val = np.mean(all_perf_values)
    # Calculate standard error of the mean
    std_error = np.std(all_perf_values, ddof=1) / np.sqrt(len(all_perf_values)) if len(all_perf_values) > 1 else 0

    return mean_val, std_error, all_perf_values

# Collect data for all techniques and difficulty levels
print("Extracting performance data...")
data_records = []

for technique in techniques_order:
    for difficulty_name, difficulty_info in difficulty_levels.items():
        avg_perf, std_error, all_values = calculate_difficulty_average(
            difficulty_info,
            helmet_performance_df,
            longproc_performance_df,
            technique,
            filtered_models
        )

        data_records.append({
            'Technique': technique_labels[technique],
            'Difficulty_Level': difficulty_name,
            'Difficulty_Label': difficulty_info['label'],
            'Average_Performance': avg_perf if avg_perf is not None else np.nan,
            'Std_Error': std_error if std_error is not None else np.nan,
            'N_Samples': len(all_values),
            'Min_Performance': np.min(all_values) if len(all_values) > 0 else np.nan,
            'Max_Performance': np.max(all_values) if len(all_values) > 0 else np.nan,
            'Std_Dev': np.std(all_values, ddof=1) if len(all_values) > 1 else np.nan
        })

        avg_str = f"{avg_perf:.2f}" if avg_perf is not None else "N/A"
        se_str = f"{std_error:.2f}" if std_error is not None else "N/A"
        print(f"  {technique_labels[technique]:15s} | {difficulty_name:6s} | "
              f"Avg: {avg_str:>6s} | SE: {se_str:>6s} | N: {len(all_values):4d}")

# Create DataFrame and save to CSV
output_df = pd.DataFrame(data_records)
output_csv_path = os.path.join(results_dir, 'quadrant_plot_data.csv')
output_df.to_csv(output_csv_path, index=False)

print(f"\n✓ Data saved to: {output_csv_path}")
print(f"  Total records: {len(output_df)}")
print(f"  Techniques: {len(techniques_order)}")
print(f"  Difficulty levels: {len(difficulty_levels)}")