#!/usr/bin/env python3
"""
Extract the raw data used in task_deltas_averaged_configs.py plot
Outputs a CSV file with performance deltas for each technique and task group.
"""

import pandas as pd
import numpy as np
import os

# Define paths
results_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/results'
output_dir = results_dir

# Load data
helmet_performance_df = pd.read_csv(os.path.join(results_dir, 'helmet_results', 'helmet_performance.csv'))
longproc_performance_df = pd.read_csv(os.path.join(results_dir, 'longproc_results', 'longproc_performance.csv'))

# Define constants
HELMET_CONTEXT_16K = '16k'
HELMET_CONTEXT_32K = '32k'
LONGPROC_CONTEXT_2K = '2k'
LONGPROC_CONTEXT_5K = '5k'

# Models to include
all_models = ['Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct',
              'DeepSeek-R1-Distill-Llama-8B', 'DeepSeek-R1-Distill-Qwen-7B']
duoattn_models = ['Llama-3.1-8B-Instruct']  # Only Llama for DuoAttn

# Task definitions with display names
helmet_tasks = {
    'niah': 'niah',
    'recall_jsonkv': 'recall_jsonkv',
    'rag_hotpotqa': 'rag_hotpotqa',
    'rag_nq': 'rag_nq',
    'cite_str_em': 'cite_str_em',
    'cite_citation_rec': 'cite_rec',
    'cite_citation_prec': 'cite_prec',
    'rerank': 'rerank',
    'icl_clinic': 'icl_clinc',
    'icl_banking': 'icl_banking',
    'summ_multilex': 'multi_lexsum'
}

longproc_tasks = {
    'pseudo_to_code': 'pseudo_to_code',
    'html_to_tsv': 'html_to_tsv',
    'travel_planning': 'travel_planning'
}

# Ordered task list for x-axis
task_order = list(helmet_tasks.values()) + list(longproc_tasks.values())

# Technique configurations - averaging both cache configs for SnapKV/PyramidKV
techniques_config = [
    {'name': 'NF4', 'technique': 'INT4', 'cache_sizes': None, 'models': all_models},
    {'name': 'Int8', 'technique': 'INT8', 'cache_sizes': None, 'models': all_models},
    {'name': 'SnapKV', 'technique': 'snapkv', 'cache_sizes': ['w256_c2048_k7_maxpool', 'w2048_c8192_k7_maxpool'], 'models': all_models},
    {'name': 'PyramidKV', 'technique': 'pyramidkv', 'cache_sizes': ['w256_c2048_k7_avgpool', 'w2048_c8192_k7_avgpool'], 'models': all_models},
    {'name': 'StreamingLLM', 'technique': 'streamingllm', 'cache_sizes': ['n_local_4092_n_init_4', 'n_local_4096_n_init_4'], 'models': all_models},
    {'name': 'DuoAttention', 'technique': 'duoattn', 'cache_sizes': None, 'models': duoattn_models},
]

# Grouped task labels
grouped_task_labels = ['NIAH', 'Recall', 'RAG', 'ICL', 'Cite', 'Re-rank', 'Summ', 'Pseudo', 'HTML', 'Travel']

# Map original tasks to grouped labels
task_grouping = {
    'niah': 'NIAH',
    'recall_jsonkv': 'Recall',
    'rag_hotpotqa': 'RAG',
    'rag_nq': 'RAG',
    'icl_clinc': 'ICL',
    'icl_banking': 'ICL',
    'cite_str_em': 'Cite',
    'cite_rec': 'Cite',
    'cite_prec': 'Cite',
    'rerank': 'Re-rank',
    'multi_lexsum': 'Summ',
    'pseudo_to_code': 'Pseudo',
    'html_to_tsv': 'HTML',
    'travel_planning': 'Travel'
}

def get_performance_values(df, technique, model, cache_sizes, context_length, task_mapping):
    """Get performance values for a technique-model-context combination"""
    values = []

    if cache_sizes is None:
        # No cache size (INT4, INT8, baseline)
        rows = df[
            (df['technique'] == technique) &
            (df['model'] == model) &
            (df['context_length'] == context_length)
        ]
    else:
        # With cache sizes
        rows = df[
            (df['technique'] == technique) &
            (df['model'] == model) &
            (df['context_length'] == context_length) &
            (df['cache_size'].isin(cache_sizes))
        ]

    if rows.empty:
        return {}

    # Extract task values
    task_values = {}
    for original_task, display_task in task_mapping.items():
        if original_task in rows.columns:
            vals = rows[original_task].values
            vals = vals[~pd.isna(vals) & (vals != 0)]
            if len(vals) > 0:
                task_values[display_task] = np.mean(vals)

    return task_values

def compute_deltas_for_technique(tech_config):
    """Compute performance deltas (technique - baseline) for all tasks"""
    technique = tech_config['technique']
    cache_sizes = tech_config['cache_sizes']
    models = tech_config['models']

    # Aggregate across models and contexts
    all_task_deltas = {task: [] for task in task_order}

    for model in models:
        # Get baseline performance
        baseline_helmet_16k = get_performance_values(helmet_performance_df, 'baseline', model, None, HELMET_CONTEXT_16K, helmet_tasks)
        baseline_helmet_32k = get_performance_values(helmet_performance_df, 'baseline', model, None, HELMET_CONTEXT_32K, helmet_tasks)
        baseline_longproc_2k = get_performance_values(longproc_performance_df, 'baseline', model, None, LONGPROC_CONTEXT_2K, longproc_tasks)
        baseline_longproc_5k = get_performance_values(longproc_performance_df, 'baseline', model, None, LONGPROC_CONTEXT_5K, longproc_tasks)

        # Get technique performance
        tech_helmet_16k = get_performance_values(helmet_performance_df, technique, model, cache_sizes, HELMET_CONTEXT_16K, helmet_tasks)
        tech_helmet_32k = get_performance_values(helmet_performance_df, technique, model, cache_sizes, HELMET_CONTEXT_32K, helmet_tasks)
        tech_longproc_2k = get_performance_values(longproc_performance_df, technique, model, cache_sizes, LONGPROC_CONTEXT_2K, longproc_tasks)
        tech_longproc_5k = get_performance_values(longproc_performance_df, technique, model, cache_sizes, LONGPROC_CONTEXT_5K, longproc_tasks)

        # Compute deltas for each task
        for task in task_order:
            deltas = []

            # HELMET 16K
            if task in tech_helmet_16k and task in baseline_helmet_16k:
                deltas.append(tech_helmet_16k[task] - baseline_helmet_16k[task])

            # HELMET 32K
            if task in tech_helmet_32k and task in baseline_helmet_32k:
                deltas.append(tech_helmet_32k[task] - baseline_helmet_32k[task])

            # LongProc 2K
            if task in tech_longproc_2k and task in baseline_longproc_2k:
                deltas.append(tech_longproc_2k[task] - baseline_longproc_2k[task])

            # LongProc 5K
            if task in tech_longproc_5k and task in baseline_longproc_5k:
                deltas.append(tech_longproc_5k[task] - baseline_longproc_5k[task])

            if deltas:
                all_task_deltas[task].extend(deltas)

    # Average deltas across all models and contexts
    averaged_deltas = {}
    for task, deltas in all_task_deltas.items():
        if deltas:
            averaged_deltas[task] = np.mean(deltas)
        else:
            averaged_deltas[task] = np.nan

    return averaged_deltas, all_task_deltas

# Compute deltas for all techniques
print("Computing performance deltas for all techniques...")
all_results = []

for tech_config in techniques_config:
    print(f"  Processing {tech_config['name']}...")
    averaged_deltas, raw_deltas = compute_deltas_for_technique(tech_config)

    # Compute grouped deltas
    grouped_deltas = {}
    for task, delta in averaged_deltas.items():
        group = task_grouping.get(task, task)
        if group not in grouped_deltas:
            grouped_deltas[group] = []
        if not np.isnan(delta):
            grouped_deltas[group].append(delta)

    # Average deltas within each group
    result_row = {'Technique': tech_config['name']}
    for label in grouped_task_labels:
        if label in grouped_deltas and grouped_deltas[label]:
            result_row[label] = np.mean(grouped_deltas[label])
        else:
            result_row[label] = np.nan

    all_results.append(result_row)

# Create DataFrame
df_results = pd.DataFrame(all_results)

# Save to CSV
output_path = os.path.join(output_dir, 'task_deltas_data.csv')
df_results.to_csv(output_path, index=False)
print(f"\nSaved data to: {output_path}")
print("\nData preview:")
print(df_results.to_string())

# Also save a detailed version with individual task deltas (not grouped)
print("\n\nCreating detailed (ungrouped) version...")
detailed_results = []

for tech_config in techniques_config:
    print(f"  Processing {tech_config['name']}...")
    averaged_deltas, raw_deltas = compute_deltas_for_technique(tech_config)

    result_row = {'Technique': tech_config['name']}
    for task in task_order:
        result_row[task] = averaged_deltas.get(task, np.nan)

    detailed_results.append(result_row)

df_detailed = pd.DataFrame(detailed_results)
output_path_detailed = os.path.join(output_dir, 'task_deltas_data_detailed.csv')
df_detailed.to_csv(output_path_detailed, index=False)
print(f"\nSaved detailed data to: {output_path_detailed}")
print("\nDetailed data preview:")
print(df_detailed.to_string())