#!/usr/bin/env python3
"""
Analyze which summ_multilex experiments need to be rerun
"""
import pandas as pd
import numpy as np

# Load the CSV
df = pd.read_csv('results/helmet_results/helmet_performance.csv')

# Filter to only rows where summ_multilex is missing (NaN or 0)
# and to the 4 main models and main techniques
main_models = ['Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct',
               'DeepSeek-R1-Distill-Llama-8B', 'DeepSeek-R1-Distill-Qwen-7B']

main_techniques = ['baseline', 'INT4', 'INT8', 'duoattn', 'snapkv', 'pyramidkv', 'streamingllm']

# Get all rows for main models and techniques
df_main = df[
    (df['model'].isin(main_models)) &
    (df['technique'].isin(main_techniques))
]

# Identify missing summ_multilex data
df_main['has_multilex'] = df_main['summ_multilex'].notna() & (df_main['summ_multilex'] != 0)

# Group by technique, model, context_length, cache_size and check if multilex data exists
print('='*80)
print('MISSING SUMM_MULTILEX EXPERIMENTS')
print('='*80)
print()

missing = df_main[~df_main['has_multilex']]

if len(missing) > 0:
    print(f'Total missing experiments: {len(missing)}')
    print()

    # Group by technique to see patterns
    by_technique = missing.groupby('technique').size()
    print('Missing experiments by technique:')
    for tech, count in by_technique.items():
        print(f'  {tech}: {count} experiments')
    print()

    # Show detailed list
    print('Detailed list of missing experiments:')
    print('-'*80)
    for idx, row in missing.iterrows():
        cache_info = f" (cache: {row['cache_size']})" if pd.notna(row['cache_size']) and row['cache_size'] != 'default' else ''
        print(f"{row['technique']:20s} | {row['context_length']:4s} | {row['model']:35s}{cache_info}")
    print()

    # Also show which ones DO have multilex data for comparison
    has_data = df_main[df_main['has_multilex']]
    print('='*80)
    print(f'EXPERIMENTS WITH SUMM_MULTILEX DATA: {len(has_data)}')
    print('='*80)
    print()
    for idx, row in has_data.iterrows():
        cache_info = f" (cache: {row['cache_size']})" if pd.notna(row['cache_size']) and row['cache_size'] != 'default' else ''
        score = f"{row['summ_multilex']:.4f}" if pd.notna(row['summ_multilex']) else 'N/A'
        print(f"{row['technique']:20s} | {row['context_length']:4s} | {row['model']:35s}{cache_info} | score: {score}")
else:
    print('All main experiments have summ_multilex data!')
