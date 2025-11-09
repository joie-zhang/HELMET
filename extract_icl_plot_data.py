"""
Extract the exact data points used in the ICL memory-only plots.
This script replicates the filtering and data processing from plot_icl_memory_only.py
and saves the data points to CSV files for analysis.
"""

import pandas as pd
import os

# Load data
helmet_memory_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_memory_usage.csv')
helmet_performance_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_performance.csv')

# Filter to 16k context only
helmet_memory_df = helmet_memory_df[helmet_memory_df['context_length'] == '16k']
helmet_performance_df = helmet_performance_df[helmet_performance_df['context_length'] == '16k']

# Remove Qwen models
excluded_models = ['Qwen2.5-7B-Instruct', 'DeepSeek-R1-Distill-Qwen-7B', 'Qwen3-8B', 'Yarn-Qwen3-8B']
helmet_memory_df = helmet_memory_df[~helmet_memory_df['model'].isin(excluded_models)]
helmet_performance_df = helmet_performance_df[~helmet_performance_df['model'].isin(excluded_models)]

# Remove quantization techniques
quantization_techniques = ['INT4', 'INT8']
helmet_memory_df = helmet_memory_df[~helmet_memory_df['technique'].isin(quantization_techniques)]
helmet_performance_df = helmet_performance_df[~helmet_performance_df['technique'].isin(quantization_techniques)]

# Remove unwanted techniques
unwanted_techniques = ['quest', 'streamingllm_original', 'minference']
helmet_memory_df = helmet_memory_df[~helmet_memory_df['technique'].isin(unwanted_techniques)]
helmet_performance_df = helmet_performance_df[~helmet_performance_df['technique'].isin(unwanted_techniques)]

# Filter StreamingLLM to only n_local = 4092 or 4096
streamingllm_mask = (
    (helmet_memory_df['technique'] != 'streamingllm') |
    (helmet_memory_df['cache_size'].str.contains('n_local_4092|n_local_4096', na=False))
)
helmet_memory_df = helmet_memory_df[streamingllm_mask]

streamingllm_mask_perf = (
    (helmet_performance_df['technique'] != 'streamingllm') |
    (helmet_performance_df['cache_size'].str.contains('n_local_4092|n_local_4096', na=False))
)
helmet_performance_df = helmet_performance_df[streamingllm_mask_perf]

# Filter SnapKV and PyramidKV to specific cache sizes
allowed_snapkv_pyramidkv_configs = ['w256_c2048_k7_avgpool', 'w256_c2048_k7_maxpool',
                                     'w2048_c8192_k7_avgpool', 'w2048_c8192_k7_maxpool']

snapkv_pyramidkv_mask = (
    (~helmet_memory_df['technique'].isin(['snapkv', 'pyramidkv'])) |
    (helmet_memory_df['cache_size'].isin(allowed_snapkv_pyramidkv_configs))
)
helmet_memory_df = helmet_memory_df[snapkv_pyramidkv_mask]

snapkv_pyramidkv_mask_perf = (
    (~helmet_performance_df['technique'].isin(['snapkv', 'pyramidkv'])) |
    (helmet_performance_df['cache_size'].isin(allowed_snapkv_pyramidkv_configs))
)
helmet_performance_df = helmet_performance_df[snapkv_pyramidkv_mask_perf]

# Helper function to format cache size for display
def format_cache_size(cache_size: str) -> str:
    if cache_size == "default":
        return "baseline"
    elif cache_size.startswith("n_local_"):
        parts = cache_size.split('_')
        n_local = parts[2]
        n_init = parts[4]
        return f"L={n_local},I={n_init}"
    elif cache_size.startswith("w") and "_c" in cache_size and "_k" in cache_size:
        parts = cache_size.split('_')
        window_val = parts[0][1:]
        cache_val = parts[1][1:]
        pooling = parts[3] if len(parts) > 3 else ''
        return f"W={window_val},C={cache_val},{pooling}"
    elif cache_size.startswith("sp"):
        parts = cache_size.split('_')
        sparsity = parts[0].replace('sp', '')
        return f"S={sparsity}"
    else:
        return cache_size

# Extract data points for ICL tasks
data_points = []

for (technique, model), group in helmet_memory_df.groupby(['technique', 'model']):
    # Sort by cache size if applicable
    if 'cache_size' in group.columns:
        if technique == "streamingllm":
            group['sort_key'] = group['cache_size'].apply(
                lambda x: int(x.split('_')[2]) if x.startswith('n_local_') else 0
            )
            group = group.sort_values('sort_key').drop('sort_key', axis=1)
        elif technique in ["snapkv", "pyramidkv"]:
            def extract_cache_and_k(cache_size):
                if not cache_size.startswith('w'):
                    return (0, 0, 0)
                parts = cache_size.split('_')
                cache_val = int(parts[1][1:])
                k_val = int(parts[2][1:])
                return (cache_val, k_val, 0)

            group['sort_key'] = group['cache_size'].apply(extract_cache_and_k)
            group = group.sort_values('sort_key').drop('sort_key', axis=1)

    # Collect data points
    for idx, row in group.iterrows():
        # Average memory usage across icl_clinic and icl_banking
        memory_cols = [col for col in ['icl_clinic', 'icl_banking'] if col in row.index]
        if not memory_cols:
            continue

        # Calculate average ICL memory
        icl_clinic_mem = row['icl_clinic']
        icl_banking_mem = row['icl_banking']

        if pd.isna(icl_clinic_mem) or pd.isna(icl_banking_mem):
            continue

        avg_memory = (icl_clinic_mem + icl_banking_mem) / 2

        # Find corresponding performance value
        perf_row = helmet_performance_df[
            (helmet_performance_df['technique'] == row['technique']) &
            (helmet_performance_df['context_length'] == row['context_length']) &
            (helmet_performance_df['model'] == row['model']) &
            (helmet_performance_df['cache_size'] == row['cache_size'])
        ]

        if perf_row.empty:
            continue

        # Calculate average ICL performance
        icl_clinic_perf = perf_row.iloc[0]['icl_clinic']
        icl_banking_perf = perf_row.iloc[0]['icl_banking']

        if pd.isna(icl_clinic_perf) or pd.isna(icl_banking_perf) or icl_clinic_perf == 0 or icl_banking_perf == 0:
            continue

        avg_performance = (icl_clinic_perf + icl_banking_perf) / 2

        data_points.append({
            'technique': row['technique'],
            'model': row['model'],
            'context_length': row['context_length'],
            'cache_size': row['cache_size'],
            'cache_size_display': format_cache_size(row['cache_size']),
            'icl_clinic_memory_gb': icl_clinic_mem,
            'icl_banking_memory_gb': icl_banking_mem,
            'avg_icl_memory_gb': avg_memory,
            'icl_clinic_performance': icl_clinic_perf,
            'icl_banking_performance': icl_banking_perf,
            'avg_icl_performance': avg_performance
        })

# Create DataFrame
df_plot_data = pd.DataFrame(data_points)

# Sort by technique, model, and memory for readability
df_plot_data = df_plot_data.sort_values(['technique', 'model', 'avg_icl_memory_gb'])

# Save to CSV
output_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/results'
output_path = os.path.join(output_dir, 'icl_plot_data_16k.csv')
df_plot_data.to_csv(output_path, index=False)

print(f"Data extracted successfully!")
print(f"Saved to: {output_path}")
print(f"Total data points: {len(df_plot_data)}")
print("\nData points by technique and model:")
print(df_plot_data.groupby(['technique', 'model']).size())
print("\nFirst few rows:")
print(df_plot_data.head(10))
print("\nSummary statistics:")
print(df_plot_data.describe())
