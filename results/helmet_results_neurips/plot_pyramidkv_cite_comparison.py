import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Read the data
perf_df = pd.read_csv('helmet_performance.csv')
memory_df = pd.read_csv('helmet_memory_usage.csv')
throughput_df = pd.read_csv('helmet_throughput.csv')

# Overlapping configurations
overlapping_configs = [
    'w32_c64_k5_avgpool',
    'w32_c2048_k5_avgpool',
    'w32_c4096_k5_avgpool',
    'w256_c2048_k7_avgpool',
    'w2048_c8192_k7_avgpool'
]

# Filter for PyramidKV with both models at 16k
models = ['Llama-3.1-8B-Instruct', 'DeepSeek-R1-Distill-Llama-8B']

# Filter and merge data
results = []
for model in models:
    pyramidkv_perf = perf_df[
        (perf_df['technique'] == 'pyramidkv') &
        (perf_df['context_length'] == '16k') &
        (perf_df['model'] == model) &
        (perf_df['cache_size'].isin(overlapping_configs))
    ].copy()

    pyramidkv_mem = memory_df[
        (memory_df['technique'] == 'pyramidkv') &
        (memory_df['context_length'] == '16k') &
        (memory_df['model'] == model) &
        (memory_df['cache_size'].isin(overlapping_configs))
    ].copy()

    pyramidkv_thr = throughput_df[
        (throughput_df['technique'] == 'pyramidkv') &
        (throughput_df['context_length'] == '16k') &
        (throughput_df['model'] == model) &
        (throughput_df['cache_size'].isin(overlapping_configs))
    ].copy()

    # Merge
    merged_mem = pyramidkv_perf.merge(
        pyramidkv_mem[['cache_size', 'cite']],
        on='cache_size',
        how='inner',
        suffixes=('', '_mem')
    )

    merged_thr = pyramidkv_perf.merge(
        pyramidkv_thr[['cache_size', 'cite']],
        on='cache_size',
        how='inner',
        suffixes=('', '_thr')
    )

    merged_mem_clean = merged_mem.dropna(subset=['cite_str_em', 'cite']).copy()
    merged_thr_clean = merged_thr.dropna(subset=['cite_str_em', 'cite']).copy()

    # Convert memory to GB
    merged_mem_clean['memory_gb'] = merged_mem_clean['cite'] / (1024**3)

    # Add model info
    merged_mem_clean['model'] = model
    merged_thr_clean['model'] = model

    results.append((merged_mem_clean, merged_thr_clean))

# Combine results
mem_data = pd.concat([r[0] for r in results])
thr_data = pd.concat([r[1] for r in results])

# Extract cache size and window size for labeling
def extract_cache_size(cache_size):
    match = re.search(r'_c(\d+)', str(cache_size))
    return int(match.group(1)) if match else 0

def extract_window_size(cache_size):
    match = re.search(r'w(\d+)_', str(cache_size))
    return int(match.group(1)) if match else 0

mem_data['c_value'] = mem_data['cache_size'].apply(extract_cache_size)
thr_data['c_value'] = thr_data['cache_size'].apply(extract_cache_size)
mem_data['w_value'] = mem_data['cache_size'].apply(extract_window_size)
thr_data['w_value'] = thr_data['cache_size'].apply(extract_window_size)

# Sort by window size, then cache size
mem_data = mem_data.sort_values(['w_value', 'c_value'])
thr_data = thr_data.sort_values(['w_value', 'c_value'])

# Create the plots - 1 row, 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

# Define colors for window sizes
window_colors = {
    32: '#AB63FA',
    256: '#00CC96',
    2048: '#EF553B'
}

# Define markers for models
model_markers = {
    'Llama-3.1-8B-Instruct': 'o',
    'DeepSeek-R1-Distill-Llama-8B': 's'
}

# Plot 1: cite_str_em vs memory
for model in models:
    for w_size in sorted(mem_data['w_value'].unique()):
        model_window_mem = mem_data[(mem_data['model'] == model) & (mem_data['w_value'] == w_size)]

        # Skip if no data
        if len(model_window_mem) == 0:
            continue

        ax1.scatter(model_window_mem['memory_gb'], model_window_mem['cite_str_em'],
                    alpha=0.7, s=150, c=window_colors[w_size],
                    marker=model_markers[model],
                    label=f"{model.split('-')[0]} w={w_size}")

        # Connect points with lines for same window size (only if multiple points)
        if len(model_window_mem) > 1:
            ax1.plot(model_window_mem['memory_gb'], model_window_mem['cite_str_em'],
                    color=window_colors[w_size], alpha=0.3, linestyle='--', linewidth=1.5)

        # Annotate each point with c number
        for _, row in model_window_mem.iterrows():
            ax1.annotate(f"c={row['c_value']}",
                        (row['memory_gb'], row['cite_str_em']),
                        fontsize=8, alpha=0.7,
                        xytext=(5, 5), textcoords='offset points')

ax1.set_xlabel('Memory Usage (GB)', fontsize=13, fontweight='bold')
ax1.set_ylabel('cite_str_em Score', fontsize=13, fontweight='bold')
ax1.set_title('PyramidKV: cite_str_em vs Memory\n(Llama vs DeepSeek-R1, 16k)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Plot 2: cite_str_em vs throughput (inverse of latency)
# Calculate throughput as 1/latency, handling division by zero
thr_data['throughput'] = thr_data['cite'].apply(lambda x: 1/x if x != 0 else 0)

for model in models:
    for w_size in sorted(thr_data['w_value'].unique()):
        model_window_thr = thr_data[(thr_data['model'] == model) & (thr_data['w_value'] == w_size)]

        # Skip if no data
        if len(model_window_thr) == 0:
            continue

        ax2.scatter(model_window_thr['throughput'], model_window_thr['cite_str_em'],
                    alpha=0.7, s=150, c=window_colors[w_size],
                    marker=model_markers[model],
                    label=f"{model.split('-')[0]} w={w_size}")

        # Connect points with lines for same window size (only if multiple points)
        if len(model_window_thr) > 1:
            ax2.plot(model_window_thr['throughput'], model_window_thr['cite_str_em'],
                    color=window_colors[w_size], alpha=0.3, linestyle='--', linewidth=1.5)

        # Annotate each point with c number
        for _, row in model_window_thr.iterrows():
            ax2.annotate(f"c={row['c_value']}",
                        (row['throughput'], row['cite_str_em']),
                        fontsize=8, alpha=0.7,
                        xytext=(5, 5), textcoords='offset points')

ax2.set_xlabel('Latency (sec/token)', fontsize=13, fontweight='bold')
ax2.set_ylabel('cite_str_em Score', fontsize=13, fontweight='bold')
ax2.set_title('PyramidKV: cite_str_em vs Latency\n(Llama vs DeepSeek-R1, 16k)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig('pyramidkv_cite_llama_vs_deepseek.pdf', bbox_inches='tight')
plt.savefig('pyramidkv_cite_llama_vs_deepseek.png', dpi=300, bbox_inches='tight')

print(f"Number of data points (Memory): {len(mem_data)}")
print(f"Number of data points (Throughput): {len(thr_data)}")
print(f"\nOverlapping configurations: {overlapping_configs}")
print(f"Plots saved to: pyramidkv_cite_llama_vs_deepseek.pdf and pyramidkv_cite_llama_vs_deepseek.png")
