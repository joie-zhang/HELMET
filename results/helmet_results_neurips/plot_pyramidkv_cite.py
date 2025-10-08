import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Read the data
perf_df = pd.read_csv('helmet_performance.csv')
memory_df = pd.read_csv('helmet_memory_usage.csv')
throughput_df = pd.read_csv('helmet_throughput.csv')

# Filter for PyramidKV with DeepSeek-R1-Distill-Llama-8B at 16k
pyramidkv_perf = perf_df[
    (perf_df['technique'] == 'pyramidkv') &
    (perf_df['context_length'] == '16k') &
    (perf_df['model'] == 'DeepSeek-R1-Distill-Llama-8B')
].copy()

pyramidkv_mem = memory_df[
    (memory_df['technique'] == 'pyramidkv') &
    (memory_df['context_length'] == '16k') &
    (memory_df['model'] == 'DeepSeek-R1-Distill-Llama-8B')
].copy()

pyramidkv_thr = throughput_df[
    (throughput_df['technique'] == 'pyramidkv') &
    (throughput_df['context_length'] == '16k') &
    (throughput_df['model'] == 'DeepSeek-R1-Distill-Llama-8B')
].copy()

# Merge the dataframes
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

# Remove rows with NaN values in cite_str_em, memory, or throughput
merged_mem_clean = merged_mem.dropna(subset=['cite_str_em', 'cite']).copy()
merged_thr_clean = merged_thr.dropna(subset=['cite_str_em', 'cite']).copy()

# Extract window size and k value from cache_size
def extract_window_size(cache_size):
    match = re.search(r'w(\d+)_', str(cache_size))
    return int(match.group(1)) if match else 0

def extract_k_value(cache_size):
    match = re.search(r'_k(\d+)_', str(cache_size))
    return int(match.group(1)) if match else 0

merged_mem_clean['window_size'] = merged_mem_clean['cache_size'].apply(extract_window_size)
merged_thr_clean['window_size'] = merged_thr_clean['cache_size'].apply(extract_window_size)
merged_mem_clean['k_value'] = merged_mem_clean['cache_size'].apply(extract_k_value)
merged_thr_clean['k_value'] = merged_thr_clean['cache_size'].apply(extract_k_value)

# Convert memory from bytes to GB
merged_mem_clean['memory_gb'] = merged_mem_clean['cite'] / (1024**3)

# Sort by window size
merged_mem_clean = merged_mem_clean.sort_values('window_size')
merged_thr_clean = merged_thr_clean.sort_values('window_size')

# Filter for k=5 only
k5_mem = merged_mem_clean[merged_mem_clean['k_value'] == 5]
k5_thr = merged_thr_clean[merged_thr_clean['k_value'] == 5]

# Get baseline data for DeepSeek-R1-Distill-Llama-8B at 16k
baseline_perf = perf_df[
    (perf_df['technique'] == 'baseline') &
    (perf_df['context_length'] == '16k') &
    (perf_df['model'] == 'DeepSeek-R1-Distill-Llama-8B')
].copy()

baseline_mem = memory_df[
    (memory_df['technique'] == 'baseline') &
    (memory_df['context_length'] == '16k') &
    (memory_df['model'] == 'DeepSeek-R1-Distill-Llama-8B')
].copy()

baseline_thr = throughput_df[
    (throughput_df['technique'] == 'baseline') &
    (throughput_df['context_length'] == '16k') &
    (throughput_df['model'] == 'DeepSeek-R1-Distill-Llama-8B')
].copy()

# Extract baseline values
if len(baseline_perf) > 0:
    baseline_cite_str_em = baseline_perf['cite_str_em'].values[0]
else:
    baseline_cite_str_em = None

if len(baseline_mem) > 0:
    baseline_memory_gb = baseline_mem['cite'].values[0] / (1024**3)
else:
    baseline_memory_gb = None

if len(baseline_thr) > 0:
    baseline_latency = baseline_thr['cite'].values[0]
else:
    baseline_latency = None

# Create the plots - 1 row, 2 columns (memory and throughput)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

# Define colors and markers for different window sizes
unique_windows = sorted(k5_mem['window_size'].unique())
custom_colors = ['#AB63FA', '#00CC96', '#EF553B', '#FFA15A', '#19D3F3', '#FF97FF', '#B6E880']
colors = [custom_colors[i % len(custom_colors)] for i in range(len(unique_windows))]
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

# Plot 1: k=5, cite_str_em vs memory
for i, window_size in enumerate(unique_windows):
    mem_subset = k5_mem[k5_mem['window_size'] == window_size]
    if len(mem_subset) > 0:
        ax1.scatter(mem_subset['memory_gb'], mem_subset['cite_str_em'],
                    alpha=0.7, s=120, c=[colors[i]],
                    marker=markers[i % len(markers)],
                    label=f'w={window_size}')
        # Annotate each point with c number
        for _, row in mem_subset.iterrows():
            match = re.search(r'_c(\d+)', str(row['cache_size']))
            if match:
                c_num = match.group(1)
                ax1.annotate(c_num,
                            (row['memory_gb'], row['cite_str_em']),
                            fontsize=7, alpha=0.7,
                            xytext=(5, 5), textcoords='offset points')

# Add baseline marker to plot 1
if baseline_cite_str_em is not None and baseline_memory_gb is not None:
    ax1.scatter(baseline_memory_gb, baseline_cite_str_em,
                marker='*', s=500, c='black', edgecolors='gold', linewidths=2,
                label='Baseline', zorder=10)
    ax1.annotate('Baseline',
                (baseline_memory_gb, baseline_cite_str_em),
                fontsize=10, fontweight='bold',
                xytext=(10, 10), textcoords='offset points')

ax1.set_xlabel('Memory Usage (GB)', fontsize=12)
ax1.set_ylabel('cite_str_em Score', fontsize=12)
ax1.set_title('PyramidKV (k=5): cite_str_em vs Memory\n(DeepSeek-R1-Distill-Llama-8B, 16k)', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Plot 2: k=5, cite_str_em vs throughput
for i, window_size in enumerate(unique_windows):
    thr_subset = k5_thr[k5_thr['window_size'] == window_size].copy()
    if len(thr_subset) > 0:
        # Calculate throughput as 1/latency, handling division by zero
        thr_subset['throughput'] = thr_subset['cite'].apply(lambda x: 1/x if x != 0 else 0)

        ax2.scatter(thr_subset['throughput'], thr_subset['cite_str_em'],
                    alpha=0.7, s=120, c=[colors[i]],
                    marker=markers[i % len(markers)],
                    label=f'w={window_size}')
        # Annotate each point with c number
        for _, row in thr_subset.iterrows():
            match = re.search(r'_c(\d+)', str(row['cache_size']))
            if match:
                c_num = match.group(1)
                ax2.annotate(c_num,
                            (row['throughput'], row['cite_str_em']),
                            fontsize=7, alpha=0.7,
                            xytext=(5, 5), textcoords='offset points')

# Add baseline marker to plot 2
if baseline_cite_str_em is not None and baseline_latency is not None:
    ax2.scatter(baseline_latency, baseline_cite_str_em,
                marker='*', s=500, c='black', edgecolors='gold', linewidths=2,
                label='Baseline', zorder=10)
    ax2.annotate('Baseline',
                (baseline_latency, baseline_cite_str_em),
                fontsize=10, fontweight='bold',
                xytext=(10, 10), textcoords='offset points')

ax2.set_xlabel('Latency (sec/token)', fontsize=12)
ax2.set_ylabel('cite_str_em Score', fontsize=12)
ax2.set_title('PyramidKV (k=5): cite_str_em vs Latency\n(DeepSeek-R1-Distill-Llama-8B, 16k)', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig('pyramidkv_cite_analysis.pdf', bbox_inches='tight')
plt.savefig('pyramidkv_cite_analysis.png', dpi=300, bbox_inches='tight')

print(f"Number of data points (Memory): {len(k5_mem)}")
print(f"Number of data points (Throughput): {len(k5_thr)}")
print(f"\nWindow sizes: {unique_windows}")
print(f"Plots saved to: pyramidkv_cite_analysis.pdf and pyramidkv_cite_analysis.png")
