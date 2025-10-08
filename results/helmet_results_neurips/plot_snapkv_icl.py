import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Read the data
perf_df = pd.read_csv('helmet_performance.csv')
memory_df = pd.read_csv('helmet_memory_usage.csv')
throughput_df = pd.read_csv('helmet_throughput.csv')

# Filter for SnapKV with DeepSeek-R1-Distill-Llama-8B at 16k
snapkv_perf = perf_df[
    (perf_df['technique'] == 'snapkv') &
    (perf_df['context_length'] == '16k') &
    (perf_df['model'] == 'DeepSeek-R1-Distill-Llama-8B')
].copy()

snapkv_mem = memory_df[
    (memory_df['technique'] == 'snapkv') &
    (memory_df['context_length'] == '16k') &
    (memory_df['model'] == 'DeepSeek-R1-Distill-Llama-8B')
].copy()

snapkv_thr = throughput_df[
    (throughput_df['technique'] == 'snapkv') &
    (throughput_df['context_length'] == '16k') &
    (throughput_df['model'] == 'DeepSeek-R1-Distill-Llama-8B')
].copy()

# Merge the dataframes
merged_mem = snapkv_perf.merge(
    snapkv_mem[['cache_size', 'icl_clinic']],
    on='cache_size',
    how='inner',
    suffixes=('', '_mem')
)

merged_thr = snapkv_perf.merge(
    snapkv_thr[['cache_size', 'icl_clinic']],
    on='cache_size',
    how='inner',
    suffixes=('', '_thr')
)

# Remove rows with NaN values in icl_clinic, memory, or throughput
merged_mem_clean = merged_mem.dropna(subset=['icl_clinic', 'icl_clinic_mem']).copy()
merged_thr_clean = merged_thr.dropna(subset=['icl_clinic', 'icl_clinic_thr']).copy()

# Extract window size from cache_size
def extract_window_size(cache_size):
    match = re.search(r'w(\d+)_', str(cache_size))
    return int(match.group(1)) if match else 0

merged_mem_clean['window_size'] = merged_mem_clean['cache_size'].apply(extract_window_size)
merged_thr_clean['window_size'] = merged_thr_clean['cache_size'].apply(extract_window_size)

# Convert memory from bytes to GB
merged_mem_clean['memory_gb'] = merged_mem_clean['icl_clinic_mem'] / (1024**3)

# Sort by window size
merged_mem_clean = merged_mem_clean.sort_values('window_size')
merged_thr_clean = merged_thr_clean.sort_values('window_size')

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
    baseline_icl_clinic = baseline_perf['icl_clinic'].values[0]
else:
    baseline_icl_clinic = None

if len(baseline_mem) > 0:
    baseline_memory_gb = baseline_mem['icl_clinic'].values[0] / (1024**3)
else:
    baseline_memory_gb = None

if len(baseline_thr) > 0:
    baseline_latency = baseline_thr['icl_clinic'].values[0]
else:
    baseline_latency = None

# Create the plots - 1 row, 2 columns (memory and throughput)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

# Define colors and markers for different window sizes
unique_windows = sorted(merged_mem_clean['window_size'].unique())
custom_colors = ['#AB63FA', '#00CC96', '#EF553B', '#FFA15A', '#19D3F3', '#FF97FF', '#B6E880']
colors = [custom_colors[i % len(custom_colors)] for i in range(len(unique_windows))]
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

# Plot 1: icl_clinic vs memory
for i, window_size in enumerate(unique_windows):
    mem_subset = merged_mem_clean[merged_mem_clean['window_size'] == window_size]
    if len(mem_subset) > 0:
        ax1.scatter(mem_subset['memory_gb'], mem_subset['icl_clinic'],
                    alpha=0.7, s=120, c=[colors[i]],
                    marker=markers[i % len(markers)],
                    label=f'w={window_size}')
        # Annotate each point with c number
        for _, row in mem_subset.iterrows():
            match = re.search(r'_c(\d+)', str(row['cache_size']))
            if match:
                c_num = match.group(1)
                ax1.annotate(c_num,
                            (row['memory_gb'], row['icl_clinic']),
                            fontsize=7, alpha=0.7,
                            xytext=(5, 5), textcoords='offset points')

# Add baseline marker to plot 1
if baseline_icl_clinic is not None and baseline_memory_gb is not None:
    ax1.scatter(baseline_memory_gb, baseline_icl_clinic,
                marker='*', s=500, c='black', edgecolors='gold', linewidths=2,
                label='Baseline', zorder=10)
    ax1.annotate('Baseline',
                (baseline_memory_gb, baseline_icl_clinic),
                fontsize=10, fontweight='bold',
                xytext=(10, 10), textcoords='offset points')

ax1.set_xlabel('Memory Usage (GB)', fontsize=12)
ax1.set_ylabel('ICL Clinic Score', fontsize=12)
ax1.set_title('SnapKV: ICL Clinic vs Memory\n(DeepSeek-R1-Distill-Llama-8B, 16k)', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Plot 2: icl_clinic vs throughput
for i, window_size in enumerate(unique_windows):
    thr_subset = merged_thr_clean[merged_thr_clean['window_size'] == window_size].copy()
    if len(thr_subset) > 0:
        # Calculate throughput as 1/latency, handling division by zero
        thr_subset['throughput'] = thr_subset['icl_clinic_thr'].apply(lambda x: 1/x if x != 0 else 0)

        ax2.scatter(thr_subset['throughput'], thr_subset['icl_clinic'],
                    alpha=0.7, s=120, c=[colors[i]],
                    marker=markers[i % len(markers)],
                    label=f'w={window_size}')
        # Annotate each point with c number
        for _, row in thr_subset.iterrows():
            match = re.search(r'_c(\d+)', str(row['cache_size']))
            if match:
                c_num = match.group(1)
                ax2.annotate(c_num,
                            (row['throughput'], row['icl_clinic']),
                            fontsize=7, alpha=0.7,
                            xytext=(5, 5), textcoords='offset points')

# Add baseline marker to plot 2
if baseline_icl_clinic is not None and baseline_latency is not None:
    ax2.scatter(baseline_latency, baseline_icl_clinic,
                marker='*', s=500, c='black', edgecolors='gold', linewidths=2,
                label='Baseline', zorder=10)
    ax2.annotate('Baseline',
                (baseline_latency, baseline_icl_clinic),
                fontsize=10, fontweight='bold',
                xytext=(10, 10), textcoords='offset points')

ax2.set_xlabel('Latency (sec/token)', fontsize=12)
ax2.set_ylabel('ICL Clinic Score', fontsize=12)
ax2.set_title('SnapKV: ICL Clinic vs Latency\n(DeepSeek-R1-Distill-Llama-8B, 16k)', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig('snapkv_icl_analysis.pdf', bbox_inches='tight')
plt.savefig('snapkv_icl_analysis.png', dpi=300, bbox_inches='tight')

print(f"Number of data points (Memory): {len(merged_mem_clean)}")
print(f"Number of data points (Throughput): {len(merged_thr_clean)}")
print(f"\nWindow sizes: {unique_windows}")
print(f"Plots saved to: snapkv_icl_analysis.pdf and snapkv_icl_analysis.png")
