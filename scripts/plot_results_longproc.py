import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Load the LongProc data
longproc_memory_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/longproc_results/longproc_memory_usage.csv')
longproc_throughput_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/longproc_results/longproc_throughput.csv')
longproc_performance_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/longproc_results/longproc_performance.csv')

# Filter out the 'quest' technique
longproc_memory_df = longproc_memory_df[longproc_memory_df['technique'] != 'quest']
longproc_throughput_df = longproc_throughput_df[longproc_throughput_df['technique'] != 'quest']
longproc_performance_df = longproc_performance_df[longproc_performance_df['technique'] != 'quest']

# Create output directory for plots
plots_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/results/plots'
os.makedirs(plots_dir, exist_ok=True)

# Plot 1: LongProc Memory vs Performance
plt.figure(figsize=(12, 8))
tasks = longproc_memory_df['technique'].unique()
colors = plt.cm.rainbow(np.linspace(0, 1, len(tasks)))
for task, color in zip(tasks, colors):
    task_data = longproc_memory_df[longproc_memory_df['technique'] == task]
    for _, row in task_data.iterrows():
        perf_row = longproc_performance_df[
            (longproc_performance_df['technique'] == row['technique']) &
            (longproc_performance_df['context_length'] == row['context_length']) &
            (longproc_performance_df['model'] == row['model'])
        ].iloc[0]
        label = f"{row['context_length']}\n{row['model']}"
        plt.scatter(row['html_to_tsv'], perf_row['html_to_tsv'], alpha=0.7, color=color, label=task)
        plt.annotate(label, (row['html_to_tsv'], perf_row['html_to_tsv']))
plt.title('LongProc: Memory vs Performance')
plt.xlabel('Memory Usage (GB)')
plt.ylabel('Performance')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'longproc_memory_vs_performance.png'), bbox_inches='tight')
plt.close()

# Plot 2: LongProc Throughput vs Performance
plt.figure(figsize=(12, 8))
tasks = longproc_throughput_df['technique'].unique()
colors = plt.cm.rainbow(np.linspace(0, 1, len(tasks)))
for task, color in zip(tasks, colors):
    task_data = longproc_throughput_df[longproc_throughput_df['technique'] == task]
    for _, row in task_data.iterrows():
        perf_row = longproc_performance_df[
            (longproc_performance_df['technique'] == row['technique']) &
            (longproc_performance_df['context_length'] == row['context_length']) &
            (longproc_performance_df['model'] == row['model'])
        ].iloc[0]
        label = f"{row['context_length']}\n{row['model']}"
        plt.scatter(row['html_to_tsv'], perf_row['html_to_tsv'], alpha=0.7, color=color, label=task)
        plt.annotate(label, (row['html_to_tsv'], perf_row['html_to_tsv']))
plt.title('LongProc: Throughput vs Performance')
plt.xlabel('Throughput (samples/s)')
plt.ylabel('Performance')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'longproc_throughput_vs_performance.png'), bbox_inches='tight')
plt.close()