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

# Define colors and markers for techniques
techniques = longproc_memory_df['technique'].unique()
colors = plt.cm.rainbow(np.linspace(0, 1, len(techniques)))
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd']

# Plot 1: LongProc Memory vs Performance for each task
tasks = ['html_to_tsv', 'pseudo_to_code', 'travel_planning']
for task in tasks:
    plt.figure(figsize=(12, 8))
    for technique, color, marker in zip(techniques, colors, markers):
        task_data = longproc_memory_df[longproc_memory_df['technique'] == technique]
        for _, row in task_data.iterrows():
            perf_row = longproc_performance_df[
                (longproc_performance_df['technique'] == row['technique']) &
                (longproc_performance_df['context_length'] == row['context_length']) &
                (longproc_performance_df['model'] == row['model'])
            ].iloc[0]
            label = f"{task}\n{row['context_length']}\n{row['model']}"
            plt.scatter(row[task], perf_row[task], alpha=0.7, color=color, marker=marker, label=technique)
            plt.annotate(label, (row[task], perf_row[task]))
    plt.title(f'LongProc: Memory vs Performance for {task}')
    plt.xlabel('Memory Usage (GB)')
    plt.ylabel('Performance')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'longproc_memory_vs_performance_{task}.png'), bbox_inches='tight')
    plt.close()

# Plot 2: LongProc Throughput vs Performance for each task
for task in tasks:
    plt.figure(figsize=(12, 8))
    for technique, color, marker in zip(techniques, colors, markers):
        task_data = longproc_throughput_df[longproc_throughput_df['technique'] == technique]
        for _, row in task_data.iterrows():
            perf_row = longproc_performance_df[
                (longproc_performance_df['technique'] == row['technique']) &
                (longproc_performance_df['context_length'] == row['context_length']) &
                (longproc_performance_df['model'] == row['model'])
            ].iloc[0]
            label = f"{task}\n{row['context_length']}\n{row['model']}"
            plt.scatter(row[task], perf_row[task], alpha=0.7, color=color, marker=marker, label=technique)
            plt.annotate(label, (row[task], perf_row[task]))
    plt.title(f'LongProc: Throughput vs Performance for {task}')
    plt.xlabel('Throughput (samples/s)')
    plt.ylabel('Performance')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'longproc_throughput_vs_performance_{task}.png'), bbox_inches='tight')
    plt.close()