import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the HELMET data
helmet_memory_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_memory_usage.csv')
helmet_throughput_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_throughput.csv')
helmet_performance_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_performance.csv')

# Filter out the 'quest' technique
helmet_memory_df = helmet_memory_df[helmet_memory_df['technique'] != 'quest']
helmet_throughput_df = helmet_throughput_df[helmet_throughput_df['technique'] != 'quest']
helmet_performance_df = helmet_performance_df[helmet_performance_df['technique'] != 'quest']

# Create output directory for plots
plots_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/results/plots'
os.makedirs(plots_dir, exist_ok=True)

# Define task columns
tasks = ["rag_nq", "recall_jsonkv", "rerank", "rag_hotpotqa", "cite"]

# Plot Memory vs Performance for each task
for task in tasks:
    plt.figure(figsize=(12, 8))
    for _, row in helmet_memory_df.iterrows():
        perf_row = helmet_performance_df[
            (helmet_performance_df['technique'] == row['technique']) &
            (helmet_performance_df['context_length'] == row['context_length']) &
            (helmet_performance_df['model'] == row['model'])
        ].iloc[0]
        label = f"{task}\n{row['context_length']}\n{row['model']}"
        if (task == "cite"):
            # cite_labels = ['str_em', 'citation_rec', 'citation_prec']
            # for cite in cite_labels:
            #     plt.scatter(row[cite], perf_row[cite], alpha=0.7)
            #     plt.annotate(label, (row[cite], perf_row[cite])) 
            continue
        else:
            plt.scatter(row[task], perf_row[task], alpha=0.7)
            plt.annotate(label, (row[task], perf_row[task]))
    plt.title(f'HELMET: Memory vs Performance for {task}')
    plt.xlabel('Memory Usage (GB)')
    plt.ylabel('Performance')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'helmet_memory_vs_performance_{task}.png'))
    plt.close()

# Plot Throughput vs Performance for each task
for task in tasks:
    plt.figure(figsize=(12, 8))
    for _, row in helmet_throughput_df.iterrows():
        perf_row = helmet_performance_df[
            (helmet_performance_df['technique'] == row['technique']) &
            (helmet_performance_df['context_length'] == row['context_length']) &
            (helmet_performance_df['model'] == row['model'])
        ].iloc[0]
        label = f"{task}\n{row['context_length']}\n{row['model']}"
        if (task == "cite"):
            # cite_labels = ['str_em', 'citation_rec', 'citation_prec']
            # for cite in cite_labels:
            #     plt.scatter(row[cite], perf_row[cite], alpha=0.7)
            #     plt.annotate(label, (row[cite], perf_row[cite])) 
            continue
        else:
            plt.scatter(row[task], perf_row[task], alpha=0.7)
            plt.annotate(label, (row[task], perf_row[task]))
    plt.title(f'HELMET: Throughput vs Performance for {task}')
    plt.xlabel('Throughput (samples/s)')
    plt.ylabel('Performance')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'helmet_throughput_vs_performance_{task}.png'))
    plt.close()