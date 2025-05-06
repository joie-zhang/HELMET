import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the LongProc data
longproc_memory_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/longproc_results/longproc_memory_usage.csv')
longproc_throughput_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/longproc_results/longproc_throughput.csv')
longproc_performance_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/longproc_results/longproc_performance.csv')

# Load the HELMET data
helmet_memory_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_memory_usage.csv')
helmet_throughput_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_throughput.csv')
helmet_performance_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_performance.csv')

# Print column names to debug
print("HELMET Memory DF columns:", helmet_memory_df.columns)
print("HELMET Throughput DF columns:", helmet_throughput_df.columns) 
print("HELMET Performance DF columns:", helmet_performance_df.columns)

print("LongProc Memory DF columns:", longproc_memory_df.columns)
print("LongProc Throughput DF columns:", longproc_throughput_df.columns)
print("LongProc Performance DF columns:", longproc_performance_df.columns)

# Create output directory for plots
plots_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/results/plots'
os.makedirs(plots_dir, exist_ok=True)

# Plot 1: HELMET Memory vs Performance
plt.figure(figsize=(12, 8))
for _, row in helmet_memory_df.iterrows():
    perf_row = helmet_performance_df[
        (helmet_performance_df['technique'] == row['technique']) &
        (helmet_performance_df['context_length'] == row['context_length']) &
        (helmet_performance_df['model'] == row['model'])
    ].iloc[0]
    label = f"{row['technique']}\n{row['context_length']}\n{row['model']}"
    plt.scatter(row['rag_nq'], perf_row['rag_nq'], alpha=0.7)
    plt.annotate(label, (row['rag_nq'], perf_row['rag_nq']))
plt.title('HELMET: Memory vs Performance')
plt.xlabel('Memory Usage (GB)')
plt.ylabel('Performance')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'helmet_memory_vs_performance.png'))
plt.close()

# Plot 2: HELMET Throughput vs Performance
plt.figure(figsize=(12, 8))
for _, row in helmet_throughput_df.iterrows():
    perf_row = helmet_performance_df[
        (helmet_performance_df['technique'] == row['technique']) &
        (helmet_performance_df['context_length'] == row['context_length']) &
        (helmet_performance_df['model'] == row['model'])
    ].iloc[0]
    label = f"{row['technique']}\n{row['context_length']}\n{row['model']}"
    plt.scatter(row['rag_nq'], perf_row['rag_nq'], alpha=0.7)
    plt.annotate(label, (row['rag_nq'], perf_row['rag_nq']))
plt.title('HELMET: Throughput vs Performance')
plt.xlabel('Throughput (samples/s)')
plt.ylabel('Performance')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'helmet_throughput_vs_performance.png'))
plt.close()

# Plot 3: LongProc Memory vs Performance
plt.figure(figsize=(12, 8))
for _, row in longproc_memory_df.iterrows():
    perf_row = longproc_performance_df[
        (longproc_performance_df['technique'] == row['technique']) &
        (longproc_performance_df['context_length'] == row['context_length']) &
        (longproc_performance_df['model'] == row['model'])
    ].iloc[0]
    label = f"{row['technique']}\n{row['context_length']}\n{row['model']}"
    plt.scatter(row['html_to_tsv'], perf_row['html_to_tsv'], alpha=0.7)
    plt.annotate(label, (row['html_to_tsv'], perf_row['html_to_tsv']))
plt.title('LongProc: Memory vs Performance')
plt.xlabel('Memory Usage (GB)')
plt.ylabel('Performance')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'longproc_memory_vs_performance.png'))
plt.close()

# Plot 4: LongProc Throughput vs Performance
plt.figure(figsize=(12, 8))
for _, row in longproc_throughput_df.iterrows():
    perf_row = longproc_performance_df[
        (longproc_performance_df['technique'] == row['technique']) &
        (longproc_performance_df['context_length'] == row['context_length']) &
        (longproc_performance_df['model'] == row['model'])
    ].iloc[0]
    label = f"{row['technique']}\n{row['context_length']}\n{row['model']}"
    plt.scatter(row['html_to_tsv'], perf_row['html_to_tsv'], alpha=0.7)
    plt.annotate(label, (row['html_to_tsv'], perf_row['html_to_tsv']))
plt.title('LongProc: Throughput vs Performance')
plt.xlabel('Throughput (samples/s)')
plt.ylabel('Performance')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'longproc_throughput_vs_performance.png'))
plt.close()