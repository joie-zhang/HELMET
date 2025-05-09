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

# Merge LongProc dataframes
longproc_merged_df = longproc_memory_df.merge(longproc_throughput_df, on=['technique', 'context_length', 'model'])
longproc_merged_df = longproc_merged_df.merge(longproc_performance_df, on=['technique', 'context_length', 'model'])

# Merge HELMET dataframes
helmet_merged_df = helmet_memory_df.merge(helmet_throughput_df, on=['technique', 'context_length', 'model'])
helmet_merged_df = helmet_merged_df.merge(helmet_performance_df, on=['technique', 'context_length', 'model'])

print("\nMerged HELMET DF columns:", helmet_merged_df.columns)
print("Merged LongProc DF columns:", longproc_merged_df.columns)

# Create output directory for plots
plots_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/results/plots'
os.makedirs(plots_dir, exist_ok=True)

# Plot 1: HELMET Memory vs Performance
plt.figure(figsize=(10, 6))
plt.scatter(helmet_merged_df['rag_nq_x'], helmet_merged_df['rag_nq'], alpha=0.7)
for i, txt in enumerate(helmet_merged_df['technique']):
    plt.annotate(txt, (helmet_merged_df['rag_nq_x'].iloc[i], helmet_merged_df['rag_nq'].iloc[i]))
plt.title('HELMET: Memory vs Performance')
plt.xlabel('Memory Usage (GB)')
plt.ylabel('Performance')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'helmet_memory_vs_performance.png'))
plt.close()

# Plot 2: HELMET Throughput vs Performance
plt.figure(figsize=(10, 6))
plt.scatter(helmet_merged_df['rag_nq_y'], helmet_merged_df['rag_nq'], alpha=0.7)
for i, txt in enumerate(helmet_merged_df['technique']):
    plt.annotate(txt, (helmet_merged_df['rag_nq_y'].iloc[i], helmet_merged_df['rag_nq'].iloc[i]))
plt.title('HELMET: Throughput vs Performance')
plt.xlabel('Throughput (samples/s)')
plt.ylabel('Performance')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'helmet_throughput_vs_performance.png'))
plt.close()

# Plot 3: LongProc Memory vs Performance
plt.figure(figsize=(10, 6))
plt.scatter(longproc_merged_df['html_to_tsv_x'], longproc_merged_df['html_to_tsv'], alpha=0.7)
for i, txt in enumerate(longproc_merged_df['technique']):
    plt.annotate(txt, (longproc_merged_df['html_to_tsv_x'].iloc[i], longproc_merged_df['html_to_tsv'].iloc[i]))
plt.title('LongProc: Memory vs Performance')
plt.xlabel('Memory Usage (GB)')
plt.ylabel('Performance')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'longproc_memory_vs_performance.png'))
plt.close()

# Plot 4: LongProc Throughput vs Performance
plt.figure(figsize=(10, 6))
plt.scatter(longproc_merged_df['html_to_tsv_y'], longproc_merged_df['html_to_tsv'], alpha=0.7)
for i, txt in enumerate(longproc_merged_df['technique']):
    plt.annotate(txt, (longproc_merged_df['html_to_tsv_y'].iloc[i], longproc_merged_df['html_to_tsv'].iloc[i]))
plt.title('LongProc: Throughput vs Performance')
plt.xlabel('Throughput (samples/s)')
plt.ylabel('Performance')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'longproc_throughput_vs_performance.png'))
plt.close()