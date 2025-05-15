import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib.lines import Line2D
from tqdm import tqdm

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

# 1) Seaborn style
sns.set(style='whitegrid')

# 2) Define the 7 performance tasks (including the three cite metrics) and the two contexts
perf_tasks = [
    'recall_jsonkv', 'rag_nq', 'rag_hotpotqa', 'rerank',
    'cite_str_em', 'cite_citation_rec', 'cite_citation_prec'
]
contexts = ['16k', '32k']

# 3) Color palette for models, marker shapes for techniques
model_palette = {
    'Llama-3.1-8B-Instruct': 'tab:orange',
    'Qwen2.5-7B-Instruct':    'tab:blue'
}
marker_dict = {
    'baseline':     'o',
    'INT8':         's',
    'INT4':         '^',
    'minference':   'D',
    'pyramidkv':    'P',
    'snapkv':       'X',
    'streamingllm': '*',
    'streamingllm_original': '8',
}

# Add marker size dictionary to compensate for visual differences
marker_size_dict = {
    'o': 100,  # baseline
    's': 100,  # INT8
    '^': 100,  # INT4
    'D': 100,  # minference
    'P': 120,  # pyramidkv - slightly larger
    'X': 120,  # snapkv - slightly larger
    '*': 200,  # streamingllm - make stars bigger
    '8': 85,  # streamingllm_original - make octagons smaller
}

# 4) Create a 7×4 grid
fig, axes = plt.subplots(
    nrows=len(perf_tasks),
    ncols=4,
    figsize=(20, 30),
)

# 5) Populate each subplot
for i, perf_task in tqdm(enumerate(perf_tasks), desc='Processing tasks', total=len(perf_tasks)):
    # memory/throughput column uses same underlying column 'cite' for all cite_* tasks
    mem_thr_col = 'cite' if perf_task.startswith('cite_') else perf_task

    for j in range(4):
        ax = axes[i, j]
        if j < 2:
            df      = helmet_memory_df
            context = contexts[j]
            x_label = 'Memory (GB)'
        else:
            df      = helmet_throughput_df
            context = contexts[j - 2]
            # Calculate latency as 1/throughput with zero handling
            df = df.copy()  # Create a copy to avoid modifying the original
            for task in ['recall_jsonkv', 'rag_nq', 'rag_hotpotqa', 'rerank', 'cite']:
                # Replace 0 with NaN to avoid division by zero
                df[task] = df[task].replace(0, float('nan'))
                # Convert non-zero throughput to latency
                df[task] = 1 / df[task]
            x_label = 'Latency (s/sample)'

        # plot one dot per (technique, model)
        subset = df[df['context_length'] == context]
        for _, row in tqdm(subset.iterrows(), desc=f'Plotting {perf_task} ({context})', leave=False):
            x = row[mem_thr_col]
            # find the matching performance
            perf_row = helmet_performance_df[
                (helmet_performance_df['technique']     == row['technique']) &
                (helmet_performance_df['context_length'] == context) &
                (helmet_performance_df['model']         == row['model'])
            ]
            if perf_row.empty or pd.isna(x):
                continue
            y = perf_row.iloc[0][perf_task]

            ax.scatter(
                x, y,
                color = model_palette[row['model']],
                marker= marker_dict[row['technique']],
                s      = marker_size_dict[marker_dict[row['technique']]],
                edgecolor='k',
                linewidth=0.5,
            )

        # only label the leftmost column with the task name
        if j == 0:
            ax.set_ylabel(perf_task.replace('_', ' ').title(), fontsize=8)
        # only label the bottom row with the x‐axis label
        if i == len(perf_tasks) - 1:
            ax.set_xlabel(x_label, fontsize=8)
        # only title the top row with the column header
        if i == 0:
            col_title = (f"Memory {context.replace('k', 'K')}" if j < 2 else 
                        f"Latency {context.replace('k', 'K')}")
            ax.set_title(col_title, fontsize=10)

# 6) Build a shared legend to the right
model_handles = [
    Line2D([0], [0],
           marker='o', color='w',
           markerfacecolor=color,
           markersize=10,
           label=model)
    for model, color in model_palette.items()
]
tech_handles = [
    Line2D([0], [0],
           marker=marker, color='k',
           linestyle='None',
           markersize=10,
           label=tech)
    for tech, marker in marker_dict.items()
]

all_handles = model_handles + tech_handles
all_labels  = list(model_palette.keys()) + list(marker_dict.keys())

fig.legend(
    all_handles,
    all_labels,
    loc='center',
    ncol=len(all_handles),
    bbox_to_anchor=(0.5, 0.05),  # Changed from 0.02 to 0.05 to move legend up
    bbox_transform=fig.transFigure,
    title='Legend',
    fontsize=10,
)

# Adjust the bottom margin in tight_layout
plt.tight_layout(rect=[0, 0.06, 1, 0.98])  # Changed from [0, 0.08, 1, 0.98] to reduce bottom margin

print("Saving plot...")
# Save with a higher DPI to ensure legend is clear
fig.savefig(os.path.join(plots_dir, 'helmet_overall_plot.png'), 
            bbox_inches='tight',  # Added to ensure legend is included
            dpi=300)  # Increased DPI for better quality
print("Done!")