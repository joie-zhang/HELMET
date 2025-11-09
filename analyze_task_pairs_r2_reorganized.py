#!/usr/bin/env python3
"""
Calculate R² coefficients for task pairs with reorganized axes
Y-axis: niah, recall_jsonkv, rag_hotpotqa, rag_nq
X-axis: icl_clinic, icl_banking, rerank, summ_multilex, cite (averaged)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Set font to Arial
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

def calculate_r2(x, y):
    """Calculate R² coefficient between two arrays"""
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 2:
        return np.nan

    # Calculate correlation coefficient
    correlation = np.corrcoef(x_clean, y_clean)[0, 1]
    r2 = correlation ** 2

    return r2

# Read the CSV files
helmet_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_performance.csv')
longproc_df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/longproc_results/longproc_performance.csv')

# Merge the datasets on technique, model, and cache_size (ignoring context_length)
# This allows us to correlate HELMET tasks (tested at 16k/32k) with longproc tasks (tested at 5k/8k)
# for the same technique/model/cache combinations
df = pd.merge(helmet_df, longproc_df,
              on=['technique', 'model', 'cache_size'],
              how='outer',
              suffixes=('', '_longproc'))

# Drop the duplicate context_length column from longproc
if 'context_length_longproc' in df.columns:
    df = df.drop(columns=['context_length_longproc'])

# Calculate averaged cite metric
cite_cols = ['cite_str_em', 'cite_citation_rec', 'cite_citation_prec']
df['cite_avg'] = df[cite_cols].mean(axis=1)

# Average RAG tasks
df['rag_avg'] = df[['rag_hotpotqa', 'rag_nq']].mean(axis=1)

# Average ICL tasks
df['icl_avg'] = df[['icl_clinic', 'icl_banking']].mean(axis=1)

# Define Y-axis tasks (rows) - with averaged RAG
y_tasks = ['niah', 'recall_jsonkv', 'rag_avg']

# Define X-axis tasks (columns) - with averaged ICL and including longproc tasks (reordered)
x_tasks = ['icl_avg', 'cite_avg', 'rerank', 'summ_multilex',
           'pseudo_to_code', 'html_to_tsv', 'travel_planning']

# Define clean labels for tasks
y_labels = ['NIAH', 'Recall', 'RAG']
x_labels = ['ICL', 'Cite', 'Re-rank', 'Summ',
            'Pseudo', 'HTML', 'Travel']

print(f"Y-axis tasks: {y_tasks}")
print(f"X-axis tasks: {x_tasks}")

# Create R² matrix
n_y = len(y_tasks)
n_x = len(x_tasks)
r2_matrix = np.zeros((n_y, n_x))

print(f"\nCalculating R² for {n_y * n_x} task pairs...")

for i, y_task in enumerate(y_tasks):
    for j, x_task in enumerate(x_tasks):
        r2 = calculate_r2(df[y_task].values, df[x_task].values)
        r2_matrix[i, j] = r2
        print(f"{y_task} vs {x_task}: R² = {r2:.4f}")

# Print the R² matrix
print(f"\n{'='*80}")
print("R² MATRIX")
print(f"{'='*80}")
r2_df = pd.DataFrame(r2_matrix, index=y_tasks, columns=x_tasks)
print(r2_df.to_string())

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 6))

# Use the RdYlGn colormap from the original script
cmap = 'RdYlGn'

# Create heatmap
im = ax.imshow(r2_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

# Set ticks and labels
ax.set_xticks(np.arange(n_x))
ax.set_yticks(np.arange(n_y))
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=26)
ax.set_yticklabels(y_labels, fontsize=26)

# Add text annotations
for i in range(n_y):
    for j in range(n_x):
        value = r2_matrix[i, j]
        if not np.isnan(value):
            # Use white for red colors (low values < 0.3), black for everything else
            text_color = 'white' if value < 0.3 else 'black'
            text = ax.text(j, i, f'{value:.3f}',
                          ha="center", va="center", color=text_color,
                          fontsize=18)

# Add axis labels
ax.set_xlabel('High-Dispersion Tasks', fontsize=28, labelpad=10)
ax.set_ylabel('Low-Dispersion Tasks', fontsize=28, labelpad=10)

# Remove spines (bounding box)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Add thin white horizontal borders between rows
# Between NIAH (row 0) and Recall (row 1)
ax.axhline(y=0.5, color='white', linewidth=4, zorder=10)
# Between Recall (row 1) and RAG (row 2)
ax.axhline(y=1.5, color='white', linewidth=4, zorder=10)

plt.tight_layout()

# Save figure
output_path = '/scratch/gpfs/DANQIC/jz4391/HELMET/results/plots/task_pairs_r2_reorganized.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nHeatmap saved to: {output_path}")

# Also save as PDF
pdf_path = output_path.replace('.png', '.pdf')
plt.savefig(pdf_path, bbox_inches='tight')
print(f"Heatmap also saved as PDF: {pdf_path}")

# Print summary statistics
print(f"\n{'='*80}")
print("SUMMARY STATISTICS")
print(f"{'='*80}")

all_values = r2_matrix.flatten()
all_values = all_values[~np.isnan(all_values)]

print(f"Mean R²: {np.mean(all_values):.4f}")
print(f"Median R²: {np.median(all_values):.4f}")
print(f"Std R²: {np.std(all_values):.4f}")
print(f"Min R²: {np.min(all_values):.4f}")
print(f"Max R²: {np.max(all_values):.4f}")

# Print all correlations sorted
print(f"\n{'='*80}")
print("ALL CORRELATIONS (sorted by R²)")
print(f"{'='*80}")

correlations = []
for i, y_task in enumerate(y_tasks):
    for j, x_task in enumerate(x_tasks):
        if not np.isnan(r2_matrix[i, j]):
            correlations.append((y_task, x_task, r2_matrix[i, j]))

correlations.sort(key=lambda x: x[2], reverse=True)

for idx, (y_task, x_task, r2_val) in enumerate(correlations, 1):
    print(f"{idx:2d}. {y_task:20s} <-> {x_task:15s}: R² = {r2_val:.4f}")

plt.show()
