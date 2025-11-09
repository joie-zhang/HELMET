#!/usr/bin/env python3
"""
Calculate Spearman correlation coefficients for task pairs with reorganized axes
Y-axis: niah, recall_jsonkv, rag_hotpotqa, rag_nq
X-axis: icl_clinic, icl_banking, rerank, summ_multilex, cite (averaged)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def spearman_correlation(x, y):
    """Calculate Spearman correlation coefficient between two arrays"""
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 2:
        return np.nan

    # Rank the data
    x_rank = pd.Series(x_clean).rank()
    y_rank = pd.Series(y_clean).rank()

    # Calculate Pearson correlation of ranks (which is Spearman)
    correlation = np.corrcoef(x_rank, y_rank)[0, 1]

    return correlation

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

# Define Y-axis tasks (rows)
y_tasks = ['niah', 'recall_jsonkv', 'rag_hotpotqa', 'rag_nq']

# Define X-axis tasks (columns) - including longproc tasks
x_tasks = ['icl_clinic', 'icl_banking', 'rerank', 'summ_multilex', 'cite_avg',
           'travel_planning', 'html_to_tsv', 'pseudo_to_code']

print(f"Y-axis tasks: {y_tasks}")
print(f"X-axis tasks: {x_tasks}")

# Create Spearman correlation matrix
n_y = len(y_tasks)
n_x = len(x_tasks)
spearman_matrix = np.zeros((n_y, n_x))

print(f"\nCalculating Spearman correlation for {n_y * n_x} task pairs...")

for i, y_task in enumerate(y_tasks):
    for j, x_task in enumerate(x_tasks):
        rho = spearman_correlation(df[y_task].values, df[x_task].values)
        spearman_matrix[i, j] = rho
        print(f"{y_task} vs {x_task}: ρ = {rho:.4f}")

# Print the Spearman correlation matrix
print(f"\n{'='*80}")
print("SPEARMAN CORRELATION MATRIX")
print(f"{'='*80}")
spearman_df = pd.DataFrame(spearman_matrix, index=y_tasks, columns=x_tasks)
print(spearman_df.to_string())

# Create heatmap
fig, ax = plt.subplots(figsize=(14, 6))

# Create heatmap with diverging colormap centered at 0
im = ax.imshow(spearman_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

# Set ticks and labels
ax.set_xticks(np.arange(n_x))
ax.set_yticks(np.arange(n_y))
ax.set_xticklabels(x_tasks, rotation=45, ha='right', fontsize=12, fontweight='bold')
ax.set_yticklabels(y_tasks, fontsize=12, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Spearman ρ', fontsize=12, fontweight='bold')

# Add text annotations
for i in range(n_y):
    for j in range(n_x):
        value = spearman_matrix[i, j]
        if not np.isnan(value):
            # Use white text for extreme values, black for middle values
            text_color = 'white' if abs(value) > 0.5 else 'black'
            text = ax.text(j, i, f'{value:.3f}',
                          ha="center", va="center", color=text_color,
                          fontsize=11, fontweight='bold')

# Title and labels
ax.set_title('Spearman Correlation Coefficients: Task Pair Correlations\n(HELMET + Longproc Performance Data)',
            fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Downstream Tasks', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('Needle/Retrieval Tasks', fontsize=13, fontweight='bold', labelpad=10)

plt.tight_layout()

# Save figure
output_path = '/scratch/gpfs/DANQIC/jz4391/HELMET/results/plots/task_pairs_spearman.png'
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

all_values = spearman_matrix.flatten()
all_values = all_values[~np.isnan(all_values)]

print(f"Mean Spearman ρ: {np.mean(all_values):.4f}")
print(f"Median Spearman ρ: {np.median(all_values):.4f}")
print(f"Std Spearman ρ: {np.std(all_values):.4f}")
print(f"Min Spearman ρ: {np.min(all_values):.4f}")
print(f"Max Spearman ρ: {np.max(all_values):.4f}")

# Print all correlations sorted
print(f"\n{'='*80}")
print("ALL CORRELATIONS (sorted by Spearman ρ)")
print(f"{'='*80}")

correlations = []
for i, y_task in enumerate(y_tasks):
    for j, x_task in enumerate(x_tasks):
        if not np.isnan(spearman_matrix[i, j]):
            correlations.append((y_task, x_task, spearman_matrix[i, j]))

correlations.sort(key=lambda x: x[2], reverse=True)

for idx, (y_task, x_task, rho_val) in enumerate(correlations, 1):
    print(f"{idx:2d}. {y_task:20s} <-> {x_task:15s}: ρ = {rho_val:.4f}")

plt.show()
