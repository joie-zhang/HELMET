#!/usr/bin/env python3
"""
Calculate R² coefficients for all task pairs in helmet_performance.csv
Creates a heatmap visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

print(f"Merged dataset shape: {df.shape}")
print(f"Number of rows with both HELMET and longproc data: {df['niah'].notna().sum() if 'niah' in df.columns else 0}")
print(f"Number of rows with longproc data: {df['travel_planning'].notna().sum() if 'travel_planning' in df.columns else 0}")

# Identify task columns (exclude metadata columns)
metadata_cols = ['technique', 'context_length', 'model', 'cache_size']
task_columns = [col for col in df.columns if col not in metadata_cols]

print(f"Found {len(task_columns)} task columns:")
print(task_columns)

# Create R² matrix
n_tasks = len(task_columns)
r2_matrix = np.zeros((n_tasks, n_tasks))

print(f"\nCalculating R² for all {n_tasks * n_tasks} task pairs...")

for i, task1 in enumerate(task_columns):
    for j, task2 in enumerate(task_columns):
        if i == j:
            r2_matrix[i, j] = 1.0  # Perfect correlation with itself
        else:
            r2 = calculate_r2(df[task1].values, df[task2].values)
            r2_matrix[i, j] = r2

# Print the R² matrix
print(f"\n{'='*80}")
print("R² MATRIX")
print(f"{'='*80}")
r2_df = pd.DataFrame(r2_matrix, index=task_columns, columns=task_columns)
print(r2_df.to_string())

# Create heatmap
fig, ax = plt.subplots(figsize=(14, 12))

# Create heatmap
im = ax.imshow(r2_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# Set ticks and labels
ax.set_xticks(np.arange(n_tasks))
ax.set_yticks(np.arange(n_tasks))
ax.set_xticklabels(task_columns, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(task_columns, fontsize=10)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('R² Coefficient', fontsize=12, fontweight='bold')

# Add text annotations
for i in range(n_tasks):
    for j in range(n_tasks):
        value = r2_matrix[i, j]
        if not np.isnan(value):
            text_color = 'white' if value < 0.5 else 'black'
            text = ax.text(j, i, f'{value:.2f}',
                          ha="center", va="center", color=text_color,
                          fontsize=8, fontweight='bold')

# Title
ax.set_title('R² Coefficients Between All Task Pairs\n(HELMET + Longproc Performance Data)',
            fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()

# Save figure
output_path = '/scratch/gpfs/DANQIC/jz4391/HELMET/results/plots/task_pairs_r2_heatmap.png'
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

# Get upper triangle (excluding diagonal) for summary stats
upper_tri_indices = np.triu_indices(n_tasks, k=1)
upper_tri_values = r2_matrix[upper_tri_indices]
upper_tri_values = upper_tri_values[~np.isnan(upper_tri_values)]

print(f"Mean R² (excluding diagonal): {np.mean(upper_tri_values):.4f}")
print(f"Median R² (excluding diagonal): {np.median(upper_tri_values):.4f}")
print(f"Std R² (excluding diagonal): {np.std(upper_tri_values):.4f}")
print(f"Min R² (excluding diagonal): {np.min(upper_tri_values):.4f}")
print(f"Max R² (excluding diagonal): {np.max(upper_tri_values):.4f}")

# Find strongest correlations
print(f"\n{'='*80}")
print("TOP 10 STRONGEST CORRELATIONS (excluding diagonal)")
print(f"{'='*80}")

correlations = []
for i in range(n_tasks):
    for j in range(i+1, n_tasks):
        if not np.isnan(r2_matrix[i, j]):
            correlations.append((task_columns[i], task_columns[j], r2_matrix[i, j]))

correlations.sort(key=lambda x: x[2], reverse=True)

for idx, (task1, task2, r2_val) in enumerate(correlations[:10], 1):
    print(f"{idx}. {task1} <-> {task2}: R² = {r2_val:.4f}")

# Find weakest correlations
print(f"\n{'='*80}")
print("TOP 10 WEAKEST CORRELATIONS (excluding diagonal)")
print(f"{'='*80}")

for idx, (task1, task2, r2_val) in enumerate(correlations[-10:], 1):
    print(f"{idx}. {task1} <-> {task2}: R² = {r2_val:.4f}")

plt.show()
