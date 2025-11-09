#!/usr/bin/env python3
"""
Analyze NIAH vs RECALL_JSONKV performance from helmet_performance.csv
Creates a scatter plot and calculates regression coefficient
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

# Read the CSV file
df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_performance.csv')

# Extract NIAH and RECALL_JSONKV columns
# Filter out rows where either value is NaN
data = df[['niah', 'recall_jsonkv', 'model', 'technique']].dropna()

print(f"Total data points: {len(data)}")
print(f"\nNIAH statistics:")
print(data['niah'].describe())
print(f"\nRECALL_JSONKV statistics:")
print(data['recall_jsonkv'].describe())

# Calculate regression
x = data['niah'].values
y = data['recall_jsonkv'].values

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
r_squared = r_value ** 2

print(f"\n{'='*60}")
print(f"LINEAR REGRESSION RESULTS")
print(f"{'='*60}")
print(f"Regression equation: y = {slope:.4f}x + {intercept:.4f}")
print(f"Correlation coefficient (r): {r_value:.4f}")
print(f"R-squared (R²): {r_squared:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Standard error: {std_err:.4f}")
print(f"{'='*60}")

# Create scatter plot
plt.figure(figsize=(12, 8))

# Use different colors for different techniques
techniques = data['technique'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(techniques)))

for idx, technique in enumerate(techniques):
    technique_data = data[data['technique'] == technique]
    plt.scatter(technique_data['niah'], technique_data['recall_jsonkv'],
               label=technique, alpha=0.7, s=100, color=colors[idx])

# Add regression line
x_line = np.linspace(x.min(), x.max(), 100)
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, 'r--', linewidth=2,
         label=f'Regression line (R²={r_squared:.3f})')

# Labels and title
plt.xlabel('NIAH Performance', fontsize=14, fontweight='bold')
plt.ylabel('RECALL_JSONKV Performance', fontsize=14, fontweight='bold')
plt.title('NIAH vs RECALL_JSONKV Performance\nScatter Plot with Linear Regression',
         fontsize=16, fontweight='bold')

# Add grid
plt.grid(True, alpha=0.3, linestyle='--')

# Legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

# Add text box with statistics
textstr = f'$R^2 = {r_squared:.4f}$\n$r = {r_value:.4f}$\n$p = {p_value:.6f}$\n$n = {len(data)}$'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

plt.tight_layout()

# Save figure
output_path = '/scratch/gpfs/DANQIC/jz4391/HELMET/results/plots/niah_vs_recall_jsonkv_regression.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

# Also save as PDF
pdf_path = output_path.replace('.png', '.pdf')
plt.savefig(pdf_path, bbox_inches='tight')
print(f"Plot also saved as PDF: {pdf_path}")

plt.show()

# Print some additional statistics
print(f"\n{'='*60}")
print(f"ADDITIONAL STATISTICS")
print(f"{'='*60}")
print(f"Pearson correlation coefficient: {r_value:.4f}")
print(f"Spearman correlation coefficient: {stats.spearmanr(x, y)[0]:.4f}")
print(f"Kendall tau correlation coefficient: {stats.kendalltau(x, y)[0]:.4f}")

# Print breakdown by technique
print(f"\n{'='*60}")
print(f"BREAKDOWN BY TECHNIQUE")
print(f"{'='*60}")
for technique in techniques:
    technique_data = data[data['technique'] == technique]
    print(f"\n{technique}:")
    print(f"  Number of data points: {len(technique_data)}")
    print(f"  NIAH mean: {technique_data['niah'].mean():.2f} ± {technique_data['niah'].std():.2f}")
    print(f"  RECALL_JSONKV mean: {technique_data['recall_jsonkv'].mean():.2f} ± {technique_data['recall_jsonkv'].std():.2f}")
