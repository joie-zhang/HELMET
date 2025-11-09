#!/usr/bin/env python3
"""
Script to create a consolidated notebook with all figure scripts.
This reads the existing Python scripts and creates a notebook.
"""
import json
import os

# Read the Python scripts
scripts_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/scripts'
root_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET'

# Script files to include (in order)
figure_scripts = [
    ('Figure 1: Main Comparison Plot', 'scripts/plot_from_averaged_csv.py'),
    ('Figure 2: Quadrant Comparison Plot (1x1)', 'scripts/plot_quadrant_comparison_1x1.py'),
    ('Figure 3: Task Performance Deltas', 'scripts/plot_task_deltas_averaged_configs.py'),
    ('Figure 4: Task Correlation Heatmaps', 'analyze_task_pairs_r2_reorganized.py'),
    ('Figure 6: ICL Memory-Only Analysis', 'scripts/plot_icl_memory_only.py'),
]

# Create notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Add header cell
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# All Paper Figures - Consolidated Notebook\n\n",
        "This notebook contains all the plotting scripts for the paper figures:\n",
        "- **Figure 1**: Main Comparison Plot (Memory vs Performance)\n",
        "- **Figure 2**: Quadrant Comparison Plot (1x1 Grouped Bar)\n",
        "- **Figure 3**: Task Performance Deltas\n",
        "- **Figure 4**: Task Correlation Heatmaps\n",
        "- **Figure 6**: ICL Memory-Only Analysis\n\n",
        "Run cells in order to generate all figures."
    ]
})

# Add setup cell
setup_code = """# Common imports and setup
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
import os

# Set style for professional appearance
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Define paths
results_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/results'
output_dir = os.path.join(results_dir, 'plots')
os.makedirs(output_dir, exist_ok=True)

print("Setup complete!")"""

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": setup_code.split('\n')
})

# Add each figure script as a cell
for fig_title, script_path in figure_scripts:
    # Add markdown header
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## {fig_title}\n\n", f"Script: `{script_path}`"]
    })
    
    # Read and add script content
    full_path = os.path.join(root_dir, script_path)
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            script_content = f.read()
        # Remove shebang if present
        if script_content.startswith('#!/'):
            script_content = script_content.split('\n', 1)[1]
        # Remove docstring if it's a triple-quoted string at the start
        if script_content.strip().startswith('"""'):
            lines = script_content.split('\n')
            if lines[0].strip().startswith('"""'):
                # Find closing """
                for i, line in enumerate(lines[1:], 1):
                    if '"""' in line:
                        script_content = '\n'.join(lines[i+1:])
                        break
        
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": script_content.split('\n')
        })
    else:
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [f"# Script not found: {script_path}"]
        })

# Write notebook
output_path = os.path.join(root_dir, 'scripts/all_figures.ipynb')
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook created: {output_path}")
