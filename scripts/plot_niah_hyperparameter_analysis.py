#!/usr/bin/env python3
"""
NIAH Hyperparameter Analysis Script

Specialized script for analyzing NIAH hyperparameter sweep results on Yarn-Qwen3-8B
to identify optimal cache configurations for SnapKV and PyramidKV techniques.

This script creates:
1. Performance vs Memory usage scatter plots
2. Performance vs Latency scatter plots
3. Detailed heatmaps showing performance/memory/efficiency across hyperparameters
4. Pareto frontier analysis for optimal configurations
5. Comparative analysis between SnapKV and PyramidKV
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

def parse_cache_params(cache_size: str, technique: str) -> dict:
    """Parse cache string to extract hyperparameters."""
    params = {}

    try:
        if technique in ['pyramidkv', 'snapkv']:
            # Format: w{window}_c{capacity}_k{kernel}_{pooling}
            # Example: w2048_c8192_k7_maxpool
            parts = cache_size.split('_')
            print(f"DEBUG: {technique} parsing cache_size='{cache_size}', parts: {parts}")

            if len(parts) >= 4:
                w_part = parts[0]  # e.g., 'w2048'
                c_part = parts[1]  # e.g., 'c8192'
                k_part = parts[2]  # e.g., 'k7'
                pool_part = parts[3]  # e.g., 'maxpool'

                # Extract numbers by removing prefixes
                w_size = w_part.replace('w', '') if w_part.startswith('w') else w_part
                c_size = c_part.replace('c', '') if c_part.startswith('c') else c_part
                k_size = k_part.replace('k', '') if k_part.startswith('k') else k_part

                params['window_size'] = int(w_size)
                params['max_capacity'] = int(c_size)
                params['kernel_size'] = int(k_size)
                params['pooling'] = pool_part

                print(f"DEBUG: Successfully parsed {technique} - window_size={params['window_size']}, max_capacity={params['max_capacity']}")
            else:
                print(f"ERROR: Not enough parts in {technique} cache_size '{cache_size}' - expected format: w2048_c8192_k7_maxpool")
                raise ValueError(f"Invalid {technique} cache_size format: {cache_size}")

    except (IndexError, ValueError) as e:
        print(f"Warning: Could not parse cache parameters for {technique}: {cache_size}")
        params['cache_config'] = cache_size

    return params

def load_and_filter_data(model='Yarn-Qwen3-8B', context_length='16k', task='niah'):
    """Load the helmet results and filter for NIAH analysis."""
    base_path = '/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results'

    try:
        # Load data
        memory_df = pd.read_csv(os.path.join(base_path, 'helmet_memory_usage.csv'))
        throughput_df = pd.read_csv(os.path.join(base_path, 'helmet_throughput.csv'))
        performance_df = pd.read_csv(os.path.join(base_path, 'helmet_performance.csv'))
    except FileNotFoundError as e:
        print(f"Error: Could not find results files in {base_path}")
        print("Make sure you have run the collect_results_new.py script first.")
        return None, None, None

    # Filter for specified model/context/task
    print(f"Filtering for: {model} + {context_length} + NIAH task")

    # Filter for both snapkv and pyramidkv techniques
    techniques = ['snapkv', 'pyramidkv']

    memory_filtered = memory_df[
        (memory_df['technique'].isin(techniques)) &
        (memory_df['model'] == model) &
        (memory_df['context_length'] == context_length)
    ].copy()

    throughput_filtered = throughput_df[
        (throughput_df['technique'].isin(techniques)) &
        (throughput_df['model'] == model) &
        (throughput_df['context_length'] == context_length)
    ].copy()

    performance_filtered = performance_df[
        (performance_df['technique'].isin(techniques)) &
        (performance_df['model'] == model) &
        (performance_df['context_length'] == context_length)
    ].copy()

    print(f"Found {len(memory_filtered)} memory records")
    print(f"Found {len(throughput_filtered)} throughput records")
    print(f"Found {len(performance_filtered)} performance records")

    return memory_filtered, throughput_filtered, performance_filtered

def add_hyperparameter_columns(df, technique_col='technique'):
    """Add hyperparameter columns by parsing cache_size for each technique."""
    if df.empty:
        return df

    print(f"\nDEBUG: Processing {len(df)} rows")

    # Parse cache parameters for each row
    param_rows = []

    for idx, row in df.iterrows():
        technique = row[technique_col]
        cache_size = row['cache_size']
        print(f"DEBUG: Processing row {idx}: technique='{technique}', cache_size='{cache_size}'")

        params = parse_cache_params(cache_size, technique)
        param_rows.append(params)

    # Add all parameter columns
    param_df = pd.DataFrame(param_rows)
    df_combined = pd.concat([df.reset_index(drop=True), param_df.reset_index(drop=True)], axis=1)

    return df_combined

def create_comparative_analysis(memory_df, throughput_df, performance_df, model, context_length, task, output_dir):
    """Create comprehensive comparative analysis between SnapKV and PyramidKV."""

    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Colors for different techniques
    technique_colors = {'snapkv': '#FF6B6B', 'pyramidkv': '#4ECDC4'}

    # Plot 1: Memory vs Performance (top-left)
    ax1 = axes[0, 0]
    plot_memory_vs_performance_comparative(ax1, memory_df, performance_df, task, technique_colors)

    # Plot 2: Latency vs Performance (top-right)
    ax2 = axes[0, 1]
    plot_latency_vs_performance_comparative(ax2, throughput_df, performance_df, task, technique_colors)

    # Plot 3: Window Size vs Performance (bottom-left)
    ax3 = axes[1, 0]
    plot_window_vs_performance(ax3, memory_df, performance_df, task, technique_colors)

    # Plot 4: Cache Size vs Performance (bottom-right)
    ax4 = axes[1, 1]
    plot_cache_vs_performance(ax4, memory_df, performance_df, task, technique_colors)

    # Add shared legend
    legend_elements = [
        Line2D([0], [0], marker='o', color=technique_colors['snapkv'],
               linestyle='-', markersize=10, linewidth=3, label='SnapKV'),
        Line2D([0], [0], marker='o', color=technique_colors['pyramidkv'],
               linestyle='-', markersize=10, linewidth=3, label='PyramidKV')
    ]

    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02),
               ncol=2, fontsize=14, title='KV-Cache Technique', title_fontsize=16)

    # Add title
    fig.suptitle(f'NIAH Hyperparameter Analysis: SnapKV vs PyramidKV\n{model} | {context_length}',
                 fontsize=18, y=0.98)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f'niah_comparative_analysis_{model.replace("-", "_")}_{context_length}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()

def plot_memory_vs_performance_comparative(ax, memory_df, performance_df, task, technique_colors):
    """Plot Memory vs Performance comparing techniques."""

    # Merge memory and performance data
    merged = pd.merge(memory_df, performance_df,
                     on=['technique', 'context_length', 'model', 'cache_size'],
                     suffixes=('_mem', '_perf'))

    # Find memory and task columns
    memory_col = 'niah' if 'niah' in merged.columns else None
    task_col = 'niah' if 'niah' in merged.columns else None

    if memory_col is None or task_col is None:
        print("Warning: Could not find NIAH columns for memory vs performance plot")
        print(f"Available columns: {list(merged.columns)}")
        return

    # Check if we have meaningful data (non-zero performance values)
    non_zero_performance = merged[merged[task_col] > 0.01]  # Allow small non-zero values
    if len(non_zero_performance) == 0:
        print("Warning: No meaningful NIAH performance data found (all values are 0 or very small)")
        print("This suggests NIAH experiments haven't completed yet for these cache configurations")
        return

    # Plot each technique
    for technique in ['snapkv', 'pyramidkv']:
        tech_data = merged[merged['technique'] == technique]

        if len(tech_data) == 0:
            continue

        # Extract values
        memory_values = tech_data[memory_col].values
        performance_values = tech_data[task_col].values

        # Filter NaN values
        valid_indices = ~(np.isnan(memory_values) | np.isnan(performance_values))
        memory_values = memory_values[valid_indices]
        performance_values = performance_values[valid_indices]

        if len(memory_values) == 0:
            continue

        # Plot points
        ax.scatter(memory_values, performance_values,
                  color=technique_colors[technique], s=100, alpha=0.8,
                  edgecolors='black', linewidth=1, label=technique.upper())

        # Add trend line
        if len(memory_values) > 1:
            z = np.polyfit(memory_values, performance_values, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(memory_values.min(), memory_values.max(), 100)
            ax.plot(x_trend, p(x_trend), color=technique_colors[technique],
                   linestyle='--', alpha=0.7, linewidth=2)

    ax.set_xlabel('Memory Usage (GB)', fontsize=12)
    ax.set_ylabel('NIAH Performance', fontsize=12)
    ax.set_title('Memory vs Performance', fontsize=14)
    ax.grid(True, alpha=0.3)

def plot_latency_vs_performance_comparative(ax, throughput_df, performance_df, task, technique_colors):
    """Plot Latency vs Performance comparing techniques."""

    # Calculate latency from throughput
    throughput_df = throughput_df.copy()
    throughput_col = 'niah' if 'niah' in throughput_df.columns else None

    if throughput_col is None:
        print("Warning: Could not find NIAH column for latency analysis")
        return

    throughput_df[throughput_col] = throughput_df[throughput_col].replace(0, np.nan)
    throughput_df['latency'] = 1 / throughput_df[throughput_col]

    # Merge with performance data
    merged = pd.merge(throughput_df, performance_df,
                     on=['technique', 'context_length', 'model', 'cache_size'],
                     suffixes=('_thr', '_perf'))

    task_col = 'niah' if 'niah' in merged.columns else None
    if task_col is None:
        print("Warning: Could not find NIAH performance column")
        print(f"Available columns: {list(merged.columns)}")
        return

    # Check if we have meaningful data (non-zero performance values)
    non_zero_performance = merged[merged[task_col] > 0.01]
    if len(non_zero_performance) == 0:
        print("Warning: No meaningful NIAH performance data found for latency analysis")
        return

    # Plot each technique
    for technique in ['snapkv', 'pyramidkv']:
        tech_data = merged[merged['technique'] == technique]

        if len(tech_data) == 0:
            continue

        # Extract values
        latency_values = tech_data['latency'].values
        performance_values = tech_data[task_col].values

        # Filter NaN values
        valid_indices = ~(np.isnan(latency_values) | np.isnan(performance_values))
        latency_values = latency_values[valid_indices]
        performance_values = performance_values[valid_indices]

        if len(latency_values) == 0:
            continue

        # Plot points
        ax.scatter(latency_values, performance_values,
                  color=technique_colors[technique], s=100, alpha=0.8,
                  edgecolors='black', linewidth=1)

        # Add trend line
        if len(latency_values) > 1:
            z = np.polyfit(latency_values, performance_values, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(latency_values.min(), latency_values.max(), 100)
            ax.plot(x_trend, p(x_trend), color=technique_colors[technique],
                   linestyle='--', alpha=0.7, linewidth=2)

    ax.set_xlabel('Latency (s/token)', fontsize=12)
    ax.set_ylabel('NIAH Performance', fontsize=12)
    ax.set_title('Latency vs Performance', fontsize=14)
    ax.grid(True, alpha=0.3)

def plot_window_vs_performance(ax, memory_df, performance_df, task, technique_colors):
    """Plot Window Size vs Performance."""

    # Merge data
    merged = pd.merge(memory_df, performance_df,
                     on=['technique', 'context_length', 'model', 'cache_size'],
                     suffixes=('_mem', '_perf'))

    task_col = 'niah' if 'niah' in merged.columns else None
    if task_col is None or 'window_size' not in merged.columns:
        print("Warning: Missing columns for window size analysis")
        return

    # Plot each technique
    for technique in ['snapkv', 'pyramidkv']:
        tech_data = merged[merged['technique'] == technique]

        if len(tech_data) == 0:
            continue

        # Group by window size
        for window_size in sorted(tech_data['window_size'].unique()):
            window_data = tech_data[tech_data['window_size'] == window_size]

            performance_values = window_data[task_col].values
            cache_sizes = window_data['max_capacity'].values

            # Filter NaN values
            valid_indices = ~np.isnan(performance_values)
            performance_values = performance_values[valid_indices]
            cache_sizes = cache_sizes[valid_indices]

            if len(performance_values) == 0:
                continue

            # Plot points for this window size
            ax.scatter([window_size] * len(performance_values), performance_values,
                      color=technique_colors[technique], s=80, alpha=0.8,
                      edgecolors='black', linewidth=0.5)

            # Annotate with cache sizes
            for perf, cache_size in zip(performance_values, cache_sizes):
                ax.annotate(f'{cache_size}', (window_size, perf),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)

    ax.set_xlabel('Window Size', fontsize=12)
    ax.set_ylabel('NIAH Performance', fontsize=12)
    ax.set_title('Window Size vs Performance\n(Numbers show cache size)', fontsize=14)
    ax.grid(True, alpha=0.3)

def plot_cache_vs_performance(ax, memory_df, performance_df, task, technique_colors):
    """Plot Cache Size vs Performance."""

    # Merge data
    merged = pd.merge(memory_df, performance_df,
                     on=['technique', 'context_length', 'model', 'cache_size'],
                     suffixes=('_mem', '_perf'))

    task_col = 'niah' if 'niah' in merged.columns else None
    if task_col is None or 'max_capacity' not in merged.columns:
        print("Warning: Missing columns for cache size analysis")
        return

    # Plot each technique
    for technique in ['snapkv', 'pyramidkv']:
        tech_data = merged[merged['technique'] == technique]

        if len(tech_data) == 0:
            continue

        # Group by cache size
        for cache_size in sorted(tech_data['max_capacity'].unique()):
            cache_data = tech_data[tech_data['max_capacity'] == cache_size]

            performance_values = cache_data[task_col].values
            window_sizes = cache_data['window_size'].values

            # Filter NaN values
            valid_indices = ~np.isnan(performance_values)
            performance_values = performance_values[valid_indices]
            window_sizes = window_sizes[valid_indices]

            if len(performance_values) == 0:
                continue

            # Plot points for this cache size
            ax.scatter([cache_size] * len(performance_values), performance_values,
                      color=technique_colors[technique], s=80, alpha=0.8,
                      edgecolors='black', linewidth=0.5)

            # Annotate with window sizes
            for perf, window_size in zip(performance_values, window_sizes):
                ax.annotate(f'{window_size}', (cache_size, perf),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)

    ax.set_xlabel('Cache Size', fontsize=12)
    ax.set_ylabel('NIAH Performance', fontsize=12)
    ax.set_title('Cache Size vs Performance\n(Numbers show window size)', fontsize=14)
    ax.grid(True, alpha=0.3)

def create_detailed_heatmaps(memory_df, performance_df, model, context_length, task, output_dir):
    """Create detailed heatmaps for each technique."""

    # Create heatmaps for each technique
    for technique in ['snapkv', 'pyramidkv']:
        tech_memory = memory_df[memory_df['technique'] == technique]
        tech_performance = performance_df[performance_df['technique'] == technique]

        if tech_memory.empty or tech_performance.empty:
            print(f"Warning: No data found for {technique}")
            continue

        # Merge data
        merged = pd.merge(tech_memory, tech_performance,
                         on=['technique', 'context_length', 'model', 'cache_size'],
                         suffixes=('_mem', '_perf'))

        if 'window_size' not in merged.columns or 'max_capacity' not in merged.columns:
            print(f"Warning: Missing hyperparameter columns for {technique}")
            continue

        task_col = 'niah' if 'niah' in merged.columns else None
        memory_col = 'niah' if 'niah' in merged.columns else None

        if task_col is None or memory_col is None:
            print(f"Warning: Missing NIAH columns for {technique}")
            continue

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        try:
            # Performance heatmap
            perf_pivot = merged.pivot(index='window_size', columns='max_capacity', values=task_col)
            sns.heatmap(perf_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0])
            axes[0].set_title(f'{technique.upper()} - NIAH Performance')
            axes[0].set_xlabel('Cache Size (max_capacity)')
            axes[0].set_ylabel('Window Size')

            # Memory heatmap
            mem_pivot = merged.pivot(index='window_size', columns='max_capacity', values=memory_col)
            sns.heatmap(mem_pivot, annot=True, fmt='.1f', cmap='YlGnBu', ax=axes[1])
            axes[1].set_title(f'{technique.upper()} - Memory Usage (GB)')
            axes[1].set_xlabel('Cache Size (max_capacity)')
            axes[1].set_ylabel('Window Size')

            # Efficiency heatmap (Performance / Memory)
            efficiency = merged.copy()
            efficiency['efficiency'] = efficiency[task_col] / efficiency[memory_col]
            eff_pivot = efficiency.pivot(index='window_size', columns='max_capacity', values='efficiency')
            sns.heatmap(eff_pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[2])
            axes[2].set_title(f'{technique.upper()} - Efficiency (Performance/Memory)')
            axes[2].set_xlabel('Cache Size (max_capacity)')
            axes[2].set_ylabel('Window Size')

            # Add title
            fig.suptitle(f'{technique.upper()} Heatmap Analysis | {model} | {context_length} | NIAH',
                         fontsize=16, y=0.98)

            plt.tight_layout()
            filename = f'niah_{technique}_heatmap_{model.replace("-", "_")}_{context_length}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.show()

        except Exception as e:
            print(f"Error creating heatmap for {technique}: {e}")

def analyze_optimal_configurations(memory_df, performance_df, model, context_length):
    """Analyze and report optimal configurations."""

    print(f"\n{'='*80}")
    print(f"OPTIMAL CONFIGURATION ANALYSIS - NIAH")
    print(f"Model: {model} | Context: {context_length}")
    print(f"{'='*80}")

    # Analyze each technique separately
    for technique in ['snapkv', 'pyramidkv']:
        tech_memory = memory_df[memory_df['technique'] == technique]
        tech_performance = performance_df[performance_df['technique'] == technique]

        if tech_memory.empty or tech_performance.empty:
            continue

        # Merge data
        merged = pd.merge(tech_memory, tech_performance,
                         on=['technique', 'context_length', 'model', 'cache_size'],
                         suffixes=('_mem', '_perf'))

        task_col = 'niah' if 'niah' in merged.columns else None
        memory_col = 'niah' if 'niah' in merged.columns else None

        if task_col is None or memory_col is None:
            continue

        # Calculate efficiency
        merged['efficiency'] = merged[task_col] / merged[memory_col]

        # Find optimal configurations
        best_performance = merged.loc[merged[task_col].idxmax()]
        best_memory = merged.loc[merged[memory_col].idxmin()]
        best_efficiency = merged.loc[merged['efficiency'].idxmax()]

        print(f"\n--- {technique.upper()} OPTIMAL CONFIGURATIONS ---")
        print(f"Best Performance: window_size={best_performance.get('window_size', 'N/A')}, "
              f"cache_size={best_performance.get('max_capacity', 'N/A')}, "
              f"performance={best_performance[task_col]:.4f}, "
              f"memory={best_performance[memory_col]:.1f}GB")
        print(f"Best Memory:      window_size={best_memory.get('window_size', 'N/A')}, "
              f"cache_size={best_memory.get('max_capacity', 'N/A')}, "
              f"performance={best_memory[task_col]:.4f}, "
              f"memory={best_memory[memory_col]:.1f}GB")
        print(f"Best Efficiency:  window_size={best_efficiency.get('window_size', 'N/A')}, "
              f"cache_size={best_efficiency.get('max_capacity', 'N/A')}, "
              f"performance={best_efficiency[task_col]:.4f}, "
              f"memory={best_efficiency[memory_col]:.1f}GB, "
              f"efficiency={best_efficiency['efficiency']:.4f}")

def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze NIAH hyperparameter sweep results for Yarn-Qwen3-8B')
    parser.add_argument('--model', default='Yarn-Qwen3-8B', help='Model name')
    parser.add_argument('--context', default='16k', help='Context length')
    parser.add_argument('--task', default='niah', help='Task to analyze')
    parser.add_argument('--output_dir', default='/scratch/gpfs/DANQIC/jz4391/HELMET/results/niah_hyperparameter_analysis',
                        help='Output directory for plots')

    args = parser.parse_args()

    print(f"NIAH Hyperparameter Analysis for {args.model}")
    print(f"Context: {args.context} | Task: {args.task}")

    # Load and filter data
    memory_df, throughput_df, performance_df = load_and_filter_data(args.model, args.context, args.task)

    if memory_df is None or memory_df.empty:
        print(f"Error: No results found for {args.model} with {args.context} context.")
        return

    # Add hyperparameter columns
    memory_df = add_hyperparameter_columns(memory_df, 'technique')
    throughput_df = add_hyperparameter_columns(throughput_df, 'technique')
    performance_df = add_hyperparameter_columns(performance_df, 'technique')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create comparative analysis
    create_comparative_analysis(memory_df, throughput_df, performance_df,
                              args.model, args.context, args.task, args.output_dir)

    # Create detailed heatmaps
    create_detailed_heatmaps(memory_df, performance_df,
                           args.model, args.context, args.task, args.output_dir)

    # Analyze optimal configurations
    analyze_optimal_configurations(memory_df, performance_df, args.model, args.context)

if __name__ == "__main__":
    main()