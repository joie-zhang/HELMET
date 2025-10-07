#!/usr/bin/env python3
"""
KV Cache Sweep Analysis Script

This script analyzes the PyramidKV and SnapKV hyperparameter sweep results for
DeepSeek-R1-Distill-Llama-8B on HELMET (16k) and LongProc (2k) benchmarks.

It generates:
1. Performance vs Memory plots
2. Performance vs Latency plots
3. Heatmaps for different hyperparameter combinations
4. Pareto frontier analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# Configuration for the sweep
MODEL = 'DeepSeek-R1-Distill-Llama-8B'
BENCHMARKS = {
    'helmet': {
        'context_length': '16k',
        'tasks': ['niah', 'rag_hotpotqa', 'rag_nq', 'cite_str_em', 'cite_citation_rec',
                  'cite_citation_prec', 'recall_jsonkv', 'rerank']
    },
    'longproc': {
        'context_length': '2k',
        'tasks': ['travel_planning', 'html_to_tsv', 'pseudo_to_code']
    }
}

KV_METHODS = ['pyramidkv', 'snapkv']

# Specific window-capacity pairs from the sweep
WINDOW_CAPACITY_PAIRS = {
    64: [512, 1024, 2048],
    128: [512],
    256: [512]
}

def parse_cache_params(cache_size: str) -> dict:
    """Parse cache string to extract hyperparameters."""
    params = {}
    try:
        # Format: w{window}_c{capacity}_k{kernel}_{pooling}
        parts = cache_size.split('_')
        if len(parts) >= 4:
            params['window_size'] = int(parts[0].replace('w', ''))
            params['max_capacity'] = int(parts[1].replace('c', ''))
            params['kernel_size'] = int(parts[2].replace('k', ''))
            params['pooling'] = parts[3]
    except (IndexError, ValueError) as e:
        print(f"Warning: Could not parse cache parameters: {cache_size}")
    return params

def load_benchmark_data(benchmark='helmet'):
    """Load data for a specific benchmark."""
    base_path = f'/scratch/gpfs/DANQIC/jz4391/HELMET/results/{benchmark}_results'

    try:
        memory_df = pd.read_csv(os.path.join(base_path, f'{benchmark}_memory_usage.csv'))
        throughput_df = pd.read_csv(os.path.join(base_path, f'{benchmark}_throughput.csv'))
        performance_df = pd.read_csv(os.path.join(base_path, f'{benchmark}_performance.csv'))
    except FileNotFoundError as e:
        print(f"Error: Could not find results files in {base_path}")
        return None, None, None

    return memory_df, throughput_df, performance_df

def filter_sweep_data(df, benchmark):
    """Filter data for the specific sweep configurations."""
    context_length = BENCHMARKS[benchmark]['context_length']

    # Filter for model, techniques, and context length
    filtered = df[
        (df['model'] == MODEL) &
        (df['technique'].isin(KV_METHODS)) &
        (df['context_length'] == context_length)
    ].copy()

    # Add hyperparameter columns
    params_list = [parse_cache_params(cs) for cs in filtered['cache_size']]
    params_df = pd.DataFrame(params_list)
    filtered = pd.concat([filtered.reset_index(drop=True), params_df], axis=1)

    # Filter for kernel_size = 7 and specific window-capacity pairs
    filtered = filtered[filtered['kernel_size'] == 7]

    # Filter for specific window-capacity combinations
    valid_configs = []
    for window, capacities in WINDOW_CAPACITY_PAIRS.items():
        for capacity in capacities:
            valid_configs.append((window, capacity))

    filtered = filtered[
        filtered.apply(lambda row: (row['window_size'], row['max_capacity']) in valid_configs, axis=1)
    ]

    print(f"\n{benchmark.upper()} - Filtered to {len(filtered)} configurations")
    print(f"Window-Capacity pairs: {filtered[['window_size', 'max_capacity']].drop_duplicates().values.tolist()}")

    return filtered

def create_analysis_plots(memory_df, throughput_df, performance_df, benchmark, task, output_dir):
    """Create comprehensive analysis plots for a task."""

    plt.style.use('default')
    sns.set_palette("husl")

    # Merge data
    merged = pd.merge(
        memory_df,
        performance_df,
        on=['technique', 'context_length', 'model', 'cache_size', 'window_size', 'max_capacity', 'kernel_size', 'pooling'],
        suffixes=('_mem', '_perf')
    )

    # Determine memory column (task-specific)
    memory_col = None
    if benchmark == 'helmet':
        memory_col = 'cite' if 'cite' in merged.columns else 'recall_jsonkv'
    else:
        memory_col = 'travel_planning'

    if memory_col not in merged.columns:
        print(f"Warning: Memory column {memory_col} not found")
        return

    # Check if task exists
    if task not in merged.columns:
        print(f"Warning: Task {task} not found in data")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Plot for each technique separately
    for idx, technique in enumerate(KV_METHODS):
        tech_data = merged[merged['technique'] == technique].copy()

        if len(tech_data) == 0:
            print(f"No data for {technique}")
            continue

        # Get unique window sizes and create color map
        window_sizes = sorted(tech_data['window_size'].unique())
        colors = plt.cm.Set1(np.linspace(0, 1, len(window_sizes)))
        color_map = dict(zip(window_sizes, colors))

        # Plot 1: Memory vs Performance
        ax1 = axes[idx, 0]
        plot_memory_vs_performance(ax1, tech_data, task, memory_col, color_map)
        ax1.set_title(f'{technique.upper()}: Memory vs Performance')

        # Plot 2: Latency vs Performance
        ax2 = axes[idx, 1]
        tech_throughput = throughput_df[throughput_df['technique'] == technique].copy()
        if len(tech_throughput) > 0:
            plot_latency_vs_performance(ax2, tech_throughput, tech_data, task, memory_col, color_map)
            ax2.set_title(f'{technique.upper()}: Latency vs Performance')

    # Add overall title
    fig.suptitle(f'KV Cache Sweep Analysis: {MODEL}\n{benchmark.upper()} ({BENCHMARKS[benchmark]["context_length"]}) - {task.replace("_", " ").title()}',
                 fontsize=16, y=0.995)

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f'kv_sweep_{benchmark}_{task.replace("_", "-")}_analysis.png'
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def plot_memory_vs_performance(ax, data, task, memory_col, color_map):
    """Plot Memory vs Performance with lines connecting same window_size."""

    for window_size in sorted(data['window_size'].unique()):
        window_data = data[data['window_size'] == window_size].sort_values('max_capacity')

        if len(window_data) == 0:
            continue

        # Extract values
        memory_values = window_data[memory_col].values
        performance_values = window_data[task].values
        capacity_values = window_data['max_capacity'].values

        # Filter out NaN
        valid = ~(np.isnan(memory_values) | np.isnan(performance_values))
        if valid.sum() == 0:
            continue

        memory_values = memory_values[valid]
        performance_values = performance_values[valid]
        capacity_values = capacity_values[valid]

        # Plot
        ax.scatter(memory_values, performance_values,
                  color=color_map[window_size], s=100, alpha=0.8,
                  edgecolors='black', linewidth=1)
        ax.plot(memory_values, performance_values,
               color=color_map[window_size], alpha=0.6, linewidth=2, linestyle='-',
               label=f'window={window_size}')

        # Annotate with capacity
        for mem, perf, cap in zip(memory_values, performance_values, capacity_values):
            ax.annotate(f'{int(cap)}', (mem, perf), xytext=(5, 5),
                       textcoords='offset points', fontsize=9, alpha=0.8)

    ax.set_xlabel('Memory Usage (GB)', fontsize=11)
    ax.set_ylabel(f'{task.replace("_", " ").title()}', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

def plot_latency_vs_performance(ax, throughput_df, performance_df, task, memory_col, color_map):
    """Plot Latency vs Performance."""

    # Convert throughput to latency
    throughput_df = throughput_df.copy()
    throughput_df[memory_col] = throughput_df[memory_col].replace(0, np.nan)
    throughput_df['latency'] = 1 / throughput_df[memory_col]

    # Merge with performance
    merged = pd.merge(
        throughput_df,
        performance_df,
        on=['technique', 'context_length', 'model', 'cache_size', 'window_size', 'max_capacity', 'kernel_size', 'pooling']
    )

    if task not in merged.columns:
        return

    for window_size in sorted(merged['window_size'].unique()):
        window_data = merged[merged['window_size'] == window_size].sort_values('max_capacity')

        if len(window_data) == 0:
            continue

        latency_values = window_data['latency'].values
        performance_values = window_data[task].values
        capacity_values = window_data['max_capacity'].values

        # Filter out NaN
        valid = ~(np.isnan(latency_values) | np.isnan(performance_values))
        if valid.sum() == 0:
            continue

        latency_values = latency_values[valid]
        performance_values = performance_values[valid]
        capacity_values = capacity_values[valid]

        # Plot
        ax.scatter(latency_values, performance_values,
                  color=color_map[window_size], s=100, alpha=0.8,
                  edgecolors='black', linewidth=1)
        ax.plot(latency_values, performance_values,
               color=color_map[window_size], alpha=0.6, linewidth=2, linestyle='-',
               label=f'window={window_size}')

        # Annotate
        for lat, perf, cap in zip(latency_values, performance_values, capacity_values):
            ax.annotate(f'{int(cap)}', (lat, perf), xytext=(5, 5),
                       textcoords='offset points', fontsize=9, alpha=0.8)

    ax.set_xlabel('Latency (s/token)', fontsize=11)
    ax.set_ylabel(f'{task.replace("_", " ").title()}', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

def create_comparison_heatmap(memory_df, performance_df, benchmark, task, output_dir):
    """Create heatmap comparing PyramidKV vs SnapKV."""

    # Merge data
    merged = pd.merge(
        memory_df,
        performance_df,
        on=['technique', 'context_length', 'model', 'cache_size', 'window_size', 'max_capacity', 'kernel_size', 'pooling']
    )

    if task not in merged.columns:
        return

    # Create figure with 2 heatmaps (one per technique)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, technique in enumerate(KV_METHODS):
        tech_data = merged[merged['technique'] == technique]

        if len(tech_data) == 0:
            continue

        # Create pivot table
        try:
            pivot = tech_data.pivot(index='window_size', columns='max_capacity', values=task)

            # Sort by window size and capacity
            pivot = pivot.sort_index()
            pivot = pivot[sorted(pivot.columns)]

            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd',
                       ax=axes[idx], cbar_kws={'label': task.replace('_', ' ').title()})
            axes[idx].set_title(f'{technique.upper()} Performance')
            axes[idx].set_xlabel('Max Capacity')
            axes[idx].set_ylabel('Window Size')
        except Exception as e:
            print(f"Error creating heatmap for {technique}: {e}")

    fig.suptitle(f'Performance Heatmap: {benchmark.upper()} - {task.replace("_", " ").title()}',
                 fontsize=16, y=1.02)

    plt.tight_layout()
    filename = f'kv_sweep_{benchmark}_{task.replace("_", "-")}_heatmap.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def analyze_pareto_frontier(memory_df, performance_df, benchmark, task):
    """Find Pareto optimal configurations."""

    # Determine memory column
    memory_col = 'cite' if benchmark == 'helmet' else 'travel_planning'
    if memory_col not in memory_df.columns:
        memory_col = 'recall_jsonkv'

    # Merge data
    merged = pd.merge(
        memory_df,
        performance_df,
        on=['technique', 'context_length', 'model', 'cache_size', 'window_size', 'max_capacity', 'kernel_size', 'pooling']
    )

    if task not in merged.columns:
        print(f"Task {task} not found in data")
        return

    # Find Pareto optimal configs
    print(f"\n{'='*80}")
    print(f"PARETO OPTIMAL CONFIGURATIONS: {benchmark.upper()} - {task.upper()}")
    print(f"{'='*80}\n")

    for technique in KV_METHODS:
        tech_data = merged[merged['technique'] == technique].copy()
        tech_data = tech_data.dropna(subset=[memory_col, task])

        if len(tech_data) == 0:
            continue

        pareto_configs = []

        for _, row in tech_data.iterrows():
            is_pareto = True
            for _, other_row in tech_data.iterrows():
                # Check if other_row dominates row (lower memory, higher performance)
                if (other_row[memory_col] <= row[memory_col] and
                    other_row[task] >= row[task] and
                    not (other_row[memory_col] == row[memory_col] and other_row[task] == row[task])):
                    is_pareto = False
                    break

            if is_pareto:
                pareto_configs.append(row)

        if len(pareto_configs) > 0:
            pareto_df = pd.DataFrame(pareto_configs)
            pareto_df = pareto_df.sort_values(task, ascending=False)

            print(f"\n{technique.upper()}:")
            print("-" * 80)
            display_cols = ['window_size', 'max_capacity', 'pooling', memory_col, task]
            display_cols = [col for col in display_cols if col in pareto_df.columns]
            print(pareto_df[display_cols].to_string(index=False, float_format='%.3f'))

            if len(pareto_df) > 0:
                best = pareto_df.iloc[0]
                most_efficient = pareto_df.iloc[-1]
                print(f"\nBest performance: window={best['window_size']}, capacity={best['max_capacity']}, {task}={best[task]:.3f}")
                print(f"Most efficient: window={most_efficient['window_size']}, capacity={most_efficient['max_capacity']}, memory={most_efficient[memory_col]:.2f}GB")

def main():
    parser = argparse.ArgumentParser(description='Analyze KV cache sweep results')
    parser.add_argument('--benchmark', choices=['helmet', 'longproc', 'both'], default='both',
                       help='Which benchmark to analyze')
    parser.add_argument('--task', default=None,
                       help='Specific task to analyze (if not specified, analyzes all)')
    parser.add_argument('--output_dir', default='/scratch/gpfs/DANQIC/jz4391/HELMET/results/kv_sweep_analysis',
                       help='Output directory for plots')

    args = parser.parse_args()

    benchmarks_to_analyze = ['helmet', 'longproc'] if args.benchmark == 'both' else [args.benchmark]

    for benchmark in benchmarks_to_analyze:
        print(f"\n{'='*80}")
        print(f"ANALYZING {benchmark.upper()} BENCHMARK")
        print(f"{'='*80}")

        # Load data
        memory_df, throughput_df, performance_df = load_benchmark_data(benchmark)

        if memory_df is None:
            continue

        # Filter for sweep configurations
        memory_filtered = filter_sweep_data(memory_df, benchmark)
        throughput_filtered = filter_sweep_data(throughput_df, benchmark)
        performance_filtered = filter_sweep_data(performance_df, benchmark)

        if len(memory_filtered) == 0:
            print(f"No matching configurations found for {benchmark}")
            continue

        # Get available tasks
        available_tasks = BENCHMARKS[benchmark]['tasks']
        tasks_to_analyze = [args.task] if args.task else available_tasks

        # Filter for tasks that exist in the data
        tasks_to_analyze = [t for t in tasks_to_analyze if t in performance_filtered.columns]

        print(f"\nAnalyzing tasks: {tasks_to_analyze}")

        # Analyze each task
        for task in tasks_to_analyze:
            print(f"\n{'-'*60}")
            print(f"Task: {task.upper()}")
            print(f"{'-'*60}")

            # Create analysis plots
            create_analysis_plots(memory_filtered, throughput_filtered, performance_filtered,
                                benchmark, task, args.output_dir)

            # Create heatmap
            create_comparison_heatmap(memory_filtered, performance_filtered,
                                    benchmark, task, args.output_dir)

            # Pareto analysis
            analyze_pareto_frontier(memory_filtered, performance_filtered, benchmark, task)

if __name__ == "__main__":
    main()
