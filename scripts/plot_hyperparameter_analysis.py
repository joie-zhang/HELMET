#!/usr/bin/env python3
"""
Extensible Hyperparameter Analysis Script

This script analyzes hyperparameter sweep results for any KV-cache technique to help identify
optimal configurations by plotting:
1. Performance vs Memory usage
2. Performance vs Latency (throughput)
3. Heatmap analysis showing performance/memory/efficiency across hyperparameters
4. Pareto frontier analysis

Supports multiple models, techniques, and context lengths.
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
    """Parse cache string to extract hyperparameters for different techniques."""
    params = {}

    try:
        if technique in ['pyramidkv', 'snapkv']:
            # Format: w{window}_c{capacity}_k{kernel}_{pooling}
            # Example: w32_c4096_k5_avgpool
            parts = cache_size.split('_')
            print(f"DEBUG: {technique} parsing cache_size='{cache_size}', parts: {parts}")

            if len(parts) >= 4:
                w_part = parts[0]  # e.g., 'w32'
                c_part = parts[1]  # e.g., 'c4096'
                k_part = parts[2]  # e.g., 'k5'
                pool_part = parts[3]  # e.g., 'avgpool'

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
                print(f"ERROR: Not enough parts in {technique} cache_size '{cache_size}' - expected format: w32_c4096_k5_avgpool")
                raise ValueError(f"Invalid {technique} cache_size format: {cache_size}")

        elif technique == 'streamingllm':
            # Handle both formats:
            # 1. Original cache dir format: local3968_init128
            # 2. Processed format in CSV: n_local_3968_n_init_128
            if cache_size.startswith('n_local_'):
                # Format: n_local_X_n_init_Y
                parts = cache_size.split('_')
                print(f"DEBUG: StreamingLLM n_local format, parts: {parts}")
                params['n_local'] = int(parts[2])
                params['n_init'] = int(parts[4])
            elif 'local' in cache_size and 'init' in cache_size:
                # Format: local3968_init128
                parts = cache_size.split('_')
                print(f"DEBUG: StreamingLLM local/init format, parts: {parts}")
                n_local = parts[0].replace('local', '')  # Get 3968 from local3968
                n_init = parts[1].replace('init', '')    # Get 128 from init128
                params['n_local'] = int(n_local)
                params['n_init'] = int(n_init)
            else:
                print(f"ERROR: Unknown StreamingLLM cache format: '{cache_size}'")

        elif technique == 'duoattn':
            # Format: spX.X_pfYYYY
            parts = cache_size.split('_')
            params['sparsity'] = float(parts[0].replace('sp', ''))
            params['prefill'] = int(parts[1].replace('pf', ''))

        elif cache_size.startswith('cache_'):
            # Generic cache size format
            params['cache_size'] = int(cache_size.replace('cache_', ''))

        else:
            params['cache_config'] = cache_size

    except (IndexError, ValueError) as e:
        print(f"Warning: Could not parse cache parameters for {technique}: {cache_size}")
        params['cache_config'] = cache_size

    return params

def load_and_filter_data(technique='pyramidkv', model='DeepSeek-R1-Distill-Llama-8B', context_length='16k'):
    """Load the helmet results and filter for specified technique/model/context."""
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

    # Filter for specified technique/model/context
    filters = {
        'technique': technique,
        'model': model,
        'context_length': context_length
    }

    print(f"Filtering for: {technique} + {model} + {context_length}")

    memory_filtered = memory_df[
        (memory_df['technique'] == filters['technique']) &
        (memory_df['model'] == filters['model']) &
        (memory_df['context_length'] == filters['context_length'])
    ].copy()

    throughput_filtered = throughput_df[
        (throughput_df['technique'] == filters['technique']) &
        (throughput_df['model'] == filters['model']) &
        (throughput_df['context_length'] == filters['context_length'])
    ].copy()

    performance_filtered = performance_df[
        (performance_df['technique'] == filters['technique']) &
        (performance_df['model'] == filters['model']) &
        (performance_df['context_length'] == filters['context_length'])
    ].copy()

    return memory_filtered, throughput_filtered, performance_filtered

def add_hyperparameter_columns(df, technique):
    """Add hyperparameter columns by parsing cache_size."""
    if df.empty:
        return df

    print(f"\nDEBUG: Processing {len(df)} rows for technique '{technique}'")

    # Parse cache parameters for each row
    param_rows = []
    parse_success_count = 0
    parse_error_count = 0

    for idx, row in df.iterrows():
        print(f"\nDEBUG: Processing row {idx}: cache_size='{row['cache_size']}'")
        params = parse_cache_params(row['cache_size'], technique)
        param_rows.append(params)

        # Check if parsing was successful
        primary_params = get_primary_hyperparams(technique)
        if len(primary_params) >= 2 and all(param in params for param in primary_params):
            parse_success_count += 1
        else:
            parse_error_count += 1
            print(f"WARNING: Failed to parse hyperparameters for row {idx}, cache_size='{row['cache_size']}'")

    # Add all parameter columns
    param_df = pd.DataFrame(param_rows)
    df_combined = pd.concat([df.reset_index(drop=True), param_df.reset_index(drop=True)], axis=1)

    print(f"\nDEBUG: Parsing summary:")
    print(f"  - Successfully parsed: {parse_success_count} rows")
    print(f"  - Failed to parse: {parse_error_count} rows")
    print(f"  - Total rows: {len(df_combined)}")

    # Show unique cache_size values that failed
    primary_params = get_primary_hyperparams(technique)
    if len(primary_params) >= 2:
        failed_rows = df_combined[df_combined[primary_params[0]].isna()]
        if not failed_rows.empty:
            print(f"\nDEBUG: Failed cache_size values:")
            for cache_size in failed_rows['cache_size'].unique():
                print(f"  - '{cache_size}'")

    return df_combined

def get_primary_hyperparams(technique):
    """Get the primary hyperparameters for a technique."""
    if technique in ['pyramidkv', 'snapkv']:
        return ['window_size', 'max_capacity']
    elif technique == 'streamingllm':
        return ['n_local', 'n_init']
    elif technique == 'duoattn':
        return ['sparsity', 'prefill']
    else:
        return ['cache_config']

def get_available_tasks(performance_df, context_length):
    """Get available performance tasks based on context length."""
    if context_length in ['16k', '32k']:
        # HELMET tasks
        base_tasks = ['recall_jsonkv', 'rag_nq', 'rag_hotpotqa', 'rerank', 'niah']
        cite_tasks = [col for col in performance_df.columns if col.startswith('cite_')]
        return base_tasks + cite_tasks
    else:
        # LongProc tasks
        return ['html_to_tsv', 'pseudo_to_code', 'travel_planning']

def create_analysis_plots(memory_df, throughput_df, performance_df, technique, model, context_length, task, output_dir):
    """Create comprehensive analysis plots."""

    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")

    # Get primary hyperparameters
    primary_params = get_primary_hyperparams(technique)

    if len(primary_params) < 2:
        print(f"Warning: Only one primary hyperparameter found for {technique}")
        return

    param1, param2 = primary_params[0], primary_params[1]

    # Create figure with subplots: Memory vs Performance, Latency vs Performance
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Colors for different values of first parameter
    param1_values = sorted(memory_df[param1].unique()) if param1 in memory_df.columns else []
    if len(param1_values) == 0:
        print(f"No data found for parameter {param1}")
        return

    colors = plt.cm.Set1(np.linspace(0, 1, len(param1_values)))
    param1_color_map = dict(zip(param1_values, colors))

    # Plot 1: Memory vs Performance
    ax1 = axes[0]
    plot_memory_vs_performance(ax1, memory_df, performance_df, task, param1, param2, param1_color_map)

    # Plot 2: Latency vs Performance
    ax2 = axes[1]
    plot_latency_vs_performance(ax2, throughput_df, performance_df, task, param1, param2, param1_color_map)

    # Create shared legend
    create_shared_legend(fig, param1, param1_color_map)

    # Add title with configuration info
    fig.suptitle(f'{technique.upper()} Hyperparameter Analysis\n{model} | {context_length} | {task.replace("_", " ").title()}',
                 fontsize=16, y=0.98)

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{technique}_{model.replace("-", "_")}_{context_length}_{task.replace("_", "-")}_analysis.png'
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()

def plot_memory_vs_performance(ax, memory_df, performance_df, task, param1, param2, color_map):
    """Plot Memory vs Performance with connecting lines."""

    # Debug: Check if hyperparameter columns exist in both dataframes
    print(f"DEBUG: Memory DF columns with hyperparams: {[col for col in memory_df.columns if col in [param1, param2]]}")
    print(f"DEBUG: Performance DF columns with hyperparams: {[col for col in performance_df.columns if col in [param1, param2]]}")

    # Merge memory and performance data
    merged = pd.merge(memory_df, performance_df, on=['technique', 'context_length', 'model', 'cache_size'], suffixes=('_mem', '_perf'))

    print(f"DEBUG: Merged DF columns: {list(merged.columns)}")
    print(f"DEBUG: Merged DF shape: {merged.shape}")

    # Check if hyperparameter columns exist after merge (they might have suffixes)
    param1_col = None
    param2_col = None

    # Look for the parameter columns with or without suffixes
    for col in merged.columns:
        if col == param1 or col.startswith(f"{param1}_"):
            param1_col = col
            break

    for col in merged.columns:
        if col == param2 or col.startswith(f"{param2}_"):
            param2_col = col
            break

    if param1_col is None:
        print(f"ERROR: {param1} column missing after merge!")
        print(f"Available columns: {list(merged.columns)}")
        return
    if param2_col is None:
        print(f"ERROR: {param2} column missing after merge!")
        print(f"Available columns: {list(merged.columns)}")
        return

    print(f"DEBUG: Using columns {param1_col} and {param2_col} for plotting")

    # Determine which column to use for memory (try common task names)
    memory_cols = ['cite', 'recall_jsonkv', 'html_to_tsv', 'pseudo_to_code', 'travel_planning']
    memory_col = None
    for col in memory_cols:
        if col in merged.columns and merged[col].notna().any():
            memory_col = col
            break

    if memory_col is None:
        print("Warning: No suitable memory column found")
        return

    # Group by first parameter and plot
    for param1_val in sorted(merged[param1_col].unique()):
        param_data = merged[merged[param1_col] == param1_val].sort_values(param2_col)

        if len(param_data) == 0:
            continue

        # Find the correct task column (might have suffix)
        task_col = None
        for col in param_data.columns:
            if col == task or col.startswith(f"{task}_"):
                task_col = col
                break

        if task_col is None:
            print(f"ERROR: Task column '{task}' not found in merged data!")
            print(f"Available task-like columns: {[col for col in param_data.columns if task in col]}")
            continue

        # Extract memory and performance values
        memory_values = param_data[memory_col].values
        performance_values = param_data[task_col].values
        param2_values = param_data[param2_col].values

        # Filter out NaN values
        valid_indices = ~(np.isnan(memory_values) | np.isnan(performance_values))
        memory_values = memory_values[valid_indices]
        performance_values = performance_values[valid_indices]
        param2_values = param2_values[valid_indices]

        if len(memory_values) == 0:
            continue

        # Plot points
        ax.scatter(memory_values, performance_values,
                  color=color_map[param1_val],
                  s=80, alpha=0.8, edgecolors='black', linewidth=0.5,
                  label=f'{param1}={param1_val}')

        # Connect points with lines
        ax.plot(memory_values, performance_values,
               color=color_map[param1_val], alpha=0.6, linewidth=2, linestyle='-')

        # Annotate points with second parameter values
        for i, (mem, perf, p2) in enumerate(zip(memory_values, performance_values, param2_values)):
            ax.annotate(f'{p2}', (mem, perf), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)

    ax.set_xlabel('Memory Usage (GB)', fontsize=12)
    ax.set_ylabel(f'{task.replace("_", " ").title()}', fontsize=12)
    ax.set_title(f'Memory vs Performance\n(Numbers show {param2})', fontsize=14)
    ax.grid(True, alpha=0.3)

def plot_latency_vs_performance(ax, throughput_df, performance_df, task, param1, param2, color_map):
    """Plot Latency vs Performance with connecting lines."""

    # Debug: Check if hyperparameter columns exist in both dataframes
    print(f"DEBUG: Throughput DF columns with hyperparams: {[col for col in throughput_df.columns if col in [param1, param2]]}")
    print(f"DEBUG: Performance DF columns with hyperparams: {[col for col in performance_df.columns if col in [param1, param2]]}")

    # Determine which column to use for throughput
    throughput_cols = ['cite', 'recall_jsonkv', 'html_to_tsv', 'pseudo_to_code', 'travel_planning']
    throughput_col = None
    for col in throughput_cols:
        if col in throughput_df.columns and throughput_df[col].notna().any():
            throughput_col = col
            break

    if throughput_col is None:
        print("Warning: No suitable throughput column found")
        return

    # Convert throughput to latency (1/throughput)
    throughput_df = throughput_df.copy()
    throughput_df[throughput_col] = throughput_df[throughput_col].replace(0, np.nan)
    throughput_df['latency'] = 1 / throughput_df[throughput_col]

    # Merge latency and performance data
    merged = pd.merge(throughput_df, performance_df, on=['technique', 'context_length', 'model', 'cache_size'], suffixes=('_thr', '_perf'))

    print(f"DEBUG: Latency merged DF columns: {list(merged.columns)}")
    print(f"DEBUG: Latency merged DF shape: {merged.shape}")

    # Check if hyperparameter columns exist after merge (they might have suffixes)
    param1_col = None
    param2_col = None

    # Look for the parameter columns with or without suffixes
    for col in merged.columns:
        if col == param1 or col.startswith(f"{param1}_"):
            param1_col = col
            break

    for col in merged.columns:
        if col == param2 or col.startswith(f"{param2}_"):
            param2_col = col
            break

    if param1_col is None:
        print(f"ERROR: {param1} column missing after latency merge!")
        print(f"Available columns: {list(merged.columns)}")
        return
    if param2_col is None:
        print(f"ERROR: {param2} column missing after latency merge!")
        print(f"Available columns: {list(merged.columns)}")
        return

    print(f"DEBUG: Using latency columns {param1_col} and {param2_col} for plotting")

    # Group by first parameter and plot
    for param1_val in sorted(merged[param1_col].unique()):
        param_data = merged[merged[param1_col] == param1_val].sort_values(param2_col)

        if len(param_data) == 0:
            continue

        # Find the correct task column (might have suffix)
        task_col = None
        for col in param_data.columns:
            if col == task or col.startswith(f"{task}_"):
                task_col = col
                break

        if task_col is None:
            print(f"ERROR: Task column '{task}' not found in latency merged data!")
            print(f"Available task-like columns: {[col for col in param_data.columns if task in col]}")
            continue

        # Extract latency and performance values
        latency_values = param_data['latency'].values
        performance_values = param_data[task_col].values
        param2_values = param_data[param2_col].values

        # Filter out NaN values
        valid_indices = ~(np.isnan(latency_values) | np.isnan(performance_values))
        latency_values = latency_values[valid_indices]
        performance_values = performance_values[valid_indices]
        param2_values = param2_values[valid_indices]

        if len(latency_values) == 0:
            continue

        # Plot points
        ax.scatter(latency_values, performance_values,
                  color=color_map[param1_val],
                  s=80, alpha=0.8, edgecolors='black', linewidth=0.5,
                  label=f'{param1}={param1_val}')

        # Connect points with lines
        ax.plot(latency_values, performance_values,
               color=color_map[param1_val], alpha=0.6, linewidth=2, linestyle='-')

        # Annotate points with second parameter values
        for i, (lat, perf, p2) in enumerate(zip(latency_values, performance_values, param2_values)):
            ax.annotate(f'{p2}', (lat, perf), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)

    ax.set_xlabel('Latency (s/token)', fontsize=12)
    ax.set_ylabel(f'{task.replace("_", " ").title()}', fontsize=12)
    ax.set_title(f'Latency vs Performance\n(Numbers show {param2})', fontsize=14)
    ax.grid(True, alpha=0.3)

def create_shared_legend(fig, param1, color_map):
    """Create a shared legend for the figure."""
    legend_elements = []
    for param1_val in sorted(color_map.keys()):
        legend_elements.append(
            Line2D([0], [0], marker='o', color=color_map[param1_val],
                   linestyle='-', markersize=8, linewidth=2,
                   label=f'{param1} = {param1_val}')
        )

    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02),
               ncol=len(legend_elements), fontsize=11,
               title=f'Hyperparameter: {param1}', title_fontsize=12)

def create_heatmap_analysis(memory_df, performance_df, technique, model, context_length, task, output_dir):
    """Create heatmap analysis showing performance across different hyperparameter combinations."""

    primary_params = get_primary_hyperparams(technique)
    if len(primary_params) < 2:
        return

    param1, param2 = primary_params[0], primary_params[1]

    # Determine memory column
    memory_cols = ['cite', 'recall_jsonkv', 'html_to_tsv', 'pseudo_to_code', 'travel_planning']
    memory_col = None
    for col in memory_cols:
        if col in memory_df.columns and memory_df[col].notna().any():
            memory_col = col
            break

    if memory_col is None:
        print("Warning: No suitable memory column found for heatmap")
        return

    # Merge all data
    memory_perf = pd.merge(memory_df, performance_df, on=['technique', 'context_length', 'model', 'cache_size'], suffixes=('_mem', '_perf'))

    # Find the correct hyperparameter columns (might have suffixes)
    param1_col = None
    param2_col = None
    task_col = None

    for col in memory_perf.columns:
        if col == param1 or col.startswith(f"{param1}_"):
            param1_col = col
        if col == param2 or col.startswith(f"{param2}_"):
            param2_col = col
        if col == task or col.startswith(f"{task}_"):
            task_col = col

    if not all([param1_col, param2_col, task_col]):
        print(f"ERROR: Missing columns for heatmap - param1: {param1_col}, param2: {param2_col}, task: {task_col}")
        return

    # Create pivot tables for heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    try:
        # Performance heatmap
        perf_pivot = memory_perf.pivot(index=param1_col, columns=param2_col, values=task_col)
        sns.heatmap(perf_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0])
        axes[0].set_title(f'{task.replace("_", " ").title()} Performance')
        axes[0].set_xlabel(param2.replace('_', ' ').title())
        axes[0].set_ylabel(param1.replace('_', ' ').title())

        # Memory heatmap
        mem_pivot = memory_perf.pivot(index=param1_col, columns=param2_col, values=memory_col)
        sns.heatmap(mem_pivot, annot=True, fmt='.1f', cmap='YlGnBu', ax=axes[1])
        axes[1].set_title('Memory Usage (GB)')
        axes[1].set_xlabel(param2.replace('_', ' ').title())
        axes[1].set_ylabel(param1.replace('_', ' ').title())

        # Efficiency heatmap (Performance / Memory)
        efficiency = memory_perf.copy()
        efficiency['efficiency'] = efficiency[task_col] / efficiency[memory_col]
        eff_pivot = efficiency.pivot(index=param1_col, columns=param2_col, values='efficiency')
        sns.heatmap(eff_pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[2])
        axes[2].set_title('Efficiency (Performance/Memory)')
        axes[2].set_xlabel(param2.replace('_', ' ').title())
        axes[2].set_ylabel(param1.replace('_', ' ').title())

        # Add title
        fig.suptitle(f'{technique.upper()} Heatmap Analysis | {model} | {context_length} | {task.replace("_", " ").title()}',
                     fontsize=16, y=0.98)

        plt.tight_layout()
        filename = f'{technique}_{model.replace("-", "_")}_{context_length}_{task.replace("_", "-")}_heatmap.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.show()

    except Exception as e:
        print(f"Error creating heatmap: {e}")

def analyze_pareto_frontier(memory_df, performance_df, technique, task):
    """Identify Pareto optimal configurations."""
    # Determine memory column
    memory_cols = ['cite', 'recall_jsonkv', 'html_to_tsv', 'pseudo_to_code', 'travel_planning']
    memory_col = None
    for col in memory_cols:
        if col in memory_df.columns and memory_df[col].notna().any():
            memory_col = col
            break

    if memory_col is None:
        print("Warning: No suitable memory column found for Pareto analysis")
        return

    # Merge data
    merged = pd.merge(memory_df, performance_df, on=['technique', 'context_length', 'model', 'cache_size'], suffixes=('_mem', '_perf'))

    # Find the correct task column (might have suffix)
    task_col = None
    for col in merged.columns:
        if col == task or col.startswith(f"{task}_"):
            task_col = col
            break

    if merged.empty or task_col is None:
        print(f"No data available for Pareto analysis of {task}")
        return

    # Calculate Pareto frontier (minimize memory, maximize performance)
    pareto_configs = []

    for _, row in merged.iterrows():
        if pd.isna(row[memory_col]) or pd.isna(row[task_col]):
            continue

        is_pareto = True
        for _, other_row in merged.iterrows():
            if pd.isna(other_row[memory_col]) or pd.isna(other_row[task_col]):
                continue

            if (other_row[memory_col] <= row[memory_col] and other_row[task_col] >= row[task_col] and
                not (other_row[memory_col] == row[memory_col] and other_row[task_col] == row[task_col])):
                is_pareto = False
                break
        if is_pareto:
            pareto_configs.append(row)

    pareto_df = pd.DataFrame(pareto_configs)

    print(f"\n=== PARETO OPTIMAL CONFIGURATIONS FOR {task.upper()} ({technique.upper()}) ===")
    print("(Configurations that are not dominated by any other configuration)")
    print("-" * 80)

    if not pareto_df.empty:
        # Get hyperparameter columns for display (handle suffixes)
        param_cols = get_primary_hyperparams(technique)
        param_cols_actual = []

        for param in param_cols:
            for col in pareto_df.columns:
                if col == param or col.startswith(f"{param}_"):
                    param_cols_actual.append(col)
                    break

        display_cols = param_cols_actual + [memory_col, task_col]

        # Filter for existing columns
        display_cols = [col for col in display_cols if col in pareto_df.columns]

        pareto_df_display = pareto_df[display_cols].sort_values(task_col, ascending=False)
        print(pareto_df_display.to_string(index=False, float_format='%.3f'))

        if len(pareto_df_display) > 0:
            try:
                print(f"\nBest performance: {dict(pareto_df_display.iloc[0][param_cols_actual])}")
                print(f"Most memory efficient: {dict(pareto_df_display.iloc[-1][param_cols_actual])}")
            except Exception as e:
                print(f"Note: Could not extract hyperparameter details: {e}")
    else:
        print("No Pareto optimal configurations found.")

def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze hyperparameter sweep results')
    parser.add_argument('--technique', default='pyramidkv',
                        help='KV-cache technique (pyramidkv, snapkv, streamingllm, duoattn)')
    parser.add_argument('--model', default='DeepSeek-R1-Distill-Llama-8B',
                        help='Model name')
    parser.add_argument('--context', default='16k',
                        help='Context length')
    parser.add_argument('--task', default=None,
                        help='Specific task to analyze (if not specified, analyzes all)')
    parser.add_argument('--output_dir', default='/scratch/gpfs/DANQIC/jz4391/HELMET/results/hyperparameter_analysis',
                        help='Output directory for plots')

    args = parser.parse_args()

    print(f"Loading {args.technique} hyperparameter sweep results...")
    print(f"Configuration: {args.technique} + {args.model} + {args.context}")

    # Load and filter data
    memory_df, throughput_df, performance_df = load_and_filter_data(args.technique, args.model, args.context)

    if memory_df is None or memory_df.empty:
        print(f"Error: No {args.technique} results found for {args.model} with {args.context} context.")
        print("Make sure your experiments have completed and results have been collected.")
        return

    # Add hyperparameter columns
    memory_df = add_hyperparameter_columns(memory_df, args.technique)
    throughput_df = add_hyperparameter_columns(throughput_df, args.technique)
    performance_df = add_hyperparameter_columns(performance_df, args.technique)

    print(f"Found {len(memory_df)} memory measurements")
    print(f"Found {len(throughput_df)} throughput measurements")
    print(f"Found {len(performance_df)} performance measurements")

    # Print available hyperparameter combinations
    primary_params = get_primary_hyperparams(args.technique)
    for param in primary_params:
        if param in memory_df.columns:
            print(f"Available {param}: {sorted(memory_df[param].unique())}")

    # Get available tasks
    available_tasks = get_available_tasks(performance_df, args.context)
    performance_tasks = [task for task in available_tasks if task in performance_df.columns]
    print(f"Available performance metrics: {performance_tasks}")

    # Determine which tasks to analyze
    if args.task:
        if args.task in performance_tasks:
            tasks_to_analyze = [args.task]
        else:
            print(f"Error: Task '{args.task}' not available. Choose from: {performance_tasks}")
            return
    else:
        tasks_to_analyze = performance_tasks

    # Analyze each task
    for task in tasks_to_analyze:
        print(f"\n{'='*60}")
        print(f"ANALYZING {task.upper()}")
        print(f"{'='*60}")

        # Create analysis plots
        create_analysis_plots(memory_df, throughput_df, performance_df,
                            args.technique, args.model, args.context, task, args.output_dir)

        # Create heatmap analysis
        create_heatmap_analysis(memory_df, performance_df,
                              args.technique, args.model, args.context, task, args.output_dir)

        # Analyze Pareto frontier
        analyze_pareto_frontier(memory_df, performance_df, args.technique, task)

if __name__ == "__main__":
    main()