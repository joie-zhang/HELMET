#!/usr/bin/env python3
"""
Deep dive analysis into task vulnerability patterns across efficient inference techniques.
Explores what makes certain tasks more vulnerable to different optimization methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'Arial'

# Define paths
results_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/results'
output_dir = os.path.join(results_dir, 'plots')
os.makedirs(output_dir, exist_ok=True)

# Load data
df_grouped = pd.read_csv(os.path.join(results_dir, 'task_deltas_data.csv'))
df_detailed = pd.read_csv(os.path.join(results_dir, 'task_deltas_data_detailed.csv'))

print("=" * 100)
print("TASK VULNERABILITY DEEP DIVE ANALYSIS")
print("=" * 100)

# =======================================================================================
# SECTION 1: Task Vulnerability Rankings
# =======================================================================================
print("\n\n1. TASK VULNERABILITY RANKINGS")
print("-" * 100)

# Calculate vulnerability for each task (average absolute degradation across all techniques)
task_groups = [col for col in df_grouped.columns if col != 'Technique']
task_vulnerabilities = {}

for task in task_groups:
    values = df_grouped[task].values.astype(float)
    valid_values = values[~np.isnan(values)]
    if len(valid_values) > 0:
        # Use absolute value to measure magnitude of change
        task_vulnerabilities[task] = {
            'mean_delta': np.mean(valid_values),
            'std_delta': np.std(valid_values),
            'abs_mean': np.mean(np.abs(valid_values)),
            'range': np.max(valid_values) - np.min(valid_values),
            'min': np.min(valid_values),
            'max': np.max(valid_values)
        }

# Sort by absolute mean degradation (vulnerability)
sorted_tasks = sorted(task_vulnerabilities.items(), key=lambda x: x[1]['abs_mean'], reverse=True)

print("\nMost Vulnerable Tasks (by average absolute degradation):")
print(f"{'Rank':<6} {'Task':<12} {'Mean Δ':<12} {'Std Δ':<12} {'Range':<12} {'Min':<12} {'Max':<12}")
print("-" * 100)
for i, (task, metrics) in enumerate(sorted_tasks, 1):
    print(f"{i:<6} {task:<12} {metrics['mean_delta']:>10.2f}% {metrics['std_delta']:>10.2f}% "
          f"{metrics['range']:>10.2f}% {metrics['min']:>10.2f}% {metrics['max']:>10.2f}%")

# =======================================================================================
# SECTION 2: Technique-Specific Task Vulnerabilities
# =======================================================================================
print("\n\n2. TECHNIQUE-SPECIFIC TASK VULNERABILITIES")
print("-" * 100)

techniques = df_grouped['Technique'].tolist()

# For each technique, identify most/least vulnerable tasks
print("\nMost vulnerable task for each technique:")
for idx, row in df_grouped.iterrows():
    tech_name = row['Technique']
    values = row[task_groups].values.astype(float)

    # Find most degraded (most negative or largest absolute change)
    valid_mask = ~np.isnan(values)
    if valid_mask.any():
        valid_values = values[valid_mask]
        valid_tasks = [task_groups[i] for i in range(len(task_groups)) if valid_mask[i]]

        worst_idx = np.argmin(valid_values)
        best_idx = np.argmax(valid_values)

        print(f"\n{tech_name}:")
        print(f"  Most vulnerable: {valid_tasks[worst_idx]} ({valid_values[worst_idx]:.2f}%)")
        print(f"  Least vulnerable: {valid_tasks[best_idx]} ({valid_values[best_idx]:.2f}%)")

# =======================================================================================
# SECTION 3: Task Type Analysis
# =======================================================================================
print("\n\n3. TASK TYPE ANALYSIS")
print("-" * 100)

# Categorize tasks by type
task_types = {
    'Memory Retrieval': ['NIAH', 'Recall'],
    'Information Retrieval': ['RAG', 'Re-rank'],
    'In-Context Learning': ['ICL'],
    'Citation': ['Cite'],
    'Reasoning': ['Summ', 'Pseudo'],
    'Structured Generation': ['HTML', 'Travel']
}

print("\nAverage degradation by task type:")
for task_type, tasks in task_types.items():
    type_values = []
    for task in tasks:
        if task in df_grouped.columns:
            values = df_grouped[task].values.astype(float)
            valid_values = values[~np.isnan(values)]
            type_values.extend(valid_values)

    if type_values:
        print(f"\n{task_type}:")
        print(f"  Average: {np.mean(type_values):.2f}% ± {np.std(type_values):.2f}%")
        print(f"  Range: [{np.min(type_values):.2f}%, {np.max(type_values):.2f}%]")

# =======================================================================================
# SECTION 4: Technique Class Comparison by Task
# =======================================================================================
print("\n\n4. TECHNIQUE CLASS COMPARISON BY TASK")
print("-" * 100)

# Group techniques into classes
technique_classes = {
    'Quantization': ['NF4', 'Int8'],
    'Token Eviction': ['SnapKV', 'PyramidKV', 'StreamingLLM'],
    'Attention Optimization': ['DuoAttention']
}

print("\nTask vulnerability by technique class:")
print(f"\n{'Task':<12} | {'Quantization':<15} | {'Token Eviction':<15} | {'Attn Opt':<15}")
print("-" * 100)

for task in task_groups:
    row_data = {}
    for class_name, tech_list in technique_classes.items():
        values = []
        for tech in tech_list:
            if tech in df_grouped['Technique'].values:
                val = df_grouped[df_grouped['Technique'] == tech][task].values[0]
                if not np.isnan(val):
                    values.append(val)
        if values:
            row_data[class_name] = np.mean(values)
        else:
            row_data[class_name] = np.nan

    quant = f"{row_data.get('Quantization', np.nan):.2f}%" if not np.isnan(row_data.get('Quantization', np.nan)) else "N/A"
    evict = f"{row_data.get('Token Eviction', np.nan):.2f}%" if not np.isnan(row_data.get('Token Eviction', np.nan)) else "N/A"
    attn = f"{row_data.get('Attention Optimization', np.nan):.2f}%" if not np.isnan(row_data.get('Attention Optimization', np.nan)) else "N/A"

    print(f"{task:<12} | {quant:^15} | {evict:^15} | {attn:^15}")

# =======================================================================================
# SECTION 5: Correlation Analysis
# =======================================================================================
print("\n\n5. CORRELATION ANALYSIS: TASK SIMILARITY IN VULNERABILITY")
print("-" * 100)

# Create task-task correlation matrix based on their vulnerability patterns
task_matrix = df_grouped[task_groups].T.astype(float)
task_correlation = task_matrix.corr()

print("\nHighly correlated task pairs (similar vulnerability patterns):")
correlations = []
for i in range(len(task_groups)):
    for j in range(i+1, len(task_groups)):
        corr = task_correlation.iloc[i, j]
        if not np.isnan(corr):
            correlations.append((task_groups[i], task_groups[j], corr))

correlations.sort(key=lambda x: abs(x[2]), reverse=True)
for task1, task2, corr in correlations[:10]:
    print(f"  {task1} <-> {task2}: r = {corr:.3f}")

# =======================================================================================
# SECTION 6: Identify Critical Failure Cases
# =======================================================================================
print("\n\n6. CRITICAL FAILURE CASES (Degradation > 50%)")
print("-" * 100)

critical_failures = []
for idx, row in df_grouped.iterrows():
    tech_name = row['Technique']
    for task in task_groups:
        val = row[task]
        if not np.isnan(val) and val < -50:
            critical_failures.append({
                'Technique': tech_name,
                'Task': task,
                'Degradation': val
            })

if critical_failures:
    df_failures = pd.DataFrame(critical_failures)
    df_failures = df_failures.sort_values('Degradation')
    print("\nCritical failure cases:")
    print(df_failures.to_string(index=False))
else:
    print("\nNo critical failures detected (degradation > 50%)")

# =======================================================================================
# SECTION 7: Robustness Score
# =======================================================================================
print("\n\n7. TASK ROBUSTNESS SCORES")
print("-" * 100)
print("(Lower score = more robust across techniques, Higher score = more vulnerable)\n")

robustness_scores = {}
for task in task_groups:
    values = df_grouped[task].values.astype(float)
    valid_values = values[~np.isnan(values)]
    if len(valid_values) > 0:
        # Robustness score: combination of mean absolute degradation and variance
        robustness_scores[task] = {
            'score': np.mean(np.abs(valid_values)) + 0.5 * np.std(valid_values),
            'mean_abs': np.mean(np.abs(valid_values)),
            'std': np.std(valid_values)
        }

sorted_robustness = sorted(robustness_scores.items(), key=lambda x: x[1]['score'])

print(f"{'Rank':<6} {'Task':<12} {'Robustness Score':<18} {'Mean |Δ|':<12} {'Std Δ':<12}")
print("-" * 100)
for i, (task, metrics) in enumerate(sorted_robustness, 1):
    print(f"{i:<6} {task:<12} {metrics['score']:>16.2f} {metrics['mean_abs']:>10.2f}% {metrics['std']:>10.2f}%")

# =======================================================================================
# SECTION 8: Technique Selectivity
# =======================================================================================
print("\n\n8. TECHNIQUE SELECTIVITY (Variance in task performance)")
print("-" * 100)
print("(Higher variance = technique is more selective/task-dependent)\n")

technique_selectivity = []
for idx, row in df_grouped.iterrows():
    tech_name = row['Technique']
    values = row[task_groups].values.astype(float)
    valid_values = values[~np.isnan(values)]
    if len(valid_values) > 0:
        technique_selectivity.append({
            'Technique': tech_name,
            'Task Variance': np.var(valid_values),
            'Task Std': np.std(valid_values),
            'Range': np.max(valid_values) - np.min(valid_values)
        })

df_selectivity = pd.DataFrame(technique_selectivity)
df_selectivity = df_selectivity.sort_values('Task Variance', ascending=False)
print(df_selectivity.to_string(index=False))

# =======================================================================================
# Generate Summary Report
# =======================================================================================
print("\n\n" + "=" * 100)
print("SUMMARY: KEY INSIGHTS")
print("=" * 100)

# Find most vulnerable task overall
most_vulnerable_task = sorted_tasks[0]
least_vulnerable_task = sorted_tasks[-1]

print(f"""
1. MOST VULNERABLE TASK: {most_vulnerable_task[0]}
   - Average degradation: {most_vulnerable_task[1]['mean_delta']:.2f}% ± {most_vulnerable_task[1]['std_delta']:.2f}%
   - Range: [{most_vulnerable_task[1]['min']:.2f}%, {most_vulnerable_task[1]['max']:.2f}%]

2. MOST ROBUST TASK: {least_vulnerable_task[0]}
   - Average degradation: {least_vulnerable_task[1]['mean_delta']:.2f}% ± {least_vulnerable_task[1]['std_delta']:.2f}%
   - Range: [{least_vulnerable_task[1]['min']:.2f}%, {least_vulnerable_task[1]['max']:.2f}%]

3. MOST SELECTIVE TECHNIQUE: {df_selectivity.iloc[0]['Technique']}
   - Task variance: {df_selectivity.iloc[0]['Task Variance']:.2f}
   - This technique shows the most task-dependent performance

4. LEAST SELECTIVE TECHNIQUE: {df_selectivity.iloc[-1]['Technique']}
   - Task variance: {df_selectivity.iloc[-1]['Task Variance']:.2f}
   - This technique shows the most uniform performance across tasks

5. NUMBER OF CRITICAL FAILURES: {len(critical_failures)}
   - Critical failures are defined as degradation > 50%
""")

# Save detailed analysis
output_path = os.path.join(results_dir, 'task_vulnerability_analysis.txt')
with open(output_path, 'w') as f:
    f.write("TASK VULNERABILITY DEEP DIVE ANALYSIS\n")
    f.write("=" * 100 + "\n\n")

    f.write("MOST VULNERABLE TASKS:\n")
    for i, (task, metrics) in enumerate(sorted_tasks[:5], 1):
        f.write(f"{i}. {task}: {metrics['mean_delta']:.2f}% ± {metrics['std_delta']:.2f}%\n")

    f.write("\nMOST ROBUST TASKS:\n")
    for i, (task, metrics) in enumerate(sorted_tasks[-5:], 1):
        f.write(f"{i}. {task}: {metrics['mean_delta']:.2f}% ± {metrics['std_delta']:.2f}%\n")

    f.write(f"\nMOST SELECTIVE TECHNIQUE: {df_selectivity.iloc[0]['Technique']}\n")
    f.write(f"LEAST SELECTIVE TECHNIQUE: {df_selectivity.iloc[-1]['Technique']}\n")

print(f"\nSaved detailed analysis to: {output_path}")

# Save data tables
task_vuln_df = pd.DataFrame([
    {'Task': task, **metrics}
    for task, metrics in sorted_tasks
])
task_vuln_df.to_csv(os.path.join(results_dir, 'task_vulnerability_rankings.csv'), index=False)
print(f"Saved task vulnerability rankings to: {results_dir}/task_vulnerability_rankings.csv")

df_selectivity.to_csv(os.path.join(results_dir, 'technique_selectivity.csv'), index=False)
print(f"Saved technique selectivity to: {results_dir}/technique_selectivity.csv")
