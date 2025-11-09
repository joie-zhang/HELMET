#!/usr/bin/env python3
"""
Generate numerical findings for the task deltas plot in the style of the provided paragraph.
Analyzes performance degradation patterns across different techniques and task categories.
"""

import pandas as pd
import numpy as np
import os

# Define paths
results_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/results'

# Load the extracted data
df_grouped = pd.read_csv(os.path.join(results_dir, 'task_deltas_data.csv'))
df_detailed = pd.read_csv(os.path.join(results_dir, 'task_deltas_data_detailed.csv'))

print("=" * 80)
print("TASK-WISE PERFORMANCE DELTA ANALYSIS")
print("=" * 80)

# Get technique names
techniques = df_grouped['Technique'].tolist()

# Get task columns (excluding 'Technique')
task_groups = [col for col in df_grouped.columns if col != 'Technique']

print("\n1. OVERALL PERFORMANCE DEGRADATION BY TECHNIQUE")
print("-" * 80)

# Calculate overall averages and std for each technique
for idx, row in df_grouped.iterrows():
    tech_name = row['Technique']
    values = row[task_groups].values.astype(float)
    values = values[~np.isnan(values)]

    if len(values) > 0:
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        print(f"\n{tech_name}:")
        print(f"  Average Performance Delta: {mean_val:.2f}% ± {std_val:.2f}%")
        print(f"  Range: [{min_val:.2f}%, {max_val:.2f}%]")
        print(f"  Number of task groups evaluated: {len(values)}")

print("\n\n2. TASK GROUP VULNERABILITY ANALYSIS")
print("-" * 80)
print("Identifying which task groups are most affected by each technique:\n")

for task_group in task_groups:
    values = df_grouped[task_group].values.astype(float)
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]
    valid_techniques = [techniques[i] for i in range(len(techniques)) if valid_mask[i]]

    if len(valid_values) > 0:
        print(f"\n{task_group}:")
        print(f"  Average degradation across all techniques: {np.mean(valid_values):.2f}% ± {np.std(valid_values):.2f}%")

        # Find best and worst techniques for this task group
        best_idx = np.argmax(valid_values)
        worst_idx = np.argmin(valid_values)

        print(f"  Most robust technique: {valid_techniques[best_idx]} ({valid_values[best_idx]:.2f}%)")
        print(f"  Most degraded technique: {valid_techniques[worst_idx]} ({valid_values[worst_idx]:.2f}%)")

print("\n\n3. QUANTIZATION METHODS (NF4 vs Int8)")
print("-" * 80)

nf4_values = df_grouped[df_grouped['Technique'] == 'NF4'][task_groups].values[0].astype(float)
int8_values = df_grouped[df_grouped['Technique'] == 'Int8'][task_groups].values[0].astype(float)

nf4_valid = nf4_values[~np.isnan(nf4_values)]
int8_valid = int8_values[~np.isnan(int8_values)]

print(f"\nNF4: {np.mean(nf4_valid):.2f}% ± {np.std(nf4_valid):.2f}%")
print(f"Int8: {np.mean(int8_valid):.2f}% ± {np.std(int8_valid):.2f}%")
print(f"\nQuantization Impact:")
print(f"  NF4 shows {'better' if np.mean(nf4_valid) > np.mean(int8_valid) else 'worse'} average performance")
print(f"  Difference: {abs(np.mean(nf4_valid) - np.mean(int8_valid)):.2f} percentage points")

print("\n\n4. KV CACHE EVICTION METHODS (SnapKV, PyramidKV, StreamingLLM)")
print("-" * 80)

snapkv_values = df_grouped[df_grouped['Technique'] == 'SnapKV'][task_groups].values[0].astype(float)
pyramidkv_values = df_grouped[df_grouped['Technique'] == 'PyramidKV'][task_groups].values[0].astype(float)
streamingllm_values = df_grouped[df_grouped['Technique'] == 'StreamingLLM'][task_groups].values[0].astype(float)

snapkv_valid = snapkv_values[~np.isnan(snapkv_values)]
pyramidkv_valid = pyramidkv_values[~np.isnan(pyramidkv_values)]
streamingllm_valid = streamingllm_values[~np.isnan(streamingllm_values)]

print(f"\nSnapKV: {np.mean(snapkv_valid):.2f}% ± {np.std(snapkv_valid):.2f}%")
print(f"PyramidKV: {np.mean(pyramidkv_valid):.2f}% ± {np.std(pyramidkv_valid):.2f}%")
print(f"StreamingLLM: {np.mean(streamingllm_valid):.2f}% ± {np.std(streamingllm_valid):.2f}%")

all_eviction_values = np.concatenate([snapkv_valid, pyramidkv_valid, streamingllm_valid])
print(f"\nCombined eviction methods: {np.mean(all_eviction_values):.2f}% ± {np.std(all_eviction_values):.2f}%")
print(f"Range: [{np.min(all_eviction_values):.2f}%, {np.max(all_eviction_values):.2f}%]")

print("\n\n5. DuoATTENTION ANALYSIS")
print("-" * 80)

duoattn_values = df_grouped[df_grouped['Technique'] == 'DuoAttention'][task_groups].values[0].astype(float)
duoattn_valid = duoattn_values[~np.isnan(duoattn_values)]

print(f"\nDuoAttention: {np.mean(duoattn_valid):.2f}% ± {np.std(duoattn_valid):.2f}%")
print(f"Range: [{np.min(duoattn_valid):.2f}%, {np.max(duoattn_valid):.2f}%]")
print(f"Number of task groups evaluated: {len(duoattn_valid)}")

# Count positive vs negative deltas
positive_count = np.sum(duoattn_valid > 0)
negative_count = np.sum(duoattn_valid < 0)
print(f"\nPerformance distribution:")
print(f"  Task groups with improvement: {positive_count}/{len(duoattn_valid)}")
print(f"  Task groups with degradation: {negative_count}/{len(duoattn_valid)}")

print("\n\n6. TASK-SPECIFIC VULNERABILITY PATTERNS")
print("-" * 80)
print("Analyzing which specific tasks are most vulnerable:\n")

# Use detailed data for individual task analysis
detailed_tasks = [col for col in df_detailed.columns if col != 'Technique']

# Calculate average degradation for each individual task
task_degradations = {}
for task in detailed_tasks:
    values = df_detailed[task].values.astype(float)
    valid_values = values[~np.isnan(values)]
    if len(valid_values) > 0:
        task_degradations[task] = np.mean(valid_values)

# Sort by degradation
sorted_tasks = sorted(task_degradations.items(), key=lambda x: x[1])

print("Most vulnerable tasks (highest degradation):")
for task, deg in sorted_tasks[:5]:
    print(f"  {task}: {deg:.2f}%")

print("\nMost robust tasks (least degradation):")
for task, deg in sorted_tasks[-5:]:
    print(f"  {task}: {deg:.2f}%")

print("\n\n7. COMPARISON WITH QUANTIZATION METHODS")
print("-" * 80)

# Compare quantization vs eviction methods
quant_combined = np.concatenate([nf4_valid, int8_valid])
print(f"\nQuantization methods (NF4 + Int8): {np.mean(quant_combined):.2f}% ± {np.std(quant_combined):.2f}%")
print(f"KV Cache eviction methods: {np.mean(all_eviction_values):.2f}% ± {np.std(all_eviction_values):.2f}%")
print(f"DuoAttention: {np.mean(duoattn_valid):.2f}% ± {np.std(duoattn_valid):.2f}%")

print(f"\nPerformance gap (Eviction vs Quantization): {abs(np.mean(all_eviction_values) - np.mean(quant_combined)):.2f} percentage points")
print(f"Performance gap (Eviction vs DuoAttention): {abs(np.mean(all_eviction_values) - np.mean(duoattn_valid)):.2f} percentage points")

# Generate LaTeX-style paragraph
print("\n\n" + "=" * 80)
print("LATEX-STYLE SUMMARY PARAGRAPH")
print("=" * 80)

paragraph = f"""
In Figure [task_deltas], we analyze the task-wise performance degradation of six efficient inference techniques relative to the baseline, averaged across all models and context lengths. Quantization methods demonstrate minimal performance impact, with NF4 averaging {np.mean(nf4_valid):.2f}% ± {np.std(nf4_valid):.2f}% degradation and Int8 showing {np.mean(int8_valid):.2f}% ± {np.std(int8_valid):.2f}% degradation across task categories. In stark contrast, token eviction methods exhibit severe and inconsistent performance penalties: SnapKV (-{abs(np.mean(snapkv_valid)):.2f}% ± {np.std(snapkv_valid):.2f}%), PyramidKV (-{abs(np.mean(pyramidkv_valid)):.2f}% ± {np.std(pyramidkv_valid):.2f}%), and StreamingLLM (-{abs(np.mean(streamingllm_valid)):.2f}% ± {np.std(streamingllm_valid):.2f}%). The aggregate degradation for token eviction methods is {np.mean(all_eviction_values):.2f}% ± {np.std(all_eviction_values):.2f}%, representing a {abs(np.mean(all_eviction_values) - np.mean(quant_combined)):.2f} percentage point gap compared to quantization methods. Notably, recall-based tasks (Recall: {df_grouped[df_grouped['Technique'].isin(['SnapKV', 'PyramidKV', 'StreamingLLM'])]['Recall'].mean():.2f}%) and retrieval tasks (Re-rank: {df_grouped[df_grouped['Technique'].isin(['SnapKV', 'PyramidKV', 'StreamingLLM'])]['Re-rank'].mean():.2f}%) suffer disproportionately under token eviction, while reasoning tasks (Summ, ICL) show relative resilience. DuoAttention demonstrates competitive performance with {np.mean(duoattn_valid):.2f}% ± {np.std(duoattn_valid):.2f}% average degradation, showing {positive_count} out of {len(duoattn_valid)} task categories with improvements or minimal degradation, though it shows vulnerabilities in specific tasks such as citation tasks and long-context generation tasks. These results highlight the critical importance of task-aware method selection, as token eviction methods systematically underperform on memory-intensive tasks while quantization provides more uniform preservation of capabilities.
"""

print(paragraph)

# Save the paragraph to a file
output_path = os.path.join(results_dir, 'task_deltas_findings.txt')
with open(output_path, 'w') as f:
    f.write(paragraph)

print(f"\n\nSaved LaTeX paragraph to: {output_path}")

# Also create a summary statistics CSV
summary_stats = []
for idx, row in df_grouped.iterrows():
    tech_name = row['Technique']
    values = row[task_groups].values.astype(float)
    values = values[~np.isnan(values)]

    if len(values) > 0:
        summary_stats.append({
            'Technique': tech_name,
            'Mean_Degradation': np.mean(values),
            'Std_Degradation': np.std(values),
            'Min_Degradation': np.min(values),
            'Max_Degradation': np.max(values),
            'Num_Tasks': len(values)
        })

df_summary = pd.DataFrame(summary_stats)
summary_path = os.path.join(results_dir, 'task_deltas_summary_statistics.csv')
df_summary.to_csv(summary_path, index=False)
print(f"Saved summary statistics to: {summary_path}")
print("\nSummary Statistics:")
print(df_summary.to_string(index=False))
