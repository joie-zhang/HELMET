#!/usr/bin/env python3
"""
Generate numerical findings from the quadrant comparison plot data.
Produces analysis similar to the style of the provided example paragraph.
"""

import pandas as pd
import numpy as np
import os

# Load the extracted data
results_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/results'
data_path = os.path.join(results_dir, 'quadrant_plot_data.csv')
df = pd.read_csv(data_path)

print("="*80)
print("QUADRANT COMPARISON ANALYSIS: PERFORMANCE BY TASK DIFFICULTY")
print("="*80)
print()

# Get baseline performance for each difficulty level
baseline_data = df[df['Technique'] == 'Baseline'].set_index('Difficulty_Level')

def calculate_relative_change(technique_name, difficulty_level, metric='Average_Performance'):
    """Calculate percent change relative to baseline."""
    technique_val = df[(df['Technique'] == technique_name) &
                       (df['Difficulty_Level'] == difficulty_level)][metric].values[0]
    baseline_val = baseline_data.loc[difficulty_level, metric]

    if pd.isna(technique_val) or pd.isna(baseline_val) or baseline_val == 0:
        return np.nan

    return ((technique_val - baseline_val) / baseline_val) * 100

def get_avg_and_se(technique_name, metric='Average_Performance'):
    """Get average across all difficulty levels with standard error."""
    tech_data = df[df['Technique'] == technique_name][metric]
    avg = tech_data.mean()
    se = tech_data.std(ddof=1) / np.sqrt(len(tech_data)) if len(tech_data) > 1 else 0
    return avg, se

# Calculate performance changes for each technique across difficulty levels
print("TECHNIQUE PERFORMANCE SUMMARY")
print("-" * 80)

techniques = ['NF4', 'Int8', 'DuoAttention', 'SnapKV', 'PyramidKV', 'StreamingLLM']

# Store all performance changes for summary statistics
all_perf_changes = {tech: [] for tech in techniques}

for technique in techniques:
    print(f"\n{technique}:")
    for difficulty in ['Easy', 'Medium', 'Hard']:
        perf_change = calculate_relative_change(technique, difficulty)
        all_perf_changes[technique].append(perf_change)

        tech_perf = df[(df['Technique'] == technique) &
                       (df['Difficulty_Level'] == difficulty)]['Average_Performance'].values[0]
        base_perf = baseline_data.loc[difficulty, 'Average_Performance']

        print(f"  {difficulty:6s}: {tech_perf:5.1f} vs {base_perf:5.1f} (baseline) = {perf_change:+6.2f}%")

print("\n" + "="*80)
print("CROSS-DIFFICULTY ANALYSIS")
print("="*80)

# Calculate average performance change and std error across all difficulties
print("\nAverage Performance Change Across All Difficulty Levels:")
print("-" * 80)

for technique in techniques:
    changes = all_perf_changes[technique]
    avg_change = np.mean(changes)
    std_change = np.std(changes, ddof=1)
    se_change = std_change / np.sqrt(len(changes)) if len(changes) > 1 else 0

    print(f"{technique:15s}: {avg_change:+7.2f}% ± {se_change:.2f}% (std: {std_change:.2f}%)")

print("\n" + "="*80)
print("DIFFICULTY-SPECIFIC PATTERNS")
print("="*80)

# Analyze performance degradation patterns across difficulties
for difficulty in ['Easy', 'Medium', 'Hard']:
    print(f"\n{difficulty} Tasks ({df[df['Difficulty_Level']==difficulty]['Difficulty_Label'].values[0]}):")
    print("-" * 80)

    baseline_perf = baseline_data.loc[difficulty, 'Average_Performance']
    print(f"Baseline: {baseline_perf:.2f}")

    for technique in techniques:
        perf_change = calculate_relative_change(technique, difficulty)
        tech_perf = df[(df['Technique'] == technique) &
                       (df['Difficulty_Level'] == difficulty)]['Average_Performance'].values[0]
        print(f"  {technique:15s}: {tech_perf:5.2f} ({perf_change:+6.2f}%)")

print("\n" + "="*80)
print("GENERATED PARAGRAPH (Similar to provided style)")
print("="*80)

# Calculate statistics for the generated paragraph
# Quantization techniques (NF4, Int8)
nf4_perf_changes = all_perf_changes['NF4']
int8_perf_changes = all_perf_changes['Int8']

nf4_avg_perf = np.mean(nf4_perf_changes)
nf4_se_perf = np.std(nf4_perf_changes, ddof=1) / np.sqrt(len(nf4_perf_changes))
int8_avg_perf = np.mean(int8_perf_changes)
int8_se_perf = np.std(int8_perf_changes, ddof=1) / np.sqrt(len(int8_perf_changes))

# DuoAttention
duo_perf_changes = all_perf_changes['DuoAttention']
duo_avg_perf = np.mean(duo_perf_changes)
duo_se_perf = np.std(duo_perf_changes, ddof=1) / np.sqrt(len(duo_perf_changes))

# Token eviction methods (SnapKV, PyramidKV, StreamingLLM)
eviction_techniques = ['SnapKV', 'PyramidKV', 'StreamingLLM']
eviction_perf_changes = []
for tech in eviction_techniques:
    eviction_perf_changes.extend(all_perf_changes[tech])

eviction_avg_perf = np.mean(eviction_perf_changes)
eviction_se_perf = np.std(eviction_perf_changes, ddof=1) / np.sqrt(len(eviction_perf_changes))

# Calculate performance drop from easy to hard for baseline
baseline_easy = baseline_data.loc['Easy', 'Average_Performance']
baseline_medium = baseline_data.loc['Medium', 'Average_Performance']
baseline_hard = baseline_data.loc['Hard', 'Average_Performance']
baseline_drop_easy_to_hard = ((baseline_hard - baseline_easy) / baseline_easy) * 100

# Calculate how techniques perform relative to baseline as difficulty increases
# Focus on the degradation pattern
print("\nGenerated Paragraph:")
print("-" * 80)

paragraph = f"""
In the quadrant comparison analysis (Figure X), we examine the average performance of
baseline and six different techniques across three difficulty levels: Easy (Short Output,
Low Dispersion), Medium (Short Output, High Dispersion), and Hard (Long Output, High
Dispersion), averaged across all models and context lengths from both HELMET and LongProc
benchmarks.

Quantization methods (NF4 and Int8) demonstrate consistent performance across difficulty
levels, with NF4 achieving {nf4_avg_perf:+.2f}% ± {nf4_se_perf:.2f}% average change relative
to baseline, and Int8 achieving {int8_avg_perf:+.2f}% ± {int8_se_perf:.2f}%. These methods
maintain near-baseline performance with minimal degradation across all task difficulties.

DuoAttention represents a notable exception among KV cache methods, achieving improved
performance of {duo_avg_perf:+.2f}% ± {duo_se_perf:.2f}% across all difficulty levels.
The improvement is most pronounced on Easy tasks ({all_perf_changes['DuoAttention'][0]:+.2f}%),
with sustained gains on Medium ({all_perf_changes['DuoAttention'][1]:+.2f}%) and Hard
({all_perf_changes['DuoAttention'][2]:+.2f}%) tasks.

In contrast, token eviction methods (SnapKV, PyramidKV, StreamingLLM) show substantial
performance degradation averaging {eviction_avg_perf:+.2f}% ± {eviction_se_perf:.2f}%
across all difficulty levels. StreamingLLM exhibits the most severe degradation, particularly
on Easy tasks ({all_perf_changes['StreamingLLM'][0]:+.2f}%), while maintaining relatively
better performance on Hard tasks ({all_perf_changes['StreamingLLM'][2]:+.2f}%). SnapKV and
PyramidKV show more uniform degradation patterns, with SnapKV performing at
{all_perf_changes['SnapKV'][0]:+.2f}%, {all_perf_changes['SnapKV'][1]:+.2f}%, and
{all_perf_changes['SnapKV'][2]:+.2f}% on Easy, Medium, and Hard tasks respectively, while
PyramidKV achieves {all_perf_changes['PyramidKV'][0]:+.2f}%, {all_perf_changes['PyramidKV'][1]:+.2f}%,
and {all_perf_changes['PyramidKV'][2]:+.2f}%.

The baseline itself shows a natural performance degradation as task difficulty increases,
declining from {baseline_easy:.1f} on Easy tasks to {baseline_medium:.1f} on Medium tasks
(−{((baseline_medium - baseline_easy) / baseline_easy * 100):.1f}%) and {baseline_hard:.1f}
on Hard tasks (−{((baseline_hard - baseline_easy) / baseline_easy * 100):.1f}% from Easy).
This {baseline_drop_easy_to_hard:.1f}% drop highlights the inherent challenge increase, which
token eviction methods exacerbate while DuoAttention and quantization methods manage more
gracefully.
"""

print(paragraph)

print("\n" + "="*80)
print("KEY STATISTICS TABLE")
print("="*80)

# Create a summary table
summary_data = []
for technique in ['Baseline'] + techniques:
    row = {'Technique': technique}
    for difficulty in ['Easy', 'Medium', 'Hard']:
        perf = df[(df['Technique'] == technique) &
                  (df['Difficulty_Level'] == difficulty)]['Average_Performance'].values[0]
        se = df[(df['Technique'] == technique) &
                (df['Difficulty_Level'] == difficulty)]['Std_Error'].values[0]

        if technique != 'Baseline':
            change = calculate_relative_change(technique, difficulty)
            row[f'{difficulty}_Perf'] = f"{perf:.1f} ({change:+.1f}%)"
        else:
            row[f'{difficulty}_Perf'] = f"{perf:.1f}"

    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save summary statistics
summary_csv_path = os.path.join(results_dir, 'quadrant_plot_summary_statistics.csv')
summary_df.to_csv(summary_csv_path, index=False)
print(f"\n✓ Summary statistics saved to: {summary_csv_path}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)