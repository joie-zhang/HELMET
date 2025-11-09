#!/usr/bin/env python3
"""
Generate comprehensive key findings summary for the quadrant comparison plot.
Focus on:
1. Baseline degradation from Easy → Medium → Hard
2. Average degradation across all efficient inference methods
3. DuoAttention's improvement over baseline
4. Token eviction methods' poor performance compared to baseline, NF4, Int8
"""

import pandas as pd
import numpy as np
import os

# Load the extracted data
results_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/results'
data_path = os.path.join(results_dir, 'quadrant_plot_data.csv')
df = pd.read_csv(data_path)

# Get baseline performance for each difficulty level
baseline_data = df[df['Technique'] == 'Baseline'].set_index('Difficulty_Level')

print("="*80)
print("QUADRANT COMPARISON PLOT: KEY FINDINGS SUMMARY")
print("="*80)
print()

# ============================================================================
# FINDING 1: Baseline performance degradation across difficulty levels
# ============================================================================
print("FINDING 1: Baseline Performance Degradation Across Task Difficulty")
print("-" * 80)

baseline_easy = baseline_data.loc['Easy', 'Average_Performance']
baseline_medium = baseline_data.loc['Medium', 'Average_Performance']
baseline_hard = baseline_data.loc['Hard', 'Average_Performance']

baseline_easy_se = baseline_data.loc['Easy', 'Std_Error']
baseline_medium_se = baseline_data.loc['Medium', 'Std_Error']
baseline_hard_se = baseline_data.loc['Hard', 'Std_Error']

# Calculate absolute and relative drops
easy_to_medium_drop = baseline_easy - baseline_medium
easy_to_medium_pct = (easy_to_medium_drop / baseline_easy) * 100

medium_to_hard_drop = baseline_medium - baseline_hard
medium_to_hard_pct = (medium_to_hard_drop / baseline_medium) * 100

easy_to_hard_drop = baseline_easy - baseline_hard
easy_to_hard_pct = (easy_to_hard_drop / baseline_easy) * 100

print(f"Easy tasks:   {baseline_easy:.2f} ± {baseline_easy_se:.2f}")
print(f"Medium tasks: {baseline_medium:.2f} ± {baseline_medium_se:.2f} "
      f"({easy_to_medium_drop:.2f} point drop, {easy_to_medium_pct:.1f}% decrease from Easy)")
print(f"Hard tasks:   {baseline_hard:.2f} ± {baseline_hard_se:.2f} "
      f"({medium_to_hard_drop:.2f} point drop, {medium_to_hard_pct:.1f}% decrease from Medium)")
print(f"\nOverall Easy → Hard: {easy_to_hard_drop:.2f} point drop ({easy_to_hard_pct:.1f}% decrease)")

# ============================================================================
# FINDING 2: Average degradation across ALL efficient inference methods
# ============================================================================
print("\n" + "="*80)
print("FINDING 2: Average Performance Across All Efficient Inference Methods")
print("-" * 80)

all_methods = ['NF4', 'Int8', 'DuoAttention', 'SnapKV', 'PyramidKV', 'StreamingLLM']

# Calculate average performance for all methods combined at each difficulty level
all_methods_data = {}
for difficulty in ['Easy', 'Medium', 'Hard']:
    all_perfs = []
    for method in all_methods:
        perf = df[(df['Technique'] == method) &
                  (df['Difficulty_Level'] == difficulty)]['Average_Performance'].values[0]
        all_perfs.append(perf)

    avg_perf = np.mean(all_perfs)
    se_perf = np.std(all_perfs, ddof=1) / np.sqrt(len(all_perfs))
    all_methods_data[difficulty] = {'avg': avg_perf, 'se': se_perf}

    # Compare to baseline
    baseline_perf = baseline_data.loc[difficulty, 'Average_Performance']
    diff_from_baseline = avg_perf - baseline_perf
    pct_change = (diff_from_baseline / baseline_perf) * 100

    print(f"{difficulty} tasks: {avg_perf:.2f} ± {se_perf:.2f} vs {baseline_perf:.2f} baseline "
          f"({pct_change:+.1f}%)")

# Calculate degradation from Easy to Medium to Hard for all methods
all_easy_to_medium_drop = all_methods_data['Easy']['avg'] - all_methods_data['Medium']['avg']
all_easy_to_medium_pct = (all_easy_to_medium_drop / all_methods_data['Easy']['avg']) * 100

all_medium_to_hard_drop = all_methods_data['Medium']['avg'] - all_methods_data['Hard']['avg']
all_medium_to_hard_pct = (all_medium_to_hard_drop / all_methods_data['Medium']['avg']) * 100

all_easy_to_hard_drop = all_methods_data['Easy']['avg'] - all_methods_data['Hard']['avg']
all_easy_to_hard_pct = (all_easy_to_hard_drop / all_methods_data['Easy']['avg']) * 100

print(f"\nAverage degradation across all methods:")
print(f"  Easy → Medium: {all_easy_to_medium_drop:.2f} points ({all_easy_to_medium_pct:.1f}%)")
print(f"  Medium → Hard: {all_medium_to_hard_drop:.2f} points ({all_medium_to_hard_pct:.1f}%)")
print(f"  Easy → Hard:   {all_easy_to_hard_drop:.2f} points ({all_easy_to_hard_pct:.1f}%)")

# ============================================================================
# FINDING 3: DuoAttention's improvement over baseline
# ============================================================================
print("\n" + "="*80)
print("FINDING 3: DuoAttention Performance vs Baseline")
print("-" * 80)

duo_improvements = []
for difficulty in ['Easy', 'Medium', 'Hard']:
    duo_perf = df[(df['Technique'] == 'DuoAttention') &
                  (df['Difficulty_Level'] == difficulty)]['Average_Performance'].values[0]
    duo_se = df[(df['Technique'] == 'DuoAttention') &
                (df['Difficulty_Level'] == difficulty)]['Std_Error'].values[0]

    baseline_perf = baseline_data.loc[difficulty, 'Average_Performance']
    baseline_se = baseline_data.loc[difficulty, 'Std_Error']

    improvement = duo_perf - baseline_perf
    improvement_pct = (improvement / baseline_perf) * 100
    duo_improvements.append(improvement_pct)

    print(f"{difficulty} tasks: {duo_perf:.2f} ± {duo_se:.2f} vs {baseline_perf:.2f} ± {baseline_se:.2f} "
          f"({improvement:+.2f} points, {improvement_pct:+.1f}%)")

avg_duo_improvement = np.mean(duo_improvements)
se_duo_improvement = np.std(duo_improvements, ddof=1) / np.sqrt(len(duo_improvements))

print(f"\nAverage improvement across all difficulty levels: {avg_duo_improvement:+.2f}% ± {se_duo_improvement:.2f}%")

# ============================================================================
# FINDING 4: Token eviction methods vs baseline, NF4, Int8
# ============================================================================
print("\n" + "="*80)
print("FINDING 4: Token Eviction Methods (SnapKV, PyramidKV, StreamingLLM)")
print("            vs Baseline, NF4, and Int8")
print("-" * 80)

eviction_methods = ['SnapKV', 'PyramidKV', 'StreamingLLM']
comparison_methods = ['Baseline', 'NF4', 'Int8']

# For each difficulty level, show token eviction vs comparisons
for difficulty in ['Easy', 'Medium', 'Hard']:
    print(f"\n{difficulty} Tasks:")
    print("-" * 40)

    # Get baseline, NF4, Int8 performance
    comparison_perfs = {}
    for method in comparison_methods:
        perf = df[(df['Technique'] == method) &
                  (df['Difficulty_Level'] == difficulty)]['Average_Performance'].values[0]
        comparison_perfs[method] = perf
        print(f"  {method:12s}: {perf:.2f}")

    avg_comparison = np.mean(list(comparison_perfs.values()))
    print(f"  {'Avg (Baseline/NF4/Int8)':>25s}: {avg_comparison:.2f}")
    print()

    # Show token eviction methods
    for method in eviction_methods:
        perf = df[(df['Technique'] == method) &
                  (df['Difficulty_Level'] == difficulty)]['Average_Performance'].values[0]

        # Compare to baseline
        baseline_perf = comparison_perfs['Baseline']
        vs_baseline = perf - baseline_perf
        vs_baseline_pct = (vs_baseline / baseline_perf) * 100

        # Compare to average of baseline/NF4/Int8
        vs_avg = perf - avg_comparison
        vs_avg_pct = (vs_avg / avg_comparison) * 100

        print(f"  {method:12s}: {perf:.2f} "
              f"(vs baseline: {vs_baseline:+.2f} / {vs_baseline_pct:+.1f}%, "
              f"vs avg: {vs_avg:+.2f} / {vs_avg_pct:+.1f}%)")

# Calculate overall averages for token eviction methods
print("\n" + "-" * 80)
print("Overall Average Performance (across all difficulty levels):")
print("-" * 80)

for method in eviction_methods:
    method_perfs = df[df['Technique'] == method]['Average_Performance'].values
    avg_perf = np.mean(method_perfs)

    # Compare to average baseline performance
    baseline_perfs = baseline_data['Average_Performance'].values
    avg_baseline = np.mean(baseline_perfs)

    # Compare to average NF4/Int8 performance
    nf4_perfs = df[df['Technique'] == 'NF4']['Average_Performance'].values
    int8_perfs = df[df['Technique'] == 'Int8']['Average_Performance'].values
    avg_quant = np.mean(np.concatenate([nf4_perfs, int8_perfs]))

    vs_baseline = avg_perf - avg_baseline
    vs_baseline_pct = (vs_baseline / avg_baseline) * 100

    vs_quant = avg_perf - avg_quant
    vs_quant_pct = (vs_quant / avg_quant) * 100

    print(f"{method:12s}: {avg_perf:.2f} "
          f"(vs baseline: {vs_baseline:+.2f} / {vs_baseline_pct:+.1f}%, "
          f"vs NF4/Int8: {vs_quant:+.2f} / {vs_quant_pct:+.1f}%)")

# Average of all three token eviction methods
all_eviction_perfs = df[df['Technique'].isin(eviction_methods)]['Average_Performance'].values
avg_eviction = np.mean(all_eviction_perfs)
se_eviction = np.std(all_eviction_perfs, ddof=1) / np.sqrt(len(all_eviction_perfs))

baseline_avg = np.mean(baseline_perfs)
nf4_int8_avg = avg_quant

print(f"\n{'Avg (all token eviction)':>25s}: {avg_eviction:.2f} ± {se_eviction:.2f}")
print(f"{'Avg (Baseline)':>25s}: {baseline_avg:.2f}")
print(f"{'Avg (NF4/Int8)':>25s}: {nf4_int8_avg:.2f}")
print(f"\nToken eviction gap from baseline: {avg_eviction - baseline_avg:+.2f} ({(avg_eviction - baseline_avg)/baseline_avg*100:+.1f}%)")
print(f"Token eviction gap from NF4/Int8:  {avg_eviction - nf4_int8_avg:+.2f} ({(avg_eviction - nf4_int8_avg)/nf4_int8_avg*100:+.1f}%)")

print("\n" + "="*80)
print("SUMMARY PARAGRAPH")
print("="*80)
print(f"""
Across three difficulty levels (Easy, Medium, Hard), the baseline model shows natural
performance degradation, declining from {baseline_easy:.1f} on Easy tasks to {baseline_medium:.1f}
on Medium tasks (-{easy_to_medium_pct:.1f}%) and {baseline_hard:.1f} on Hard tasks (-{medium_to_hard_pct:.1f}%
from Medium), representing an overall {easy_to_hard_pct:.1f}% drop from Easy to Hard tasks.
Averaged across all six efficient inference methods, performance follows a similar pattern,
dropping {all_easy_to_hard_pct:.1f}% from Easy to Hard tasks ({all_methods_data['Easy']['avg']:.1f} →
{all_methods_data['Medium']['avg']:.1f} → {all_methods_data['Hard']['avg']:.1f}).

DuoAttention consistently outperforms baseline across all difficulty levels, achieving
{duo_improvements[0]:+.1f}%, {duo_improvements[1]:+.1f}%, and {duo_improvements[2]:+.1f}% improvements
on Easy, Medium, and Hard tasks respectively (average: {avg_duo_improvement:+.1f}% ± {se_duo_improvement:.2f}%).
This represents the only KV cache method that maintains or improves performance relative to
baseline across the full difficulty spectrum.

In stark contrast, token eviction methods (SnapKV, PyramidKV, StreamingLLM) show severe
degradation compared to both baseline and quantization methods (NF4, Int8). Averaging across
all difficulty levels, these methods achieve {avg_eviction:.1f} ± {se_eviction:.2f} performance
compared to {baseline_avg:.1f} for baseline and {nf4_int8_avg:.1f} for NF4/Int8, representing
gaps of {(avg_eviction - baseline_avg)/baseline_avg*100:+.1f}% and {(avg_eviction - nf4_int8_avg)/nf4_int8_avg*100:+.1f}%
respectively. StreamingLLM performs worst overall, particularly struggling on Easy tasks
({df[(df['Technique']=='StreamingLLM')&(df['Difficulty_Level']=='Easy')]['Average_Performance'].values[0]:.1f},
{(df[(df['Technique']=='StreamingLLM')&(df['Difficulty_Level']=='Easy')]['Average_Performance'].values[0] - baseline_easy)/baseline_easy*100:+.1f}%
vs baseline), while SnapKV and PyramidKV show more uniform degradation patterns averaging
{np.mean(df[df['Technique']=='SnapKV']['Average_Performance'].values):.1f} and
{np.mean(df[df['Technique']=='PyramidKV']['Average_Performance'].values):.1f} respectively.
Notably, quantization methods (NF4: {nf4_perfs.mean():.1f}, Int8: {int8_perfs.mean():.1f})
maintain near-baseline performance with minimal degradation, making them substantially more
effective than token eviction approaches across the difficulty spectrum.
""")

print("\n" + "="*80)
