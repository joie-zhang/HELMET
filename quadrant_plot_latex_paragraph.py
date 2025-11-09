#!/usr/bin/env python3
"""
Generate LaTeX-friendly paragraph for the quadrant comparison plot findings.
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

# Calculate baseline degradation
baseline_easy = baseline_data.loc['Easy', 'Average_Performance']
baseline_medium = baseline_data.loc['Medium', 'Average_Performance']
baseline_hard = baseline_data.loc['Hard', 'Average_Performance']

easy_to_medium_pct = ((baseline_easy - baseline_medium) / baseline_easy) * 100
medium_to_hard_pct = ((baseline_medium - baseline_hard) / baseline_medium) * 100
easy_to_hard_pct = ((baseline_easy - baseline_hard) / baseline_easy) * 100

# Quantization methods (NF4, Int8)
nf4_easy_pct = ((df[(df['Technique']=='NF4')&(df['Difficulty_Level']=='Easy')]['Average_Performance'].values[0] - baseline_easy) / baseline_easy) * 100
nf4_medium_pct = ((df[(df['Technique']=='NF4')&(df['Difficulty_Level']=='Medium')]['Average_Performance'].values[0] - baseline_medium) / baseline_medium) * 100
nf4_hard_pct = ((df[(df['Technique']=='NF4')&(df['Difficulty_Level']=='Hard')]['Average_Performance'].values[0] - baseline_hard) / baseline_hard) * 100
nf4_changes = [nf4_easy_pct, nf4_medium_pct, nf4_hard_pct]
nf4_avg = np.mean(nf4_changes)
nf4_se = np.std(nf4_changes, ddof=1) / np.sqrt(len(nf4_changes))

int8_easy_pct = ((df[(df['Technique']=='Int8')&(df['Difficulty_Level']=='Easy')]['Average_Performance'].values[0] - baseline_easy) / baseline_easy) * 100
int8_medium_pct = ((df[(df['Technique']=='Int8')&(df['Difficulty_Level']=='Medium')]['Average_Performance'].values[0] - baseline_medium) / baseline_medium) * 100
int8_hard_pct = ((df[(df['Technique']=='Int8')&(df['Difficulty_Level']=='Hard')]['Average_Performance'].values[0] - baseline_hard) / baseline_hard) * 100
int8_changes = [int8_easy_pct, int8_medium_pct, int8_hard_pct]
int8_avg = np.mean(int8_changes)
int8_se = np.std(int8_changes, ddof=1) / np.sqrt(len(int8_changes))

# DuoAttention
duo_easy_pct = ((df[(df['Technique']=='DuoAttention')&(df['Difficulty_Level']=='Easy')]['Average_Performance'].values[0] - baseline_easy) / baseline_easy) * 100
duo_medium_pct = ((df[(df['Technique']=='DuoAttention')&(df['Difficulty_Level']=='Medium')]['Average_Performance'].values[0] - baseline_medium) / baseline_medium) * 100
duo_hard_pct = ((df[(df['Technique']=='DuoAttention')&(df['Difficulty_Level']=='Hard')]['Average_Performance'].values[0] - baseline_hard) / baseline_hard) * 100
duo_changes = [duo_easy_pct, duo_medium_pct, duo_hard_pct]
duo_avg = np.mean(duo_changes)
duo_se = np.std(duo_changes, ddof=1) / np.sqrt(len(duo_changes))

# Token eviction methods
eviction_methods = ['SnapKV', 'PyramidKV', 'StreamingLLM']
eviction_perf_changes = []
for method in eviction_methods:
    for difficulty, baseline_perf in [('Easy', baseline_easy), ('Medium', baseline_medium), ('Hard', baseline_hard)]:
        method_perf = df[(df['Technique']==method)&(df['Difficulty_Level']==difficulty)]['Average_Performance'].values[0]
        pct_change = ((method_perf - baseline_perf) / baseline_perf) * 100
        eviction_perf_changes.append(pct_change)

eviction_avg = np.mean(eviction_perf_changes)
eviction_se = np.std(eviction_perf_changes, ddof=1) / np.sqrt(len(eviction_perf_changes))

# Individual token eviction methods
snapkv_changes = []
pyramidkv_changes = []
streamingllm_changes = []

for difficulty, baseline_perf in [('Easy', baseline_easy), ('Medium', baseline_medium), ('Hard', baseline_hard)]:
    snapkv_perf = df[(df['Technique']=='SnapKV')&(df['Difficulty_Level']==difficulty)]['Average_Performance'].values[0]
    snapkv_changes.append(((snapkv_perf - baseline_perf) / baseline_perf) * 100)

    pyramidkv_perf = df[(df['Technique']=='PyramidKV')&(df['Difficulty_Level']==difficulty)]['Average_Performance'].values[0]
    pyramidkv_changes.append(((pyramidkv_perf - baseline_perf) / baseline_perf) * 100)

    streamingllm_perf = df[(df['Technique']=='StreamingLLM')&(df['Difficulty_Level']==difficulty)]['Average_Performance'].values[0]
    streamingllm_changes.append(((streamingllm_perf - baseline_perf) / baseline_perf) * 100)

snapkv_avg = np.mean(snapkv_changes)
snapkv_se = np.std(snapkv_changes, ddof=1) / np.sqrt(len(snapkv_changes))

pyramidkv_avg = np.mean(pyramidkv_changes)
pyramidkv_se = np.std(pyramidkv_changes, ddof=1) / np.sqrt(len(pyramidkv_changes))

streamingllm_avg = np.mean(streamingllm_changes)
streamingllm_se = np.std(streamingllm_changes, ddof=1) / np.sqrt(len(streamingllm_changes))

# Specific performance on Easy tasks (where differences are most pronounced)
snapkv_easy = df[(df['Technique']=='SnapKV')&(df['Difficulty_Level']=='Easy')]['Average_Performance'].values[0]
pyramidkv_easy = df[(df['Technique']=='PyramidKV')&(df['Difficulty_Level']=='Easy')]['Average_Performance'].values[0]
streamingllm_easy = df[(df['Technique']=='StreamingLLM')&(df['Difficulty_Level']=='Easy')]['Average_Performance'].values[0]

snapkv_easy_pct = ((snapkv_easy - baseline_easy) / baseline_easy) * 100
pyramidkv_easy_pct = ((pyramidkv_easy - baseline_easy) / baseline_easy) * 100
streamingllm_easy_pct = ((streamingllm_easy - baseline_easy) / baseline_easy) * 100

# Performance degradation from Easy to Hard for token eviction methods
snapkv_easy_to_hard = df[(df['Technique']=='SnapKV')&(df['Difficulty_Level']=='Easy')]['Average_Performance'].values[0]
snapkv_hard_to_hard = df[(df['Technique']=='SnapKV')&(df['Difficulty_Level']=='Hard')]['Average_Performance'].values[0]
snapkv_degradation = ((snapkv_easy_to_hard - snapkv_hard_to_hard) / snapkv_easy_to_hard) * 100

pyramidkv_easy_to_hard = df[(df['Technique']=='PyramidKV')&(df['Difficulty_Level']=='Easy')]['Average_Performance'].values[0]
pyramidkv_hard_to_hard = df[(df['Technique']=='PyramidKV')&(df['Difficulty_Level']=='Hard')]['Average_Performance'].values[0]
pyramidkv_degradation = ((pyramidkv_easy_to_hard - pyramidkv_hard_to_hard) / pyramidkv_easy_to_hard) * 100

# Average degradation for token eviction
avg_eviction_degradation = (snapkv_degradation + pyramidkv_degradation) / 2

print("="*80)
print("LaTeX-FRIENDLY PARAGRAPH FOR QUADRANT COMPARISON PLOT")
print("="*80)
print()

latex_paragraph = f"""In \\Cref{{fig:quadrant_comparison}}, we examine the average performance of baseline and six different methods across three difficulty levels---Easy (Short Output, Low Dispersion), Medium (Short Output, High Dispersion), and Hard (Long Output, High Dispersion)---averaged across all models and context lengths from both HELMET (16K/32K) and LongProc (2K/5K) benchmarks. The baseline model exhibits natural performance degradation as task difficulty increases, declining {easy_to_medium_pct:.1f}\\% from Easy to Medium tasks and an additional {medium_to_hard_pct:.1f}\\% from Medium to Hard tasks, representing an overall {easy_to_hard_pct:.1f}\\% drop across the difficulty spectrum. NF4 and Int8 demonstrate minimal performance degradation of {nf4_avg:+.2f}\\% ± {nf4_se:.2f}\\% and {int8_avg:+.2f}\\% ± {int8_se:.2f}\\% respectively, maintaining near-baseline performance across all difficulty levels. DuoAttention represents a notable exception among KV cache methods, achieving improved performance of {duo_avg:+.2f}\\% ± {duo_se:.2f}\\% across all difficulty levels, with particularly strong gains on Easy tasks ({duo_easy_pct:+.1f}\\%) that diminish but remain positive on Medium ({duo_medium_pct:+.1f}\\%) and Hard ({duo_hard_pct:+.1f}\\%) tasks. In contrast, token eviction methods (SnapKV, PyramidKV, StreamingLLM) show substantial performance degradation averaging {eviction_avg:.2f}\\% ± {eviction_se:.2f}\\% across all difficulty levels. StreamingLLM exhibits the most severe degradation at {streamingllm_avg:.2f}\\% ± {streamingllm_se:.2f}\\%, particularly struggling on Easy tasks ({streamingllm_easy_pct:.1f}\\%), while SnapKV and PyramidKV show more uniform degradation patterns at {snapkv_avg:.2f}\\% ± {snapkv_se:.2f}\\% and {pyramidkv_avg:.2f}\\% ± {pyramidkv_se:.2f}\\% respectively. Notably, token eviction methods exhibit substantial internal degradation as difficulty increases, with SnapKV and PyramidKV dropping {snapkv_degradation:.1f}\\% and {pyramidkv_degradation:.1f}\\% respectively from Easy to Hard tasks, exacerbating the {easy_to_hard_pct:.1f}\\% baseline degradation. Thus, quantization methods maintain near-baseline performance with minimal variation across difficulty levels, DuoAttention consistently improves performance, while token eviction methods fail to achieve competitive performance across the difficulty spectrum."""

print(latex_paragraph)
print()
print("="*80)
print("FORMATTED FOR EASIER READING")
print("="*80)
print()

# Print with line breaks for readability
lines = latex_paragraph.split('. ')
for i, line in enumerate(lines):
    if i < len(lines) - 1:
        print(line + '.')
    else:
        print(line)
    print()

print("="*80)
print("KEY STATISTICS SUMMARY")
print("="*80)
print(f"""
Baseline degradation:
  Easy → Medium: {easy_to_medium_pct:.1f}%
  Medium → Hard: {medium_to_hard_pct:.1f}%
  Easy → Hard:   {easy_to_hard_pct:.1f}%

Quantization methods (vs baseline):
  NF4:  {nf4_avg:+.2f}% ± {nf4_se:.2f}%
  Int8: {int8_avg:+.2f}% ± {int8_se:.2f}%

DuoAttention (vs baseline):
  Average:      {duo_avg:+.2f}% ± {duo_se:.2f}%
  Easy tasks:   {duo_easy_pct:+.1f}%
  Medium tasks: {duo_medium_pct:+.1f}%
  Hard tasks:   {duo_hard_pct:+.1f}%

Token eviction methods (vs baseline):
  Average (all):  {eviction_avg:.2f}% ± {eviction_se:.2f}%
  SnapKV:         {snapkv_avg:.2f}% ± {snapkv_se:.2f}%
  PyramidKV:      {pyramidkv_avg:.2f}% ± {pyramidkv_se:.2f}%
  StreamingLLM:   {streamingllm_avg:.2f}% ± {streamingllm_se:.2f}%

  StreamingLLM on Easy tasks: {streamingllm_easy_pct:.1f}%

  Internal degradation (Easy → Hard):
    SnapKV:    {snapkv_degradation:.1f}%
    PyramidKV: {pyramidkv_degradation:.1f}%
""")

# Save to file
output_file = os.path.join(results_dir, 'quadrant_plot_latex_paragraph.txt')
with open(output_file, 'w') as f:
    f.write(latex_paragraph)

print(f"✓ LaTeX paragraph saved to: {output_file}")
print("="*80)
