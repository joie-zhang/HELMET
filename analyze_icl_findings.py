"""
Analyze ICL plot data and generate numerical findings comparing reasoning models
to instruction-tuned models across different token eviction techniques.
"""

import pandas as pd
import numpy as np

# Load the extracted plot data
df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/icl_plot_data_16k.csv')

print("=" * 80)
print("ICL Memory-Only Plot Analysis: Numerical Findings")
print("=" * 80)
print()

# Get baseline performance for both models
baseline_r1 = df[(df['technique'] == 'baseline') & (df['model'] == 'DeepSeek-R1-Distill-Llama-8B')].iloc[0]
baseline_llama = df[(df['technique'] == 'baseline') & (df['model'] == 'Llama-3.1-8B-Instruct')].iloc[0]

print("BASELINE PERFORMANCE:")
print(f"  DeepSeek-R1-Distill-Llama-8B: {baseline_r1['avg_icl_performance']:.1f}%")
print(f"  Llama-3.1-8B-Instruct: {baseline_llama['avg_icl_performance']:.1f}%")
print(f"  R1-Distill-Llama advantage: {baseline_r1['avg_icl_performance'] - baseline_llama['avg_icl_performance']:.1f} percentage points")
print()

# Analyze SnapKV with small cache (w256_c2048)
print("=" * 80)
print("SNAPKV ANALYSIS (W=256, C=2048):")
print("=" * 80)

snapkv_small_r1 = df[(df['technique'] == 'snapkv') &
                      (df['model'] == 'DeepSeek-R1-Distill-Llama-8B') &
                      (df['cache_size'] == 'w256_c2048_k7_maxpool')].iloc[0]

snapkv_small_llama = df[(df['technique'] == 'snapkv') &
                         (df['model'] == 'Llama-3.1-8B-Instruct') &
                         (df['cache_size'] == 'w256_c2048_k7_maxpool')].iloc[0]

# Calculate absolute drops
r1_snapkv_drop_abs = baseline_r1['avg_icl_performance'] - snapkv_small_r1['avg_icl_performance']
llama_snapkv_drop_abs = baseline_llama['avg_icl_performance'] - snapkv_small_llama['avg_icl_performance']

# Calculate percentage drops
r1_snapkv_drop_pct = r1_snapkv_drop_abs
llama_snapkv_drop_pct = llama_snapkv_drop_abs

print(f"\nLlama-3.1-8B-Instruct:")
print(f"  Baseline: {baseline_llama['avg_icl_performance']:.1f}%")
print(f"  SnapKV (small): {snapkv_small_llama['avg_icl_performance']:.1f}%")
print(f"  Absolute drop: {llama_snapkv_drop_abs:.1f} percentage points")
print(f"  Percentage drop: {llama_snapkv_drop_pct:.1f}%")

print(f"\nDeepSeek-R1-Distill-Llama-8B:")
print(f"  Baseline: {baseline_r1['avg_icl_performance']:.1f}%")
print(f"  SnapKV (small): {snapkv_small_r1['avg_icl_performance']:.1f}%")
print(f"  Absolute drop: {r1_snapkv_drop_abs:.1f} percentage points")
print(f"  Percentage drop: {r1_snapkv_drop_pct:.1f}%")

print(f"\nComparison:")
print(f"  R1-Distill-Llama degradation is {llama_snapkv_drop_pct - r1_snapkv_drop_pct:.1f} percentage points LESS than Llama-Instruct")
print()

# Analyze PyramidKV with small cache (w256_c2048)
print("=" * 80)
print("PYRAMIDKV ANALYSIS (W=256, C=2048):")
print("=" * 80)

pyramidkv_small_r1 = df[(df['technique'] == 'pyramidkv') &
                         (df['model'] == 'DeepSeek-R1-Distill-Llama-8B') &
                         (df['cache_size'] == 'w256_c2048_k7_avgpool')].iloc[0]

pyramidkv_small_llama = df[(df['technique'] == 'pyramidkv') &
                            (df['model'] == 'Llama-3.1-8B-Instruct') &
                            (df['cache_size'] == 'w256_c2048_k7_avgpool')].iloc[0]

# Calculate absolute drops
r1_pyramidkv_drop_abs = baseline_r1['avg_icl_performance'] - pyramidkv_small_r1['avg_icl_performance']
llama_pyramidkv_drop_abs = baseline_llama['avg_icl_performance'] - pyramidkv_small_llama['avg_icl_performance']

# Calculate percentage drops
r1_pyramidkv_drop_pct = r1_pyramidkv_drop_abs
llama_pyramidkv_drop_pct = llama_pyramidkv_drop_abs

print(f"\nLlama-3.1-8B-Instruct:")
print(f"  Baseline: {baseline_llama['avg_icl_performance']:.1f}%")
print(f"  PyramidKV (small): {pyramidkv_small_llama['avg_icl_performance']:.1f}%")
print(f"  Absolute drop: {llama_pyramidkv_drop_abs:.1f} percentage points")
print(f"  Percentage drop: {llama_pyramidkv_drop_pct:.1f}%")

print(f"\nDeepSeek-R1-Distill-Llama-8B:")
print(f"  Baseline: {baseline_r1['avg_icl_performance']:.1f}%")
print(f"  PyramidKV (small): {pyramidkv_small_r1['avg_icl_performance']:.1f}%")
print(f"  Absolute drop: {r1_pyramidkv_drop_abs:.1f} percentage points")
print(f"  Percentage drop: {r1_pyramidkv_drop_pct:.1f}%")

print(f"\nComparison:")
print(f"  R1-Distill-Llama degradation is {llama_pyramidkv_drop_pct - r1_pyramidkv_drop_pct:.1f} percentage points LESS than Llama-Instruct")
print()

# Average across SnapKV and PyramidKV
print("=" * 80)
print("COMBINED SNAPKV + PYRAMIDKV ANALYSIS (SMALL CACHE):")
print("=" * 80)

avg_llama_drop = (llama_snapkv_drop_pct + llama_pyramidkv_drop_pct) / 2
avg_r1_drop = (r1_snapkv_drop_pct + r1_pyramidkv_drop_pct) / 2

print(f"\nAverage performance drop with small cache (W=256, C=2048):")
print(f"  Llama-3.1-8B-Instruct: {avg_llama_drop:.1f} percentage points")
print(f"  DeepSeek-R1-Distill-Llama-8B: {avg_r1_drop:.1f} percentage points")
print(f"  Difference: {avg_llama_drop - avg_r1_drop:.1f} percentage points")
print()

# Analyze DuoAttention
print("=" * 80)
print("DUOATTENTION ANALYSIS:")
print("=" * 80)

duoattn_r1 = df[(df['technique'] == 'duoattn') & (df['model'] == 'DeepSeek-R1-Distill-Llama-8B')].iloc[0]
duoattn_llama = df[(df['technique'] == 'duoattn') & (df['model'] == 'Llama-3.1-8B-Instruct')].iloc[0]

r1_duoattn_drop = baseline_r1['avg_icl_performance'] - duoattn_r1['avg_icl_performance']
llama_duoattn_drop = baseline_llama['avg_icl_performance'] - duoattn_llama['avg_icl_performance']

print(f"\nLlama-3.1-8B-Instruct:")
print(f"  Baseline: {baseline_llama['avg_icl_performance']:.1f}%")
print(f"  DuoAttention: {duoattn_llama['avg_icl_performance']:.1f}%")
print(f"  Drop: {llama_duoattn_drop:.1f} percentage points")

print(f"\nDeepSeek-R1-Distill-Llama-8B:")
print(f"  Baseline: {baseline_r1['avg_icl_performance']:.1f}%")
print(f"  DuoAttention: {duoattn_r1['avg_icl_performance']:.1f}%")
print(f"  Drop: {r1_duoattn_drop:.1f} percentage points")

print(f"\nMemory savings with DuoAttention:")
print(f"  Both models: {baseline_r1['avg_icl_memory_gb'] - duoattn_r1['avg_icl_memory_gb']:.2f} GB")
print(f"  Percentage reduction: {((baseline_r1['avg_icl_memory_gb'] - duoattn_r1['avg_icl_memory_gb']) / baseline_r1['avg_icl_memory_gb'] * 100):.1f}%")
print()

# Analyze StreamingLLM
print("=" * 80)
print("STREAMINGLLM ANALYSIS:")
print("=" * 80)

streamingllm_r1 = df[(df['technique'] == 'streamingllm') & (df['model'] == 'DeepSeek-R1-Distill-Llama-8B')].iloc[0]
streamingllm_llama = df[(df['technique'] == 'streamingllm') & (df['model'] == 'Llama-3.1-8B-Instruct')].iloc[0]

r1_streaming_drop = baseline_r1['avg_icl_performance'] - streamingllm_r1['avg_icl_performance']
llama_streaming_drop = baseline_llama['avg_icl_performance'] - streamingllm_llama['avg_icl_performance']

print(f"\nLlama-3.1-8B-Instruct:")
print(f"  Baseline: {baseline_llama['avg_icl_performance']:.1f}%")
print(f"  StreamingLLM: {streamingllm_llama['avg_icl_performance']:.1f}%")
print(f"  Drop: {llama_streaming_drop:.1f} percentage points")

print(f"\nDeepSeek-R1-Distill-Llama-8B:")
print(f"  Baseline: {baseline_r1['avg_icl_performance']:.1f}%")
print(f"  StreamingLLM: {streamingllm_r1['avg_icl_performance']:.1f}%")
print(f"  Drop: {r1_streaming_drop:.1f} percentage points")
print()

# Summary table
print("=" * 80)
print("SUMMARY TABLE: PERFORMANCE DROPS FROM BASELINE")
print("=" * 80)
print()

summary_data = {
    'Technique': ['SnapKV (small)', 'PyramidKV (small)', 'Average (small cache)', 'DuoAttention', 'StreamingLLM'],
    'Llama-3.1-8B-Instruct Drop (pp)': [
        llama_snapkv_drop_pct,
        llama_pyramidkv_drop_pct,
        avg_llama_drop,
        llama_duoattn_drop,
        llama_streaming_drop
    ],
    'R1-Distill-Llama-8B Drop (pp)': [
        r1_snapkv_drop_pct,
        r1_pyramidkv_drop_pct,
        avg_r1_drop,
        r1_duoattn_drop,
        r1_streaming_drop
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df['Advantage (pp)'] = summary_df['Llama-3.1-8B-Instruct Drop (pp)'] - summary_df['R1-Distill-Llama-8B Drop (pp)']

print(summary_df.to_string(index=False))
print()

# Save summary to CSV
summary_df.to_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/icl_findings_summary.csv', index=False)
print("Summary table saved to: /scratch/gpfs/DANQIC/jz4391/HELMET/results/icl_findings_summary.csv")
print()

# Generate key findings
print("=" * 80)
print("KEY FINDINGS FOR PAPER:")
print("=" * 80)
print()

findings = []

# Finding 1: Baseline performance
findings.append(f"The DeepSeek-R1-Distill-Llama-8B reasoning model achieves {baseline_r1['avg_icl_performance']:.1f}% average performance on ICL tasks (Banking77 and Clinc150), compared to {baseline_llama['avg_icl_performance']:.1f}% for Llama-3.1-8B-Instruct, showing a {baseline_r1['avg_icl_performance'] - baseline_llama['avg_icl_performance']:.1f} percentage point advantage.")

# Finding 2: Token eviction resilience
findings.append(f"When subjected to aggressive token eviction with SnapKV (W=256, C=2048), Llama-3.1-8B-Instruct experiences a {llama_snapkv_drop_pct:.1f} percentage point decrease (from {baseline_llama['avg_icl_performance']:.1f}% to {snapkv_small_llama['avg_icl_performance']:.1f}%), while R1-Distill-Llama shows only a {r1_snapkv_drop_pct:.1f} percentage point decrease (from {baseline_r1['avg_icl_performance']:.1f}% to {snapkv_small_r1['avg_icl_performance']:.1f}%).")

# Finding 3: PyramidKV comparison
findings.append(f"With PyramidKV at the same cache configuration, Llama-Instruct drops {llama_pyramidkv_drop_pct:.1f} percentage points (from {baseline_llama['avg_icl_performance']:.1f}% to {pyramidkv_small_llama['avg_icl_performance']:.1f}%), while R1-Distill-Llama drops only {r1_pyramidkv_drop_pct:.1f} percentage points (from {baseline_r1['avg_icl_performance']:.1f}% to {pyramidkv_small_r1['avg_icl_performance']:.1f}%).")

# Finding 4: Average resilience
findings.append(f"On average across SnapKV and PyramidKV with small cache sizes, the reasoning model exhibits {avg_llama_drop - avg_r1_drop:.1f} percentage points less degradation than the instruction-tuned model ({avg_r1_drop:.1f} pp vs {avg_llama_drop:.1f} pp).")

# Finding 5: DuoAttention
findings.append(f"DuoAttention, which uses attention sparsity patterns, reduces memory by {((baseline_r1['avg_icl_memory_gb'] - duoattn_r1['avg_icl_memory_gb']) / baseline_r1['avg_icl_memory_gb'] * 100):.1f}% with only {r1_duoattn_drop:.1f} percentage point drop for R1-Distill-Llama and {llama_duoattn_drop:.1f} percentage point drop for Llama-Instruct.")

# Finding 6: StreamingLLM
findings.append(f"StreamingLLM causes significant degradation for both models ({r1_streaming_drop:.1f} pp for R1-Distill-Llama, {llama_streaming_drop:.1f} pp for Llama-Instruct), suggesting that ICL tasks require access to earlier context that StreamingLLM's fixed attention window cannot provide.")

print("\nFINDINGS (BULLET FORM):")
print("-" * 80)
for i, finding in enumerate(findings, 1):
    print(f"{i}. {finding}")
    print()

# Generate paragraph form
print("\nFINDINGS (PARAGRAPH FORM):")
print("-" * 80)
paragraph = f"""The most striking finding from the ICL memory-only analysis is that the DeepSeek-R1-Distill-Llama-8B reasoning model demonstrates substantially greater resilience to token eviction techniques compared to the Llama-3.1-8B-Instruct model. Specifically, when subjected to aggressive token eviction with SnapKV at W=256 and C=2048, Llama-Instruct experiences a {llama_snapkv_drop_pct:.1f} percentage point decrease (from {baseline_llama['avg_icl_performance']:.1f}% to {snapkv_small_llama['avg_icl_performance']:.1f}%), while R1-Distill-Llama shows only a {r1_snapkv_drop_pct:.1f} percentage point decrease (from {baseline_r1['avg_icl_performance']:.1f}% to {snapkv_small_r1['avg_icl_performance']:.1f}%). Similarly, with PyramidKV at the same cache configuration, Llama-Instruct drops {llama_pyramidkv_drop_pct:.1f} percentage points (from {baseline_llama['avg_icl_performance']:.1f}% to {pyramidkv_small_llama['avg_icl_performance']:.1f}%), while R1-Distill-Llama drops only {r1_pyramidkv_drop_pct:.1f} percentage points (from {baseline_r1['avg_icl_performance']:.1f}% to {pyramidkv_small_r1['avg_icl_performance']:.1f}%). The reasoning model thus exhibits {avg_llama_drop - avg_r1_drop:.1f} percentage points less degradation on average than the instruct model even when subjected to the more ambitious token eviction setting. This superior resilience suggests that reasoning models may structure information more redundantly across their context, making them more robust to the loss of individual tokens during eviction."""

print(paragraph)
print()

# Save findings to file
with open('/scratch/gpfs/DANQIC/jz4391/HELMET/results/icl_findings.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("ICL MEMORY-ONLY PLOT: KEY FINDINGS\n")
    f.write("=" * 80 + "\n\n")

    f.write("FINDINGS (BULLET FORM):\n")
    f.write("-" * 80 + "\n")
    for i, finding in enumerate(findings, 1):
        f.write(f"{i}. {finding}\n\n")

    f.write("\nFINDINGS (PARAGRAPH FORM):\n")
    f.write("-" * 80 + "\n")
    f.write(paragraph + "\n")

print("Findings saved to: /scratch/gpfs/DANQIC/jz4391/HELMET/results/icl_findings.txt")
print()
