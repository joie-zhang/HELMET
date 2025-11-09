import pandas as pd
import numpy as np

# Load the exact plot data
df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/plot_data_with_connections_incl_duo.csv')

print("="*80)
print("RECALCULATING PARAGRAPH STATISTICS FROM ACTUAL PLOT DATA")
print("="*80)

# Get baseline statistics
baseline_data = df[df['technique'] == 'baseline']
baseline_mem_mean = baseline_data['memory_gb'].mean()
baseline_perf_mean = baseline_data['performance_score'].mean()

print(f"\nBASELINE (averaged across all 4 models):")
print(f"  Average Memory: {baseline_mem_mean:.3f} GB")
print(f"  Average Performance: {baseline_perf_mean:.3f}")
print(f"  Individual baselines:")
for _, row in baseline_data.iterrows():
    print(f"    {row['model']}: {row['memory_gb']:.3f} GB, {row['performance_score']:.3f} score")

# Function to calculate statistics for a technique
def calc_technique_stats(df, technique_name, baseline_mem_mean, baseline_perf_mean):
    technique_data = df[df['technique'] == technique_name]

    if len(technique_data) == 0:
        return None

    # Calculate changes relative to baseline for each model
    mem_changes = []
    perf_changes = []

    for _, row in technique_data.iterrows():
        # Find the corresponding baseline for this model
        model_baseline = df[(df['technique'] == 'baseline') & (df['model'] == row['model'])]
        if len(model_baseline) > 0:
            baseline_mem = model_baseline.iloc[0]['memory_gb']
            baseline_perf = model_baseline.iloc[0]['performance_score']

            mem_change_pct = ((row['memory_gb'] - baseline_mem) / baseline_mem) * 100
            perf_change_pct = ((row['performance_score'] - baseline_perf) / baseline_perf) * 100

            mem_changes.append(mem_change_pct)
            perf_changes.append(perf_change_pct)

    return {
        'technique': technique_name,
        'n_points': len(technique_data),
        'mem_mean': technique_data['memory_gb'].mean(),
        'mem_std': technique_data['memory_gb'].std(),
        'perf_mean': technique_data['performance_score'].mean(),
        'perf_std': technique_data['performance_score'].std(),
        'mem_change_mean': np.mean(mem_changes),
        'mem_change_std': np.std(mem_changes),
        'perf_change_mean': np.mean(perf_changes),
        'perf_change_std': np.std(perf_changes),
        'mem_changes': mem_changes,
        'perf_changes': perf_changes
    }

# Calculate for each technique
print("\n" + "="*80)
print("TECHNIQUE STATISTICS (relative to per-model baseline)")
print("="*80)

techniques = ['INT4', 'INT8', 'duoattn', 'snapkv', 'pyramidkv', 'streamingllm']

all_stats = {}
for tech in techniques:
    stats = calc_technique_stats(df, tech, baseline_mem_mean, baseline_perf_mean)
    if stats:
        all_stats[tech] = stats

        print(f"\n{tech.upper()}:")
        print(f"  Memory Change: {stats['mem_change_mean']:+.2f}% ± {stats['mem_change_std']:.2f}%")
        print(f"  Performance Change: {stats['perf_change_mean']:+.2f}% ± {stats['perf_change_std']:.2f}%")
        print(f"  Absolute Memory: {stats['mem_mean']:.2f} ± {stats['mem_std']:.2f} GB")
        print(f"  Absolute Performance: {stats['perf_mean']:.2f} ± {stats['perf_std']:.2f}")
        print(f"  N={stats['n_points']} models")

# Aggregate token eviction methods
print("\n" + "="*80)
print("TOKEN EVICTION METHODS AGGREGATED (SnapKV, PyramidKV, StreamingLLM)")
print("="*80)

kv_methods = ['snapkv', 'pyramidkv', 'streamingllm']
all_kv_mem_changes = []
all_kv_perf_changes = []
all_kv_mems = []
all_kv_perfs = []

for tech in kv_methods:
    if tech in all_stats:
        all_kv_mem_changes.extend(all_stats[tech]['mem_changes'])
        all_kv_perf_changes.extend(all_stats[tech]['perf_changes'])
        tech_data = df[df['technique'] == tech]
        all_kv_mems.extend(tech_data['memory_gb'].values)
        all_kv_perfs.extend(tech_data['performance_score'].values)

print(f"\nALL TOKEN EVICTION (n={len(all_kv_mem_changes)} configurations):")
print(f"  Memory Change: {np.mean(all_kv_mem_changes):+.2f}% ± {np.std(all_kv_mem_changes):.2f}%")
print(f"  Performance Change: {np.mean(all_kv_perf_changes):+.2f}% ± {np.std(all_kv_perf_changes):.2f}%")

# Split into small and large cache based on memory
sorted_indices = np.argsort(all_kv_mems)
half = len(sorted_indices) // 2

small_mem_changes = [all_kv_mem_changes[i] for i in sorted_indices[:half]]
small_perf_changes = [all_kv_perf_changes[i] for i in sorted_indices[:half]]
small_mems = [all_kv_mems[i] for i in sorted_indices[:half]]
small_perfs = [all_kv_perfs[i] for i in sorted_indices[:half]]

large_mem_changes = [all_kv_mem_changes[i] for i in sorted_indices[half:]]
large_perf_changes = [all_kv_perf_changes[i] for i in sorted_indices[half:]]
large_mems = [all_kv_mems[i] for i in sorted_indices[half:]]
large_perfs = [all_kv_perfs[i] for i in sorted_indices[half:]]

print(f"\nSMALL CACHE CONFIGS (n={len(small_mems)}):")
print(f"  Memory Change: {np.mean(small_mem_changes):+.2f}% ± {np.std(small_mem_changes):.2f}%")
print(f"  Performance Change: {np.mean(small_perf_changes):+.2f}% ± {np.std(small_perf_changes):.2f}%")
print(f"  Absolute Memory: {np.mean(small_mems):.2f} ± {np.std(small_mems):.2f} GB")
print(f"  Absolute Performance: {np.mean(small_perfs):.2f} ± {np.std(small_perfs):.2f}")

print(f"\nLARGE CACHE CONFIGS (n={len(large_mems)}):")
print(f"  Memory Change: {np.mean(large_mem_changes):+.2f}% ± {np.std(large_mem_changes):.2f}%")
print(f"  Performance Change: {np.mean(large_perf_changes):+.2f}% ± {np.std(large_perf_changes):.2f}%")
print(f"  Absolute Memory: {np.mean(large_mems):.2f} ± {np.std(large_mems):.2f} GB")
print(f"  Absolute Performance: {np.mean(large_perfs):.2f} ± {np.std(large_perfs):.2f}")

# Calculate improvement from small to large
avg_small_mem = np.mean(small_mems)
avg_large_mem = np.mean(large_mems)
avg_small_perf = np.mean(small_perfs)
avg_large_perf = np.mean(large_perfs)

mem_increase_small_to_large = ((avg_large_mem - avg_small_mem) / avg_small_mem) * 100
perf_increase_small_to_large = ((avg_large_perf - avg_small_perf) / avg_small_perf) * 100

print(f"\nSMALL → LARGE CACHE CHANGE:")
print(f"  Memory increase: +{mem_increase_small_to_large:.2f}%")
print(f"  Performance improvement: {perf_increase_small_to_large:+.2f}%")

# Generate the corrected paragraph
print("\n" + "="*80)
print("CORRECTED PARAGRAPH")
print("="*80)

paragraph = f"""
In \\Cref{{fig:main_figure_model_comparison}}, we compare the average performance and
peak memory usage of the baseline and seven different methods on four models, averaged
across all tasks from both HELMET (16K context) and LongProc (2K context) benchmarks.
NF4 and Int8 achieve substantial memory savings of {all_stats['INT4']['mem_change_mean']:.2f}% ± {all_stats['INT4']['mem_change_std']:.2f}%
and {all_stats['INT8']['mem_change_mean']:.2f}% ± {all_stats['INT8']['mem_change_std']:.2f}% respectively, with minimal performance
degradation of only {all_stats['INT4']['perf_change_mean']:.2f}% ± {all_stats['INT4']['perf_change_std']:.2f}% and {all_stats['INT8']['perf_change_mean']:.2f}% ± {all_stats['INT8']['perf_change_std']:.2f}%.
DuoAttn represents a notable exception among KV cache methods, achieving improved
performance ({all_stats['duoattn']['perf_change_mean']:+.2f}% ± {all_stats['duoattn']['perf_change_std']:.2f}%) with negligible memory
overhead ({all_stats['duoattn']['mem_change_mean']:+.2f}% ± {all_stats['duoattn']['mem_change_std']:.2f}%). In contrast, token eviction methods
(SnapKV, PyramidKV, StreamingLLM) struggle with Pareto optimality: smaller cache sizes
result in modest memory overhead ({np.mean(small_mem_changes):+.2f}% ± {np.std(small_mem_changes):.2f}%) but severe performance
degradation ({np.mean(small_perf_changes):+.2f}% ± {np.std(small_perf_changes):.2f}%), while larger cache sizes reduce performance
degradation to {np.mean(large_perf_changes):+.2f}% ± {np.std(large_perf_changes):.2f}% but increase memory consumption by
{np.mean(large_mem_changes):+.2f}% ± {np.std(large_mem_changes):.2f}%. Specifically, increasing cache size improves performance by
{perf_increase_small_to_large:+.2f}%, but requires {mem_increase_small_to_large:+.2f}% more memory. Thus, on average, token eviction
methods fail to achieve Pareto optimality: small caches sacrifice too much performance,
while large caches still underperform the baseline while consuming more memory.
"""

print(paragraph)

# Create a detailed breakdown table
print("\n" + "="*80)
print("DETAILED PER-MODEL BREAKDOWN")
print("="*80)

models = df['model'].unique()
for model in sorted(models):
    print(f"\n{model}:")
    model_baseline = df[(df['technique'] == 'baseline') & (df['model'] == model)].iloc[0]
    baseline_mem = model_baseline['memory_gb']
    baseline_perf = model_baseline['performance_score']

    print(f"  Baseline: {baseline_mem:.3f} GB, {baseline_perf:.3f} score")

    for tech in ['INT4', 'INT8', 'duoattn', 'snapkv', 'pyramidkv', 'streamingllm']:
        model_tech = df[(df['technique'] == tech) & (df['model'] == model)]
        if len(model_tech) > 0:
            row = model_tech.iloc[0]
            mem_change = ((row['memory_gb'] - baseline_mem) / baseline_mem) * 100
            perf_change = ((row['performance_score'] - baseline_perf) / baseline_perf) * 100
            print(f"  {tech:15s}: {row['memory_gb']:.3f} GB ({mem_change:+6.2f}%), {row['performance_score']:.3f} score ({perf_change:+6.2f}%)")

print("\n" + "="*80)
print("SUMMARY FOR PAPER")
print("="*80)
print("\nKey numbers to cite:")
print(f"  - NF4 memory savings: {all_stats['INT4']['mem_change_mean']:.2f}% ± {all_stats['INT4']['mem_change_std']:.2f}%")
print(f"  - NF4 performance change: {all_stats['INT4']['perf_change_mean']:.2f}% ± {all_stats['INT4']['perf_change_std']:.2f}%")
print(f"  - Int8 memory savings: {all_stats['INT8']['mem_change_mean']:.2f}% ± {all_stats['INT8']['mem_change_std']:.2f}%")
print(f"  - Int8 performance change: {all_stats['INT8']['perf_change_mean']:.2f}% ± {all_stats['INT8']['perf_change_std']:.2f}%")
print(f"  - DuoAttn memory change: {all_stats['duoattn']['mem_change_mean']:+.2f}% ± {all_stats['duoattn']['mem_change_std']:.2f}%")
print(f"  - DuoAttn performance change: {all_stats['duoattn']['perf_change_mean']:+.2f}% ± {all_stats['duoattn']['perf_change_std']:.2f}%")
print(f"  - Small KV cache overhead: {np.mean(small_mem_changes):+.2f}% ± {np.std(small_mem_changes):.2f}%")
print(f"  - Small KV cache perf degradation: {np.mean(small_perf_changes):+.2f}% ± {np.std(small_perf_changes):.2f}%")
print(f"  - Large KV cache overhead: {np.mean(large_mem_changes):+.2f}% ± {np.std(large_mem_changes):.2f}%")
print(f"  - Large KV cache perf degradation: {np.mean(large_perf_changes):+.2f}% ± {np.std(large_perf_changes):.2f}%")
print(f"  - Small→Large: {perf_increase_small_to_large:+.2f}% perf for {mem_increase_small_to_large:+.2f}% mem")
