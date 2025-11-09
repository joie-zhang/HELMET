import pandas as pd
import numpy as np

# Load the averaged data (which averages across both small and large cache configs)
df_averaged = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/plot_data_averaged_across_cache_sizes.csv')

# Load the full data to see individual cache sizes
df_full = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/plot_data_all_cache_sizes.csv')

print("="*80)
print("RECALCULATING STATISTICS WITH AVERAGED CACHE CONFIGURATIONS")
print("="*80)
print("\nFor SnapKV/PyramidKV: Averaging across BOTH w256 (small) and w2048 (large) configs")
print("="*80)

# Get baseline statistics
baseline_data = df_averaged[df_averaged['technique'] == 'baseline']
baseline_mem_mean = baseline_data['avg_memory_gb'].mean()
baseline_perf_mean = baseline_data['avg_performance_score'].mean()

print(f"\nBASELINE (averaged across all 4 models):")
print(f"  Average Memory: {baseline_mem_mean:.3f} GB")
print(f"  Average Performance: {baseline_perf_mean:.3f}")

# Function to calculate statistics for a technique
def calc_technique_stats(df_averaged, technique_name):
    technique_data = df_averaged[df_averaged['technique'] == technique_name]

    if len(technique_data) == 0:
        return None

    # Calculate changes relative to baseline for each model
    mem_changes = []
    perf_changes = []

    for _, row in technique_data.iterrows():
        # Find the corresponding baseline for this model
        model_baseline = df_averaged[(df_averaged['technique'] == 'baseline') & (df_averaged['model'] == row['model'])]
        if len(model_baseline) > 0:
            baseline_mem = model_baseline.iloc[0]['avg_memory_gb']
            baseline_perf = model_baseline.iloc[0]['avg_performance_score']

            mem_change_pct = ((row['avg_memory_gb'] - baseline_mem) / baseline_mem) * 100
            perf_change_pct = ((row['avg_performance_score'] - baseline_perf) / baseline_perf) * 100

            mem_changes.append(mem_change_pct)
            perf_changes.append(perf_change_pct)

    return {
        'technique': technique_name,
        'n_points': len(technique_data),
        'mem_mean': technique_data['avg_memory_gb'].mean(),
        'mem_std': technique_data['avg_memory_gb'].std(),
        'perf_mean': technique_data['avg_performance_score'].mean(),
        'perf_std': technique_data['avg_performance_score'].std(),
        'mem_change_mean': np.mean(mem_changes),
        'mem_change_std': np.std(mem_changes),
        'perf_change_mean': np.mean(perf_changes),
        'perf_change_std': np.std(perf_changes),
        'mem_changes': mem_changes,
        'perf_changes': perf_changes
    }

# Calculate for each technique
print("\n" + "="*80)
print("TECHNIQUE STATISTICS (with cache sizes averaged per model)")
print("="*80)

techniques = ['INT4', 'INT8', 'duoattn', 'snapkv', 'pyramidkv', 'streamingllm']

all_stats = {}
for tech in techniques:
    stats = calc_technique_stats(df_averaged, tech)
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
print("Each SnapKV/PyramidKV point is averaged across both cache sizes")
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
        tech_data = df_averaged[df_averaged['technique'] == tech]
        all_kv_mems.extend(tech_data['avg_memory_gb'].values)
        all_kv_perfs.extend(tech_data['avg_performance_score'].values)

print(f"\nALL TOKEN EVICTION (n={len(all_kv_mem_changes)} models × techniques):")
print(f"  Memory Change: {np.mean(all_kv_mem_changes):+.2f}% ± {np.std(all_kv_mem_changes):.2f}%")
print(f"  Performance Change: {np.mean(all_kv_perf_changes):+.2f}% ± {np.std(all_kv_perf_changes):.2f}%")

# Now analyze the INDIVIDUAL small and large cache sizes (not averaged)
print("\n" + "="*80)
print("INDIVIDUAL CACHE SIZE ANALYSIS (Small vs Large)")
print("="*80)

df_kv_only = df_full[df_full['technique'].isin(['snapkv', 'pyramidkv'])]

small_cache_data = []
large_cache_data = []

for _, row in df_kv_only.iterrows():
    # Get baseline for this model
    baseline = df_averaged[(df_averaged['technique'] == 'baseline') & (df_averaged['model'] == row['model'])].iloc[0]
    baseline_mem = baseline['avg_memory_gb']
    baseline_perf = baseline['avg_performance_score']

    mem_change = ((row['memory_gb'] - baseline_mem) / baseline_mem) * 100
    perf_change = ((row['performance_score'] - baseline_perf) / baseline_perf) * 100

    if row['cache_type'] == 'small':
        small_cache_data.append({
            'mem': row['memory_gb'],
            'perf': row['performance_score'],
            'mem_change': mem_change,
            'perf_change': perf_change
        })
    elif row['cache_type'] == 'large':
        large_cache_data.append({
            'mem': row['memory_gb'],
            'perf': row['performance_score'],
            'mem_change': mem_change,
            'perf_change': perf_change
        })

print(f"\nSMALL CACHE CONFIGS (w256_c2048) - n={len(small_cache_data)}:")
small_mem_changes = [d['mem_change'] for d in small_cache_data]
small_perf_changes = [d['perf_change'] for d in small_cache_data]
small_mems = [d['mem'] for d in small_cache_data]
small_perfs = [d['perf'] for d in small_cache_data]

print(f"  Memory Change: {np.mean(small_mem_changes):+.2f}% ± {np.std(small_mem_changes):.2f}%")
print(f"  Performance Change: {np.mean(small_perf_changes):+.2f}% ± {np.std(small_perf_changes):.2f}%")
print(f"  Absolute Memory: {np.mean(small_mems):.2f} ± {np.std(small_mems):.2f} GB")
print(f"  Absolute Performance: {np.mean(small_perfs):.2f} ± {np.std(small_perfs):.2f}")

print(f"\nLARGE CACHE CONFIGS (w2048_c8192) - n={len(large_cache_data)}:")
large_mem_changes = [d['mem_change'] for d in large_cache_data]
large_perf_changes = [d['perf_change'] for d in large_cache_data]
large_mems = [d['mem'] for d in large_cache_data]
large_perfs = [d['perf'] for d in large_cache_data]

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
print("CORRECTED PARAGRAPH (using averaged cache sizes)")
print("="*80)

paragraph = f"""
In \\Cref{{fig:main_figure_model_comparison}}, we compare the average performance and
peak memory usage of the baseline and seven different methods on four models, averaged
across all tasks from both HELMET (16K context) and LongProc (2K context) benchmarks.
NF4 and Int8 achieve substantial memory savings of {all_stats['INT4']['mem_change_mean']:.2f}% ± {all_stats['INT4']['mem_change_std']:.2f}%
and {all_stats['INT8']['mem_change_mean']:.2f}% ± {all_stats['INT8']['mem_change_std']:.2f}% respectively, with minimal performance
degradation of only {all_stats['INT4']['perf_change_mean']:.2f}% ± {all_stats['INT4']['perf_change_std']:.2f}% and {all_stats['INT8']['perf_change_mean']:.2f}% ± {all_stats['INT8']['perf_change_std']:.2f}%.
DuoAttn achieves near-baseline performance ({all_stats['duoattn']['perf_change_mean']:+.2f}% ± {all_stats['duoattn']['perf_change_std']:.2f}%)
with minimal memory savings ({all_stats['duoattn']['mem_change_mean']:.2f}% ± {all_stats['duoattn']['mem_change_std']:.2f}%).
In contrast, token eviction methods (SnapKV, PyramidKV, StreamingLLM), averaged across
multiple cache configurations, show substantial memory overhead
({np.mean(all_kv_mem_changes):+.2f}% ± {np.std(all_kv_mem_changes):.2f}%) with severe performance degradation
({np.mean(all_kv_perf_changes):+.2f}% ± {np.std(all_kv_perf_changes):.2f}%).
When examining individual cache sizes for SnapKV and PyramidKV, we observe that
smaller caches (w256_c2048) result in modest memory overhead ({np.mean(small_mem_changes):+.2f}% ± {np.std(small_mem_changes):.2f}%)
but severe performance degradation ({np.mean(small_perf_changes):+.2f}% ± {np.std(small_perf_changes):.2f}%),
while larger caches (w2048_c8192) substantially increase memory consumption
({np.mean(large_mem_changes):+.2f}% ± {np.std(large_mem_changes):.2f}%) and improve performance to
{np.mean(large_perf_changes):+.2f}% ± {np.std(large_perf_changes):.2f}%.
Specifically, increasing cache size from w256 to w2048 improves performance by
{perf_increase_small_to_large:+.2f}%, but requires {mem_increase_small_to_large:+.2f}% more memory. Thus, token eviction
methods fail to achieve Pareto optimality: small caches sacrifice too much performance,
while large caches consume substantially more memory than baseline while still
underperforming it.
"""

print(paragraph)

# Detailed breakdown
print("\n" + "="*80)
print("DETAILED PER-MODEL BREAKDOWN (with averaged cache sizes)")
print("="*80)

models = df_averaged['model'].unique()
for model in sorted(models):
    print(f"\n{model}:")
    model_baseline = df_averaged[(df_averaged['technique'] == 'baseline') & (df_averaged['model'] == model)].iloc[0]
    baseline_mem = model_baseline['avg_memory_gb']
    baseline_perf = model_baseline['avg_performance_score']

    print(f"  Baseline: {baseline_mem:.3f} GB, {baseline_perf:.3f} score")

    for tech in ['INT4', 'INT8', 'duoattn', 'snapkv', 'pyramidkv', 'streamingllm']:
        model_tech = df_averaged[(df_averaged['technique'] == tech) & (df_averaged['model'] == model)]
        if len(model_tech) > 0:
            row = model_tech.iloc[0]
            mem_change = ((row['avg_memory_gb'] - baseline_mem) / baseline_mem) * 100
            perf_change = ((row['avg_performance_score'] - baseline_perf) / baseline_perf) * 100
            num_configs = row['num_cache_configs']
            config_label = f"(avg of {num_configs})" if num_configs > 1 else ""
            print(f"  {tech:15s}: {row['avg_memory_gb']:6.2f} GB ({mem_change:+6.2f}%), {row['avg_performance_score']:6.2f} score ({perf_change:+6.2f}%) {config_label}")

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
print(f"  - Token eviction (all, averaged): {np.mean(all_kv_mem_changes):+.2f}% mem, {np.mean(all_kv_perf_changes):+.2f}% perf")
print(f"  - Small KV cache (w256): {np.mean(small_mem_changes):+.2f}% mem, {np.mean(small_perf_changes):+.2f}% perf")
print(f"  - Large KV cache (w2048): {np.mean(large_mem_changes):+.2f}% mem, {np.mean(large_perf_changes):+.2f}% perf")
print(f"  - Small→Large: {perf_increase_small_to_large:+.2f}% perf for {mem_increase_small_to_large:+.2f}% mem")
