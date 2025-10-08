import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
helmet_perf = pd.read_csv('/Users/qw281/Downloads/helmet_results/helmet_performance.csv')
longproc_perf = pd.read_csv('/Users/qw281/Downloads/helmet_results/longproc_performance.csv')
helmet_mem = pd.read_csv('/Users/qw281/Downloads/helmet_results/helmet_memory_usage.csv')
longproc_mem = pd.read_csv('/Users/qw281/Downloads/helmet_results/longproc_memory_usage.csv')

# Filter for baseline techniques only
helmet_perf_baseline = helmet_perf[helmet_perf['technique'] == 'baseline']
longproc_perf_baseline = longproc_perf[longproc_perf['technique'] == 'baseline']
helmet_mem_baseline = helmet_mem[helmet_mem['technique'] == 'baseline']
longproc_mem_baseline = longproc_mem[longproc_mem['technique'] == 'baseline']

# Define model pairs (reasoning vs regular)
model_pairs = [
    ('DeepSeek-R1-Distill-Llama-8B', 'Llama-3.1-8B-Instruct', 'Llama-8B'),
    ('DeepSeek-R1-Distill-Qwen-7B', 'Qwen2.5-7B-Instruct', 'Qwen-7B')
]

# Helmet performance tasks
helmet_perf_tasks = ['niah', 'rag_hotpotqa', 'rag_nq', 'cite_str_em', 'cite_citation_rec',
                     'cite_citation_prec', 'recall_jsonkv', 'rerank', 'icl_clinic', 'icl_banking']

# Longproc performance tasks
longproc_perf_tasks = ['travel_planning', 'html_to_tsv', 'pseudo_to_code']

# Helmet memory tasks
helmet_mem_tasks = ['niah', 'rag_hotpotqa', 'rag_nq', 'cite', 'recall_jsonkv', 'rerank', 'icl_clinic', 'icl_banking']

# Longproc memory tasks
longproc_mem_tasks = ['travel_planning', 'html_to_tsv', 'pseudo_to_code']

# Create 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle('Reasoning vs Instruct Models: Performance and Memory Comparison',
             fontsize=18, fontweight='bold')

# Top row: Performance differences
for idx, (reasoning_model, regular_model, label) in enumerate(model_pairs):
    ax = axes[0, idx]

    # Helmet tasks
    reasoning_helmet = helmet_perf_baseline[(helmet_perf_baseline['model'] == reasoning_model) &
                                            (helmet_perf_baseline['context_length'] == '16k')]
    regular_helmet = helmet_perf_baseline[(helmet_perf_baseline['model'] == regular_model) &
                                          (helmet_perf_baseline['context_length'] == '16k')]

    # Longproc tasks
    reasoning_longproc = longproc_perf_baseline[(longproc_perf_baseline['model'] == reasoning_model) &
                                                (longproc_perf_baseline['context_length'] == '2k')]
    regular_longproc = longproc_perf_baseline[(longproc_perf_baseline['model'] == regular_model) &
                                              (longproc_perf_baseline['context_length'] == '2k')]

    all_tasks = helmet_perf_tasks + longproc_perf_tasks
    diffs = []

    for task in helmet_perf_tasks:
        if len(reasoning_helmet) > 0 and len(regular_helmet) > 0:
            reg_val = regular_helmet[task].values[0]
            reas_val = reasoning_helmet[task].values[0]
            if pd.notna(reg_val) and pd.notna(reas_val):
                diffs.append(reas_val - reg_val)
            else:
                diffs.append(0)
        else:
            diffs.append(0)

    for task in longproc_perf_tasks:
        if len(reasoning_longproc) > 0 and len(regular_longproc) > 0:
            reg_val = regular_longproc[task].values[0]
            reas_val = reasoning_longproc[task].values[0]
            if pd.notna(reg_val) and pd.notna(reas_val):
                diffs.append(reas_val - reg_val)
            else:
                diffs.append(0)
        else:
            diffs.append(0)

    x = np.arange(len(all_tasks))
    colors = ['#EF553B' if d < 0 else '#00CC96' for d in diffs]
    bars = ax.bar(x, diffs, color=colors, alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Tasks', fontweight='bold', fontsize=11)
    ax.set_ylabel('Performance Difference', fontweight='bold', fontsize=11)
    ax.set_title(f'{label} Performance\n(Green=Better)', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(all_tasks, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

# Bottom row: Memory savings (negative of difference, so green is up)
for idx, (reasoning_model, regular_model, label) in enumerate(model_pairs):
    ax = axes[1, idx]

    # Helmet tasks
    reasoning_helmet = helmet_mem_baseline[(helmet_mem_baseline['model'] == reasoning_model) &
                                           (helmet_mem_baseline['context_length'] == '16k')]
    regular_helmet = helmet_mem_baseline[(helmet_mem_baseline['model'] == regular_model) &
                                         (helmet_mem_baseline['context_length'] == '16k')]

    # Longproc tasks
    reasoning_longproc = longproc_mem_baseline[(longproc_mem_baseline['model'] == reasoning_model) &
                                               (longproc_mem_baseline['context_length'] == '2k')]
    regular_longproc = longproc_mem_baseline[(longproc_mem_baseline['model'] == regular_model) &
                                             (longproc_mem_baseline['context_length'] == '2k')]

    all_tasks = helmet_mem_tasks + longproc_mem_tasks
    savings = []  # Negative of difference (regular - reasoning), so positive = savings

    for task in helmet_mem_tasks:
        if len(reasoning_helmet) > 0 and len(regular_helmet) > 0:
            reg_val = regular_helmet[task].values[0]
            reas_val = reasoning_helmet[task].values[0]
            if pd.notna(reg_val) and pd.notna(reas_val):
                savings.append(reg_val - reas_val)  # Regular - Reasoning
            else:
                savings.append(0)
        else:
            savings.append(0)

    for task in longproc_mem_tasks:
        if len(reasoning_longproc) > 0 and len(regular_longproc) > 0:
            reg_val = regular_longproc[task].values[0]
            reas_val = reasoning_longproc[task].values[0]
            if pd.notna(reg_val) and pd.notna(reas_val):
                savings.append(reg_val - reas_val)  # Regular - Reasoning
            else:
                savings.append(0)
        else:
            savings.append(0)

    x = np.arange(len(all_tasks))
    colors = ['#00CC96' if s > 0 else '#EF553B' for s in savings]
    bars = ax.bar(x, savings, color=colors, alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Tasks', fontweight='bold', fontsize=11)
    ax.set_ylabel('Memory Savings (GB)', fontweight='bold', fontsize=11)
    ax.set_title(f'{label} Memory\n(Green=Savings)', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(all_tasks, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/qw281/Downloads/helmet_results/reasoning_vs_regular_difference.pdf', dpi=300, bbox_inches='tight')
print("2x2 plot saved to: /Users/qw281/Downloads/helmet_results/reasoning_vs_regular_difference.pdf")
plt.close()
