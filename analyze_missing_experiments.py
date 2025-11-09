#!/usr/bin/env python3
"""
Analyze missing experiments from HELMET and LongProc results
"""
import csv
from typing import Dict, Set, Tuple, List
from collections import defaultdict

# Define expected configurations
TECHNIQUES = {
    "baseline": ["baseline", "INT4", "INT8"],
    "snapkv": ["snapkv"],
    "pyramidkv": ["pyramidkv"],
    "streamingllm": ["streamingllm"],
    "duoattn": ["duoattn"],  # Only for Llama models
    "minference": ["minference"],  # Only for instruct models
}

MODELS = {
    "R1 Distill Llama": "DeepSeek-R1-Distill-Llama-8B",
    "R1 Distill Qwen": "DeepSeek-R1-Distill-Qwen-7B",
    "Llama instruct": "Llama-3.1-8B-Instruct",
    "Qwen instruct": "Qwen2.5-7B-Instruct",
    "Qwen3-8B (thinking/default)": "Qwen3-8B",
    "Qwen3-8B-thinking (separate)": "Qwen3-8B-thinking",
    "Qwen3-8B-nothinking": "Qwen3-8B-nothinking",
    "Yarn-Qwen3-8B (thinking/default)": "Yarn-Qwen3-8B",
    "Yarn-Qwen3-8B-thinking (separate)": "Yarn-Qwen3-8B-thinking",
    "Yarn-Qwen3-8B-nothinking": "Yarn-Qwen3-8B-nothinking",
}

# Context lengths for each task type
HELMET_CONTEXTS = ["16k", "32k"]
LONGPROC_CONTEXTS = ["5k", "2k", "8k"]  # 0.5k maps to 5k

# Task columns for each benchmark
HELMET_TASKS = ["niah", "rag_hotpotqa", "rag_nq", "cite_str_em", "cite_citation_rec", "cite_citation_prec",
                "recall_jsonkv", "rerank", "summ_multilex", "icl_clinic", "icl_banking"]
LONGPROC_TASKS = ["travel_planning", "html_to_tsv", "pseudo_to_code"]

def load_csv(filepath):
    """Load CSV file into list of dicts"""
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def load_data():
    """Load all CSV files"""
    helmet_perf = load_csv("/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_performance.csv")
    helmet_mem = load_csv("/scratch/gpfs/DANQIC/jz4391/HELMET/results/helmet_results/helmet_memory_usage.csv")

    longproc_perf = load_csv("/scratch/gpfs/DANQIC/jz4391/HELMET/results/longproc_results/longproc_performance.csv")
    longproc_mem = load_csv("/scratch/gpfs/DANQIC/jz4391/HELMET/results/longproc_results/longproc_memory_usage.csv")

    return helmet_perf, helmet_mem, longproc_perf, longproc_mem

def analyze_missing_combinations(data: List[Dict], context_list: List[str], task_columns: List[str], benchmark_name: str):
    """Analyze what combinations are missing"""
    print(f"\n{'='*80}")
    print(f"Analyzing {benchmark_name}")
    print(f"{'='*80}\n")

    # Get unique values
    techniques_in_data = set(row['technique'] for row in data)
    models_in_data = set(row['model'] for row in data)
    contexts_in_data = set(row['context_length'] for row in data)

    print(f"Techniques found: {sorted(techniques_in_data)}")
    print(f"Models found: {sorted(models_in_data)}")
    print(f"Contexts found: {sorted(contexts_in_data)}")
    print()

    # Expected baseline experiments (baseline, INT4, INT8)
    print("="*80)
    print("MISSING BASELINE/INT4/INT8 EXPERIMENTS")
    print("="*80)

    for context in context_list:
        if context not in contexts_in_data:
            print(f"\nWARNING: Context {context} not found in data at all!")
            continue

        print(f"\n--- Context: {context} ---")

        for model_name in MODELS.values():
            if model_name not in models_in_data:
                print(f"  Model '{model_name}' - NOT FOUND IN DATA")
                continue

            # Check baseline, INT4, INT8
            for tech in ["baseline", "INT4", "INT8"]:
                rows = [r for r in data if r['technique'] == tech and
                        r['model'] == model_name and
                        r['context_length'] == context]

                if not rows:
                    print(f"  Model '{model_name}' - {tech:10s} - MISSING COMPLETELY")
                else:
                    # Check which tasks have data
                    row = rows[0]
                    missing_tasks = []
                    for task in task_columns:
                        if task not in row or not row[task] or row[task] == '' or float(row[task]) == 0:
                            missing_tasks.append(task)

                    if missing_tasks:
                        print(f"  Model '{model_name}' - {tech:10s} - Missing tasks: {', '.join(missing_tasks)}")

    # Check MInference (only for instruct models)
    print("\n" + "="*80)
    print("MISSING MINFERENCE EXPERIMENTS (Instruct models only)")
    print("="*80)

    instruct_models = [m for name, m in MODELS.items() if "instruct" in name.lower()]

    for context in context_list:
        if context not in contexts_in_data:
            continue

        print(f"\n--- Context: {context} ---")

        for model in instruct_models:
            if model not in models_in_data:
                continue

            rows = [r for r in data if r['technique'] == "minference" and
                    r['model'] == model and
                    r['context_length'] == context]

            if not rows:
                print(f"  Model '{model}' - minference - MISSING COMPLETELY")
            else:
                row = rows[0]
                missing_tasks = []
                for task in task_columns:
                    if task not in row or not row[task] or row[task] == '' or float(row[task]) == 0:
                        missing_tasks.append(task)

                if missing_tasks:
                    print(f"  Model '{model}' - minference - Missing tasks: {', '.join(missing_tasks)}")

    # Check DuoAttention (only for Llama models)
    print("\n" + "="*80)
    print("MISSING DUOATTENTION EXPERIMENTS (Llama models only)")
    print("="*80)

    llama_models = [m for name, m in MODELS.items() if "llama" in name.lower()]

    for context in context_list:
        if context not in contexts_in_data:
            continue

        print(f"\n--- Context: {context} ---")

        for model in llama_models:
            if model not in models_in_data:
                continue

            rows = [r for r in data if r['technique'] == "duoattn" and
                    r['model'] == model and
                    r['context_length'] == context]

            if not rows:
                print(f"  Model '{model}' - duoattn - MISSING COMPLETELY")
            else:
                row = rows[0]
                missing_tasks = []
                for task in task_columns:
                    if task not in row or not row[task] or row[task] == '' or float(row[task]) == 0:
                        missing_tasks.append(task)

                if missing_tasks:
                    print(f"  Model '{model}' - duoattn - Missing tasks: {', '.join(missing_tasks)}")

    # Check StreamingLLM
    print("\n" + "="*80)
    print("STREAMINGLLM EXPERIMENTS")
    print("="*80)

    streamingllm_rows = [r for r in data if r['technique'] == 'streamingllm']
    print(f"\nTotal StreamingLLM rows: {len(streamingllm_rows)}")

    # Check for 4092 vs 4096 inconsistency
    print("\nChecking 4092 vs 4096 inconsistency...")
    cache_sizes = set(r['cache_size'] for r in streamingllm_rows)
    for cs in sorted(cache_sizes):
        if 'n_local_4092' in cs or 'n_local_4096' in cs:
            count = len([r for r in streamingllm_rows if r['cache_size'] == cs])
            print(f"  {cs}: {count} experiments")

    # Check for missing models/contexts
    for context in context_list:
        if context not in contexts_in_data:
            continue

        print(f"\n--- Context: {context} ---")

        for model_name in MODELS.values():
            if model_name not in models_in_data:
                continue

            rows = [r for r in data if r['technique'] == "streamingllm" and
                    r['model'] == model_name and
                    r['context_length'] == context]

            if not rows:
                print(f"  Model '{model_name}' - streamingllm - MISSING COMPLETELY")
            else:
                # Show what cache sizes exist
                cache_sizes_for_model = set(r['cache_size'] for r in rows)
                print(f"  Model '{model_name}' - streamingllm - Cache sizes: {sorted(cache_sizes_for_model)}")

                # Check for missing tasks
                for row in rows:
                    missing_tasks = []
                    for task in task_columns:
                        if task not in row or not row[task] or row[task] == '' or float(row[task]) == 0:
                            missing_tasks.append(task)

                    if missing_tasks:
                        print(f"    Cache size '{row['cache_size']}' - Missing tasks: {', '.join(missing_tasks)}")

def main():
    helmet_perf, helmet_mem, longproc_perf, longproc_mem = load_data()

    # Analyze HELMET
    analyze_missing_combinations(helmet_perf, HELMET_CONTEXTS, HELMET_TASKS, "HELMET")

    # Analyze LongProc
    analyze_missing_combinations(longproc_perf, LONGPROC_CONTEXTS, LONGPROC_TASKS, "LongProc")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
