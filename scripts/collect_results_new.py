import os
import json
import pandas as pd
from collections import defaultdict
from typing import Dict, Tuple, List
from tqdm import tqdm

# --- MERGED CONSTANTS ---
TASK_MAP = {
    "alce_asqa": "cite",
    "json_kv": "recall_jsonkv", 
    "kilt_hotpotqa": "rag_hotpotqa",
    "kilt_nq": "rag_nq",
    "msmarco_rerank": "rerank",
    "ruler_niah_s_2": "niah",
    "icl_banking77": "banking77",
    "icl_clinic150": "clinic150",
    "multi_lexsum": "multi_lexsum"
}

TASKS_BY_CONTEXT = {
    "2k": ["html_to_tsv", "pseudo_to_code", "travel_planning"],
    "5k": ["html_to_tsv", "pseudo_to_code"],
    "8k": ["html_to_tsv", "travel_planning"],
    "16k": ["recall_jsonkv", "rag_nq", "rag_hotpotqa", "rerank", "cite", "niah", "banking77", "clinic150", "multi_lexsum"],
    "32k": ["recall_jsonkv", "rag_nq", "rag_hotpotqa", "rerank", "cite", "niah", "banking77", "clinic150", "multi_lexsum"]
}

SCORE_KEYS = {
    "html_to_tsv": ["f1"],
    "pseudo_to_code": ["accuracy"],
    "travel_planning": ["accuracy"], 
    "cite": ["str_em", "citation_rec", "citation_prec"],
    "recall_jsonkv": ["substring_exact_match"],
    "rag_nq": ["substring_exact_match"],
    "rag_hotpotqa": ["substring_exact_match"],
    "rerank": ["NDCG@10"],
    "niah": ["ruler_recall"],
    "banking77": ["exact_match"],
    "clinic150": ["exact_match"],
    "multi_lexsum": ["gpt-4-f1"]
}

# --- HELPER FUNCTIONS ---

def get_task_from_filename(filename: str) -> str:
    """Maps a file prefix to its corresponding task name."""
    for prefix, task in TASK_MAP.items():
        if filename.startswith(prefix):
            return task
    for task in SCORE_KEYS:
        if filename.startswith(task):
            return task
    return None

def parse_cache_params(cache_dir: str, technique: str) -> str:
    """Parses cache parameter strings from directory names for specific techniques."""
    if technique == "streamingllm":
        try:
            # Format: local3968_init128
            parts = cache_dir.split('_')
            n_local = parts[0].replace('local', '')
            n_init = parts[1].replace('init', '')
            return f"n_local_{n_local}_n_init_{n_init}"
        except (ValueError, IndexError):
            print(f"Warning: Could not parse streamingllm cache directory name: {cache_dir}\n")
            return None
    elif technique in ["snapkv", "pyramidkv"]:
        try:
            # Format: w32_c4096_k5_avgpool
            parts = cache_dir.split('_')
            w_size = parts[0].replace('w', '')
            c_size = parts[1].replace('c', '')
            k_size = parts[2].replace('k', '')
            pool = parts[3]
            return f"w{w_size}_c{c_size}_k{k_size}_{pool}"
        except (ValueError, IndexError):
            print(f"Warning: Could not parse {technique} cache directory name: {cache_dir}")
            return None
    return None

def should_skip_dir(dirname: str) -> bool:
    """Checks if a directory should be skipped."""
    return dirname.startswith('gen_max') or dirname.startswith('archive')

# --- DATA INITIALIZATION ---
longproc_memory_data = defaultdict(lambda: defaultdict(float))
longproc_throughput_data = defaultdict(lambda: defaultdict(float))
longproc_performance_data = defaultdict(lambda: defaultdict(float))

helmet_memory_data = defaultdict(lambda: defaultdict(float))
helmet_throughput_data = defaultdict(lambda: defaultdict(float))
helmet_performance_data = defaultdict(lambda: defaultdict(float))

# Base directory
base_dir = "/scratch/gpfs/DANQIC/jz4391/HELMET/output"

# --- DIRECTORY TRAVERSAL AND DATA PARSING ---
for technique in tqdm(os.listdir(base_dir), desc="Processing techniques"):
    technique_path = os.path.join(base_dir, technique)
    if not os.path.isdir(technique_path) or technique == "quest":
        continue

    for context_length in tqdm(os.listdir(technique_path), desc=f"Processing {technique} contexts", leave=False):
        context_path = os.path.join(technique_path, context_length)
        if not os.path.isdir(context_path):
            continue

        for model in tqdm(os.listdir(context_path), desc=f"Processing {context_length} models", leave=False):
            model_path = os.path.join(context_path, model)
            if not os.path.isdir(model_path):
                continue

            subdirs = []
            
            # Handle directory structures based on technique
            if technique == "baseline":
                for q in os.listdir(model_path):
                    if should_skip_dir(q): continue
                    if q.startswith("4bit"):
                        row_key = ("INT4", context_length, model, "default")
                        subdirs.append((os.path.join(model_path, q), row_key))
                    elif q.startswith("8bit"):
                        row_key = ("INT8", context_length, model, "default")
                        subdirs.append((os.path.join(model_path, q), row_key))
                    elif q.startswith("16bit"):
                        row_key = (technique, context_length, model, "default")
                        subdirs.append((os.path.join(model_path, q), row_key))
            elif technique == "duoattn":
                for subdir in os.listdir(model_path):
                    if should_skip_dir(subdir) or not subdir.startswith("_sp"): continue
                    try:
                        parts = subdir.split('_')
                        sparsity = parts[1].replace('sp', '')
                        prefill = parts[2].replace('pf', '')
                        cache_params = f"sp{sparsity}_pf{prefill}"
                        row_key = (technique, context_length, model, cache_params)
                        subdirs.append((os.path.join(model_path, subdir), row_key))
                    except (ValueError, IndexError):
                        print(f"Warning: Could not parse duoattn directory name: {subdir}")
            elif technique in ["streamingllm", "snapkv", "pyramidkv"]:
                for cache_dir in os.listdir(model_path):
                    if should_skip_dir(cache_dir): continue
                    cache_path = os.path.join(model_path, cache_dir)
                    if os.path.isdir(cache_path):
                        cache_params = parse_cache_params(cache_dir, technique)
                        if cache_params:
                            row_key = (technique, context_length, model, cache_params)
                            subdirs.append((cache_path, row_key))
            else:
                row_key = (technique, context_length, model, "default")
                subdirs.append((model_path, row_key))

            # Process files in each identified subdirectory
            for subdir, row_key in subdirs:
                if not os.path.isdir(subdir): continue

                tasks = TASKS_BY_CONTEXT.get(context_length, [])
                is_longproc = context_length in ["2k", "5k", "8k"]
                
                # Assign data to the correct dictionary (LongProc or HELMET)
                memory_data = longproc_memory_data if is_longproc else helmet_memory_data
                throughput_data = longproc_throughput_data if is_longproc else helmet_throughput_data
                performance_data = longproc_performance_data if is_longproc else helmet_performance_data

                for file in os.listdir(subdir):
                    filepath = os.path.join(subdir, file)
                    if not os.path.isfile(filepath): continue

                    task = get_task_from_filename(file)
                    if task not in tasks: continue

                    # Handle GPT-4 evaluation files (e.g., for multi_lexsum)
                    if file.endswith("-gpt4eval_o.json"):
                        with open(filepath) as f:
                            score_data = json.load(f)
                            if "averaged_metrics" in score_data:
                                for key in SCORE_KEYS.get(task, []):
                                    value = score_data['averaged_metrics'].get(key)
                                    if value is not None:
                                        perf_key = f"{task}_{key}" if len(SCORE_KEYS[task]) > 1 else task
                                        performance_data[row_key][perf_key] = float(value)
                            else:
                                print(f"Warning: No averaged_metrics found in {filepath}")
                    
                    # Handle standard score files
                    elif file.endswith(".json.score"):
                        with open(filepath) as f:
                            score_data = json.load(f)
                            for key in SCORE_KEYS.get(task, []):
                                value = score_data.get(key)
                                if value is not None:
                                    perf_key = f"{task}_{key}" if len(SCORE_KEYS[task]) > 1 else task
                                    performance_data[row_key][perf_key] = float(value)
                                else:
                                    print(f"Warning: No value found for key {key} in {filepath}")
                    
                    # Handle main result files for memory and throughput
                    elif file.endswith(".json"):
                        with open(filepath) as f:
                            data = json.load(f)
                            if "memory_usage" in data:
                                memory_data[row_key][task] = float(data["memory_usage"]) / 1e9 # Convert to GB
                            if "throughput" in data and "averaged_metrics" in data and "output_len" in data["averaged_metrics"]:
                                # Calculate throughput in tokens/second
                                samples_per_second = float(data["throughput"])
                                avg_tokens_per_sample = float(data["averaged_metrics"]["output_len"])
                                throughput_data[row_key][task] = samples_per_second * avg_tokens_per_sample

# --- DATAFRAME CREATION ---
def create_dataframe(data_dict: Dict[Tuple[str, str, str, str], Dict[str, float]]) -> pd.DataFrame:
    """Creates a DataFrame from the collected data dictionary."""
    records = []
    for (technique, context_length, model, cache_size), task_data in data_dict.items():
        record = {
            "technique": technique,
            "context_length": context_length,
            "model": model,
            "cache_size": cache_size
        }
        record.update(task_data)
        records.append(record)
    return pd.DataFrame(records)

# Create LongProc DataFrames
longproc_memory_df = create_dataframe(longproc_memory_data)
longproc_throughput_df = create_dataframe(longproc_throughput_data)
longproc_performance_df = create_dataframe(longproc_performance_data)

# Create HELMET DataFrames
helmet_memory_df = create_dataframe(helmet_memory_data)
helmet_throughput_df = create_dataframe(helmet_throughput_data)
helmet_performance_df = create_dataframe(helmet_performance_data)

# --- SAVE RESULTS TO CSV ---
base_output_dir = "/scratch/gpfs/DANQIC/jz4391/HELMET/results"
longproc_dir = os.path.join(base_output_dir, "longproc_results")
helmet_dir = os.path.join(base_output_dir, "helmet_results")
os.makedirs(longproc_dir, exist_ok=True)
os.makedirs(helmet_dir, exist_ok=True)

# Save LongProc DataFrames
longproc_memory_df.to_csv(os.path.join(longproc_dir, "longproc_memory_usage.csv"), index=False)
longproc_throughput_df.to_csv(os.path.join(longproc_dir, "longproc_throughput.csv"), index=False)
longproc_performance_df.to_csv(os.path.join(longproc_dir, "longproc_performance.csv"), index=False)

# Save HELMET DataFrames
helmet_memory_df.to_csv(os.path.join(helmet_dir, "helmet_memory_usage.csv"), index=False)
helmet_throughput_df.to_csv(os.path.join(helmet_dir, "helmet_throughput.csv"), index=False)
helmet_performance_df.to_csv(os.path.join(helmet_dir, "helmet_performance.csv"), index=False)

# --- PRINT DATAFRAMES ---
print("\n--- LongProc DataFrames ---")
print("\nMemory Usage:")
print(longproc_memory_df)
print("\nThroughput (tokens/sec):")
print(longproc_throughput_df)
print("\nPerformance:")
print(longproc_performance_df)

print("\n--- HELMET DataFrames ---")
print("\nMemory Usage:")
print(helmet_memory_df)
print("\nThroughput (tokens/sec):")
print(helmet_throughput_df)
print("\nPerformance:")
print(helmet_performance_df)