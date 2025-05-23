import os
import json
import pandas as pd
from collections import defaultdict
from typing import Dict, Tuple, List
from tqdm import tqdm

# Constants
TASK_MAP = {
    "alce_asqa": "cite",
    "json_kv": "recall_jsonkv", 
    "kilt_hotpotqa": "rag_hotpotqa",
    "kilt_nq": "rag_nq",
    "msmarco_rerank": "rerank"
}
TASKS_BY_CONTEXT = {
    "2k": ["html_to_tsv", "pseudo_to_code", "travel_planning"],
    "5k": ["html_to_tsv", "pseudo_to_code"],
    "16k": ["recall_jsonkv", "rag_nq", "rag_hotpotqa", "rerank", "cite"],
    "32k": ["recall_jsonkv", "rag_nq", "rag_hotpotqa", "rerank", "cite"]
}
SCORE_KEYS = {
    "html_to_tsv": ["f1"],
    "pseudo_to_code": ["accuracy"],
    "travel_planning": ["accuracy"], 
    "cite": ["str_em", "citation_rec", "citation_prec"],
    "recall_jsonkv": ["substring_exact_match"],
    "rag_nq": ["substring_exact_match"],
    "rag_hotpotqa": ["substring_exact_match"],
    "rerank": ["NDCG@10"]
}

# Helper function to map file prefix to task
def get_task_from_filename(filename: str) -> str:
    for prefix, task in TASK_MAP.items():
        if filename.startswith(prefix):
            return task
    for task in SCORE_KEYS:
        if filename.startswith(task):
            return task
    return None

# Initialize data collectors for both LongProc and HELMET
longproc_memory_data = defaultdict(lambda: defaultdict(float))
longproc_throughput_data = defaultdict(lambda: defaultdict(float))
longproc_performance_data = defaultdict(lambda: defaultdict(float))

helmet_memory_data = defaultdict(lambda: defaultdict(float))
helmet_throughput_data = defaultdict(lambda: defaultdict(float))
helmet_performance_data = defaultdict(lambda: defaultdict(float))

# Base directory (adjust if needed)
base_dir = "/scratch/gpfs/DANQIC/jz4391/HELMET/output"

# Traverse directories
for technique in tqdm(os.listdir(base_dir), desc="Processing techniques"):
    technique_path = os.path.join(base_dir, technique)
    if not os.path.isdir(technique_path):
        continue
    for context_length in tqdm(os.listdir(technique_path), desc=f"Processing {technique} contexts", leave=False):
        context_path = os.path.join(technique_path, context_length)
        if not os.path.isdir(context_path):
            continue

        for model in tqdm(os.listdir(context_path), desc=f"Processing {context_length} models", leave=False):
            model_path = os.path.join(context_path, model)
            if not os.path.isdir(model_path):
                continue

            subdirs = [(model_path, (technique, context_length, model))]
            # Handle quantization folders for baseline
            if technique == "baseline":
                subdirs = []  # Reset subdirs for baseline
                for q in os.listdir(model_path):
                    if q.startswith("4bit"):
                        row_key = ("INT4", context_length, model)
                        subdirs.append((os.path.join(model_path, q), row_key))
                    elif q.startswith("8bit"):
                        row_key = ("INT8", context_length, model)
                        subdirs.append((os.path.join(model_path, q), row_key))
                    elif q.startswith("16bit"):
                        row_key = (technique, context_length, model)
                        subdirs.append((os.path.join(model_path, q), row_key))

            for subdir, row_key in subdirs:
                if not os.path.isdir(subdir):
                    continue

                tasks = TASKS_BY_CONTEXT.get(context_length, [])

                # Determine if this is LongProc or HELMET based on context length
                is_longproc = context_length in ["2k", "5k"]
                memory_data = longproc_memory_data if is_longproc else helmet_memory_data
                throughput_data = longproc_throughput_data if is_longproc else helmet_throughput_data
                performance_data = longproc_performance_data if is_longproc else helmet_performance_data

                for file in os.listdir(subdir):
                    filepath = os.path.join(subdir, file)
                    if not os.path.isfile(filepath):
                        continue

                    task = get_task_from_filename(file)
                    if task not in tasks:
                        continue

                    if file.endswith(".json") and not file.endswith(".json.score"):
                        with open(filepath) as f:
                            data = json.load(f)
                            if "memory_usage" in data:
                                # Convert memory to GB by dividing by 10^9
                                memory_data[row_key][task] = float(data["memory_usage"]) / 1e9
                            if "throughput" in data:
                                throughput_data[row_key][task] = float(data["throughput"])
                    elif file.endswith(".json.score"):
                        with open(filepath) as f:
                            score_data = json.load(f)
                            # Print for debugging
                            # print(f"Processing score file: {filepath}")
                            # print(f"Score data: {score_data}")
                            # print(f"Task: {task}, Score keys: {SCORE_KEYS.get(task, [])}")
                            
                            # Add debug prints to trace the data flow
                            # print(f"Current row_key: {row_key}")
                            # print(f"Current performance_data: {dict(performance_data)}")
                            
                            for key in SCORE_KEYS.get(task, []):
                                value = score_data.get(key)
                                if value is not None:
                                    try:
                                        perf_key = f"{task}_{key}" if len(SCORE_KEYS[task]) > 1 else task
                                        performance_data[row_key][perf_key] = float(value)
                                        # print(f"Successfully added performance data: {row_key} -> {perf_key} = {value}")
                                    except Exception as e:
                                        print(f"Error processing value: {value} for key: {key}")
                                        print(f"Exception: {str(e)}")
                                else:
                                    print(f"Warning: No value found for key {key} in score data")

# Create DataFrames
def create_dataframe(data_dict: Dict[Tuple[str, str, str], Dict[str, float]]) -> pd.DataFrame:
    records = []
    for (technique, context_length, model), task_data in data_dict.items():
        record = {"technique": technique, "context_length": context_length, "model": model}
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

# Print raw data for debugging
print("\nRaw LongProc Performance Data:")
print("Performance data dict:", dict(longproc_performance_data))
print("Keys in dict:", list(longproc_performance_data.keys()))
print("Score keys being used:", SCORE_KEYS)
print("\nRaw HELMET Performance Data:") 
print("Performance data dict:", dict(helmet_performance_data))
print("Keys in dict:", list(helmet_performance_data.keys()))
print("Score keys being used:", SCORE_KEYS)

# Create output directories
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

# Print DataFrames
print("\nLongProc Memory Usage DataFrame:")
print(longproc_memory_df)
print("\nLongProc Throughput DataFrame:")
print(longproc_throughput_df)
print("\nLongProc Performance DataFrame:")
print(longproc_performance_df)

print("\nHELMET Memory Usage DataFrame:")
print(helmet_memory_df)
print("\nHELMET Throughput DataFrame:")
print(helmet_throughput_df)
print("\nHELMET Performance DataFrame:")
print(helmet_performance_df)
