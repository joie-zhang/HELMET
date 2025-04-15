import os
import sys
import json
import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass, asdict
from tqdm import tqdm

dataset_to_metrics = {
    "json_kv": "substring_exact_match",
    "nq": "substring_exact_match",
    "popqa": "substring_exact_match",
    "triviaqa": "substring_exact_match",
    "hotpotqa": "substring_exact_match",
    
    "narrativeqa": ["gpt-4-score"],
    "msmarco_rerank_psg": "NDCG@10",
    
    "trec_coarse": "exact_match",
    "trec_fine": "exact_match",
    "banking77": "exact_match",
    "clinic150": "exact_match",
    "nlu": "exact_match",
    
    "qmsum": "rougeL_recall",
    "multi_lexsum": ["gpt-4-f1"],
    
    "ruler_niah_s_1": "ruler_recall",
    "ruler_niah_s_2": "ruler_recall",
    "ruler_niah_s_3": "ruler_recall",
    "ruler_niah_mk_1": "ruler_recall",
    "ruler_niah_mk_2": "ruler_recall",
    "ruler_niah_mk_3": "ruler_recall",
    "ruler_niah_mq": "ruler_recall",
    "ruler_niah_mv": "ruler_recall",
    "ruler_fwe": "ruler_recall",
    "ruler_cwe": "ruler_recall",
    "ruler_vt": "ruler_recall",
    "ruler_qa_1": "substring_exact_match",
    "ruler_qa_2": "substring_exact_match",
    
    "infbench_qa": ["rougeL_f1"],
    "infbench_choice": ["exact_match"],
    "infbench_sum": ["gpt-4-f1"],
    
    "alce_asqa": ["str_em", "citation_rec", "citation_prec"],
    "alce_qampari": ["qampari_rec_top5", "citation_rec", "citation_prec"],
}

dataset_to_metrics = {k: [v] if isinstance(v, str) else v for k, v in dataset_to_metrics.items()}
custom_avgs = {
    "Recall": ["json_kv substring_exact_match", "ruler_niah_mk_2 ruler_recall", "ruler_niah_mk_3 ruler_recall", "ruler_niah_mv ruler_recall"],
    "RAG": ['nq substring_exact_match', 'hotpotqa substring_exact_match', 'popqa substring_exact_match', 'triviaqa substring_exact_match',],
    "ICL": ['trec_coarse exact_match', 'trec_fine exact_match', 'banking77 exact_match', 'clinic150 exact_match', 'nlu exact_match'],
    "Cite": ['alce_asqa str_em', 'alce_asqa citation_rec', 'alce_asqa citation_prec', 'alce_qampari qampari_rec_top5', 'alce_qampari citation_rec', 'alce_qampari citation_prec', ],
    "Re-rank": ['msmarco_rerank_psg NDCG@10', ],
    "LongQA": ['narrativeqa gpt-4-score', 'infbench_qa rougeL_f1', 'infbench_choice exact_match', ],
    "Summ": ['infbench_sum gpt-4-f1', 'multi_lexsum gpt-4-f1', ],
    "RULER": ['ruler_niah_s_1 ruler_recall', 'ruler_niah_s_2 ruler_recall', 'ruler_niah_s_3 ruler_recall', 'ruler_niah_mk_1 ruler_recall', 'ruler_niah_mk_2 ruler_recall', 'ruler_niah_mk_3 ruler_recall', 'ruler_niah_mq ruler_recall', 'ruler_niah_mv ruler_recall', 'ruler_cwe ruler_recall', 'ruler_fwe ruler_recall', 'ruler_vt ruler_recall', 'ruler_qa_1 substring_exact_match', 'ruler_qa_2 substring_exact_match'],
    "Ours-Real": ['RAG', 'ICL', 'Cite', 'Re-rank', 'LongQA', 'Summ'],
    "Ours": ['Recall', 'RAG', 'ICL', 'Cite', 'Re-rank', 'LongQA', 'Summ'],
}

@dataclass
class arguments:
    tag: str = "v1"
    input_max_length: int = 131072
    generation_max_length: int = 100
    generation_min_length: int = 0
    max_test_samples: int = 100
    shots: int = 2
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    use_chat_template: bool = False
    seed: int = 42
    test_name: str = ""
    dataset: str = "nq"
    output_dir: str = "output"
    popularity_threshold: float = 3
    flenqa_ctx_size: int = 1000
    context_length: str = "8k"  # New field for context length
    experiment_type: str = "standard"  # Can be "standard", "minference", or "streamingllm"
    quantize: int = 16  # Can be 16, 8, or 4
    
    category: str = "synthetic"
    
    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
    def get_path(self):
        tag = self.tag
        if "flenqa" in self.dataset:
            tag += f"_ctx{self.flenqa_ctx_size}"
            
        # Build the base filename
        # filename = "{args.dataset}_{tag}_{args.test_name}_in{args.input_max_length}_size{args.max_test_samples}_shots{args.shots}_samp{args.do_sample}max{args.generation_max_length}min{args.generation_min_length}t{args.temperature}p{args.top_p}_chat{args.use_chat_template}_{args.seed}.json".format(args=self, tag=tag)
        filename = "{args.dataset}_{args.quantize}bit_{tag}_{args.test_name}_in{args.input_max_length}_size{args.max_test_samples}_shots{args.shots}_samp{args.do_sample}max{args.generation_max_length}min{args.generation_min_length}t{args.temperature}p{args.top_p}_chat{args.use_chat_template}_{args.seed}.json".format(args=self, tag=tag)
        
        # Determine the correct directory based on experiment type
        if self.experiment_type == "standard":
            base_dir = os.path.join(self.output_dir, self.context_length, f"bit{args.quantize}", model)
        elif self.experiment_type == "minference":
            base_dir = os.path.join(self.output_dir, "minference", self.context_length, f"bit{args.quantize}", args.model_name_or_path)
        elif self.experiment_type == "streamingllm":
            base_dir = os.path.join(self.output_dir, "minference", "streamingllm", self.context_length, f"bit{args.quantize}", args.model_name_or_path)
        
        path = os.path.join(base_dir, filename)

        # Check for alternative file formats
        if os.path.exists(path.replace(".json", "-gpt4eval_o.json")):
            return path.replace(".json", "-gpt4eval_o.json")
        if "alce" in self.dataset:
            return path.replace(".json", ".json.score")
        if os.path.exists(path + ".score"):
            return path + ".score"
        return path

    def get_metric_name(self):
        for d, m in dataset_to_metrics.items():
            if d in self.dataset:
                return d, m
        return None
    
    def get_averaged_metric(self):
        path = self.get_path()
        print(path)
        if not os.path.exists(path):
            print("path doesn't exist")
            return None
        with open(path) as f:
            results = json.load(f)
        
        _, metric = self.get_metric_name()
        if path.endswith(".score"):
            if any([m not in results for m in metric]):
                print("metric doesn't exist")
                return None
            s = {m: results[m] for m in metric}
        else:
            if any([m not in results["averaged_metrics"] for m in metric]):
                print("metric doesn't exist")
                return None
            s = {m: results['averaged_metrics'][m] for m in metric}
        
        s = {m : v * (100 if m == "gpt-4-f1" else 1) * (100/3 if m == "gpt-4-score" else 1) for m, v in s.items()}
        print("found scores:", s)
        return s
        
    def get_metric_by_depth(self):
        path = self.get_path()
        path = path.replace(".score", '')
        print(path)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            results = json.load(f)

        output = []        
        _, metric = self.get_metric_name()
        metric = metric[0]
        keys = ["depth", "k", metric]
        for d in results["data"]:
            o = {}
            for key in keys:
                if key == "k" and "ctxs" in d:
                    d["k"] = len(d['ctxs'])
                if key not in d:
                    print("no", key)
                    return None
                o[key] = d[key]
            o["metric"] = o.pop(metric)
            output.append(o)
        
        df = pd.DataFrame(output)
        dfs = df.groupby(list(output[0].keys())[:-1]).mean().reset_index()

        return dfs.to_dict("records")

if __name__ == "__main__":
    # Define context lengths to evaluate
    context_lengths = ["0.5k", "2k", "8k", "16k", "32k"]
    
    # Define experiment types
    experiment_types = ["standard", "minference", "streamingllm"]
    
    # comment out the models you don't want to include, or add the new ones 
    models_configs = [
        # {"model": "gpt-4-0125-preview", "use_chat_template": True, "training_length": 128000},
        # {"model": "gpt-4o-mini-2024-07-18", "use_chat_template": True, "training_length": 128000},
        # {"model": "gpt-4o-2024-05-13", "use_chat_template": True, "training_length": 128000},
        # {"model": "gpt-4o-2024-08-06", "use_chat_template": True, "training_length": 128000},
        # {"model": "claude-3-5-sonnet-20240620", "use_chat_template": True, "training_length": 200000},
        # {"model": "gemini-1.5-flash-001", "use_chat_template": True, "training_length": 1048576},
        # {"model": "gemini-1.5-pro-001", "use_chat_template": True, "training_length": 2097152},

        # llama 2 based models
        # {"model": "Llama-2-7B-32K", "use_chat_template": False, "training_length": 32768},
        # {"model": "Llama-2-7B-32K-Instruct", "training_length": 32768},
        # {"model": "llama-2-7b-80k", "use_chat_template": False, "training_length": 80000},
        # {"model": "Yarn-Llama-2-7b-64k", "use_chat_template": False, "training_length": 65536},
        # {"model": "Yarn-Llama-2-7b-128k", "use_chat_template": False, "training_length": 131072},
        
        # llama 3 models
        # {"model": "Meta-Llama-3-8B", "use_chat_template": False, "training_length": 8192},
        # {"model": "Meta-Llama-3-8B-Instruct", "training_length": 8192},
        # {"model": "Meta-Llama-3-8B-Theta16M", "use_chat_template": False, "training_length": 8192},
        # {"model": "Meta-Llama-3-8B-Instruct-Theta16M", "training_length": 8192},
        # {"model": "Meta-Llama-3-70B-Theta16M", "use_chat_template": False, "training_length": 8192},
        # {"model": "Meta-Llama-3-70B-Instruct-Theta16M", "training_length": 8192},
        
        # {"model": "Llama-3.1-8B", "use_chat_template": False, "training_length": 131072},
        {"model": "Llama-3.1-8B-Instruct", "training_length": 131072},
        # {"model": "Llama-3.1-70B", "use_chat_template": False, "training_length": 131072},
        # {"model": "Llama-3.1-70B-Instruct", "training_length": 131072},
        # {"model": "Llama-3.3-70B-Instruct", "training_length": 131072},
        
        # {"model": "Llama-3.2-1B", "use_chat_template": False, "training_length": 131072},
        # {"model": "Llama-3.2-1B-Instruct", "training_length": 131072},
        # {"model": "Llama-3.2-3B", "use_chat_template": False, "training_length": 131072},
        # {"model": "Llama-3.2-3B-Instruct", "training_length": 131072},
        
        # mistral models
        # {"model": "Mistral-7B-v0.1", "use_chat_template": False, "training_length": 8192},
        # {"model": "Mistral-7B-Instruct-v0.1", "training_length": 8192},
        # {"model": "Mistral-7B-Instruct-v0.2", "training_length": 32768},
        # {"model": "Mistral-7B-v0.3", "use_chat_template": False, "training_length": 32768},
        # {"model": "Mistral-7B-Instruct-v0.3", "training_length": 32768},
        # {"model": "Ministral-8B-Instruct-2410", "training_length": 131072},
        
        # {"model": "Mistral-Nemo-Base-2407", "use_chat_template": False, "training_length": 128000},
        # {"model": "Mistral-Nemo-Instruct-2407", "training_length": 128000},
        # {"model": "MegaBeam-Mistral-7B-512k", "training_length": 524288},
        
        # yi models
        # {"model": "Yi-6B-200K", "use_chat_template": False, "training_length": 200000},
        # {"model": "Yi-9B-200K", "use_chat_template": False, "training_length": 200000},
        # {"model": "Yi-34B-200K", "use_chat_template": False, "training_length": 200000},
        # {"model": "Yi-1.5-9B-32K", "use_chat_template": False, "training_length": 32768},
        
        # phi models
        # {"model": "Phi-3-mini-128k-instruct", "training_length": 131072},
        # {"model": "Phi-3-small-128k-instruct", "training_length": 131072},
        # {"model": "Phi-3-medium-128k-instruct", "training_length": 131072},
        # {"model": "Phi-3.5-mini-instruct", "training_length": 131072},
        
        # qwen models
        # {"model": "Qwen2-7B", "use_chat_template": False, "training_length": 32768},
        # {"model": "Qwen2-7B-Instruct", "training_length": 32768},
        # {"model": "Qwen2-57B-A14B", "use_chat_template": False, "training_length": 32768},
        # {"model": "Qwen2-57B-A14B-Instruct", "training_length": 32768},
        # {"model": "Qwen2.5-1.5B", "use_chat_template": False, "training_length": 32768},
        # {"model": "Qwen2.5-1.5B-Instruct", "training_length": 32768},
        # {"model": "Qwen2.5-3B", "use_chat_template": False, "training_length": 32768},
        # {"model": "Qwen2.5-3B-Instruct", "training_length": 32768},
        # {"model": "Qwen2.5-7B", "use_chat_template": False, "training_length": 131072},
        {"model": "Qwen2.5-7B-Instruct", "training_length": 131072},
        # {"model": "Qwen2.5-72B-Instruct", "training_length": 131072},
        
        # prolong
        # {"model": "Llama-3-8B-ProLong-512k-Instruct", "training_length": 524288},
        
        # gemma 2 models
        # {"model": "gemma-2-9b", "use_chat_template": False, "training_length": 8192},
        # {"model": "gemma-2-9b-it", "training_length": 8192},
        # {"model": "gemma-2-9b-it-Theta320K", "training_length": 8192},

        # {"model": "gemma-2-27b", "use_chat_template": False, "training_length": 8192},
        # {"model": "gemma-2-27b-it", "training_length": 8192},
        # {"model": "gemma-2-27b-it-Theta320K", "training_length": 8192},
        
        # others
        # {"model": "c4ai-command-r-v01", "training_length": 131072},
        # {"model": "Jamba-v0.1", "use_chat_template": False, "training_length": 262144},
        # {"model": "AI21-Jamba-1.5-Mini", "training_length": 262144},
    ]

    # set your configs here, only include the ones that you ran
    config_files = [
        # HELMET configs
        "configs/rerank_16k.yaml", "configs/rerank_32k.yaml",
        "configs/cite_16k.yaml", "configs/cite_32k.yaml",
        "configs/rag_16k.yaml", "configs/rag_32k.yaml",
        "configs/recall_jsonkv_16k.yaml", "configs/recall_jsonkv_32k.yaml",
        
        # LongProc configs
        "longproc_addon/configs/countdown_2k.yaml",
        "longproc_addon/configs/travel_planning_2k.yaml",
        "longproc_addon/configs/html_to_tsv_2k.yaml",
    ]

    dataset_configs = []
    for file in config_files:
        try: 
            c = yaml.safe_load(open(file))
        
            # Set default values for optional fields
            defaults = {
                'max_test_samples': 100,  # Default to 100 samples
                'use_chat_template': False,  # Default to False
                'shots': 2,  # Default to 2 shots
                'input_max_length': 131072,  # Default max input length
                'generation_max_length': 100,  # Default max generation length
            }
            
            # Update defaults with actual values from config
            for key, default_value in defaults.items():
                if key not in c:
                    c[key] = default_value
                        
            # Convert single values to lists
            if isinstance(c.get('datasets'), str):
                datasets = c['datasets'].split(',')
            else:
                datasets = [str(c['datasets'])]
                
            if isinstance(c.get('test_files'), str):
                test_files = c['test_files'].split(',')
            else:
                test_files = [str(c['test_files'])]
                
            if isinstance(c.get('input_max_length'), str):
                input_lengths = c['input_max_length'].split(',')
            else:
                input_lengths = [str(c['input_max_length'])] * len(datasets)
                
            if isinstance(c.get('generation_max_length'), str):
                gen_lengths = c['generation_max_length'].split(',')
            else:
                gen_lengths = [str(c['generation_max_length'])] * len(datasets)

            # Make sure all lists have the same length
            for d, t, l, g in zip(datasets, test_files, input_lengths, gen_lengths):
                dataset_configs.append({
                    "dataset": d, 
                    "test_name": os.path.basename(os.path.splitext(t)[0]), 
                    "input_max_length": int(l), 
                    "generation_max_length": int(g), 
                    "max_test_samples": c['max_test_samples'], 
                    'use_chat_template': c['use_chat_template'], 
                    'shots': c['shots']
                })
        except Exception as e:
            print(f"Error loading config file {file}: {str(e)}")
            continue

    if not dataset_configs:
        print("Warning: No valid dataset configurations were loaded!")
    else:
        print(f"Successfully loaded {len(dataset_configs)} dataset configurations")
        print(dataset_configs)

    failed_paths = []
    df = []
    
    # Add debug counters
    total_attempts = 0
    successful_metrics = 0
    
    # Iterate through experiment types and context lengths
    for exp_type in experiment_types:
        for ctx_len in context_lengths:
            for model in tqdm(models_configs, desc=f"{exp_type}-{ctx_len}"):
                args = arguments()
                args.tag = "v1"  # SET YOUR TAG HERE
                args.output_dir = "output"  # Now using root output dir
                args.context_length = ctx_len
                args.experiment_type = exp_type
                
                for dataset in dataset_configs:
                    total_attempts += 1
                    args.update(dataset)
                    args.update(model)

                    # Print the path we're trying to access
                    path = args.get_path()
                    print(f"\nTrying path: {path}")
                    
                    metric = args.get_averaged_metric()
                    if metric is None:
                        failed_paths.append(path)
                        continue
                        
                    dsimple, mnames = args.get_metric_name()
                    if dsimple is None:
                        print(f"Warning: No metric found for dataset {args.dataset}")
                        continue
                        
                    successful_metrics += 1
                    
                    # Create a base dictionary with all the fields we want to include
                    base_entry = {
                        "model": model["model"],
                        "context_length": ctx_len,
                        "experiment_type": exp_type,
                        "input_max_length": args.input_max_length,
                        "test_data": f"{args.dataset}-{args.test_name}-{args.input_max_length}"
                    }
                    
                    for k, m in metric.items():
                        entry = base_entry.copy()
                        entry.update({
                            "metric_name": k,
                            "metric": m,
                            "dataset_simple": dsimple + " " + k
                        })
                        df.append(entry)

    # Print debug information
    print(f"\nTotal attempts: {total_attempts}")
    print(f"Successful metrics: {successful_metrics}")
    print(f"Failed paths: {len(failed_paths)}")

    # Create the main results dataframe
    if not df:
        print("\nNo results were found! Check if the paths and experiment configurations are correct.")
        sys.exit(1)
        
    all_df = pd.DataFrame(df)
    print("\nDataFrame columns:", all_df.columns.tolist())
    print("\nDataFrame shape:", all_df.shape)
    
    # Create pivot tables for each experiment type
    for exp_type in experiment_types:
        exp_df = all_df[all_df["experiment_type"] == exp_type]  # Use dictionary-style access
        if len(exp_df) == 0:  # Skip if no results for this experiment type
            print(f"\nNo results found for experiment type: {exp_type}")
            continue
        
        print(f"\nFound {len(exp_df)} results for {exp_type}")
        
        lf_df = exp_df.pivot_table(
            index=["model", "context_length", "input_max_length"],
            columns="dataset_simple",
            values="metric",
            sort=False
        )
        lf_df = lf_df.reset_index()
        
        # Save to CSV
        output_file = f"results_{exp_type}.csv"
        print(f"\nSaving {exp_type} results to {output_file}")
        print(lf_df.to_csv(index=False))
        lf_df.to_csv(output_file, index=False)

    print("\nWarning, failed to get the following paths (first 10 shown):")
    for path in failed_paths[:10]:
        print(f"  - {path}")
    if len(failed_paths) > 10:
        print(f"  ... and {len(failed_paths) - 10} more")