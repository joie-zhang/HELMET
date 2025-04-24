import os
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
        
    category: str = "synthetic"
    
    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
    def get_path(self):
        tag = self.tag
        if "flenqa" in self.dataset:
            tag += f"_ctx{self.flenqa_ctx_size}"
        path = os.path.join(self.output_dir, "{args.dataset}_{tag}_{args.test_name}_in{args.input_max_length}_size{args.max_test_samples}_shots{args.shots}_samp{args.do_sample}max{args.generation_max_length}min{args.generation_min_length}t{args.temperature}p{args.top_p}_chat{args.use_chat_template}_{args.seed}.json".format(args=self, tag=tag))

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

# ... existing code until the main section ...

if __name__ == "__main__":
    output_root = "output"
    models_configs = []
    
    print("Discovering models...")
    # Walk through the output directory structure
    for method in os.listdir(output_root):
        method_dir = os.path.join(output_root, method)
        if not os.path.isdir(method_dir):
            continue
            
        print(f"\nMethod: {method}")
        # Each method directory contains context length subdirectories
        for ctx_len in os.listdir(method_dir):
            ctx_dir = os.path.join(method_dir, ctx_len)
            if not os.path.isdir(ctx_dir):
                continue
                
            print(f"  Context length: {ctx_len}")
            # Each context length directory contains model directories
            for model_name in os.listdir(ctx_dir):
                model_dir = os.path.join(ctx_dir, model_name)
                if not os.path.isdir(model_dir):
                    continue
                
                # Create model config
                try:
                    # Keep original ctx_len string for directory path
                    ctx_length = int(ctx_len.replace('k', '000'))
                    model_config = {
                        "model": model_name,
                        "use_chat_template": "Instruct" in model_name,
                        "training_length": ctx_length,
                        "method": method,
                        "context_length": ctx_length,
                        "ctx_len_str": ctx_len  # Store original string version
                    }
                    models_configs.append(model_config)
                    print(f"    Added model: {model_name}")
                except ValueError as e:
                    print(f"    Skipped invalid context length: {ctx_len}")
                    continue

    print("\nDiscovering dataset configs...")
    dataset_configs = []
    for model_config in models_configs:
        # Use original ctx_len string for path
        model_dir = os.path.join(output_root, model_config["method"], 
                               model_config["ctx_len_str"], 
                               model_config["model"])
        
        if not os.path.exists(model_dir):
            print(f"Model directory not found: {model_dir}")
            continue
            
        print(f"\nChecking model dir: {model_dir}")
        print(f"Directory contents: {os.listdir(model_dir)}")
        
        for filename in os.listdir(model_dir):
            if not filename.endswith('.json.score'):
                continue
                
            print(f"  Processing score file: {filename}")
            # Parse filename to extract dataset config parameters
            parts = filename.split('_')
            dataset = parts[0]
            if dataset == "kilt":
                dataset = parts[1]
            
            print(f"    Dataset identified: {dataset}")
            
            # Extract parameters from filename
            params = {
                'input_max_length': 32768,
                'generation_max_length': 100,
                'max_test_samples': 100,
                'shots': 2,
                'use_chat_template': False
            }
            
            for part in parts:
                if part.startswith('in'):
                    params['input_max_length'] = int(part[2:])
                elif part.startswith('max'):
                    params['generation_max_length'] = int(part[3:])
                elif part.startswith('size'):
                    params['max_test_samples'] = int(part[4:])
                elif part.startswith('shots'):
                    params['shots'] = int(part[5:])
                elif part.startswith('chat'):
                    params['use_chat_template'] = part[4:].lower() == 'true'
            
            print(f"    Extracted params: {params}")
            
            test_parts = [p for p in parts if 'test' in p or 'dev' in p]
            test_name = test_parts[0] if test_parts else 'test'
            
            dataset_config = {
                "dataset": dataset,
                "test_name": test_name,
                **params
            }
            
            if dataset_config not in dataset_configs:
                dataset_configs.append(dataset_config)
                print(f"    Added dataset config: {dataset} - {test_name}")

    print(f"\nFound {len(models_configs)} models and {len(dataset_configs)} dataset configs")
    print("\nModel configs:")
    for mc in models_configs:
        print(f"  {mc}")
    print("\nDataset configs:")
    for dc in dataset_configs:
        print(f"  {dc}")
    
    failed_paths = []
    df = []
    print("\nCollecting results...")
    for model in tqdm(models_configs):
        args = arguments()
        args.tag = "v1"
        # Use original ctx_len string for path
        args.output_dir = os.path.join("output", model['method'], 
                                     model["ctx_len_str"], 
                                     model["model"])
    
        for dataset in dataset_configs:
            args.update(dataset)
            args.update(model)

            print(f"\nProcessing {model['model']} - {dataset['dataset']}")
            full_path = args.get_path()
            print(f"Looking for file at: {full_path}")
            print(f"File exists: {os.path.exists(full_path)}")
            
            metric = args.get_averaged_metric()
            if metric is None:
                failed_paths.append(full_path)
                print(f"  Failed to get metric")
                continue
                
            dsimple, mnames = args.get_metric_name()
            if dsimple is None:
                print(f"  Unknown dataset metrics for: {dataset['dataset']}")
                continue

            print(f"  Got metrics: {metric}")
            for k, m in metric.items():
                row_data = {**asdict(args), **model,
                    "metric_name": k,
                    "metric": m, 
                    "dataset_simple": dsimple + " " + k, 
                    "test_data": f"{args.dataset}-{args.test_name}-{args.input_max_length}"
                }
                df.append(row_data)
                print(f"    Added result: {k}: {m}")

    if not df:
        print("\nNo results were collected! Check the paths and file structure.")
        print("Failed paths:", failed_paths)
        exit(1)

    print("\nCreating DataFrame...")
    all_df = pd.DataFrame(df)
    print("\nDataFrame columns:", all_df.columns.tolist())
    print("\nFirst few rows of data:")
    print(all_df.head())
    
    lf_df = all_df.pivot_table(index=["model", "input_max_length", "method", "context_length"], 
                              columns="dataset_simple", 
                              values="metric", 
                              sort=False)
    lf_df = lf_df.reset_index()

    print(lf_df.to_csv(index=False))

    print("\nWarning, failed to get the following paths, make sure that these are correct or the printed results will not be accurate:", failed_paths)