#!/usr/bin/env python3
import sys
import os
import json
import numpy as np

duo_coefficients = json.load(open("/scratch/gpfs/ab4197/p-longhead/p-prulong/eval/HELMET-w-duoattn-eval/outputs/coefficients/duo-prulong_0.9_agg.json"))
snap_coefficients = json.load(open("/scratch/gpfs/ab4197/p-longhead/p-prulong/eval/HELMET-w-duoattn-eval/outputs/coefficients/snap-pyramidkv_0.9_agg.json"))

recall_128k = """
ruler_niah_mk_2_eval_validation_131072_in131072_size100_shots2_sampFalsemax50min0t0.0p1.0_chatFalse_42.json.score
ruler_niah_mk_3_eval_validation_131072_in131072_size100_shots2_sampFalsemax100min0t0.0p1.0_chatFalse_42.json.score
ruler_niah_mv_eval_validation_131072_in131072_size100_shots2_sampFalsemax50min0t0.0p1.0_chatFalse_42.json.score
json_kv_eval_test_k1800_dep6_in131072_size100_shots2_sampFalsemax100min0t0.0p1.0_chatFalse_42.json.score
"""

json_kv_128k = """
json_kv_eval_test_k1800_dep6_in131072_size100_shots2_sampFalsemax100min0t0.0p1.0_chatFalse_42.json.score
"""

json_5k5v_128k = """
json_kv_eval_test_k360_dep6_uuids5_in131072_size100_shots2_sampFalsemax500min0t0.0p1.0_chatFalse_42.json
"""

html_to_tsv_128k = """
html_to_tsv_0.5k_eval_data_in128000_size100_shots0_sampFalsemax1024min0t0.0p1.0_chatTrue_42.json.score
html_to_tsv_2k_eval_data_in128000_size100_shots0_sampFalsemax3072min0t0.0p1.0_chatTrue_42.json.score
html_to_tsv_8k_eval_data_in128000_size100_shots0_sampFalsemax10240min0t0.0p1.0_chatTrue_42.json.score
"""

pseudo_to_code_128k = """
pseudo_to_code_0.5k_eval_data_in8000_size100_shots0_sampFalsemax1024min0t0.0p1.0_chatTrue_42.json.score
pseudo_to_code_2k_eval_data_in8000_size100_shots0_sampFalsemax3072min0t0.0p1.0_chatTrue_42.json.score
"""

travel_planning_128k = """
travel_planning_2k_eval_data_in32000_size100_shots0_sampFalsemax3072min0t0.0p1.0_chatTrue_42.json.score
travel_planning_8k_eval_data_in32000_size100_shots0_sampFalsemax10240min0t0.0p1.0_chatTrue_42.json.score
"""

countdown_128k = """
countdown_0.5k_eval_data_in32000_size100_shots0_sampFalsemax1024min0t0.0p1.0_chatTrue_42.json.score
countdown_2k_eval_data_in32000_size100_shots0_sampFalsemax3072min0t0.0p1.0_chatTrue_42.json.score
countdown_8k_eval_data_in32000_size100_shots0_sampFalsemax10240min0t0.0p1.0_chatTrue_42.json.score
"""

rag_128k = """
kilt_nq_eval_nq-dev-multikilt_1000_k1000_dep6_in131072_size100_shots2_sampFalsemax20min0t0.0p1.0_chatFalse_42.json.score
kilt_popqa_3_eval_popqa_test_1000_k1000_dep6_in131072_size100_shots2_sampFalsemax20min0t0.0p1.0_chatFalse_42.json.score
kilt_triviaqa_eval_triviaqa-dev-multikilt_1000_k1000_dep6_in131072_size100_shots2_sampFalsemax20min0t0.0p1.0_chatFalse_42.json.score
kilt_hotpotqa_eval_hotpotqa-dev-multikilt_1000_k1000_dep3_in131072_size100_shots2_sampFalsemax20min0t0.0p1.0_chatFalse_42.json.score
"""

rerank_128k = """
msmarco_rerank_psg_eval_test_reranking_data_k1000_dep3_in131072_size100_shots2_sampFalsemax200min0t0.0p1.0_chatFalse_42.json.score
"""

icl_128k = """
icl_banking77_5900shot_balance_eval__in131072_size500_shots0_sampFalsemax20min0t0.0p1.0_chatFalse_42.json.score
icl_clinic150_7050shot_balance_eval__in131072_size500_shots0_sampFalsemax20min0t0.0p1.0_chatFalse_42.json.score
icl_nlu_8296shot_balance_eval__in131072_size500_shots0_sampFalsemax20min0t0.0p1.0_chatFalse_42.json.score
icl_trec_coarse_6600shot_balance_eval__in131072_size500_shots0_sampFalsemax20min0t0.0p1.0_chatFalse_42.json.score
icl_trec_fine_6400shot_balance_eval__in131072_size500_shots0_sampFalsemax20min0t0.0p1.0_chatFalse_42.json.score
"""

json_kv_128k = json_kv_128k.strip().split("\n")
json_5k5v_128k = json_5k5v_128k.strip().split("\n")
recall_128k = recall_128k.strip().split("\n")
html_to_tsv_128k = html_to_tsv_128k.strip().split("\n")
pseudo_to_code_128k = pseudo_to_code_128k.strip().split("\n")
travel_planning_128k = travel_planning_128k.strip().split("\n")
countdown_128k = countdown_128k.strip().split("\n")
rag_128k = rag_128k.strip().split("\n")
rerank_128k = rerank_128k.strip().split("\n")
icl_128k = icl_128k.strip().split("\n")

metric = {
    "json_kv": "substring_exact_match",
    "json_5k5v": "substring_exact_match",
    "kilt": "substring_exact_match",
    "msmarco": "NDCG@10",
    "ruler": "ruler_recall",
    "html_to": "f1",
    "pseudo_to": "accuracy", 
    "travel_planning": "accuracy",
    "countdown": "accuracy",
    "icl": "exact_match"
}

def process_file(path, allf):
    data = []
    for f in allf:
        full = os.path.join(path, f)
        try:
            r = json.load(open(full))
        except:
            # missing or bad JSON → bail
            return "NA"
        # unwrap if needed
        if "averaged_metrics" in r:
            r = r["averaged_metrics"]
        # pick task key
        task = f.split("_")[0]
        if task not in metric:
            task = f.split("_")[0] + "_" + f.split("_")[1]
        try:
            if isinstance(metric[task], list):
                scores = []
                for m in metric[task]:
                    val = r[m]
                    # apply any scaling
                    if m == "gpt4-f1":      val *= 100
                    if m == "gpt-4-score":  val *= (100/3)
                    scores.append(val)
                score = np.mean(scores)
            else:
                score = r[ metric[task] ]
        except:
            return "NA"
        data.append(score)
    return f"{np.mean(data):.2f}"

def process_file_with_sparsity(path, allf):
    ref_dict = snap_coefficients if "PATCH" in path else duo_coefficients

    data = []
    sparsity = []
    for f in allf:
        full = os.path.join(path, f)
        try:
            r = json.load(open(full))
        except:
            # missing or bad JSON → bail
            return "NA"
        # unwrap if needed
        if "averaged_metrics" in r:
            r = r["averaged_metrics"]
        if "attention_sparsity" in r:
            sparsity.append(r["attention_sparsity"])
        # pick task key
        task = f.split("_")[0]
        if task not in metric:
            task = f.split("_")[0] + "_" + f.split("_")[1]
        try:
            if isinstance(metric[task], list):
                scores = []
                for m in metric[task]:
                    val = r[m]
                    # apply any scaling
                    if m == "gpt4-f1":      val *= 100
                    if m == "gpt-4-score":  val *= (100/3)
                    scores.append(val)
                score = np.mean(scores)
            else:
                score = r[ metric[task] ]
        except:
            return "NA"
        data.append(score)
    return f"{np.mean(data):.2f} @ {np.mean(sparsity):.2f}"

if __name__ == "__main__":
    # dirs = sys.argv[1:]
    # if not dirs:
    
    if len(sys.argv) > 1:
        key = sys.argv[1]
    else:
        key = "PATCH64"
    dirs = [f for f in os.listdir("outputs/Llama-3.1-8B-Instruct") if 
            (
                f.startswith(key) if key != "" else not (
                    f.startswith("LOCAL64") or f.startswith("FULL") or f.startswith("PATCH64") or f.startswith("NOPATCH64")
                )
            )
    ]
    if key == "":
        pass
    else:
        if "_pf" in dirs[0]:
            dirs = sorted(dirs, key=lambda x: int(x.split("_sp")[-1].split("_pf")[0]))
        else:
            dirs = sorted(dirs, key=lambda x: int(x.split("_sp")[-1].split("_tg_")[0]))
    dirs = [os.path.join("outputs/Llama-3.1-8B-Instruct", d) for d in dirs]
    # print("Usage: python print_prulong.py <dir1> [<dir2> …]")
    # sys.exit(1)

    # define the columns we want, in order:
    categories = [
      ("Recall_128k", recall_128k),
      ("RAG_128k",    rag_128k),
      ("Rerank_128k", rerank_128k),
      ("ICL_128k", icl_128k),
      ("HTML_to_TSV_128k", html_to_tsv_128k),
      # ("Pseudo_to_Code_128k", pseudo_to_code_128k),
      ("Travel_Planning_128k", travel_planning_128k),
      #  ("Countdown_128k", countdown_128k),
      # ("JSON_KV_128k", json_kv_128k),
      # ("JSON_5k5v_128k", json_5k5v_128k),
    ]

    # print header
    header = ["Directory"] + [name for name,_ in categories]
    print("\t".join(header))

    # one row per directory
    for d in dirs:
        row = [os.path.basename(d)]
        for _, file_list in categories:
            row.append(process_file_with_sparsity(d, file_list))
        print("\t".join(row))
