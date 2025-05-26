import sys
import os
import json
import numpy as np

path = sys.argv[1]

recall_32k = """
ruler_niah_mk_2_eval_validation_32768_in32768_size100_shots2_sampFalsemax50min0t1.0p1.0_chatFalse_42.json.score
ruler_niah_mk_3_eval_validation_32768_in32768_size100_shots2_sampFalsemax100min0t1.0p1.0_chatFalse_42.json.score
ruler_niah_mv_eval_validation_32768_in32768_size100_shots2_sampFalsemax50min0t1.0p1.0_chatFalse_42.json.score
json_kv_eval_test_k440_dep6_in32768_size100_shots2_sampFalsemax100min0t1.0p1.0_chatFalse_42.json.score
"""

recall_64k = """
ruler_niah_mk_2_eval_validation_65536_in65536_size100_shots2_sampFalsemax50min0t1.0p1.0_chatFalse_42.json.score
ruler_niah_mk_3_eval_validation_65536_in65536_size100_shots2_sampFalsemax100min0t1.0p1.0_chatFalse_42.json.score
ruler_niah_mv_eval_validation_65536_in65536_size100_shots2_sampFalsemax50min0t1.0p1.0_chatFalse_42.json.score
json_kv_eval_test_k900_dep6_in65536_size100_shots2_sampFalsemax100min0t1.0p1.0_chatFalse_42.json.score
"""

recall_128k = """
ruler_niah_mk_2_eval_validation_131072_in131072_size100_shots2_sampFalsemax50min0t1.0p1.0_chatFalse_42.json.score
ruler_niah_mk_3_eval_validation_131072_in131072_size100_shots2_sampFalsemax100min0t1.0p1.0_chatFalse_42.json.score
ruler_niah_mv_eval_validation_131072_in131072_size100_shots2_sampFalsemax50min0t1.0p1.0_chatFalse_42.json.score
json_kv_eval_test_k1800_dep6_in131072_size100_shots2_sampFalsemax100min0t1.0p1.0_chatFalse_42.json.score
"""

rag_32k = """
kilt_nq_eval_nq-dev-multikilt_1000_k220_dep6_in32768_size100_shots2_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
kilt_triviaqa_eval_triviaqa-dev-multikilt_1000_k220_dep6_in32768_size100_shots2_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
kilt_hotpotqa_eval_hotpotqa-dev-multikilt_1000_k220_dep3_in32768_size100_shots2_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
kilt_popqa_3_eval_popqa_test_1000_k220_dep6_in32768_size100_shots2_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
"""

rag_64k = """
kilt_nq_eval_nq-dev-multikilt_1000_k440_dep6_in65536_size100_shots2_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
kilt_triviaqa_eval_triviaqa-dev-multikilt_1000_k440_dep6_in65536_size100_shots2_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
kilt_hotpotqa_eval_hotpotqa-dev-multikilt_1000_k440_dep3_in65536_size100_shots2_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
kilt_popqa_3_eval_popqa_test_1000_k440_dep6_in65536_size100_shots2_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
"""

rag_128k = """
kilt_nq_eval_nq-dev-multikilt_1000_k1000_dep6_in131072_size100_shots2_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
kilt_triviaqa_eval_triviaqa-dev-multikilt_1000_k1000_dep6_in131072_size100_shots2_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
kilt_hotpotqa_eval_hotpotqa-dev-multikilt_1000_k1000_dep3_in131072_size100_shots2_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
kilt_popqa_3_eval_popqa_test_1000_k1000_dep6_in131072_size100_shots2_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
"""

rerank_32k = """
msmarco_rerank_psg_eval_test_reranking_data_k285_dep3_in32768_size100_shots2_sampFalsemax200min0t1.0p1.0_chatFalse_42.json.score
"""

rerank_64k = """
msmarco_rerank_psg_eval_test_reranking_data_k600_dep3_in65536_size100_shots2_sampFalsemax200min0t1.0p1.0_chatFalse_42.json.score
"""

rerank_128k = """
msmarco_rerank_psg_eval_test_reranking_data_k1000_dep3_in131072_size100_shots2_sampFalsemax200min0t1.0p1.0_chatFalse_42.json.score
"""

cite_32k = """
alce_asqa_165_eval_asqa_eval_gtr_top2000_in32768_size100_shots2_sampFalsemax300min0t1.0p1.0_chatTrue_42.json.score
alce_qampari_165_eval_qampari_eval_gtr_top2000_in32768_size100_shots2_sampFalsemax300min0t1.0p1.0_chatTrue_42.json.score
"""

cite_64k = """
alce_asqa_345_eval_asqa_eval_gtr_top2000_in65536_size100_shots2_sampFalsemax300min0t1.0p1.0_chatTrue_42.json.score
alce_qampari_345_eval_qampari_eval_gtr_top2000_in65536_size100_shots2_sampFalsemax300min0t1.0p1.0_chatTrue_42.json.score
"""

cite_128k = """
alce_asqa_700_eval_asqa_eval_gtr_top2000_in131072_size100_shots2_sampFalsemax300min0t1.0p1.0_chatTrue_42.json.score
alce_qampari_700_eval_qampari_eval_gtr_top2000_in131072_size100_shots2_sampFalsemax300min0t1.0p1.0_chatTrue_42.json.score
"""

longqa_32k = """
narrativeqa_32468_eval__in32768_size100_shots2_sampFalsemax100min0t1.0p1.0_chatTrue_42-gpt4eval_o.json
infbench_qa_eng_32558_eval__in32768_size100_shots2_sampFalsemax10min0t1.0p1.0_chatTrue_42.json.score
infbench_choice_eng_32558_eval__in32768_size100_shots2_sampFalsemax10min0t1.0p1.0_chatTrue_42.json.score
"""

longqa_64k = """
narrativeqa_65236_eval__in65536_size100_shots2_sampFalsemax100min0t1.0p1.0_chatTrue_42-gpt4eval_o.json
infbench_qa_eng_65326_eval__in65536_size100_shots2_sampFalsemax10min0t1.0p1.0_chatTrue_42.json.score
infbench_choice_eng_65326_eval__in65536_size100_shots2_sampFalsemax10min0t1.0p1.0_chatTrue_42.json.score
"""

longqa_128k = """
narrativeqa_130772_eval__in131072_size100_shots2_sampFalsemax100min0t1.0p1.0_chatTrue_42-gpt4eval_o.json
infbench_qa_eng_130862_eval__in131072_size100_shots2_sampFalsemax10min0t1.0p1.0_chatTrue_42.json.score
infbench_choice_eng_130862_eval__in131072_size100_shots2_sampFalsemax10min0t1.0p1.0_chatTrue_42.json.score
"""

summ_32k = """
infbench_sum_eng_31368_eval__in32768_size100_shots2_sampFalsemax1200min0t1.0p1.0_chatTrue_42-gpt4eval_o.json
multi_lexsum_32068_eval__in32768_size100_shots2_sampFalsemax400min0t1.0p1.0_chatTrue_42-gpt4eval_o.json
"""

summ_64k = """
infbench_sum_eng_64136_eval__in65536_size100_shots2_sampFalsemax1200min0t1.0p1.0_chatTrue_42-gpt4eval_o.json
multi_lexsum_64836_eval__in65536_size100_shots2_sampFalsemax400min0t1.0p1.0_chatTrue_42-gpt4eval_o.json
"""

summ_128k = """
infbench_sum_eng_129672_eval__in131072_size100_shots2_sampFalsemax1200min0t1.0p1.0_chatTrue_42-gpt4eval_o.json
multi_lexsum_130372_eval__in131072_size100_shots2_sampFalsemax400min0t1.0p1.0_chatTrue_42-gpt4eval_o.json
"""

icl_32k = """
icl_trec_coarse_1600shot_balance_eval__in32768_size100_shots0_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
icl_trec_fine_1600shot_balance_eval__in32768_size100_shots0_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
icl_banking77_1450shot_balance_eval__in32768_size100_shots0_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
icl_clinic150_1750shot_balance_eval__in32768_size100_shots0_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
icl_nlu_2040shot_balance_eval__in32768_size100_shots0_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
"""

icl_64k = """
icl_trec_coarse_3300shot_balance_eval__in65536_size100_shots0_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
icl_trec_fine_3200shot_balance_eval__in65536_size100_shots0_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
icl_banking77_2900shot_balance_eval__in65536_size100_shots0_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
icl_clinic150_3525shot_balance_eval__in65536_size100_shots0_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
icl_nlu_4080shot_balance_eval__in65536_size100_shots0_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
"""

icl_128k = """
icl_trec_coarse_6600shot_balance_eval__in131072_size100_shots0_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
icl_trec_fine_6400shot_balance_eval__in131072_size100_shots0_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
icl_banking77_5900shot_balance_eval__in131072_size100_shots0_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
icl_clinic150_7050shot_balance_eval__in131072_size100_shots0_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
icl_nlu_8296shot_balance_eval__in131072_size100_shots0_sampFalsemax20min0t1.0p1.0_chatFalse_42.json.score
"""

recall_32k = recall_32k.strip().split("\n")
recall_64k = recall_64k.strip().split("\n")
recall_128k = recall_128k.strip().split("\n")
rag_32k = rag_32k.strip().split("\n")
rag_64k = rag_64k.strip().split("\n")
rag_128k = rag_128k.strip().split("\n")
rerank_32k = rerank_32k.strip().split("\n")
rerank_64k = rerank_64k.strip().split("\n")
rerank_128k = rerank_128k.strip().split("\n")
icl_32k = icl_32k.strip().split("\n")
icl_64k = icl_64k.strip().split("\n") 
icl_128k = icl_128k.strip().split("\n")
longqa_32k = longqa_32k.strip().split("\n")
longqa_64k = longqa_64k.strip().split("\n")
longqa_128k = longqa_128k.strip().split("\n")
summ_32k = summ_32k.strip().split("\n")
summ_64k = summ_64k.strip().split("\n")
summ_128k = summ_128k.strip().split("\n")
cite_32k = cite_32k.strip().split("\n")
cite_64k = cite_64k.strip().split("\n")
cite_128k = cite_128k.strip().split("\n")


metric = {
    "json": "substring_exact_match",
    "kilt": "substring_exact_match",
    
    "narrativeqa": ["gpt-4-score",],
    "msmarco": "NDCG@10",
    
    "icl": "exact_match",
    
    "qmsum": "rougeL_recall",
    "multi": ["gpt4-f1"],
    
    "ruler": "ruler_recall",

    "infbench_qa": [ "rougeL_f1"],
    "infbench_choice": ["exact_match"],
    "infbench_sum": ["gpt4-f1"], 
    "alce_asqa": ["str_em", "citation_rec", "citation_prec"],
    "alce_qampari": ["qampari_rec_top5", "citation_rec", "citation_prec"],
}

def process_file(allf):
    data = []
    s = ""
    for f in allf:
        try:
            r = json.load(open(os.path.join(path, f)))
        except:
            print(f)
            s += "-\t"
            data.append(-1)
            continue

        if "averaged_metrics" in r:
            r = r["averaged_metrics"]

        task = f.split("_")[0]
        if task not in metric:
            task = f.split("_")[0] + "_" + f.split("_")[1]
            assert task in metric

        try:
            if isinstance(metric[task], list):
                score = [r[_] * (100 if _ == "gpt4-f1" else 1) * (100/3 if _ == "gpt-4-score" else 1) for _ in metric[task]] 
            else:
                score = [r[metric[task]]]
        except Exception as e:
            print(e)
            score = [-1]

        s += "\t".join([f"{sc:.2f}" for sc in score]) 
        data += score

    if -1 in data:
        return f"NA"
    else:
        return f"{np.mean(data):.2f}"

print("Recall 128k\tRAG 128k\tRerank 128k\tICL 128k\tLongQA 128k\tSumm 128k\tCite 128k\t\tRecall 64k\tRAG 64k\tRerank 64k\tICL 64k\tLongQA 64k\tSumm 64k\tCite 64k\t\tRecall 32k\tRAG 32k\tRerank 32k\tICL 32k\tLongQA 32k\tSumm 32k\tCite 32k\t")
print(f"{process_file(recall_128k)}\t{process_file(rag_128k)}\t{process_file(rerank_128k)}\t{process_file(icl_128k)}\t{process_file(longqa_128k)}\t{process_file(summ_128k)}\t{process_file(cite_128k)}\t\t{process_file(recall_64k)}\t{process_file(rag_64k)}\t{process_file(rerank_64k)}\t{process_file(icl_64k)}\t{process_file(longqa_64k)}\t{process_file(summ_64k)}\t{process_file(cite_64k)}\t\t{process_file(recall_32k)}\t{process_file(rag_32k)}\t{process_file(rerank_32k)}\t{process_file(icl_32k)}\t{process_file(longqa_32k)}\t{process_file(summ_32k)}\t{process_file(cite_32k)}\t")
