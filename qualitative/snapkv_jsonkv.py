import json
from collections import defaultdict
import random

def load_predictions(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        print(f"\nFile: {file_path}")
        print("Data type:", type(data))
        print("Keys:", data.keys())
        return data['data']

def compare_predictions():
    # Load predictions for JSON KV task
    snapkv_preds = load_predictions("/scratch/gpfs/DANQIC/jz4391/HELMET/output/snapkv/32k/Llama-3.1-8B-Instruct/w32_c1024_k7_maxpool/json_kv_16bit_snapkv_32k_Llama-3.1-8B-Instruct_w32_c1024_k7_maxpool_64411397_test_k440_dep6_in32768_size100_shots2_sampFalsemax100min0t0.0p1.0_chatFalse_42.json")
    baseline_preds = load_predictions("/scratch/gpfs/DANQIC/jz4391/HELMET/output/baseline/32k/Llama-3.1-8B-Instruct/16bit/json_kv_16bit_baseline_32k_Llama-3.1-8B-Instruct_63848867_test_k440_dep6_in32768_size100_shots2_sampFalsemax100min0t0.0p1.0_chatFalse_42.json")

    snapkv_dict = {pred['question']: pred for pred in snapkv_preds}
    baseline_dict = {pred['question']: pred for pred in baseline_preds}

    categories = defaultdict(list)

    for question, snapkv_pred in snapkv_dict.items():
        if question not in baseline_dict:
            print(f"Warning: Question not found in baseline predictions: {question[:100]}...")
            continue

        base_pred = baseline_dict[question]

        snapkv_correct = snapkv_pred.get('substring_exact_match', False)
        base_correct = base_pred.get('substring_exact_match', False)

        if snapkv_correct and base_correct:
            cat = 'Both Correct'
        elif not snapkv_correct and not base_correct:
            cat = 'Both Wrong'
        elif snapkv_correct and not base_correct:
            cat = 'SnapKV Wins'
        else:
            cat = 'Baseline Wins'

        categories[cat].append({
            'question': question,
            'ground_truth': snapkv_pred.get('answer', 'N/A'),
            'snapkv_pred': snapkv_pred.get('parsed_output', 'N/A'),
            'baseline_pred': base_pred.get('parsed_output', 'N/A'),
            'input_text': snapkv_pred.get('input_text', 'N/A'),
            'metrics': {
                'snapkv_substring_exact_match': snapkv_correct,
                'baseline_substring_exact_match': base_correct,
            }
        })

    # Summary counts
    print("\nCategory Breakdown:")
    for cat in ['Both Correct', 'Both Wrong', 'SnapKV Wins', 'Baseline Wins']:
        print(f"{cat}: {len(categories[cat])}")

    # Save full details
    with open('json_kv_snapkv_bidirectional_analysis.json', 'w') as f:
        json.dump(categories, f, indent=2)

    # Print examples
    print("\nSample examples from each category:")
    for cat in ['Both Correct', 'Both Wrong', 'SnapKV Wins', 'Baseline Wins']:
        print(f"\nCategory: {cat}")
        examples = categories[cat]
        for i, ex in enumerate(random.sample(examples, min(5, len(examples)))):
            print(f"\nExample {i+1}:")
            print(f"Question: {ex['question']}")
            print(f"Ground Truth: {ex['ground_truth']}")
            print(f"SnapKV: {ex['snapkv_pred']}")
            print(f"Baseline: {ex['baseline_pred']}")
            print(f"Substring Exact Match - SnapKV: {ex['metrics']['snapkv_substring_exact_match']}, Baseline: {ex['metrics']['baseline_substring_exact_match']}")

if __name__ == "__main__":
    compare_predictions()
