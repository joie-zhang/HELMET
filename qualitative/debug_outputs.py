import json

def debug_outputs():
    # Load the data
    snapkv_file = "/scratch/gpfs/DANQIC/jz4391/HELMET/output/snapkv/32k/Llama-3.1-8B-Instruct/w32_c1024_k7_maxpool/alce_asqa_165_16bit_snapkv_32k_Llama-3.1-8B-Instruct_w32_c1024_k7_maxpool_64407623_asqa_eval_gtr_top2000_in32768_size100_shots2_sampFalsemax300min0t0.0p1.0_chatTrue_42.json"
    baseline_file = "/scratch/gpfs/DANQIC/jz4391/HELMET/output/baseline/32k/Llama-3.1-8B-Instruct/16bit/alce_asqa_165_16bit_baseline_32k_Llama-3.1-8B-Instruct_63823887_asqa_eval_gtr_top2000_in32768_size100_shots2_sampFalsemax300min0t0.0p1.0_chatTrue_42.json"
    
    with open(snapkv_file, 'r') as f:
        snapkv_data = json.load(f)
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)
    
    # Look at the first few examples
    # target_question = "Who wrote give me a home among the gum trees?"
    target_question = "How many seats in rajya sabha in assam?"
    
    print("=== LOOKING FOR SPECIFIC QUESTION ===")
    for i, example in enumerate(snapkv_data['data']):
        if example['question'] == target_question:
            print(f"SnapKV Example {i}:")
            print(f"Question: {example['question']}")
            print(f"QA Pairs:")
            for qa in example['qa_pairs']:
                print(f"  Q: {qa['question']}")
                print(f"  A: {qa['short_answers']}")
            print(f"Output: {example['output'][:500]}...")
            print()
            break
    
    for i, example in enumerate(baseline_data['data']):
        if example['question'] == target_question:
            print(f"Baseline Example {i}:")
            print(f"Question: {example['question']}")
            print(f"QA Pairs:")
            for qa in example['qa_pairs']:
                print(f"  Q: {qa['question']}")
                print(f"  A: {qa['short_answers']}")
            print(f"Output: {example['output'][:500]}...")
            print()
            break
    
    # Check if the examples are in the same order
    print("=== CHECKING ORDER MATCHING ===")
    print("First 3 questions in SnapKV:")
    for i in range(3):
        print(f"  {i}: {snapkv_data['data'][i]['question']}")
    
    print("First 3 questions in Baseline:")
    for i in range(3):
        print(f"  {i}: {baseline_data['data'][i]['question']}")

if __name__ == "__main__":
    debug_outputs()