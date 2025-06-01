import json

def investigate_baseline_issue():
    # Load both datasets
    snapkv_file = "/scratch/gpfs/DANQIC/jz4391/HELMET/output/snapkv/32k/Llama-3.1-8B-Instruct/w32_c1024_k7_maxpool/alce_asqa_165_16bit_snapkv_32k_Llama-3.1-8B-Instruct_w32_c1024_k7_maxpool_64407623_asqa_eval_gtr_top2000_in32768_size100_shots2_sampFalsemax300min0t0.0p1.0_chatTrue_42.json"
    baseline_file = "/scratch/gpfs/DANQIC/jz4391/HELMET/output/baseline/32k/Llama-3.1-8B-Instruct/16bit/alce_asqa_165_16bit_baseline_32k_Llama-3.1-8B-Instruct_63823887_asqa_eval_gtr_top2000_in32768_size100_shots2_sampFalsemax300min0t0.0p1.0_chatTrue_42.json"
    
    with open(snapkv_file, 'r') as f:
        snapkv_data = json.load(f)
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)
    
    print("=== SYSTEMATIC COMPARISON OF FIRST 5 EXAMPLES ===")
    
    for i in range(5):
        snapkv_ex = snapkv_data['data'][i]
        baseline_ex = baseline_data['data'][i]
        
        print(f"\n--- Example {i+1} ---")
        print(f"Question: {snapkv_ex['question']}")
        print(f"Questions match: {snapkv_ex['question'] == baseline_ex['question']}")
        
        # Check if any expected answers appear in outputs
        expected_answers = []
        for qa in snapkv_ex['qa_pairs']:
            expected_answers.extend(qa['short_answers'])
        
        snapkv_output = snapkv_ex['output'].lower()
        baseline_output = baseline_ex['output'].lower()
        
        snapkv_has_answers = [ans.lower() in snapkv_output for ans in expected_answers]
        baseline_has_answers = [ans.lower() in baseline_output for ans in expected_answers]
        
        print(f"Expected answers: {expected_answers}")
        print(f"SnapKV contains answers: {snapkv_has_answers} (any: {any(snapkv_has_answers)})")
        print(f"Baseline contains answers: {baseline_has_answers} (any: {any(baseline_has_answers)})")
        
        print(f"SnapKV output (first 150 chars): {snapkv_ex['output'][:150]}...")
        print(f"Baseline output (first 150 chars): {baseline_ex['output'][:150]}...")
        
        # Check if outputs are suspiciously similar or different
        if snapkv_ex['output'] == baseline_ex['output']:
            print("⚠️  IDENTICAL OUTPUTS!")
        elif len(set(snapkv_ex['output'].split()) & set(baseline_ex['output'].split())) > 10:
            print("⚠️  OUTPUTS SHARE MANY WORDS!")
        else:
            print("✓ Outputs are different")
    
    print("\n=== CHECKING OVERALL STATISTICS ===")
    
    # Check the score files too
    try:
        with open(snapkv_file + ".score", 'r') as f:
            snapkv_scores = json.load(f)
        with open(baseline_file + ".score", 'r') as f:
            baseline_scores = json.load(f)
            
        print(f"SnapKV STR-EM: {snapkv_scores.get('str_em', 'N/A')}")
        print(f"Baseline STR-EM: {baseline_scores.get('str_em', 'N/A')}")
        print(f"SnapKV Citation Recall: {snapkv_scores.get('citation_rec', 'N/A')}")
        print(f"Baseline Citation Recall: {baseline_scores.get('citation_rec', 'N/A')}")
        
    except FileNotFoundError as e:
        print(f"Could not load score files: {e}")
    
    # Check averaged metrics from main files
    print(f"\nSnapKV averaged STR-EM equivalent: {snapkv_data['averaged_metrics'].get('exact_match', 'N/A')}")
    print(f"Baseline averaged STR-EM equivalent: {baseline_data['averaged_metrics'].get('exact_match', 'N/A')}")

if __name__ == "__main__":
    investigate_baseline_issue()