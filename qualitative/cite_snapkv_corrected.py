import json
from collections import defaultdict

def exact_presence(short_answers, context):
    """Check if any of the short answers are present in the context"""
    context = context.lower()
    for ans in short_answers:
        if ans.lower() in context:
            return True
    return False

def compute_str_em_for_example(example):
    """Compute STR-EM for a single example"""
    if 'qa_pairs' not in example or example['qa_pairs'] is None:
        return 0
    
    output = example['output']
    correct_answers = 0
    total_answers = len(example['qa_pairs'])
    
    for qa_pair in example['qa_pairs']:
        if exact_presence(qa_pair['short_answers'], output):
            correct_answers += 1
    
    return correct_answers / total_answers if total_answers > 0 else 0

def load_predictions(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data['data']

def compare_predictions_bidirectional():
    # Load predictions
    snapkv_preds = load_predictions("/scratch/gpfs/DANQIC/jz4391/HELMET/output/snapkv/32k/Llama-3.1-8B-Instruct/w32_c1024_k7_maxpool/alce_asqa_165_16bit_snapkv_32k_Llama-3.1-8B-Instruct_w32_c1024_k7_maxpool_64407623_asqa_eval_gtr_top2000_in32768_size100_shots2_sampFalsemax300min0t0.0p1.0_chatTrue_42.json")
    baseline_preds = load_predictions("/scratch/gpfs/DANQIC/jz4391/HELMET/output/baseline/32k/Llama-3.1-8B-Instruct/16bit/alce_asqa_165_16bit_baseline_32k_Llama-3.1-8B-Instruct_63823887_asqa_eval_gtr_top2000_in32768_size100_shots2_sampFalsemax300min0t0.0p1.0_chatTrue_42.json")
    
    # Create dictionaries mapping question to predictions
    snapkv_dict = {pred['question']: pred for pred in snapkv_preds}
    baseline_dict = {pred['question']: pred for pred in baseline_preds}
    
    baseline_wins = []
    snapkv_wins = []
    
    stats = {
        'total_compared': 0,
        'both_correct': 0,
        'both_wrong': 0,
        'baseline_wins': 0,
        'snapkv_wins': 0,
        'both_partial': 0
    }
    
    common_questions = set(snapkv_dict.keys()) & set(baseline_dict.keys())
    
    print(f"Comparing {len(common_questions)} common questions...")
    
    for question in common_questions:
        snapkv_pred = snapkv_dict[question]
        base_pred = baseline_dict[question]
        
        stats['total_compared'] += 1
        
        try:
            # Compute STR-EM for both predictions
            snapkv_str_em = compute_str_em_for_example(snapkv_pred)
            base_str_em = compute_str_em_for_example(base_pred)
            
            # Define "correct" as STR-EM = 1.0 (all QA pairs answered correctly)
            snapkv_correct = snapkv_str_em == 1.0
            base_correct = base_str_em == 1.0
            
            # Alternative: Use a threshold (uncomment if you want to be more lenient)
            # threshold = 0.5
            # snapkv_correct = snapkv_str_em >= threshold
            # base_correct = base_str_em >= threshold
            
            if base_correct and snapkv_correct:
                stats['both_correct'] += 1
            elif not base_correct and not snapkv_correct:
                stats['both_wrong'] += 1
                # Check if both have some partial credit
                if snapkv_str_em > 0 and base_str_em > 0:
                    stats['both_partial'] += 1
            elif base_correct and not snapkv_correct:
                stats['baseline_wins'] += 1
                
                baseline_wins.append({
                    'question': question,
                    'ground_truth': snapkv_pred.get('answer', 'N/A'),
                    'snapkv_pred': snapkv_pred.get('output', 'N/A'),
                    'baseline_pred': base_pred.get('output', 'N/A'),
                    'snapkv_parsed': snapkv_pred.get('parsed_output', 'N/A'),
                    'baseline_parsed': base_pred.get('parsed_output', 'N/A'),
                    'qa_pairs': snapkv_pred.get('qa_pairs', []),
                    'metrics': {
                        'snapkv_str_em': snapkv_str_em,
                        'baseline_str_em': base_str_em,
                        'snapkv_f1': snapkv_pred.get('f1', 0),
                        'baseline_f1': base_pred.get('f1', 0),
                        'snapkv_rougeL_f1': snapkv_pred.get('rougeL_f1', 0),
                        'baseline_rougeL_f1': base_pred.get('rougeL_f1', 0),
                        'qa_pairs_count': len(snapkv_pred.get('qa_pairs', [])),
                    }
                })
            elif snapkv_correct and not base_correct:
                stats['snapkv_wins'] += 1
                
                snapkv_wins.append({
                    'question': question,
                    'ground_truth': snapkv_pred.get('answer', 'N/A'),
                    'snapkv_pred': snapkv_pred.get('output', 'N/A'),
                    'baseline_pred': base_pred.get('output', 'N/A'),
                    'snapkv_parsed': snapkv_pred.get('parsed_output', 'N/A'),
                    'baseline_parsed': base_pred.get('parsed_output', 'N/A'),
                    'qa_pairs': snapkv_pred.get('qa_pairs', []),
                    'metrics': {
                        'snapkv_str_em': snapkv_str_em,
                        'baseline_str_em': base_str_em,
                        'snapkv_f1': snapkv_pred.get('f1', 0),
                        'baseline_f1': base_pred.get('f1', 0),
                        'snapkv_rougeL_f1': snapkv_pred.get('rougeL_f1', 0),
                        'baseline_rougeL_f1': base_pred.get('rougeL_f1', 0),
                        'qa_pairs_count': len(snapkv_pred.get('qa_pairs', [])),
                    }
                })
                
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            continue
    
    # Print results
    print(f"\n=== COMPARISON STATISTICS (STR-EM = 1.0) ===")
    print(f"Total questions compared: {stats['total_compared']}")
    print(f"Both models perfect (STR-EM = 1.0): {stats['both_correct']} ({stats['both_correct']/stats['total_compared']*100:.1f}%)")
    print(f"Both models imperfect (STR-EM < 1.0): {stats['both_wrong']} ({stats['both_wrong']/stats['total_compared']*100:.1f}%)")
    print(f"  - Of those, both have partial credit: {stats['both_partial']} ({stats['both_partial']/stats['total_compared']*100:.1f}%)")
    print(f"Baseline wins (perfect vs imperfect): {stats['baseline_wins']} ({stats['baseline_wins']/stats['total_compared']*100:.1f}%)")
    print(f"SnapKV wins (perfect vs imperfect): {stats['snapkv_wins']} ({stats['snapkv_wins']/stats['total_compared']*100:.1f}%)")
    
    # Also show STR-EM distribution
    all_snapkv_str_em = [compute_str_em_for_example(pred) for pred in snapkv_preds]
    all_baseline_str_em = [compute_str_em_for_example(pred) for pred in baseline_preds]
    
    print(f"\n=== STR-EM DISTRIBUTION ===")
    print(f"SnapKV - Mean STR-EM: {sum(all_snapkv_str_em)/len(all_snapkv_str_em):.3f}")
    print(f"Baseline - Mean STR-EM: {sum(all_baseline_str_em)/len(all_baseline_str_em):.3f}")
    print(f"SnapKV - Perfect scores (1.0): {sum(1 for x in all_snapkv_str_em if x == 1.0)}")
    print(f"Baseline - Perfect scores (1.0): {sum(1 for x in all_baseline_str_em if x == 1.0)}")
    
    # Save results
    results = {
        'statistics': stats,
        'baseline_wins': baseline_wins,
        'snapkv_wins': snapkv_wins,
        'str_em_distribution': {
            'snapkv_mean': sum(all_snapkv_str_em)/len(all_snapkv_str_em),
            'baseline_mean': sum(all_baseline_str_em)/len(all_baseline_str_em),
            'snapkv_perfect': sum(1 for x in all_snapkv_str_em if x == 1.0),
            'baseline_perfect': sum(1 for x in all_baseline_str_em if x == 1.0),
        }
    }
    
    with open('cite_bidirectional_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print examples
    print(f"\n=== BASELINE WINS (first 3) ===")
    for i, diff in enumerate(baseline_wins[:3]):
        print(f"\nExample {i+1}:")
        print(f"Question: {diff['question']}")
        print(f"STR-EM - SnapKV: {diff['metrics']['snapkv_str_em']:.2f}, Baseline: {diff['metrics']['baseline_str_em']:.2f}")
        print(f"F1 - SnapKV: {diff['metrics']['snapkv_f1']:.2f}, Baseline: {diff['metrics']['baseline_f1']:.2f}")
        print(f"QA Pairs to answer: {diff['metrics']['qa_pairs_count']}")
        print(f"QA Pairs: {[qa['question'] + ' -> ' + str(qa['short_answers']) for qa in diff['qa_pairs']]}")
        print(f"SnapKV output: {str(diff['snapkv_pred'])}...")
        print(f"Baseline output: {str(diff['baseline_pred'])}...")
    
    print(f"\n=== SNAPKV WINS (first 3) ===")
    for i, diff in enumerate(snapkv_wins[:3]):
        print(f"\nExample {i+1}:")
        print(f"Question: {diff['question']}")
        print(f"STR-EM - SnapKV: {diff['metrics']['snapkv_str_em']:.2f}, Baseline: {diff['metrics']['baseline_str_em']:.2f}")
        print(f"F1 - SnapKV: {diff['metrics']['snapkv_f1']:.2f}, Baseline: {diff['metrics']['baseline_f1']:.2f}")
        print(f"QA Pairs to answer: {diff['metrics']['qa_pairs_count']}")
        print(f"QA Pairs: {[qa['question'] + ' -> ' + str(qa['short_answers']) for qa in diff['qa_pairs']]}")
        print(f"SnapKV output: {str(diff['snapkv_pred'])}...")
        print(f"Baseline output: {str(diff['baseline_pred'])}...")
    
    return results

if __name__ == "__main__":
    results = compare_predictions_bidirectional()