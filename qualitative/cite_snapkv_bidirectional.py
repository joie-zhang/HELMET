import json
from collections import defaultdict

def load_predictions(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        print(f"\nFile: {file_path}")
        print("Data type:", type(data))
        print("Keys:", data.keys())
        return data['data']

def compare_predictions_bidirectional():
    # Load predictions for cite task
    snapkv_preds = load_predictions("/scratch/gpfs/DANQIC/jz4391/HELMET/output/snapkv/32k/Llama-3.1-8B-Instruct/w32_c1024_k7_maxpool/alce_asqa_165_16bit_snapkv_32k_Llama-3.1-8B-Instruct_w32_c1024_k7_maxpool_64407623_asqa_eval_gtr_top2000_in32768_size100_shots2_sampFalsemax300min0t0.0p1.0_chatTrue_42.json")
    baseline_preds = load_predictions("/scratch/gpfs/DANQIC/jz4391/HELMET/output/baseline/32k/Llama-3.1-8B-Instruct/16bit/alce_asqa_165_16bit_baseline_32k_Llama-3.1-8B-Instruct_63823887_asqa_eval_gtr_top2000_in32768_size100_shots2_sampFalsemax300min0t0.0p1.0_chatTrue_42.json")
    
    # Debug print structure of first items
    print("\nSnapKV first item structure:")
    print("Keys available:", snapkv_preds[0].keys())
    print("\nBaseline first item structure:")
    print("Keys available:", baseline_preds[0].keys())
    
    # Create dictionaries mapping question to predictions for easier lookup
    snapkv_dict = {}
    baseline_dict = {}
    
    for pred in snapkv_preds:
        question = pred['question']
        snapkv_dict[question] = pred
        
    for pred in baseline_preds:
        question = pred['question']
        baseline_dict[question] = pred
    
    # Two types of differences
    baseline_wins = []  # Cases where baseline is correct but SnapKV is wrong
    snapkv_wins = []    # Cases where SnapKV is correct but baseline is wrong
    
    # Statistics
    stats = {
        'total_compared': 0,
        'both_correct': 0,
        'both_wrong': 0,
        'baseline_wins': 0,
        'snapkv_wins': 0,
        'missing_questions': 0
    }
    
    # Iterate through all questions present in both datasets
    common_questions = set(snapkv_dict.keys()) & set(baseline_dict.keys())
    
    for question in common_questions:
        snapkv_pred = snapkv_dict[question]
        base_pred = baseline_dict[question]
        
        stats['total_compared'] += 1
        
        try:
            # You can choose different metrics for "correctness"
            # Option 1: Use string exact match (str_em)
            snapkv_correct = snapkv_pred.get('str_em', 0) > 0  # Assuming > 0 means correct
            base_correct = base_pred.get('str_em', 0) > 0
            
            # Option 2: You could also use other metrics like citation_rec or citation_prec
            # snapkv_correct = snapkv_pred.get('citation_rec', 0) > some_threshold
            # base_correct = base_pred.get('citation_rec', 0) > some_threshold
            
            if base_correct and snapkv_correct:
                stats['both_correct'] += 1
            elif not base_correct and not snapkv_correct:
                stats['both_wrong'] += 1
            elif base_correct and not snapkv_correct:
                # Baseline wins
                stats['baseline_wins'] += 1
                baseline_wins.append({
                    'question': question,
                    'ground_truth': snapkv_pred.get('answer', 'N/A'),
                    'snapkv_pred': snapkv_pred.get('output', 'N/A'),
                    'baseline_pred': base_pred.get('output', 'N/A'),
                    'snapkv_parsed': snapkv_pred.get('parsed_output', 'N/A'),
                    'baseline_parsed': base_pred.get('parsed_output', 'N/A'),
                    'metrics': {
                        'snapkv_str_em': snapkv_pred.get('str_em', 'N/A'),
                        'baseline_str_em': base_pred.get('str_em', 'N/A'),
                        'snapkv_citation_rec': snapkv_pred.get('citation_rec', 'N/A'),
                        'baseline_citation_rec': base_pred.get('citation_rec', 'N/A'),
                        'snapkv_citation_prec': snapkv_pred.get('citation_prec', 'N/A'),
                        'baseline_citation_prec': base_pred.get('citation_prec', 'N/A'),
                        'snapkv_rougeLsum': snapkv_pred.get('rougeLsum', 'N/A'),
                        'baseline_rougeLsum': base_pred.get('rougeLsum', 'N/A'),
                    }
                })
            elif snapkv_correct and not base_correct:
                # SnapKV wins
                stats['snapkv_wins'] += 1
                snapkv_wins.append({
                    'question': question,
                    'ground_truth': snapkv_pred.get('answer', 'N/A'),
                    'snapkv_pred': snapkv_pred.get('output', 'N/A'),
                    'baseline_pred': base_pred.get('output', 'N/A'),
                    'snapkv_parsed': snapkv_pred.get('parsed_output', 'N/A'),
                    'baseline_parsed': base_pred.get('parsed_output', 'N/A'),
                    'metrics': {
                        'snapkv_str_em': snapkv_pred.get('str_em', 'N/A'),
                        'baseline_str_em': base_pred.get('str_em', 'N/A'),
                        'snapkv_citation_rec': snapkv_pred.get('citation_rec', 'N/A'),
                        'baseline_citation_rec': base_pred.get('citation_rec', 'N/A'),
                        'snapkv_citation_prec': snapkv_pred.get('citation_prec', 'N/A'),
                        'baseline_citation_prec': base_pred.get('citation_prec', 'N/A'),
                        'snapkv_rougeLsum': snapkv_pred.get('rougeLsum', 'N/A'),
                        'baseline_rougeLsum': base_pred.get('rougeLsum', 'N/A'),
                    }
                })
                
        except Exception as e:
            print(f"\nError processing question: {question[:100]}...")
            print("Error:", str(e))
            print("SnapKV item keys:", snapkv_pred.keys())
            print("Baseline item keys:", base_pred.keys())
            continue
    
    # Report statistics
    print(f"\n=== COMPARISON STATISTICS ===")
    print(f"Total questions compared: {stats['total_compared']}")
    print(f"Both models correct: {stats['both_correct']} ({stats['both_correct']/stats['total_compared']*100:.1f}%)")
    print(f"Both models wrong: {stats['both_wrong']} ({stats['both_wrong']/stats['total_compared']*100:.1f}%)")
    print(f"Baseline wins (correct when SnapKV wrong): {stats['baseline_wins']} ({stats['baseline_wins']/stats['total_compared']*100:.1f}%)")
    print(f"SnapKV wins (correct when Baseline wrong): {stats['snapkv_wins']} ({stats['snapkv_wins']/stats['total_compared']*100:.1f}%)")
    
    # Save results
    results = {
        'statistics': stats,
        'baseline_wins': baseline_wins,
        'snapkv_wins': snapkv_wins
    }
    
    with open('cite_bidirectional_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print examples from both categories
    print(f"\n=== EXAMPLES WHERE BASELINE WINS (Baseline correct, SnapKV wrong) ===")
    for i, diff in enumerate(baseline_wins[:3]):  # Show first 3 examples
        print(f"\nExample {i+1}:")
        print(f"Question: {diff['question'][:200]}...")
        print(f"Ground Truth: {str(diff['ground_truth'])[:200]}...")
        print(f"SnapKV: {str(diff['snapkv_pred'])[:200]}...")
        print(f"Baseline: {str(diff['baseline_pred'])[:200]}...")
        print(f"Metrics - SnapKV str_em: {diff['metrics']['snapkv_str_em']}, Baseline str_em: {diff['metrics']['baseline_str_em']}")
        print(f"Citation Recall - SnapKV: {diff['metrics']['snapkv_citation_rec']}, Baseline: {diff['metrics']['baseline_citation_rec']}")
    
    print(f"\n=== EXAMPLES WHERE SNAPKV WINS (SnapKV correct, Baseline wrong) ===")
    for i, diff in enumerate(snapkv_wins[:3]):  # Show first 3 examples
        print(f"\nExample {i+1}:")
        print(f"Question: {diff['question'][:200]}...")
        print(f"Ground Truth: {str(diff['ground_truth'])[:200]}...")
        print(f"SnapKV: {str(diff['snapkv_pred'])[:200]}...")
        print(f"Baseline: {str(diff['baseline_pred'])[:200]}...")
        print(f"Metrics - SnapKV str_em: {diff['metrics']['snapkv_str_em']}, Baseline str_em: {diff['metrics']['baseline_str_em']}")
        print(f"Citation Recall - SnapKV: {diff['metrics']['snapkv_citation_rec']}, Baseline: {diff['metrics']['baseline_citation_rec']}")
    
    return results

def analyze_patterns(results):
    """Analyze patterns in the differences"""
    print(f"\n=== PATTERN ANALYSIS ===")
    
    baseline_wins = results['baseline_wins']
    snapkv_wins = results['snapkv_wins']
    
    # Analyze citation metrics
    if baseline_wins:
        baseline_cit_rec_avg = sum([float(d['metrics']['baseline_citation_rec']) for d in baseline_wins if d['metrics']['baseline_citation_rec'] != 'N/A']) / len(baseline_wins)
        snapkv_cit_rec_avg = sum([float(d['metrics']['snapkv_citation_rec']) for d in baseline_wins if d['metrics']['snapkv_citation_rec'] != 'N/A']) / len(baseline_wins)
        print(f"When Baseline wins - Avg Citation Recall: Baseline {baseline_cit_rec_avg:.2f}, SnapKV {snapkv_cit_rec_avg:.2f}")
    
    if snapkv_wins:
        baseline_cit_rec_avg = sum([float(d['metrics']['baseline_citation_rec']) for d in snapkv_wins if d['metrics']['baseline_citation_rec'] != 'N/A']) / len(snapkv_wins)
        snapkv_cit_rec_avg = sum([float(d['metrics']['snapkv_citation_rec']) for d in snapkv_wins if d['metrics']['snapkv_citation_rec'] != 'N/A']) / len(snapkv_wins)
        print(f"When SnapKV wins - Avg Citation Recall: Baseline {baseline_cit_rec_avg:.2f}, SnapKV {snapkv_cit_rec_avg:.2f}")
    
    # Analyze question lengths
    if baseline_wins:
        baseline_q_lens = [len(d['question'].split()) for d in baseline_wins]
        print(f"When Baseline wins - Avg question length: {sum(baseline_q_lens)/len(baseline_q_lens):.1f} words")
    
    if snapkv_wins:
        snapkv_q_lens = [len(d['question'].split()) for d in snapkv_wins]
        print(f"When SnapKV wins - Avg question length: {sum(snapkv_q_lens)/len(snapkv_q_lens):.1f} words")

if __name__ == "__main__":
    results = compare_predictions_bidirectional()
    analyze_patterns(results)