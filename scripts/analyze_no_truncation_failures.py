import json
import sys
from pathlib import Path

# Add the path to our truncation analysis functions
sys.path.append('scripts')
from analyze_truncation_comprehensive import check_for_truncation, get_failure_mode_description

def analyze_no_truncation_failures(config_file, config_name):
    """
    Analyze failure modes that are classified as 'no truncation detected' for a specific configuration.
    """
    try:
        with open(config_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {config_file}: {e}")
        return
    
    # Extract failure cases from the detailed results
    detailed_results = data.get('detailed_results', {})
    baseline_wins = detailed_results.get('Baseline Wins', [])
    both_wrong = detailed_results.get('Both Wrong', [])
    
    # Combine all cases where SnapKV could have failed
    all_failure_cases = baseline_wins + both_wrong
    
    print(f"\nAnalyzing 'No Truncation Detected' failures for {config_name}")
    print(f"{'='*70}")
    print(f"Total failure cases: {len(all_failure_cases)}")
    
    no_truncation_failures = []
    truncation_stats = {}
    
    for i, example in enumerate(all_failure_cases):
        if isinstance(example, dict):
            snapkv_pred = example.get('snapkv_pred', '')
            question = example.get('question', '')
            
            # Check for truncation
            is_truncated, truncation_type = check_for_truncation(snapkv_pred)
            
            # Track truncation statistics
            if truncation_type not in truncation_stats:
                truncation_stats[truncation_type] = 0
            truncation_stats[truncation_type] += 1
            
            # Focus on "no truncation detected" cases
            if truncation_type == 'no_truncation_detected':
                failure_category = 'baseline_wins' if example in baseline_wins else 'both_wrong'
                
                no_truncation_failures.append({
                    'example_index': i,
                    'question': question,
                    'failure_category': failure_category,
                    'snapkv_pred': snapkv_pred,
                    'baseline_pred': example.get('baseline_pred', 'N/A'),
                    'ground_truth': example.get('ground_truth', 'N/A'),
                    'snapkv_ndcg': example['metrics'].get('snapkv_NDCG@10', False),
                    'baseline_ndcg': example['metrics'].get('baseline_NDCG@10', False)
                })
    
    print(f"\nTruncation Analysis Breakdown:")
    print(f"{'Failure Type':<40} {'Count':<8} {'Percentage':<12}")
    print("-" * 65)
    
    total_failures = len(all_failure_cases)
    for failure_type, count in sorted(truncation_stats.items()):
        percentage = (count / total_failures * 100) if total_failures > 0 else 0
        description = get_failure_mode_description(failure_type)
        print(f"{description:<40} {count:<8} {percentage:>8.1f}%")
    
    print(f"\nFound {len(no_truncation_failures)} cases with 'No Truncation Detected'")
    print(f"This represents {(len(no_truncation_failures)/total_failures*100):.1f}% of all failures")
    
    # Show examples of "no truncation detected" failures
    print(f"\n{'='*70}")
    print("EXAMPLES OF 'NO TRUNCATION DETECTED' FAILURES")
    print(f"{'='*70}")
    
    for i, failure in enumerate(no_truncation_failures[:10]):  # Show first 10 examples
        print(f"\nExample {i+1}:")
        print(f"Question: {failure['question']}")
        print(f"Failure Category: {failure['failure_category']}")
        print(f"SnapKV NDCG@10: {failure['snapkv_ndcg']}")
        print(f"Baseline NDCG@10: {failure['baseline_ndcg']}")
        print(f"Ground Truth Ranking: {failure['ground_truth'][:200]}...")
        print(f"SnapKV Prediction: {failure['snapkv_pred'][:300]}...")
        print(f"Baseline Prediction: {failure['baseline_pred'][:200]}...")
        print("-" * 50)
        
        # Try to categorize what went wrong
        analyze_failure_pattern(failure)
        print("-" * 50)
    
    return no_truncation_failures

def analyze_failure_pattern(failure):
    """
    Try to identify what went wrong in cases where there's no truncation.
    """
    snapkv_pred = failure['snapkv_pred']
    baseline_pred = failure['baseline_pred'] 
    ground_truth = failure['ground_truth']
    
    print("Failure Analysis:")
    
    # Check if SnapKV produced any ranking at all
    if 'Ranking:' not in snapkv_pred:
        print("  - SnapKV failed to produce a ranking format")
        return
    
    # Extract ranking from SnapKV prediction
    try:
        snapkv_ranking_line = None
        for line in snapkv_pred.split('\n'):
            if 'Ranking:' in line:
                snapkv_ranking_line = line.replace('Ranking:', '').strip()
                break
        
        if not snapkv_ranking_line:
            print("  - No ranking line found")
            return
            
        # Extract IDs from SnapKV ranking
        snapkv_ids = []
        for id_str in snapkv_ranking_line.split('>'):
            id_clean = id_str.strip()
            if id_clean and id_clean.isdigit():
                snapkv_ids.append(id_clean)
        
        # Extract IDs from ground truth
        ground_truth_ids = []
        for id_str in ground_truth.split('>'):
            id_clean = id_str.strip()
            if id_clean and id_clean.isdigit():
                ground_truth_ids.append(id_clean)
        
        print(f"  - SnapKV produced {len(snapkv_ids)} ranked IDs")
        print(f"  - Ground truth has {len(ground_truth_ids)} IDs")
        
        if len(snapkv_ids) == 0:
            print("  - SnapKV produced no valid IDs")
        elif len(snapkv_ids) < len(ground_truth_ids) * 0.5:
            print("  - SnapKV produced significantly fewer IDs than expected")
        elif len(set(snapkv_ids)) != len(snapkv_ids):
            print("  - SnapKV produced duplicate IDs")
        else:
            # Check if the IDs are at least somewhat relevant
            correct_ids_in_top_10 = len(set(snapkv_ids[:10]) & set(ground_truth_ids[:10]))
            print(f"  - {correct_ids_in_top_10}/10 of SnapKV's top 10 IDs match ground truth top 10")
            
            if correct_ids_in_top_10 == 0:
                print("  - Complete mismatch in top rankings - likely fundamental reasoning error")
            elif correct_ids_in_top_10 < 3:
                print("  - Poor ranking quality - partial reasoning failure")
            else:
                print("  - Some correct rankings but overall poor performance")
                
    except Exception as e:
        print(f"  - Error analyzing ranking: {e}")

def main():
    # Analyze the specific qwen_16k_c1024 configuration
    config_file = 'qualitative/rerank_analysis_results/rerank_analysis_qwen_16k_c1024.json'
    config_name = 'qwen_16k_c1024'
    
    if not Path(config_file).exists():
        print(f"Configuration file not found: {config_file}")
        print("Please run the rerank_snapkv_analysis.py script first.")
        return
    
    no_truncation_failures = analyze_no_truncation_failures(config_file, config_name)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total 'No Truncation Detected' failures analyzed: {len(no_truncation_failures)}")
    print("\nThese failures appear to be due to:")
    print("1. Fundamental reasoning errors (complete topic mismatch)")
    print("2. Poor ranking quality (some relevant IDs but wrong order)")
    print("3. Incomplete responses (too few IDs produced)")
    print("4. Format issues (malformed ranking output)")
    print("\nThese represent cases where SnapKV's failure is NOT due to truncation")
    print("but rather due to other model limitations or context compression issues.")

if __name__ == "__main__":
    main()