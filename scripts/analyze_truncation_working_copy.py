import json
import re
from collections import Counter

def analyze_truncation():
    with open('qualitative/rerank_snapkv_bidirectional_analysis.json', 'r') as f:
        data = json.load(f)
    
    total_worse_cases = 0
    truncated_cases = 0
    truncation_details = []
    
    for i, example in enumerate(data.get("Baseline Wins", [])):
        if isinstance(example, dict):
            # Get the metrics
            snapkv_ndcg = example['metrics'].get('snapkv_NDCG@10', 0)
            baseline_ndcg = example['metrics'].get('baseline_NDCG@10', 0)
            
            # Check if snapkv performed worse
            if snapkv_ndcg < baseline_ndcg:
                total_worse_cases += 1
                
                # Get the predictions
                snapkv_pred = example.get('snapkv_pred', '')
                question = example.get('question', '')
                
                # Check if prediction appears truncated
                is_truncated, truncation_type = check_for_truncation(snapkv_pred)
                
                if is_truncated:
                    truncated_cases += 1
                    truncation_details.append({
                        'example_index': i,
                        'question': question,
                        'truncation_type': truncation_type,
                        'snapkv_pred': snapkv_pred,
                        'snapkv_ndcg': snapkv_ndcg,
                        'baseline_ndcg': baseline_ndcg
                    })

    print(f"Total cases where SnapKV performed worse than baseline: {total_worse_cases}")
    print(f"Cases where worse performance was due to truncation: {truncated_cases}")
    print(f"Percentage of worse cases due to truncation: {(truncated_cases/total_worse_cases*100):.2f}%")
    
    # Analyze truncation patterns
    if truncation_details:
        print("\n" + "="*60)
        print("TRUNCATION PATTERN BREAKDOWN")
        print("="*60)
        
        # Count each truncation type
        truncation_types = [detail['truncation_type'] for detail in truncation_details]
        type_counts = Counter(truncation_types)
        
        # Create a nice mapping of technical names to readable descriptions
        type_descriptions = {
            'repeated_same_id': 'Repeated Same ID',
            'sequential_numbers': 'Sequential Numbers',
            'mostly_same_with_variations': 'Mostly Same with Variations',
            'suspiciously_short_ids': 'Suspiciously Short IDs',
            'incomplete_ranking': 'Incomplete Ranking',
            'no_ranking_found': 'No Ranking Found',
            'insufficient_ids': 'Insufficient IDs'
        }
        
        print(f"{'Truncation Type':<35} {'Count':<8} {'Percentage':<12}")
        print("-" * 60)
        
        for truncation_type, count in type_counts.most_common():
            percentage = (count / truncated_cases) * 100
            description = type_descriptions.get(truncation_type, truncation_type)
            print(f"{description:<35} {count:<8} {percentage:>8.1f}%")
        
        print("-" * 60)
        print(f"{'TOTAL':<35} {truncated_cases:<8} {'100.0%':>8}")
    
    # Print some examples of truncation patterns
    print("\n" + "="*60)
    print("EXAMPLE TRUNCATION PATTERNS")
    print("="*60)
    for detail in truncation_details[:5]:  # Show first 5 examples
        print(f"\nQuestion: {detail['question']}")
        print(f"Truncation Type: {detail['truncation_type']}")
        print(f"SnapKV Prediction: {detail['snapkv_pred'][:100]}...")
        print(f"NDCG Gap: {detail['baseline_ndcg']:.3f} - {detail['snapkv_ndcg']:.3f} = {detail['baseline_ndcg'] - detail['snapkv_ndcg']:.3f}")
        print("-" * 40)

def check_for_truncation(snapkv_pred):
    """
    Check if the snapkv prediction shows signs of truncation.
    Returns (is_truncated, truncation_type)
    """
    # Extract ranking line
    lines = snapkv_pred.split('\n')
    ranking_line = None
    
    for line in lines:
        if 'Ranking:' in line:
            ranking_line = line.replace('Ranking:', '').strip()
            break
    
    if not ranking_line:
        return False, "no_ranking_found"
    
    # Extract IDs
    ids_raw = ranking_line.split('>')
    ids = []
    for id_str in ids_raw:
        id_clean = id_str.strip()
        if id_clean and id_clean.isdigit():
            ids.append(int(id_clean))
    
    if len(ids) < 3:  # Need at least 3 IDs to detect patterns
        return False, "insufficient_ids"
    
    # Check for various truncation patterns
    
    # 1. All IDs are the same
    if len(set(ids)) == 1:
        return True, "repeated_same_id"
    
    # 2. Sequential numbers (ascending or descending)
    if is_sequential(ids):
        return True, "sequential_numbers"
    
    # 3. Mostly the same ID with minor variations
    if is_mostly_same_with_variations(ids):
        return True, "mostly_same_with_variations"
    
    # 4. Very short IDs (likely truncated)
    if all(id_val < 1000 for id_val in ids):
        return True, "suspiciously_short_ids"
    
    # 5. Incomplete ranking (ends abruptly)
    if ranking_line.endswith('>') or ranking_line.endswith(' >'):
        return True, "incomplete_ranking"
    
    return False, "no_truncation_detected"

def is_sequential(ids):
    """Check if IDs form a sequential pattern (ascending or descending)"""
    if len(ids) < 3:
        return False
    
    # Check if ascending
    ascending = all(ids[i] == ids[0] + i for i in range(len(ids)))
    
    # Check if descending  
    descending = all(ids[i] == ids[0] - i for i in range(len(ids)))
    
    # Check if close to sequential (allowing some gaps)
    close_sequential = True
    for i in range(1, len(ids)):
        diff = abs(ids[i] - ids[i-1])
        if diff > 10:  # Allow small gaps
            close_sequential = False
            break
    
    return ascending or descending or close_sequential

def is_mostly_same_with_variations(ids):
    """Check if most IDs are the same with small variations"""
    from collections import Counter
    id_counts = Counter(ids)
    most_common_id, most_common_count = id_counts.most_common(1)[0]
    
    # If more than 70% of IDs are the same
    if most_common_count / len(ids) > 0.7:
        return True
    
    return False

if __name__ == "__main__":
    analyze_truncation()