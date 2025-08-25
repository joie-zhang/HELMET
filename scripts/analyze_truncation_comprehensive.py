import json
import re
import os
import glob
from collections import Counter, defaultdict
from pathlib import Path

def analyze_truncation_for_config(analysis_file_path, config_name):
    """
    Analyze truncation patterns for a specific configuration.
    Returns detailed truncation analysis results.
    """
    try:
        with open(analysis_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Analysis file not found: {analysis_file_path}")
        return None
    except Exception as e:
        print(f"Error loading {analysis_file_path}: {e}")
        return None
    
    # Extract failure cases from the detailed results
    detailed_results = data.get('detailed_results', {})
    baseline_wins = detailed_results.get('Baseline Wins', [])
    both_wrong = detailed_results.get('Both Wrong', [])
    
    # Combine all cases where SnapKV could have failed
    all_failure_cases = baseline_wins + both_wrong
    
    total_failure_cases = len(all_failure_cases)
    truncated_cases = 0
    truncation_details = []
    failure_mode_counts = Counter()
    
    print(f"\nAnalyzing {config_name}...")
    print(f"Total failure cases (Baseline Wins + Both Wrong): {total_failure_cases}")
    
    for i, example in enumerate(all_failure_cases):
        if isinstance(example, dict):
            # Get the predictions
            snapkv_pred = example.get('snapkv_pred', '')
            question = example.get('question', '')
            
            # Get metrics
            snapkv_ndcg = example['metrics'].get('snapkv_NDCG@10', False)
            baseline_ndcg = example['metrics'].get('baseline_NDCG@10', False)
            
            # Determine failure type
            if example in baseline_wins:
                failure_category = 'baseline_wins'
            else:
                failure_category = 'both_wrong'
            
            # Check if prediction appears truncated
            is_truncated, truncation_type = check_for_truncation(snapkv_pred)
            
            failure_mode_counts[truncation_type] += 1
            
            if is_truncated:
                truncated_cases += 1
                truncation_details.append({
                    'config': config_name,
                    'example_index': i,
                    'question': question,
                    'truncation_type': truncation_type,
                    'failure_category': failure_category,
                    'snapkv_pred': snapkv_pred,
                    'snapkv_ndcg': snapkv_ndcg,
                    'baseline_ndcg': baseline_ndcg
                })
    
    # Calculate percentages
    truncation_percentage = (truncated_cases / total_failure_cases * 100) if total_failure_cases > 0 else 0
    
    results = {
        'config_name': config_name,
        'total_failure_cases': total_failure_cases,
        'truncated_cases': truncated_cases,
        'truncation_percentage': truncation_percentage,
        'failure_mode_counts': dict(failure_mode_counts),
        'truncation_details': truncation_details
    }
    
    print(f"Truncated cases: {truncated_cases} ({truncation_percentage:.1f}%)")
    
    return results

def run_comprehensive_truncation_analysis():
    """Run truncation analysis across all configurations."""
    
    # Directory containing the analysis results
    analysis_dir = 'qualitative/rerank_analysis_results'
    
    if not os.path.exists(analysis_dir):
        print(f"Error: Analysis directory not found: {analysis_dir}")
        print("Please run the rerank_snapkv_analysis.py script first to generate the analysis files.")
        return
    
    # Find all analysis files
    analysis_files = glob.glob(os.path.join(analysis_dir, 'rerank_analysis_*.json'))
    
    if not analysis_files:
        print(f"No analysis files found in {analysis_dir}")
        return
    
    # Aggregate results across all configurations
    all_config_results = {}
    aggregated_failure_modes = Counter()
    aggregated_truncation_details = []
    total_all_failure_cases = 0
    total_all_truncated_cases = 0
    
    print("="*80)
    print("COMPREHENSIVE TRUNCATION ANALYSIS ACROSS ALL CONFIGURATIONS")
    print("="*80)
    
    # Process each configuration
    for analysis_file in sorted(analysis_files):
        # Extract config name from filename
        filename = os.path.basename(analysis_file)
        config_name = filename.replace('rerank_analysis_', '').replace('.json', '')
        
        # Analyze truncation for this config
        config_results = analyze_truncation_for_config(analysis_file, config_name)
        
        if config_results:
            all_config_results[config_name] = config_results
            
            # Aggregate statistics
            aggregated_failure_modes.update(config_results['failure_mode_counts'])
            aggregated_truncation_details.extend(config_results['truncation_details'])
            total_all_failure_cases += config_results['total_failure_cases']
            total_all_truncated_cases += config_results['truncated_cases']
    
    # Create output directory for truncation analysis
    output_dir = 'qualitative/truncation_analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate individual configuration reports
    print(f"\n{'='*80}")
    print("INDIVIDUAL CONFIGURATION BREAKDOWN")
    print(f"{'='*80}")
    
    for config_name, results in all_config_results.items():
        print(f"\n{config_name.upper()}:")
        print(f"  Total Failure Cases: {results['total_failure_cases']}")
        print(f"  Truncated Cases: {results['truncated_cases']} ({results['truncation_percentage']:.1f}%)")
        
        # Print top failure modes for this config
        if results['failure_mode_counts']:
            print(f"  Top Failure Modes:")
            for mode, count in Counter(results['failure_mode_counts']).most_common(3):
                percentage = (count / results['total_failure_cases'] * 100) if results['total_failure_cases'] > 0 else 0
                description = get_failure_mode_description(mode)
                print(f"    {description}: {count} ({percentage:.1f}%)")
    
    # Generate aggregated report
    print(f"\n{'='*80}")
    print("AGGREGATED FAILURE MODE BREAKDOWN (ALL CONFIGURATIONS)")
    print(f"{'='*80}")
    
    total_aggregated_percentage = (total_all_truncated_cases / total_all_failure_cases * 100) if total_all_failure_cases > 0 else 0
    
    print(f"Total Failure Cases Across All Configs: {total_all_failure_cases}")
    print(f"Total Truncated Cases Across All Configs: {total_all_truncated_cases} ({total_aggregated_percentage:.1f}%)")
    
    print(f"\nFailure Mode Breakdown:")
    print(f"{'Failure Mode':<40} {'Count':<8} {'Percentage':<12}")
    print("-" * 65)
    
    for failure_mode, count in aggregated_failure_modes.most_common():
        percentage = (count / total_all_failure_cases) * 100 if total_all_failure_cases > 0 else 0
        description = get_failure_mode_description(failure_mode)
        print(f"{description:<40} {count:<8} {percentage:>8.1f}%")
    
    print("-" * 65)
    print(f"{'TOTAL':<40} {total_all_failure_cases:<8} {'100.0%':>8}")
    
    # Save detailed results
    comprehensive_results = {
        'summary': {
            'total_failure_cases': total_all_failure_cases,
            'total_truncated_cases': total_all_truncated_cases,
            'truncation_percentage': total_aggregated_percentage,
            'num_configurations': len(all_config_results)
        },
        'aggregated_failure_modes': dict(aggregated_failure_modes),
        'individual_config_results': all_config_results,
        'all_truncation_details': aggregated_truncation_details
    }
    
    # Save comprehensive results
    comprehensive_path = os.path.join(output_dir, 'comprehensive_truncation_analysis.json')
    with open(comprehensive_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    # Save individual config summaries
    config_summary_path = os.path.join(output_dir, 'config_breakdown_summary.json')
    config_summary = {
        config: {
            'total_failure_cases': results['total_failure_cases'],
            'truncated_cases': results['truncated_cases'],
            'truncation_percentage': results['truncation_percentage'],
            'top_failure_modes': dict(Counter(results['failure_mode_counts']).most_common(5))
        }
        for config, results in all_config_results.items()
    }
    
    with open(config_summary_path, 'w') as f:
        json.dump(config_summary, f, indent=2)
    
    # Print examples of different failure modes
    print(f"\n{'='*80}")
    print("EXAMPLE FAILURE PATTERNS BY TYPE")
    print(f"{'='*80}")
    
    # Group examples by failure mode
    examples_by_mode = defaultdict(list)
    for detail in aggregated_truncation_details:
        examples_by_mode[detail['truncation_type']].append(detail)
    
    # Show examples for each major failure mode
    for failure_mode, examples in examples_by_mode.items():
        if len(examples) >= 2:  # Only show modes with multiple examples
            print(f"\n--- {get_failure_mode_description(failure_mode).upper()} ---")
            for i, example in enumerate(examples[:2]):  # Show 2 examples per mode
                print(f"\nExample {i+1} (Config: {example['config']}):")
                print(f"Question: {example['question'][:150]}...")
                print(f"SnapKV Output: {example['snapkv_pred'][:100]}...")
                print(f"Failure Category: {example['failure_category']}")
            print("-" * 50)
    
    print(f"\nAll results saved to: {output_dir}/")
    print(f"Comprehensive analysis: {comprehensive_path}")
    print(f"Config breakdown: {config_summary_path}")

def get_failure_mode_description(mode):
    """Get human-readable description for failure modes."""
    descriptions = {
        'repeated_same_id': 'Repeated Same ID',
        'sequential_numbers': 'Sequential Numbers',
        'mostly_same_with_variations': 'Mostly Same with Variations',
        'suspiciously_short_ids': 'Suspiciously Short IDs',
        'incomplete_ranking': 'Incomplete Ranking',
        'no_ranking_found': 'No Ranking Found',
        'insufficient_ids': 'Insufficient IDs',
        'no_truncation_detected': 'No Truncation Detected'
    }
    return descriptions.get(mode, mode)

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
    run_comprehensive_truncation_analysis() 