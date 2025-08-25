import json
from collections import defaultdict
import random
import os
from pathlib import Path
import glob

def load_predictions(file_path):
    """Load predictions from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"\nLoaded: {os.path.basename(file_path)}")
            return data['data']
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def build_file_path(base_path, model_type, context_length, method, cache_config=None):
    """Build file path based on configuration parameters."""
    
    # Model name mapping
    model_names = {
        'llama': 'Llama-3.1-8B-Instruct',
        'qwen': 'Qwen2.5-7B-Instruct'  # Adjust if the actual name is different
    }
    
    model_name = model_names.get(model_type.lower(), model_type)
    
    if method == 'snapkv':
        # SnapKV path structure
        pattern = f"{base_path}/output/snapkv/{context_length}/{model_name}/{cache_config}/msmarco_rerank_psg_16bit_snapkv_{context_length}_{model_name}_{cache_config}_*_test_reranking_data_*.json"
    else:  # baseline
        # Baseline path structure  
        pattern = f"{base_path}/output/baseline/{context_length}/{model_name}/16bit/msmarco_rerank_psg_16bit_baseline_{context_length}_{model_name}_16bit_*_test_reranking_data_*.json"
    
    # Find matching files using glob
    matching_files = glob.glob(pattern)
    if matching_files:
        return matching_files[0]  # Return first match
    else:
        print(f"No files found matching pattern: {pattern}")
        return None

def compare_predictions_for_config(base_path, model_type, context_length, cache_config, output_dir):
    """Compare predictions for a specific configuration."""
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_type.upper()} {context_length} with cache {cache_config}")
    print(f"{'='*60}")
    
    # Build file paths
    snapkv_path = build_file_path(base_path, model_type, context_length, 'snapkv', cache_config)
    baseline_path = build_file_path(base_path, model_type, context_length, 'baseline')
    
    if not snapkv_path or not baseline_path:
        print(f"Skipping {model_type} {context_length} {cache_config} - missing files")
        return None
    
    # Load predictions
    snapkv_preds = load_predictions(snapkv_path)
    baseline_preds = load_predictions(baseline_path)
    
    if not snapkv_preds or not baseline_preds:
        print(f"Skipping {model_type} {context_length} {cache_config} - failed to load data")
        return None
    
    snapkv_dict = {pred['question']: pred for pred in snapkv_preds}
    baseline_dict = {pred['question']: pred for pred in baseline_preds}
    
    categories = defaultdict(list)
    
    for question, snapkv_pred in snapkv_dict.items():
        if question not in baseline_dict:
            print(f"Warning: Question not found in baseline predictions: {question[:100]}...")
            continue
        
        base_pred = baseline_dict[question]
        
        snapkv_correct = snapkv_pred.get('NDCG@10', False)
        base_correct = base_pred.get('NDCG@10', False)
        
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
            'ground_truth': base_pred.get('answer', 'N/A'),
            'snapkv_pred': snapkv_pred.get('output', 'N/A'),
            'baseline_pred': base_pred.get('output', 'N/A'),
            'input_text': snapkv_pred.get('input_text', 'N/A'),
            'metrics': {
                'snapkv_NDCG@10': snapkv_correct,
                'baseline_NDCG@10': base_correct,
            }
        })
    
    # Summary counts
    print("\nCategory Breakdown:")
    total_examples = sum(len(categories[cat]) for cat in categories)
    results_summary = {}
    
    for cat in ['Both Correct', 'Both Wrong', 'SnapKV Wins', 'Baseline Wins']:
        count = len(categories[cat])
        percentage = (count / total_examples * 100) if total_examples > 0 else 0
        print(f"{cat}: {count} ({percentage:.1f}%)")
        results_summary[cat] = {'count': count, 'percentage': percentage}
    
    # Create output filename
    cache_suffix = cache_config.replace('w32_c', 'c').replace('_k7_maxpool', '')
    output_filename = f"rerank_analysis_{model_type}_{context_length}_{cache_suffix}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save detailed results
    output_data = {
        'config': {
            'model_type': model_type,
            'context_length': context_length,
            'cache_config': cache_config,
            'snapkv_file': os.path.basename(snapkv_path),
            'baseline_file': os.path.basename(baseline_path)
        },
        'summary': results_summary,
        'total_examples': total_examples,
        'detailed_results': categories
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {output_filename}")
    
    # Print sample examples
    print("\nSample examples from each category:")
    for cat in ['Both Correct', 'Both Wrong', 'SnapKV Wins', 'Baseline Wins']:
        if categories[cat]:
            print(f"\n--- {cat} ---")
            examples = categories[cat]
            for i, ex in enumerate(random.sample(examples, min(2, len(examples)))):
                print(f"\nExample {i+1}:")
                print(f"Question: {ex['question'][:200]}...")
                print(f"SnapKV: {ex['snapkv_pred'][:100]}...")
                print(f"Baseline: {ex['baseline_pred'][:100]}...")
                print(f"NDCG@10 - SnapKV: {ex['metrics']['snapkv_NDCG@10']}, Baseline: {ex['metrics']['baseline_NDCG@10']}")
    
    return results_summary

def run_comprehensive_analysis():
    """Run analysis across all specified configurations."""
    
    # Configuration parameters
    base_path = "/scratch/gpfs/DANQIC/jz4391/HELMET"
    models = ['llama', 'qwen']
    context_lengths = ['16k', '32k']
    cache_configs = ['w32_c1024_k7_maxpool', 'w32_c4096_k7_maxpool']
    
    # Create output directory
    output_dir = 'qualitative/rerank_analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Store all results for summary
    all_results = {}
    
    print("Starting comprehensive SnapKV vs Baseline analysis...")
    print(f"Models: {models}")
    print(f"Context lengths: {context_lengths}")
    print(f"Cache configs: {cache_configs}")
    
    # Run analysis for each configuration
    for model in models:
        for context_length in context_lengths:
            for cache_config in cache_configs:
                config_key = f"{model}_{context_length}_{cache_config}"
                
                try:
                    results = compare_predictions_for_config(
                        base_path, model, context_length, cache_config, output_dir
                    )
                    if results:
                        all_results[config_key] = {
                            'model': model,
                            'context_length': context_length,
                            'cache_config': cache_config,
                            'results': results
                        }
                except Exception as e:
                    print(f"Error processing {config_key}: {e}")
                    continue
    
    # Create summary report
    summary_path = os.path.join(output_dir, 'summary_report.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    for config_key, data in all_results.items():
        print(f"\n{config_key.upper()}:")
        for category, stats in data['results'].items():
            print(f"  {category}: {stats['count']} ({stats['percentage']:.1f}%)")
    
    print(f"\nAll results saved to: {output_dir}/")
    print(f"Summary report: {summary_path}")

if __name__ == "__main__":
    run_comprehensive_analysis()
