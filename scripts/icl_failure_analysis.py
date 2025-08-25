import json
import os
from collections import Counter, defaultdict
from pathlib import Path

def analyze_icl_failure_patterns():
    """
    Analyze how SnapKV failure modes relate to in-context learning degradation
    """
    print("="*80)
    print("IN-CONTEXT LEARNING FAILURE MODE ANALYSIS")
    print("="*80)
    
    # Load a sample analysis file to understand the task structure
    config_file = 'qualitative/rerank_analysis_results/rerank_analysis_qwen_16k_c1024.json'
    
    if not Path(config_file).exists():
        print(f"Configuration file not found: {config_file}")
        return
    
    with open(config_file, 'r') as f:
        data = json.load(f)
    
    # Analyze the reranking task structure for ICL patterns
    baseline_wins = data['detailed_results'].get('Baseline Wins', [])
    both_wrong = data['detailed_results'].get('Both Wrong', [])
    
    print("\n1. TASK STRUCTURE ANALYSIS")
    print("-" * 50)
    
    if baseline_wins:
        sample_input = baseline_wins[0].get('input_text', '')
        analyze_icl_structure(sample_input)
    
    print("\n2. FAILURE MODE CATEGORIZATION BY ICL IMPACT")
    print("-" * 50)
    
    # Categorize failures by their relationship to ICL
    icl_failure_categories = {
        'format_loss': 0,  # Lost instruction format
        'example_loss': 0,  # Lost ICL examples
        'task_comprehension_loss': 0,  # Lost task understanding
        'content_hallucination': 0,  # Generated fake content
        'incomplete_processing': 0  # Partial processing
    }
    
    all_failures = baseline_wins + both_wrong
    
    for failure in all_failures:
        category = categorize_icl_failure(failure)
        if category in icl_failure_categories:
            icl_failure_categories[category] += 1
    
    total_failures = len(all_failures)
    print(f"Total failures analyzed: {total_failures}")
    print(f"{'ICL Failure Category':<30} {'Count':<8} {'Percentage':<12}")
    print("-" * 55)
    
    for category, count in icl_failure_categories.items():
        percentage = (count / total_failures * 100) if total_failures > 0 else 0
        print(f"{category.replace('_', ' ').title():<30} {count:<8} {percentage:>8.1f}%")
    
    print("\n3. ARCHITECTURAL IMPLICATIONS")
    print("-" * 50)
    analyze_snapkv_architectural_issues(icl_failure_categories, total_failures)
    
    print("\n4. SNAPKV CACHE COMPRESSION ANALYSIS")
    print("-" * 50)
    analyze_cache_compression_impact()

def analyze_icl_structure(sample_input):
    """Analyze the in-context learning structure of the task"""
    
    print("Reranking Task ICL Structure:")
    print("- Task: Document ranking based on relevance")
    print("- Input: Query + List of documents with IDs")
    print("- Expected Output: 'Ranking: ID1 > ID2 > ID3...'")
    print("- ICL Elements:")
    
    if "You are provided with a list of documents" in sample_input:
        print("  ✓ Task instruction present")
    if "Rank each document based on their relevance" in sample_input:
        print("  ✓ Task objective specified")
    if "Ranking: ID3 > ID1 > ID2" in sample_input:
        print("  ✓ Output format example provided")
    if "[ID:" in sample_input:
        print("  ✓ Document structure examples present")
    
    # Count documents in context
    doc_count = sample_input.count("[ID:")
    print(f"  📊 Documents in context: {doc_count}")
    
    # Estimate context length
    context_tokens = len(sample_input.split())
    print(f"  📏 Approximate context length: {context_tokens} words")

def categorize_icl_failure(failure):
    """Categorize failure based on ICL degradation type"""
    
    snapkv_pred = failure.get('snapkv_pred', '')
    
    # Check for format compliance
    if 'Ranking:' not in snapkv_pred:
        return 'format_loss'
    
    # Extract ranking line
    ranking_line = None
    for line in snapkv_pred.split('\n'):
        if 'Ranking:' in line:
            ranking_line = line.replace('Ranking:', '').strip()
            break
    
    if not ranking_line:
        return 'format_loss'
    
    # Extract IDs
    ids = []
    for id_str in ranking_line.split('>'):
        id_clean = id_str.strip()
        if id_clean and id_clean.isdigit():
            ids.append(int(id_clean))
    
    # Categorize based on ID patterns
    if len(ids) == 0:
        return 'format_loss'
    elif len(ids) < 5:
        return 'incomplete_processing'
    elif len(set(ids)) == 1:
        return 'content_hallucination'  # Repeated same ID
    elif all(ids[i] == ids[0] + i for i in range(len(ids))):
        return 'content_hallucination'  # Sequential fabricated IDs
    elif all(id_val < 1000 for id_val in ids):
        return 'content_hallucination'  # Suspiciously simple IDs
    else:
        # Check if IDs exist in ground truth
        ground_truth = failure.get('ground_truth', '')
        gt_ids = set()
        for id_str in ground_truth.split('>'):
            id_clean = id_str.strip()
            if id_clean and id_clean.isdigit():
                gt_ids.add(int(id_clean))
        
        snapkv_ids_set = set(ids)
        overlap = len(snapkv_ids_set & gt_ids)
        
        if overlap == 0:
            return 'task_comprehension_loss'  # No relevant IDs found
        elif overlap < len(ids) * 0.3:
            return 'example_loss'  # Some relevant but mostly wrong
        else:
            return 'incomplete_processing'  # Partial success

def analyze_snapkv_architectural_issues(failure_categories, total_failures):
    """Analyze architectural implications of failure patterns"""
    
    print("SnapKV Architectural Analysis:")
    print()
    
    format_loss_pct = (failure_categories['format_loss'] / total_failures * 100)
    content_hallucination_pct = (failure_categories['content_hallucination'] / total_failures * 100)
    task_comprehension_pct = (failure_categories['task_comprehension_loss'] / total_failures * 100)
    
    print("🏗️  **Key-Value Cache Compression Issues:**")
    if format_loss_pct > 10:
        print(f"   - HIGH format loss ({format_loss_pct:.1f}%) → Instruction tokens likely compressed away")
    if content_hallucination_pct > 20:
        print(f"   - HIGH content hallucination ({content_hallucination_pct:.1f}%) → Document content severely degraded")
    if task_comprehension_pct > 15:
        print(f"   - HIGH task comprehension loss ({task_comprehension_pct:.1f}%) → Task context corrupted")
    
    print("\n🧠 **In-Context Learning Degradation Mechanisms:**")
    print("   1. **Instruction Compression:** Critical task instructions removed from cache")
    print("   2. **Example Dilution:** ICL examples compressed, breaking pattern recognition")
    print("   3. **Content Fragmentation:** Document content scattered/lost during compression")
    print("   4. **Attention Pattern Disruption:** Key semantic relationships broken")
    
    print("\n⚙️  **SnapKV Selection Criteria Problems:**")
    print("   - **Recency Bias:** Recent tokens prioritized over important instructions")
    print("   - **Local Salience:** Misses global context relationships")
    print("   - **Task-Agnostic Compression:** Doesn't understand task-specific importance")
    print("   - **Static Windows:** Fixed compression patterns vs. dynamic task needs")

def analyze_cache_compression_impact():
    """Analyze how different cache sizes affect ICL performance"""
    
    print("Cache Size Impact Analysis:")
    print()
    
    # Load results for different cache configurations
    cache_configs = ['c1024', 'c4096']
    models = ['qwen', 'llama']
    context_lengths = ['16k', '32k']
    
    compression_analysis = {}
    
    for model in models:
        for context in context_lengths:
            for cache in cache_configs:
                config_file = f'qualitative/rerank_analysis_results/rerank_analysis_{model}_{context}_{cache}.json'
                
                if Path(config_file).exists():
                    try:
                        with open(config_file, 'r') as f:
                            data = json.load(f)
                        
                        total_examples = data.get('total_examples', 0)
                        baseline_wins = len(data['detailed_results'].get('Baseline Wins', []))
                        
                        failure_rate = (baseline_wins / total_examples * 100) if total_examples > 0 else 0
                        
                        config_key = f"{model}_{context}_{cache}"
                        compression_analysis[config_key] = {
                            'failure_rate': failure_rate,
                            'total_examples': total_examples,
                            'baseline_wins': baseline_wins
                        }
                    except Exception as e:
                        print(f"Error loading {config_file}: {e}")
    
    if compression_analysis:
        print(f"{'Configuration':<20} {'Failure Rate':<15} {'Examples':<10}")
        print("-" * 50)
        
        for config, stats in sorted(compression_analysis.items()):
            print(f"{config:<20} {stats['failure_rate']:>8.1f}%        {stats['total_examples']:<10}")
    
    print("\n🔍 **Cache Compression Insights:**")
    print("   - **c1024 vs c4096:** Larger caches should preserve more ICL context")
    print("   - **16k vs 32k:** Longer contexts test compression limits more severely")
    print("   - **Model Differences:** Qwen vs Llama may have different ICL dependencies")
    
    print("\n💡 **SnapKV Design Recommendations:**")
    print("   1. **Task-Aware Compression:** Preserve instruction and example tokens")
    print("   2. **Hierarchical Importance:** Different compression rules for different token types")
    print("   3. **ICL Pattern Recognition:** Identify and preserve demonstration patterns")
    print("   4. **Dynamic Allocation:** Adjust cache allocation based on task complexity")
    print("   5. **Semantic Coherence:** Maintain document-level semantic relationships")

if __name__ == "__main__":
    analyze_icl_failure_patterns() 