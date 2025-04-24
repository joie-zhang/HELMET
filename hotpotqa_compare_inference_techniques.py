import json
from collections import defaultdict

def load_predictions(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        print(f"\nFile: {file_path}")
        print("Data type:", type(data))
        print("Keys:", data.keys())
        return data['data']

def compare_predictions():
    # Load predictions
    streamingllm_preds = load_predictions("output/streamingllm/16k/Llama-3.1-8B-Instruct/kilt_hotpotqa_16bit_streamingllm_16k_Llama-3.1-8B-Instruct_63800165_hotpotqa-dev-multikilt_1000_k105_dep3_in16384_size100_shots2_sampFalsemax20min0t0.0p1.0_chatFalse_42.json")
    baseline_preds = load_predictions("/scratch/gpfs/DANQIC/jz4391/HELMET/output_v1_uniform/streamingllm/16k/Llama-3.1-8B-Instruct/kilt_hotpotqa_16bit_streamingllm_16k_Llama-3.1-8B-Instruct_63594798_hotpotqa-dev-multikilt_1000_k105_dep3_in16384_size100_shots2_sampFalsemax20min0t0.0p1.0_chatFalse_42.json")
    
    # Debug print structure of first items
    print("\nStreamingLLM first item structure:")
    print("Keys available:", streamingllm_preds[0].keys())
    print("\nBaseline first item structure:")
    print("Keys available:", baseline_preds[0].keys())
    
    # Create dictionaries mapping question to predictions for easier lookup
    streamingllm_dict = {}
    baseline_dict = {}
    
    for pred in streamingllm_preds:
        # Use 'question' field directly since it's available
        question = pred['question']
        streamingllm_dict[question] = pred
        
    for pred in baseline_preds:
        question = pred['question']
        baseline_dict[question] = pred
    
    differences = []
    
    # Iterate through StreamingLLM predictions
    for question, stream_pred in streamingllm_dict.items():
        if question not in baseline_dict:
            print(f"Warning: Question not found in baseline predictions: {question[:100]}...")
            continue
            
        base_pred = baseline_dict[question]
        
        try:
            # Get exact match status
            stream_correct = stream_pred.get('exact_match', False)
            base_correct = base_pred.get('exact_match', False)
            
            # If baseline is correct but StreamingLLM is wrong
            if base_correct and not stream_correct:
                differences.append({
                    'question': question,
                    'ground_truth': stream_pred.get('answer', 'N/A'),  # Changed from 'output' to 'answer'
                    'streamingllm_pred': stream_pred.get('parsed_output', 'N/A'),
                    'baseline_pred': base_pred.get('parsed_output', 'N/A'),
                    'input_text': stream_pred.get('input_text', 'N/A'),  # Added full input text for context
                    'metrics': {
                        'streamingllm_f1': stream_pred.get('f1', 'N/A'),
                        'baseline_f1': base_pred.get('f1', 'N/A')
                    }
                })
        except Exception as e:
            print(f"\nError processing question: {question[:100]}...")
            print("Error:", str(e))
            print("StreamingLLM item:", stream_pred)
            print("Baseline item:", base_pred)
            break
    
    # Save the differences to a file
    with open('hotpotqa_differences.json', 'w') as f:
        json.dump(differences, f, indent=2)
    
    print(f"\nFound {len(differences)} questions where baseline is correct but StreamingLLM is wrong")
    
    # Print a few examples
    for i, diff in enumerate(differences[:10]):
        print(f"\nExample {i+1}:")
        print(f"Question: {diff['question']}")
        print(f"Ground Truth: {diff['ground_truth']}")
        print(f"StreamingLLM: {diff['streamingllm_pred']}")
        print(f"Baseline: {diff['baseline_pred']}")
        print(f"F1 Scores - StreamingLLM: {diff['metrics']['streamingllm_f1']}, Baseline: {diff['metrics']['baseline_f1']}")

if __name__ == "__main__":
    compare_predictions()