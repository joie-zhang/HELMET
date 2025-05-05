import json
from collections import defaultdict
import argparse

def load_predictions(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        print(f"\nFile: {file_path}")
        print("Data type:", type(data))
        print("Keys:", data.keys())
        return data['data']

def parse_args():
    parser = argparse.ArgumentParser(description='Compare HTML-to-TSV predictions between models')
    parser.add_argument('--model1', type=str, default='pyramidkv', choices=['pyramidkv', 'snapkv', 'streamingllm', 'baseline'],
                      help='First model to compare')
    parser.add_argument('--model2', type=str, default='streamingllm', choices=['pyramidkv', 'snapkv', 'streamingllm', 'baseline'],
                      help='Second model to compare')
    parser.add_argument('--context_length', type=str, default='0.5k', choices=['0.5k', '2k', '5k', '16k'],
                      help='Context length for models')
    parser.add_argument('--f1_threshold', type=float, default=0.9,
                      help='F1 score threshold for considering a prediction good')
    parser.add_argument('--output_file', type=str, default=None,
                      help='Output file name for differences (default: {model1}_vs_{model2}_differences.json)')
    return parser.parse_args()

def get_model_path(model_name, context_length):
    base_path = "/scratch/gpfs/DANQIC/jz4391/HELMET"
    if model_name == 'streamingllm':
        return f"{base_path}/output/streamingllm/{context_length}/Llama-3.1-8B-Instruct/html_to_tsv_{context_length}_16bit_eval_data_in128000_size50_shots0_sampFalsemax1024min0t0.0p1.0_chatTrue_42.json"
    elif model_name == 'pyramidkv':
        # PyramidKV uses 5k instead of 0.5k
        context_dir = '5k' if context_length == '0.5k' else context_length
        return f"{base_path}/output/pyramidkv/{context_dir}/Llama-3.1-8B-Instruct/html_to_tsv_{context_length}_16bit_pyramidkv_{context_dir}_Llama-3.1-8B-Instruct_63800378_data_in128000_size50_shots0_sampFalsemax1024min0t0.0p1.0_chatTrue_42.json"
    elif model_name == 'snapkv':
        # Similar adjustment for SnapKV if needed
        context_dir = '5k' if context_length == '0.5k' else context_length
        return f"{base_path}/output/snapkv/{context_dir}/Llama-3.1-8B-Instruct/html_to_tsv_{context_length}_16bit_snapkv_{context_dir}_Llama-3.1-8B-Instruct_63800378_data_in128000_size50_shots0_sampFalsemax1024min0t0.0p1.0_chatTrue_42.json"
    else:
        raise ValueError(f"Unknown model: {model_name}")

def compare_predictions():
    args = parse_args()
    
    # Set output filename if not provided
    if args.output_file is None:
        args.output_file = f'html_to_tsv_{args.model1}_vs_{args.model2}_differences.json'
    
    # Load predictions
    model1_path = get_model_path(args.model1, args.context_length)
    model2_path = get_model_path(args.model2, args.context_length)
    
    print(f"\nComparing {args.model1} vs {args.model2} for {args.context_length} context length")
    print(f"Model 1 path: {model1_path}")
    print(f"Model 2 path: {model2_path}")
    
    model1_preds = load_predictions(model1_path)
    model2_preds = load_predictions(model2_path)
    
    # Debug print structure of first items
    print(f"\n{args.model1} first item structure:")
    print("Keys available:", model1_preds[0].keys())
    print(f"\n{args.model2} first item structure:")
    print("Keys available:", model2_preds[0].keys())
    
    # Create dictionaries mapping HTML input to predictions
    model1_dict = {pred['input_text']: pred for pred in model1_preds}
    model2_dict = {pred['input_text']: pred for pred in model2_preds}
    
    differences = []
    
    # Compare predictions
    for html_input, model1_pred in model1_dict.items():
        if html_input not in model2_dict:
            print(f"Warning: HTML input not found in {args.model2} predictions: {html_input[:100]}...")
            continue
            
        model2_pred = model2_dict[html_input]
        
        try:
            # Get F1 scores
            model1_f1 = float(model1_pred.get('f1', 0))
            model2_f1 = float(model2_pred.get('f1', 0))
            
            # If model2 has significantly better F1 score
            if model2_f1 > args.f1_threshold and model2_f1 > model1_f1 + 0.1:
                differences.append({
                    'input_html': html_input,
                    'reference_output': model1_pred.get('reference_output', 'N/A'),
                    f'{args.model1}_pred': model1_pred.get('parsed_output', 'N/A'),
                    f'{args.model2}_pred': model2_pred.get('parsed_output', 'N/A'),
                    'metrics': {
                        f'{args.model1}_f1': model1_f1,
                        f'{args.model2}_f1': model2_f1,
                        f'{args.model1}_precision': model1_pred.get('precision', 'N/A'),
                        f'{args.model1}_recall': model1_pred.get('recall', 'N/A'),
                        f'{args.model2}_precision': model2_pred.get('precision', 'N/A'),
                        f'{args.model2}_recall': model2_pred.get('recall', 'N/A')
                    }
                })
        except Exception as e:
            print(f"\nError processing HTML input: {html_input[:100]}...")
            print("Error:", str(e))
            continue
    
    # Save the differences to a file
    with open(args.output_file, 'w') as f:
        json.dump(differences, f, indent=2)
    
    print(f"\nFound {len(differences)} cases where {args.model2} performs significantly better than {args.model1}")
    print(f"(Using F1 threshold of {args.f1_threshold} and minimum difference of 0.1)")
    
    # Print statistics
    if differences:
        f1_diffs = [d['metrics'][f'{args.model2}_f1'] - d['metrics'][f'{args.model1}_f1'] for d in differences]
        avg_f1_diff = sum(f1_diffs) / len(f1_diffs)
        max_f1_diff = max(f1_diffs)
        print(f"\nStatistics:")
        print(f"Average F1 difference: {avg_f1_diff:.3f}")
        print(f"Maximum F1 difference: {max_f1_diff:.3f}")
        
        # Print a few examples
        print("\nDetailed examples:")
        for i, diff in enumerate(differences[:5]):
            print(f"\nExample {i+1}:")
            print(f"Input HTML: {diff['input_html'][:200]}...")
            print(f"Reference: {diff['reference_output']}")
            print(f"{args.model1}: {diff['metrics'][f'{args.model1}_f1']:.3f} F1")
            print(f"{args.model2}: {diff['metrics'][f'{args.model2}_f1']:.3f} F1")

if __name__ == "__main__":
    compare_predictions()