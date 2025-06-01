import json

def inspect_data_structure():
    # Load the data
    snapkv_file = "/scratch/gpfs/DANQIC/jz4391/HELMET/output/snapkv/32k/Llama-3.1-8B-Instruct/w32_c1024_k7_maxpool/alce_asqa_165_16bit_snapkv_32k_Llama-3.1-8B-Instruct_w32_c1024_k7_maxpool_64407623_asqa_eval_gtr_top2000_in32768_size100_shots2_sampFalsemax300min0t0.0p1.0_chatTrue_42.json"
    
    with open(snapkv_file, 'r') as f:
        data = json.load(f)
    
    print("=== TOP LEVEL STRUCTURE ===")
    print("Keys in data:", list(data.keys()))
    
    if 'data' in data:
        print(f"\nNumber of examples: {len(data['data'])}")
        print("\n=== FIRST EXAMPLE STRUCTURE ===")
        first_example = data['data'][0]
        print("Keys in first example:")
        for key in sorted(first_example.keys()):
            value = first_example[key]
            if isinstance(value, str):
                print(f"  {key}: {type(value).__name__} (length: {len(value)}) - '{value[:100]}...'")
            elif isinstance(value, list):
                print(f"  {key}: {type(value).__name__} (length: {len(value)})")
                if len(value) > 0:
                    print(f"    First item type: {type(value[0])}")
                    if isinstance(value[0], dict):
                        print(f"    First item keys: {list(value[0].keys())}")
            elif isinstance(value, dict):
                print(f"  {key}: {type(value).__name__} with keys: {list(value.keys())}")
            else:
                print(f"  {key}: {type(value).__name__} - {value}")
    
    # Also check if there are averaged_metrics at the top level
    if 'averaged_metrics' in data:
        print("\n=== AVERAGED METRICS ===")
        for key, value in data['averaged_metrics'].items():
            print(f"  {key}: {value}")
    
    # Check if there's a corresponding .score file
    score_file = snapkv_file + ".score"
    try:
        with open(score_file, 'r') as f:
            score_data = json.load(f)
        print(f"\n=== SCORE FILE STRUCTURE ===")
        print("Keys in score file:")
        for key, value in score_data.items():
            print(f"  {key}: {value}")
    except FileNotFoundError:
        print(f"\nNo score file found at: {score_file}")

if __name__ == "__main__":
    inspect_data_structure()