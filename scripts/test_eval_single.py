import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.eval_gpt4_summ import check_metrics
from model_utils import OpenAIModel

# Test with just one file
test_file = "output/streamingllm/16k/DeepSeek-R1-Distill-Qwen-7B/local2044_init4/multi_lexsum_15684_16bit_streamingllm_16k_DeepSeek-R1-Distill-Qwen-7B_local2044_init4_1263730__in16384_size100_shots2_sampFalsemax400min0t0.0p1.0_chatTrue_42.json"

output_file = test_file.replace(".json", "-gpt4eval_o.json")

print(f"Input file: {test_file}")
print(f"Output file: {output_file}")
print(f"\nStarting GPT-4 evaluation (this will make API calls)...")

model = OpenAIModel("gpt-4o-2024-05-13", temperature=0.1, generation_max_length=4096)
result = check_metrics(model, test_file, output_file)

print(f"\nSuccess! Created {output_file}")
print(f"GPT-4 F1 score: {result['averaged_metrics']['gpt-4-f1']}")
