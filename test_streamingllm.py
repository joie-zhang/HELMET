from model_utils_streamingllm import StreamingLLMModel
import torch

print(torch.cuda.is_available())

model = StreamingLLMModel(
    model_name="/scratch/gpfs/DANQIC/models/Llama-3.2-1B",
    cache_start_size=4,
    cache_recent_size=16,
    enable_positional_shift=False,
)

print

# test the generate function

# test how big the cache size is, actually
# 