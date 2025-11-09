# Inference technique evaluation config
# Technique: pyramidkv_k7_w256_c2048_avgpool, Model: Qwen3-8B, Task: recall_jsonkv, Context: 32k
# Enable Thinking: True (Qwen3 CoT mode)
declare -a BASE_CONFIGS=("recall_jsonkv")
declare -a CONTEXT_LENGTHS=("32k")
declare -a MODELS=("Qwen3-8B")
declare -a QUANTIZE=("")
EXP_TYPE="pyramidkv"
BENCHMARK="helmet"
USE_REASONING_CONFIG="false"
ENABLE_THINKING="True"
SEED=42

# KV Cache Configuration Parameters
KV_TYPE="pyramidkv"
WINDOW_SIZE=256
MAX_CAPACITY_PROMPT=2048
KERNEL_SIZE=7
POOLING="avgpool"

# SLURM Configuration
JOB_TIME="18:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_pyramidkv_k7_w256_c2048_avgpool_Qwen3_8B_thinking_recall_jsonkv_eval"

# Export variables
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export USE_REASONING_CONFIG
export ENABLE_THINKING
export SEED
export KV_TYPE
export WINDOW_SIZE
export MAX_CAPACITY_PROMPT
export KERNEL_SIZE
export POOLING
