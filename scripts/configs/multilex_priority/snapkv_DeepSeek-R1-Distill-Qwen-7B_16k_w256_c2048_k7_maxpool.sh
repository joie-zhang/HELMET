# Inference technique evaluation config
# Technique: snapkv, Model: DeepSeek-R1-Distill-Qwen-7B, Task: summ_multilex, Context: 16k
declare -a BASE_CONFIGS=("summ_multilex")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("DeepSeek-R1-Distill-Qwen-7B")
declare -a QUANTIZE=("16")
EXP_TYPE="snapkv"
BENCHMARK="helmet"
USE_REASONING_CONFIG="false"
SEED=42

KV_TYPE="snapkv"
WINDOW_SIZE="256"
MAX_CAPACITY_PROMPT="2048"
KERNEL_SIZE="7"
POOLING="maxpool"


# SLURM Configuration
JOB_TIME="01:00:00"
JOB_NAME="snapkv_DeepSeek_R1_Distill_Qwen_7B_summ_multilex_16k_w256_c2048_k7_maxpool"

# Export variables
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export USE_REASONING_CONFIG
export SEED
export KV_TYPE
export WINDOW_SIZE
export MAX_CAPACITY_PROMPT
export KERNEL_SIZE
export POOLING
