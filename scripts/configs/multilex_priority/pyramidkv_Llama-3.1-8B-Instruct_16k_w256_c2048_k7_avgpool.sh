# Inference technique evaluation config
# Technique: pyramidkv, Model: Llama-3.1-8B-Instruct, Task: summ_multilex, Context: 16k
declare -a BASE_CONFIGS=("summ_multilex")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("Llama-3.1-8B-Instruct")
declare -a QUANTIZE=("16")
EXP_TYPE="pyramidkv"
BENCHMARK="helmet"
USE_REASONING_CONFIG="false"
SEED=42

KV_TYPE="pyramidkv"
WINDOW_SIZE="256"
MAX_CAPACITY_PROMPT="2048"
KERNEL_SIZE="7"
POOLING="avgpool"


# SLURM Configuration
JOB_TIME="01:00:00"
JOB_NAME="pyramidkv_Llama_3.1_8B_Instruct_summ_multilex_16k_w256_c2048_k7_avgpool"

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
