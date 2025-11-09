# Inference technique evaluation config
# Technique: snapkv_k7_w256_c2048_maxpool, Model: Yarn-Qwen3-8B, Task: pseudo_to_code, Context: 2k
declare -a BASE_CONFIGS=("pseudo_to_code")
declare -a CONTEXT_LENGTHS=("2k")
declare -a MODELS=("Yarn-Qwen3-8B")
declare -a QUANTIZE=("")
EXP_TYPE="snapkv"
BENCHMARK="longproc"
USE_REASONING_CONFIG="false"
SEED=42

# KV Cache Configuration Parameters
KV_TYPE="snapkv"
WINDOW_SIZE=256
MAX_CAPACITY_PROMPT=2048
KERNEL_SIZE=7
POOLING="maxpool"

# SLURM Configuration
JOB_TIME="12:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_snapkv_k7_w256_c2048_maxpool_Yarn_Qwen3_8B_pseudo_to_code_eval"

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
