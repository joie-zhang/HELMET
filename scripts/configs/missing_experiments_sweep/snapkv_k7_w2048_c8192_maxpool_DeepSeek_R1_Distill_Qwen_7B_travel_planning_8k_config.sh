# Inference technique evaluation config
# Technique: snapkv_k7_w2048_c8192_maxpool, Model: DeepSeek-R1-Distill-Qwen-7B, Task: travel_planning, Context: 8k
declare -a BASE_CONFIGS=("travel_planning")
declare -a CONTEXT_LENGTHS=("8k")
declare -a MODELS=("DeepSeek-R1-Distill-Qwen-7B")
declare -a QUANTIZE=("")
EXP_TYPE="snapkv"
BENCHMARK="longproc"
USE_REASONING_CONFIG="false"
SEED=42

# KV Cache Configuration Parameters
KV_TYPE="snapkv"
WINDOW_SIZE=2048
MAX_CAPACITY_PROMPT=8192
KERNEL_SIZE=7
POOLING="maxpool"

# SLURM Configuration
JOB_TIME="18:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_snapkv_k7_w2048_c8192_maxpool_DeepSeek_R1_Distill_Qwen_7B_travel_planning_eval"

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
