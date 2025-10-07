# Inference technique evaluation config
# Technique: snapkv_k7_w256_c1024_maxpool, Model: DeepSeek-R1-Distill-Llama-8B, Task: icl, Context: 16k
declare -a BASE_CONFIGS=("icl")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("DeepSeek-R1-Distill-Llama-8B")
declare -a QUANTIZE=("")
EXP_TYPE="snapkv"
BENCHMARK="helmet"
USE_REASONING_CONFIG="false"
SEED=42

# KV Cache Configuration Parameters
KV_TYPE="snapkv"
WINDOW_SIZE=256
MAX_CAPACITY_PROMPT=1024
KERNEL_SIZE=7
POOLING="maxpool"

# SLURM Configuration
JOB_TIME="4:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_snapkv_k7_w256_c1024_maxpool_DeepSeek_R1_Distill_Llama_8B_icl_eval"

# Export variables so they're available to the job script
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
