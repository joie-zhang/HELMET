# Inference technique evaluation config
# Technique: pyramidkv_k7_w2048_c8192_avgpool, Model: Llama-3.1-8B-Instruct, Task: cite, Context: 16k
declare -a BASE_CONFIGS=("cite")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("Llama-3.1-8B-Instruct")
declare -a QUANTIZE=("")
EXP_TYPE="pyramidkv"
BENCHMARK="helmet"
USE_REASONING_CONFIG="false"
SEED=42

# KV Cache Configuration Parameters
KV_TYPE="pyramidkv"
WINDOW_SIZE=2048
MAX_CAPACITY_PROMPT=8192
KERNEL_SIZE=7
POOLING="avgpool"

# SLURM Configuration
JOB_TIME="12:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_pyramidkv_k7_w2048_c8192_avgpool_Llama_3_1_8B_Instruct_cite_eval"

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
