# NIAH Hyperparameter Sweep Config
# Technique: snapkv, Model: Yarn-Qwen3-8B, Window: 2048, Cache: 32768
declare -a BASE_CONFIGS=("niah")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("Yarn-Qwen3-8B")
declare -a QUANTIZE=("")
EXP_TYPE="snapkv"
BENCHMARK="helmet"
SEED=42

# KV Cache Configuration Parameters
KV_TYPE="snapkv"
WINDOW_SIZE=2048
MAX_CAPACITY_PROMPT=32768
KERNEL_SIZE=7
POOLING="maxpool"

# SLURM Configuration
JOB_TIME="1:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_snapkv_k7_w2048_c32768_maxpool_Yarn_Qwen3_8B_niah_eval"

# Export variables so they're available to the job script
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export SEED
export KV_TYPE
export WINDOW_SIZE
export MAX_CAPACITY_PROMPT
export KERNEL_SIZE
export POOLING
