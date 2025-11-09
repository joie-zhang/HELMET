# Inference technique evaluation config
# Technique: snapkv_k7_w2048_c8192_maxpool, Model: Yarn-Qwen3-8B, Task: cite, Context: 32k
# Enable Thinking: False (Qwen3 CoT mode)
declare -a BASE_CONFIGS=("cite")
declare -a CONTEXT_LENGTHS=("32k")
declare -a MODELS=("Yarn-Qwen3-8B")
declare -a QUANTIZE=("")
EXP_TYPE="snapkv"
BENCHMARK="helmet"
USE_REASONING_CONFIG="false"
ENABLE_THINKING="False"
SEED=42

# KV Cache Configuration Parameters
KV_TYPE="snapkv"
WINDOW_SIZE=2048
MAX_CAPACITY_PROMPT=8192
KERNEL_SIZE=7
POOLING="maxpool"

# SLURM Configuration
JOB_TIME="18:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_snapkv_k7_w2048_c8192_maxpool_Yarn_Qwen3_8B_nothinking_cite_eval"

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
