# PyramidKV hyperparameter sweep config
# Window Size: 128, Max Capacity: 1024
declare -a BASE_CONFIGS=("cite")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("DeepSeek-R1-Distill-Llama-8B")
declare -a QUANTIZE=("")  # non-baseline experiments should not support quantize
EXP_TYPE="pyramidkv"
BENCHMARK="helmet"
SEED=42

# KV Cache Configuration Parameters
KV_TYPE="pyramidkv"           # KV type
WINDOW_SIZE=128                # Window size for KV cache methods
MAX_CAPACITY_PROMPT=1024      # Maximum capacity for prompt in KV cache
KERNEL_SIZE=5                 # Kernel size for attention pooling
POOLING="avgpool"             # Pooling method for attention (maxpool or avgpool)

# SLURM Configuration
JOB_TIME="1:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_w128_c1024_eval"

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
