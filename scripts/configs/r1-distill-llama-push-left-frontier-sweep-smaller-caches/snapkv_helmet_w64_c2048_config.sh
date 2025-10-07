# snapkv hyperparameter sweep config
# Benchmark: helmet, Window Size: 64, Max Capacity: 2048
declare -a BASE_CONFIGS=("icl")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("DeepSeek-R1-Distill-Llama-8B")
declare -a QUANTIZE=("")  # non-baseline experiments should not support quantize
EXP_TYPE="snapkv"
BENCHMARK="helmet"
SEED=42

# KV Cache Configuration Parameters
KV_TYPE="snapkv"           # KV type
WINDOW_SIZE=64                # Window size for KV cache methods
MAX_CAPACITY_PROMPT=2048      # Maximum capacity for prompt in KV cache
KERNEL_SIZE=7                 # Kernel size for attention pooling
POOLING="maxpool"             # Pooling method for attention (maxpool or avgpool)

# SLURM Configuration
JOB_TIME="4:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_w64_c2048_eval"

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
