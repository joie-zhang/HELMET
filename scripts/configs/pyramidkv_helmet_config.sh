# pyramidkv_helmet_config.sh
# declare -a BASE_CONFIGS=("cite" "rerank" "recall_jsonkv" "rag_nq" "rag_hotpotqa" "niah")
declare -a BASE_CONFIGS=("rag_nq")
declare -a CONTEXT_LENGTHS=("32k")
# declare -a MODELS=("Llama-3.1-8B-Instruct")
declare -a MODELS=("Qwen2.5-7B-Instruct")
declare -a QUANTIZE=("")  # non-baseline experiments should not support quantize
EXP_TYPE="pyramidkv"
BENCHMARK="helmet"
SEED=42

# # pyramidkv_longproc_config.sh
# # declare -a BASE_CONFIGS=("html_to_tsv" "pseudo_to_code")
# # declare -a CONTEXT_LENGTHS=("0.5k")
# # declare -a BASE_CONFIGS=("html_to_tsv" "pseudo_to_code" "travel_planning")
# # declare -a CONTEXT_LENGTHS=("2k")
# declare -a BASE_CONFIGS=("html_to_tsv" "travel_planning")
# declare -a CONTEXT_LENGTHS=("8k")
# # declare -a MODELS=("Llama-3.1-8B-Instruct")
# declare -a MODELS=("Qwen2.5-7B-Instruct")
# declare -a QUANTIZE=("")  # non-baseline experiments should not support quantize
# EXP_TYPE="pyramidkv"
# BENCHMARK="longproc"
# SEED=42

# KV Cache Configuration Parameters
# These will be used for pyramidkv

# cache size: 64 vs 2048
KV_TYPE="pyramidkv"           # KV type
WINDOW_SIZE=32                # Window size for KV cache methods
# MAX_CAPACITY_PROMPT=64      # Maximum capacity for prompt in KV cache
MAX_CAPACITY_PROMPT=2048      # Maximum capacity for prompt in KV cache
KERNEL_SIZE=5                 # Kernel size for attention pooling
POOLING="avgpool"             # Pooling method for attention (maxpool or avgpool)

# SLURM Configuration
JOB_TIME="3:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_${CONTEXT_LENGTHS[-1]}_eval"

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