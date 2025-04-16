# streamingllm_helmet_config.sh
declare -a BASE_CONFIGS=("cite" "rerank" "recall_jsonkv" "rag")
declare -a CONTEXT_LENGTHS=("16k" "32k")
declare -a MODELS=("Llama-3.1-8B-Instruct")
EXP_TYPE="streamingllm"
SEED=42

# SLURM Configuration
JOB_TIME="1:00:00"
JOB_NAME="${EXP_TYPE}_helmet_${CONTEXT_LENGTHS[-1]}_eval"

# Export variables so they're available to the job script
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export EXP_TYPE
export SEED