# streamingllm_helmet_config.sh
# declare -a BASE_CONFIGS=("cite" "rerank" "recall_jsonkv" "rag_nq" "rag_hotpotqa" "niah")
declare -a BASE_CONFIGS=("rag_nq")
# declare -a CONTEXT_LENGTHS=("16k" "32k")
declare -a CONTEXT_LENGTHS=("16k")
# declare -a MODELS=("Llama-3.1-8B-Instruct")
declare -a MODELS=("Qwen2.5-7B-Instruct")
declare -a QUANTIZE=("")  # non-baseline experiments should not support quantize
EXP_TYPE="streamingllm"
BENCHMARK="helmet"
SEED=42

# # streamingllm_longproc_config.sh
# # declare -a BASE_CONFIGS=("html_to_tsv" "pseudo_to_code")
# # declare -a CONTEXT_LENGTHS=("0.5k")
# # declare -a BASE_CONFIGS=("html_to_tsv" "pseudo_to_code" "travel_planning")
# # declare -a CONTEXT_LENGTHS=("2k")
# # declare -a BASE_CONFIGS=("html_to_tsv" "travel_planning")
# # declare -a CONTEXT_LENGTHS=("8k")
# declare -a BASE_CONFIGS=("pseudo_to_code")
# declare -a CONTEXT_LENGTHS=("0.5k")
# declare -a MODELS=("Llama-3.1-8B-Instruct")
# # declare -a MODELS=("Qwen2.5-7B-Instruct")
# declare -a QUANTIZE=("")  # non-baseline experiments should not support quantize
# EXP_TYPE="streamingllm"
# BENCHMARK="longproc"
# SEED=42

# StreamingLLM Configuration Parameters
N_LOCAL=4092                # Number of local tokens
N_INIT=4                  # Number of initial tokens

# SLURM Configuration
JOB_TIME="2:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_${CONTEXT_LENGTHS[-1]}_eval"

# Export variables so they're available to the job script
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export SEED
export N_LOCAL
export N_INIT