# Inference technique evaluation config
# Technique: streamingllm_n2044_i4, Model: Llama-3.1-8B-Instruct, Task: html_to_tsv, Context: 2k
declare -a BASE_CONFIGS=("html_to_tsv")
declare -a CONTEXT_LENGTHS=("2k")
declare -a MODELS=("Llama-3.1-8B-Instruct")
declare -a QUANTIZE=("")
EXP_TYPE="streamingllm"
BENCHMARK="longproc"
USE_REASONING_CONFIG="false"
SEED=42

# StreamingLLM Configuration Parameters
N_LOCAL=2044
N_INIT=4

# SLURM Configuration
JOB_TIME="12:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_streamingllm_n2044_i4_Llama_3_1_8B_Instruct_html_to_tsv_eval"

# Export variables
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export USE_REASONING_CONFIG
export SEED
export N_LOCAL
export N_INIT
