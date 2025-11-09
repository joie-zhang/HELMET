# Inference technique evaluation config
# Technique: streamingllm_n4092_i4, Model: Llama-3.1-8B-Instruct, Task: rag_hotpotqa, Context: 16k
declare -a BASE_CONFIGS=("rag_hotpotqa")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("Llama-3.1-8B-Instruct")
declare -a QUANTIZE=("")
EXP_TYPE="streamingllm"
BENCHMARK="helmet"
USE_REASONING_CONFIG="false"
SEED=42

# StreamingLLM Configuration Parameters
N_LOCAL=4092
N_INIT=4

# SLURM Configuration
JOB_TIME="06:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_streamingllm_n4092_i4_Llama_3_1_8B_Instruct_rag_hotpotqa_eval"

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
