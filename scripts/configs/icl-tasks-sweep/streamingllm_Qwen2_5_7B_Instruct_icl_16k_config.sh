# Inference technique evaluation config
# Technique: streamingllm, Model: Qwen2.5-7B-Instruct, Task: icl, Context: 16k
declare -a BASE_CONFIGS=("icl")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("Qwen2.5-7B-Instruct")
declare -a QUANTIZE=("")
EXP_TYPE="streamingllm"
BENCHMARK="helmet"
USE_REASONING_CONFIG="false"
SEED=42

# StreamingLLM Configuration Parameters
N_LOCAL=4096
N_INIT=4

# SLURM Configuration
JOB_TIME="1:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_streamingllm_Qwen2_5_7B_Instruct_icl_eval"

# Export variables so they're available to the job script
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
