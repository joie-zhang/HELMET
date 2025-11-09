# Inference technique evaluation config
# Technique: streamingllm_n4092_i4, Model: Qwen2.5-7B-Instruct, Task: niah, Context: 16k
declare -a BASE_CONFIGS=("niah")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("Qwen2.5-7B-Instruct")
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
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_streamingllm_n4092_i4_Qwen2_5_7B_Instruct_niah_eval"

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
