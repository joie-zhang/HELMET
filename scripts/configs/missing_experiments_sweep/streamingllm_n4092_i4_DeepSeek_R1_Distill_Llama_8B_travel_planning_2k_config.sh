# Inference technique evaluation config
# Technique: streamingllm_n4092_i4, Model: DeepSeek-R1-Distill-Llama-8B, Task: travel_planning, Context: 2k
declare -a BASE_CONFIGS=("travel_planning")
declare -a CONTEXT_LENGTHS=("2k")
declare -a MODELS=("DeepSeek-R1-Distill-Llama-8B")
declare -a QUANTIZE=("")
EXP_TYPE="streamingllm"
BENCHMARK="longproc"
USE_REASONING_CONFIG="false"
SEED=42

# StreamingLLM Configuration Parameters
N_LOCAL=4092
N_INIT=4

# SLURM Configuration
JOB_TIME="12:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_streamingllm_n4092_i4_DeepSeek_R1_Distill_Llama_8B_travel_planning_eval"

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
