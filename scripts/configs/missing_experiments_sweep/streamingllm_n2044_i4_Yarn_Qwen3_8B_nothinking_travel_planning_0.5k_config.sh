# Inference technique evaluation config
# Technique: streamingllm_n2044_i4, Model: Yarn-Qwen3-8B, Task: travel_planning, Context: 0.5k
# Enable Thinking: False (Qwen3 CoT mode)
declare -a BASE_CONFIGS=("travel_planning")
declare -a CONTEXT_LENGTHS=("0.5k")
declare -a MODELS=("Yarn-Qwen3-8B")
declare -a QUANTIZE=("")
EXP_TYPE="streamingllm"
BENCHMARK="longproc"
USE_REASONING_CONFIG="false"
ENABLE_THINKING="False"
SEED=42

# StreamingLLM Configuration Parameters
N_LOCAL=2044
N_INIT=4

# SLURM Configuration
JOB_TIME="06:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_streamingllm_n2044_i4_Yarn_Qwen3_8B_nothinking_travel_planning_eval"

# Export variables
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export USE_REASONING_CONFIG
export ENABLE_THINKING
export SEED
export N_LOCAL
export N_INIT
