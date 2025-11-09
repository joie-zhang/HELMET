# Inference technique evaluation config
# Technique: streamingllm_n2044_i4, Model: Qwen3-8B, Task: recall_jsonkv, Context: 32k
# Enable Thinking: True (Qwen3 CoT mode)
declare -a BASE_CONFIGS=("recall_jsonkv")
declare -a CONTEXT_LENGTHS=("32k")
declare -a MODELS=("Qwen3-8B")
declare -a QUANTIZE=("")
EXP_TYPE="streamingllm"
BENCHMARK="helmet"
USE_REASONING_CONFIG="false"
ENABLE_THINKING="True"
SEED=42

# StreamingLLM Configuration Parameters
N_LOCAL=2044
N_INIT=4

# SLURM Configuration
JOB_TIME="18:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_streamingllm_n2044_i4_Qwen3_8B_thinking_recall_jsonkv_eval"

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
