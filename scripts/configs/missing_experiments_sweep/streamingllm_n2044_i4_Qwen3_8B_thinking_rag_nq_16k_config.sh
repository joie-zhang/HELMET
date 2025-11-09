# Inference technique evaluation config
# Technique: streamingllm_n2044_i4, Model: Qwen3-8B, Task: rag_nq, Context: 16k
# Enable Thinking: True (Qwen3 CoT mode)
declare -a BASE_CONFIGS=("rag_nq")
declare -a CONTEXT_LENGTHS=("16k")
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
JOB_TIME="06:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_streamingllm_n2044_i4_Qwen3_8B_thinking_rag_nq_eval"

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
