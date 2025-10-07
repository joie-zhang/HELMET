# Inference technique evaluation config
# Technique: streamingllm, Model: Qwen3-8B, Task: rerank, Context: 16k
declare -a BASE_CONFIGS=("rerank")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("Qwen3-8B")
declare -a QUANTIZE=("")
EXP_TYPE="streamingllm"
BENCHMARK="helmet"
USE_REASONING_CONFIG="true"
SEED=42

# StreamingLLM Configuration Parameters
N_LOCAL=4096
N_INIT=4

# SLURM Configuration
JOB_TIME="6:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_streamingllm_Qwen3_8B_rerank_eval"

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
