# Inference technique evaluation config
# Technique: streamingllm_n2044_i4, Model: DeepSeek-R1-Distill-Qwen-7B, Task: rag_hotpotqa, Context: 16k
declare -a BASE_CONFIGS=("rag_hotpotqa")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("DeepSeek-R1-Distill-Qwen-7B")
declare -a QUANTIZE=("")
EXP_TYPE="streamingllm"
BENCHMARK="helmet"
USE_REASONING_CONFIG="false"
SEED=42

# StreamingLLM Configuration Parameters
N_LOCAL=2044
N_INIT=4

# SLURM Configuration
JOB_TIME="06:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_streamingllm_n2044_i4_DeepSeek_R1_Distill_Qwen_7B_rag_hotpotqa_eval"

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
