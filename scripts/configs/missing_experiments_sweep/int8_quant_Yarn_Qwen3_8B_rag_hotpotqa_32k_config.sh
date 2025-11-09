# Inference technique evaluation config
# Technique: int8_quant, Model: Yarn-Qwen3-8B, Task: rag_hotpotqa, Context: 32k
declare -a BASE_CONFIGS=("rag_hotpotqa")
declare -a CONTEXT_LENGTHS=("32k")
declare -a MODELS=("Yarn-Qwen3-8B")
declare -a QUANTIZE=("8")
EXP_TYPE="baseline"
BENCHMARK="helmet"
USE_REASONING_CONFIG="false"
SEED=42



# SLURM Configuration
JOB_TIME="18:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_int8_quant_Yarn_Qwen3_8B_rag_hotpotqa_eval"

# Export variables
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export USE_REASONING_CONFIG
export SEED
