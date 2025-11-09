# Inference technique evaluation config
# Technique: int8_quant, Model: Yarn-Qwen3-8B, Task: niah, Context: 16k
# Enable Thinking: True (Qwen3 CoT mode)
declare -a BASE_CONFIGS=("niah")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("Yarn-Qwen3-8B")
declare -a QUANTIZE=("8")
EXP_TYPE="baseline"
BENCHMARK="helmet"
USE_REASONING_CONFIG="false"
ENABLE_THINKING="True"
SEED=42



# SLURM Configuration
JOB_TIME="06:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_int8_quant_Yarn_Qwen3_8B_thinking_niah_eval"

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
