# Inference technique evaluation config
# Technique: int4_quant, Model: Yarn-Qwen3-8B, Task: pseudo_to_code, Context: 0.5k
# Enable Thinking: True (Qwen3 CoT mode)
declare -a BASE_CONFIGS=("pseudo_to_code")
declare -a CONTEXT_LENGTHS=("0.5k")
declare -a MODELS=("Yarn-Qwen3-8B")
declare -a QUANTIZE=("4")
EXP_TYPE="baseline"
BENCHMARK="longproc"
USE_REASONING_CONFIG="false"
ENABLE_THINKING="True"
SEED=42



# SLURM Configuration
JOB_TIME="06:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_int4_quant_Yarn_Qwen3_8B_thinking_pseudo_to_code_eval"

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
