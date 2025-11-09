# Inference technique evaluation config
# Technique: int8_quant, Model: Qwen3-8B, Task: travel_planning, Context: 0.5k
# Enable Thinking: False (Qwen3 CoT mode)
declare -a BASE_CONFIGS=("travel_planning")
declare -a CONTEXT_LENGTHS=("0.5k")
declare -a MODELS=("Qwen3-8B")
declare -a QUANTIZE=("8")
EXP_TYPE="baseline"
BENCHMARK="longproc"
USE_REASONING_CONFIG="false"
ENABLE_THINKING="False"
SEED=42



# SLURM Configuration
JOB_TIME="06:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_int8_quant_Qwen3_8B_nothinking_travel_planning_eval"

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
