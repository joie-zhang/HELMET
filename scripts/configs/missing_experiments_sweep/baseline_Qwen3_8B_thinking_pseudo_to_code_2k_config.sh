# Inference technique evaluation config
# Technique: baseline, Model: Qwen3-8B, Task: pseudo_to_code, Context: 2k
# Enable Thinking: True (Qwen3 CoT mode)
declare -a BASE_CONFIGS=("pseudo_to_code")
declare -a CONTEXT_LENGTHS=("2k")
declare -a MODELS=("Qwen3-8B")
declare -a QUANTIZE=("16")
EXP_TYPE="baseline"
BENCHMARK="longproc"
USE_REASONING_CONFIG="false"
ENABLE_THINKING="True"
SEED=42



# SLURM Configuration
JOB_TIME="12:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_baseline_Qwen3_8B_thinking_pseudo_to_code_eval"

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
