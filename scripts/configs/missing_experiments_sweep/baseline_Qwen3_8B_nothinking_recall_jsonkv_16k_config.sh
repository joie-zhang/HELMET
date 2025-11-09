# Inference technique evaluation config
# Technique: baseline, Model: Qwen3-8B, Task: recall_jsonkv, Context: 16k
# Enable Thinking: False (Qwen3 CoT mode)
declare -a BASE_CONFIGS=("recall_jsonkv")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("Qwen3-8B")
declare -a QUANTIZE=("16")
EXP_TYPE="baseline"
BENCHMARK="helmet"
USE_REASONING_CONFIG="false"
ENABLE_THINKING="False"
SEED=42



# SLURM Configuration
JOB_TIME="06:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_baseline_Qwen3_8B_nothinking_recall_jsonkv_eval"

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
