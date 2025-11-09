# Inference technique evaluation config
# Technique: int4_quant, Model: Qwen3-8B, Task: html_to_tsv, Context: 8k
# Enable Thinking: True (Qwen3 CoT mode)
declare -a BASE_CONFIGS=("html_to_tsv")
declare -a CONTEXT_LENGTHS=("8k")
declare -a MODELS=("Qwen3-8B")
declare -a QUANTIZE=("4")
EXP_TYPE="baseline"
BENCHMARK="longproc"
USE_REASONING_CONFIG="false"
ENABLE_THINKING="True"
SEED=42



# SLURM Configuration
JOB_TIME="18:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_int4_quant_Qwen3_8B_thinking_html_to_tsv_eval"

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
