# Inference technique evaluation config
# Technique: baseline, Model: DeepSeek-R1-Distill-Qwen-7B, Task: summ_multilex, Context: 16k
declare -a BASE_CONFIGS=("summ_multilex")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("DeepSeek-R1-Distill-Qwen-7B")
declare -a QUANTIZE=("8")
EXP_TYPE="baseline"
BENCHMARK="helmet"
USE_REASONING_CONFIG="false"
SEED=42


# SLURM Configuration
JOB_TIME="01:00:00"
JOB_NAME="baseline_DeepSeek_R1_Distill_Qwen_7B_summ_multilex_16k"

# Export variables
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export USE_REASONING_CONFIG
export SEED
