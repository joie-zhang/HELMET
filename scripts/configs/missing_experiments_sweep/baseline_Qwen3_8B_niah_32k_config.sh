# Inference technique evaluation config
# Technique: baseline, Model: Qwen3-8B, Task: niah, Context: 32k
declare -a BASE_CONFIGS=("niah")
declare -a CONTEXT_LENGTHS=("32k")
declare -a MODELS=("Qwen3-8B")
declare -a QUANTIZE=("16")
EXP_TYPE="baseline"
BENCHMARK="helmet"
USE_REASONING_CONFIG="false"
SEED=42



# SLURM Configuration
JOB_TIME="18:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_baseline_Qwen3_8B_niah_eval"

# Export variables
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export USE_REASONING_CONFIG
export SEED
