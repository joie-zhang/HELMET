# Inference technique evaluation config
# Technique: int4_quant, Model: Qwen3-8B, Task: niah, Context: 16k
declare -a BASE_CONFIGS=("niah")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("Qwen3-8B")
declare -a QUANTIZE=("4")
EXP_TYPE="baseline"
BENCHMARK="helmet"
USE_REASONING_CONFIG="false"
SEED=42



# SLURM Configuration
JOB_TIME="06:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_int4_quant_Qwen3_8B_niah_eval"

# Export variables
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export USE_REASONING_CONFIG
export SEED
