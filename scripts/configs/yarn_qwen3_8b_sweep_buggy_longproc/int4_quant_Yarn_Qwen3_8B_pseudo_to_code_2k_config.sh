# Inference technique evaluation config
# Technique: int4_quant, Model: Yarn-Qwen3-8B, Task: pseudo_to_code, Context: 2k
declare -a BASE_CONFIGS=("pseudo_to_code")
declare -a CONTEXT_LENGTHS=("2k")
declare -a MODELS=("Yarn-Qwen3-8B")
declare -a QUANTIZE=("4")
EXP_TYPE="baseline"
BENCHMARK="longproc"
USE_REASONING_CONFIG="true"
SEED=42



# SLURM Configuration
JOB_TIME="6:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_int4_quant_Yarn_Qwen3_8B_pseudo_to_code_eval"

# Export variables so they're available to the job script
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export USE_REASONING_CONFIG
export SEED
