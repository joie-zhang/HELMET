# Inference technique evaluation config
# Technique: int8_quant, Model: Llama-3.1-8B-Instruct, Task: pseudo_to_code, Context: 8k
declare -a BASE_CONFIGS=("pseudo_to_code")
declare -a CONTEXT_LENGTHS=("8k")
declare -a MODELS=("Llama-3.1-8B-Instruct")
declare -a QUANTIZE=("8")
EXP_TYPE="baseline"
BENCHMARK="longproc"
USE_REASONING_CONFIG="false"
SEED=42



# SLURM Configuration
JOB_TIME="18:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_int8_quant_Llama_3_1_8B_Instruct_pseudo_to_code_eval"

# Export variables
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export USE_REASONING_CONFIG
export SEED
