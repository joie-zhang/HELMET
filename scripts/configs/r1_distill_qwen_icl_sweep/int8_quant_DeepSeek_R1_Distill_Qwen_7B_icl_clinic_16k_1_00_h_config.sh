# DeepSeek-R1-Distill-Qwen-7B evaluation config
# Technique: int8_quant, Model: DeepSeek-R1-Distill-Qwen-7B, Task: icl_clinic, Context: 16k, Time: 1:00:00
declare -a BASE_CONFIGS=("icl_clinic")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("DeepSeek-R1-Distill-Qwen-7B")
declare -a QUANTIZE=("8")
EXP_TYPE="baseline"
BENCHMARK="helmet"
USE_REASONING_CONFIG="false"
SEED=42



# SLURM Configuration
JOB_TIME="1:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_int8_quant_DeepSeek_R1_Distill_Qwen_7B_icl_clinic_eval"

# Export variables so they're available to the job script
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export USE_REASONING_CONFIG
export SEED
