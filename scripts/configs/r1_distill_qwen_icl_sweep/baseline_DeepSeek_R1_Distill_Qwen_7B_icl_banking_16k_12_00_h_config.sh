# DeepSeek-R1-Distill-Qwen-7B evaluation config
# Technique: baseline, Model: DeepSeek-R1-Distill-Qwen-7B, Task: icl_banking, Context: 16k, Time: 12:00:00
declare -a BASE_CONFIGS=("icl_banking")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("DeepSeek-R1-Distill-Qwen-7B")
declare -a QUANTIZE=("16")
EXP_TYPE="baseline"
BENCHMARK="helmet"
USE_REASONING_CONFIG="false"
SEED=42



# SLURM Configuration
JOB_TIME="12:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_baseline_DeepSeek_R1_Distill_Qwen_7B_icl_banking_eval"

# Export variables so they're available to the job script
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export USE_REASONING_CONFIG
export SEED
