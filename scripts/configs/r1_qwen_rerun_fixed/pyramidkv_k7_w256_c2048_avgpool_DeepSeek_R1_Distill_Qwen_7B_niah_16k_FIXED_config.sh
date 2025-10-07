# R1 Distill Qwen 7B Re-run Config with Chat Template Fix
# Technique: pyramidkv_k7_w256_c2048_avgpool, Model: DeepSeek-R1-Distill-Qwen-7B, Task: niah, Context: 16k
# FIX: chat template enabled to resolve hallucination issues
declare -a BASE_CONFIGS=("niah")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("DeepSeek-R1-Distill-Qwen-7B")
declare -a QUANTIZE=("16")
EXP_TYPE="pyramidkv"
BENCHMARK="helmet"
SEED=42

# CRITICAL FIX: Chat template already handled in run_job.sh for R1 models

# KV Cache Configuration Parameters
KV_TYPE="pyramidkv"
WINDOW_SIZE=256
MAX_CAPACITY_PROMPT=2048
KERNEL_SIZE=7
POOLING="avgpool"

# SLURM Configuration
JOB_TIME="3:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_pyramidkv_k7_w256_c2048_avgpool_DeepSeek_R1_Distill_Qwen_7B_niah_FIXED_eval"

# Export variables so they're available to the job script
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export SEED
# Note: Chat template handling is automatic in run_job.sh for R1 models
export KV_TYPE
export WINDOW_SIZE
export MAX_CAPACITY_PROMPT
export KERNEL_SIZE
export POOLING
