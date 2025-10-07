# Inference technique evaluation config
# Technique: int4_quant, Model: DeepSeek-R1-Distill-Llama-8B, Task: cite, Context: 16k
declare -a BASE_CONFIGS=("cite")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("DeepSeek-R1-Distill-Llama-8B")
declare -a QUANTIZE=("4")
EXP_TYPE="baseline"
BENCHMARK="helmet"
USE_REASONING_CONFIG="true"
SEED=42



# SLURM Configuration
JOB_TIME="12:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_int4_quant_DeepSeek_R1_Distill_Llama_8B_cite_eval"

# Export variables so they're available to the job script
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export USE_REASONING_CONFIG
export SEED
