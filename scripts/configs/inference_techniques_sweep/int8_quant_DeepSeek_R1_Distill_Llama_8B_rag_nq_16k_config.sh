# Inference technique evaluation config
# Technique: int8_quant, Model: DeepSeek-R1-Distill-Llama-8B, Task: rag_nq, Context: 16k
declare -a BASE_CONFIGS=("rag_nq")
declare -a CONTEXT_LENGTHS=("16k")
declare -a MODELS=("DeepSeek-R1-Distill-Llama-8B")
declare -a QUANTIZE=("8")
EXP_TYPE="baseline"
BENCHMARK="helmet"
SEED=42



# SLURM Configuration
JOB_TIME="3:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_int8_quant_DeepSeek_R1_Distill_Llama_8B_rag_nq_eval"

# Export variables so they're available to the job script
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export SEED
