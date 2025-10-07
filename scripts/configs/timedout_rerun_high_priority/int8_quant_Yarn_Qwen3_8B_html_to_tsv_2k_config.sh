# Timed-out job rerun config - 12 hour time limit
# Technique: int8_quant, Model: Yarn-Qwen3-8B, Task: html_to_tsv, Context: 2k
declare -a BASE_CONFIGS=("html_to_tsv")
declare -a CONTEXT_LENGTHS=("2k")
declare -a MODELS=("Yarn-Qwen3-8B")
declare -a QUANTIZE=("8")
EXP_TYPE="baseline"
BENCHMARK="longproc"
SEED=42



# SLURM Configuration - 12 HOUR TIME LIMIT
JOB_TIME="12:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_int8_quant_Yarn_Qwen3_8B_html_to_tsv_eval"

# Export variables so they're available to the job script
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export SEED
