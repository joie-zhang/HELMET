
# streamingllm_longproc_config.sh
declare -a BASE_CONFIGS=("countdown" "html_to_tsv")
declare -a CONTEXT_LENGTHS=("0.5k")
# declare -a BASE_CONFIGS=("countdown" "travel_planning" "html_to_tsv")
# declare -a CONTEXT_LENGTHS=("2k")
declare -a MODELS=("Llama-3.1-8B-Instruct")
EXP_TYPE="streamingllm"
BENCHMARK="longproc"
SEED=42

# SLURM Configuration
JOB_TIME="1:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_${CONTEXT_LENGTHS[-1]}_eval"

# Export variables so they're available to the job script
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export EXP_TYPE
export BENCHMARK
export SEED