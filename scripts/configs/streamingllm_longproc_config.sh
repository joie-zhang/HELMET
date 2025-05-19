# streamingllm_longproc_config.sh
# declare -a BASE_CONFIGS=("html_to_tsv" "pseudo_to_code")
# declare -a CONTEXT_LENGTHS=("0.5k")
# declare -a BASE_CONFIGS=("html_to_tsv" "pseudo_to_code" "travel_planning")
# declare -a CONTEXT_LENGTHS=("2k")
declare -a BASE_CONFIGS=("html_to_tsv" "travel_planning")
declare -a CONTEXT_LENGTHS=("8k")
# declare -a MODELS=("Llama-3.1-8B-Instruct")
declare -a MODELS=("Qwen2.5-7B-Instruct")
# declare -a QUANTIZE=(16)  # Add quantization options
declare -a QUANTIZE=("")  # non-baseline experiments should not support quantize
EXP_TYPE="streamingllm_original"
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