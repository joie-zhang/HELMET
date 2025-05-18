# streamingllm_original_helmet_config.sh
# declare -a BASE_CONFIGS=("cite" "rerank" "recall_jsonkv" "rag_nq" "rag_hotpotqa" "niah")
declare -a BASE_CONFIGS=("niah")
declare -a CONTEXT_LENGTHS=("16k" "32k")
declare -a MODELS=("Llama-3.1-8B-Instruct")
# declare -a MODELS=("Qwen2.5-7B-Instruct")
# declare -a QUANTIZE=(16)  # Add quantization options
declare -a QUANTIZE=("")  # non-baseline experiments should not support quantize
EXP_TYPE="streamingllm_original"
BENCHMARK="helmet"
SEED=42

# StreamingLLM Original Configuration Parameters
N_LOCAL=4092                # Number of local tokens
N_INIT=4                    # Number of initial tokens

# SLURM Configuration
JOB_TIME="1:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_${CONTEXT_LENGTHS[-1]}_eval"

# Export variables so they're available to the job script
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export SEED
export N_LOCAL
export N_INIT 