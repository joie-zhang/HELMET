# Inference technique evaluation config
# Technique: streamingllm_n2044_i4, Model: DeepSeek-R1-Distill-Llama-8B, Task: pseudo_to_code, Context: 0.5k
declare -a BASE_CONFIGS=("pseudo_to_code")
declare -a CONTEXT_LENGTHS=("0.5k")
declare -a MODELS=("DeepSeek-R1-Distill-Llama-8B")
declare -a QUANTIZE=("")
EXP_TYPE="streamingllm"
BENCHMARK="longproc"
USE_REASONING_CONFIG="false"
SEED=42

# StreamingLLM Configuration Parameters
N_LOCAL=2044
N_INIT=4

# SLURM Configuration
JOB_TIME="06:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_streamingllm_n2044_i4_DeepSeek_R1_Distill_Llama_8B_pseudo_to_code_eval"

# Export variables
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export USE_REASONING_CONFIG
export SEED
export N_LOCAL
export N_INIT
