# Inference technique evaluation config
# Technique: streamingllm, Model: DeepSeek-R1-Distill-Llama-8B, Task: pseudo_to_code, Context: 2k
declare -a BASE_CONFIGS=("pseudo_to_code")
declare -a CONTEXT_LENGTHS=("2k")
declare -a MODELS=("DeepSeek-R1-Distill-Llama-8B")
declare -a QUANTIZE=("")
EXP_TYPE="streamingllm"
BENCHMARK="longproc"
SEED=42

# StreamingLLM Configuration Parameters
N_LOCAL=4096
N_INIT=4

# SLURM Configuration
JOB_TIME="3:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_streamingllm_DeepSeek_R1_Distill_Llama_8B_pseudo_to_code_eval"

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
