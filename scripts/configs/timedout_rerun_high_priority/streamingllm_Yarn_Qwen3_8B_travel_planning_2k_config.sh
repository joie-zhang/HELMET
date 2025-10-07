# Timed-out job rerun config - 12 hour time limit
# Technique: streamingllm, Model: Yarn-Qwen3-8B, Task: travel_planning, Context: 2k
declare -a BASE_CONFIGS=("travel_planning")
declare -a CONTEXT_LENGTHS=("2k")
declare -a MODELS=("Yarn-Qwen3-8B")
declare -a QUANTIZE=("")
EXP_TYPE="streamingllm"
BENCHMARK="longproc"
SEED=42

# StreamingLLM Configuration Parameters
N_LOCAL=4096
N_INIT=4

# SLURM Configuration - 12 HOUR TIME LIMIT
JOB_TIME="12:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_streamingllm_Yarn_Qwen3_8B_travel_planning_eval"

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
