# Inference technique evaluation config
# Technique: baseline, Model: Qwen2.5-7B-Instruct, Task: travel_planning, Context: 8k
declare -a BASE_CONFIGS=("travel_planning")
declare -a CONTEXT_LENGTHS=("8k")
declare -a MODELS=("Qwen2.5-7B-Instruct")
declare -a QUANTIZE=("16")
EXP_TYPE="baseline"
BENCHMARK="longproc"
USE_REASONING_CONFIG="false"
SEED=42



# SLURM Configuration
JOB_TIME="18:00:00"
JOB_NAME="${EXP_TYPE}_${BENCHMARK}_baseline_Qwen2_5_7B_Instruct_travel_planning_eval"

# Export variables
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export USE_REASONING_CONFIG
export SEED
