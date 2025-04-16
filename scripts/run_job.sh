#!/bin/bash

# Verify required variables are set
if [ -z "$BASE_CONFIGS" ] || [ -z "$CONTEXT_LENGTHS" ] || [ -z "$MODELS" ] || [ -z "$EXP_TYPE" ] || [ -z "$SEED" ]; then
    echo "Error: Required variables are not set"
    echo "BASE_CONFIGS: ${BASE_CONFIGS[@]}"
    echo "CONTEXT_LENGTHS: ${CONTEXT_LENGTHS[@]}"
    echo "MODELS: ${MODELS[@]}"
    echo "EXP_TYPE: $EXP_TYPE"
    echo "SEED: $SEED"
    exit 1
fi

# Convert delimiter-separated strings back to arrays
IFS='|||' read -ra BASE_CONFIGS <<< "$BASE_CONFIGS"
IFS='|||' read -ra CONTEXT_LENGTHS <<< "$CONTEXT_LENGTHS"
IFS='|||' read -ra MODELS <<< "$MODELS"

# Debug print the arrays after conversion
echo "Debug: BASE_CONFIGS after split = ${BASE_CONFIGS[@]}"
echo "Debug: CONTEXT_LENGTHS after split = ${CONTEXT_LENGTHS[@]}"

##############################
#       Job blueprint        #
##############################
# sbatch directives are all in the submit_job.sh file

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "Array Job ID                   = $SLURM_ARRAY_JOB_ID"
echo "Array Task ID                  = $SLURM_ARRAY_TASK_ID"
echo "Cache                          = $TRANSFORMERS_CACHE"

module purge
module load anaconda3/2023.3
module load gcc/11
source /scratch/gpfs/DANQIC/jz4391/MInference/minenv/bin/activate

# Get specific model for this run (if running as array job)
MNAME="${MODELS[$SLURM_ARRAY_TASK_ID % ${#MODELS[@]}]}"
MNAME="${MNAME## }"    # Remove leading spaces
MNAME="${MNAME%% }"    # Remove trailing spaces
MODEL_NAME="/scratch/gpfs/DANQIC/models/$MNAME"

# Create flattened array of all configs
declare -a ALL_CONFIGS=()
for context in "${CONTEXT_LENGTHS[@]}"; do
    for base in "${BASE_CONFIGS[@]}"; do
        ALL_CONFIGS+=("${base}_${context}.yaml")
    done
done

# Get the configuration for this array job
CONFIG="${ALL_CONFIGS[$SLURM_ARRAY_TASK_ID]}"

# Extract context length from config name
CONTEXT_LEN=$(echo $CONFIG | grep -o '[0-9]\+k')
CONTEXT_LEN="${CONTEXT_LEN## }"    # Remove leading spaces
CONTEXT_LEN="${CONTEXT_LEN%% }"    # Remove trailing spaces
TAG="${EXP_TYPE}_${CONTEXT_LEN}_${MNAME}_${SLURM_JOB_ID}"

OUTPUT_DIR="/scratch/gpfs/DANQIC/jz4391/HELMET/output/$EXP_TYPE/$CONTEXT_LEN/$MNAME"
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Set chat template option
shopt -s nocasematch
chat_models=".*(chat|instruct|it$|nous|command|Jamba-1.5|MegaBeam).*"
OPTIONS=""
if ! [[ $MNAME =~ $chat_models ]]; then
    OPTIONS="$OPTIONS --use_chat_template False"
fi

# Print debug information
echo "Model name            = $MNAME"
echo "Context length        = $CONTEXT_LEN"
echo "Config file          = $CONFIG"
echo "Evaluation output dir = $OUTPUT_DIR"
echo "Options              = $OPTIONS"

# Set experiment-specific parameter based on EXP_TYPE
EXP_TYPE_PARAM=""
case $EXP_TYPE in
    "streamingllm")
        EXP_TYPE_PARAM="--streamingllm"
        ;;
    "minference")
        EXP_TYPE_PARAM="--minference"
        ;;
    "snapkv")
        EXP_TYPE_PARAM="--snapkv"
        ;;
    "pyramidkv")
        EXP_TYPE_PARAM="--pyramidkv"
        ;;
    "kivi")
        EXP_TYPE_PARAM="--kivi"
        ;;
    "baseline")
        EXP_TYPE_PARAM=""
        ;;
    *)
        echo "Warning: Unknown experiment type $EXP_TYPE"
        EXP_TYPE_PARAM=""
        ;;
esac

echo "Debug: ALL_CONFIGS array = ${ALL_CONFIGS[@]}"
echo "Debug: SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "Debug: Selected CONFIG = $CONFIG"
echo "Debug: Full config path = configs/$CONFIG"

# Run evaluation for the single configuration
python /scratch/gpfs/DANQIC/jz4391/HELMET/eval.py \
    --config "/scratch/gpfs/DANQIC/jz4391/HELMET/configs/$CONFIG" \
    --seed $SEED \
    --output_dir $OUTPUT_DIR \
    --tag "$TAG" \
    --model_name_or_path $MODEL_NAME \
    $EXP_TYPE_PARAM \
    $OPTIONS

echo "Finished with exit code $?"