#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <config_file>"
    echo "Example from HELMET root: $0 scripts/configs/streamingllm_helmet_config.sh"
    echo "Example from scripts dir: $0 configs/streamingllm_helmet_config.sh"
    exit 1
fi

source "$1"

# Calculate array size
ARRAY_MAX=$(( ${#BASE_CONFIGS[@]} * ${#CONTEXT_LENGTHS[@]} * ${#MODELS[@]} * ${#QUANTIZE[@]} - 1 ))

# Submit the job
sbatch \
    --export=ALL,BASE_CONFIGS="$(IFS='|||'; echo "${BASE_CONFIGS[*]}")",CONTEXT_LENGTHS="$(IFS='|||'; echo "${CONTEXT_LENGTHS[*]}")",MODELS="$(IFS='|||'; echo "${MODELS[*]}")",QUANTIZE="$(IFS='|||'; echo "${QUANTIZE[*]}")",EXP_TYPE="$EXP_TYPE",BENCHMARK="$BENCHMARK",SEED="$SEED",KV_TYPE="$KV_TYPE",WINDOW_SIZE="$WINDOW_SIZE",MAX_CAPACITY_PROMPT="$MAX_CAPACITY_PROMPT",KERNEL_SIZE="$KERNEL_SIZE",POOLING="$POOLING",N_LOCAL="$N_LOCAL",N_INIT="$N_INIT" \
    --array=0-$ARRAY_MAX \
    --time=$JOB_TIME \
    --job-name=$JOB_NAME \
    --gres=gpu:1 \
    --constraint=gpu80 \
    --ntasks-per-node=1 \
    --mail-type=ALL \
    --mail-user=joie@princeton.edu \
    --output=./joblog/%x-%A_%a.out \
    --error=./joblog/%x-%A_%a.err \
    -N 1 \
    -n 1 \
    --cpus-per-task=8 \
    --mem=50G \
    /scratch/gpfs/DANQIC/jz4391/HELMET/scripts/run_job.sh