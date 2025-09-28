#!/bin/bash

# SLURM submission script for PyramidKV hyperparameter sweep
# This script submits jobs for all generated PyramidKV configs

CONFIG_DIR="scripts/configs/pyramidkv_sweep"

# Check if config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory $CONFIG_DIR does not exist"
    echo "Please run ./scripts/generate_pyramidkv_sweep.sh first"
    exit 1
fi

# Find all config files
config_files=("$CONFIG_DIR"/*.sh)

if [ ${#config_files[@]} -eq 0 ] || [ ! -f "${config_files[0]}" ]; then
    echo "Error: No config files found in $CONFIG_DIR"
    echo "Please run ./scripts/generate_pyramidkv_sweep.sh first"
    exit 1
fi

echo "Found ${#config_files[@]} config files in $CONFIG_DIR"
echo ""

# Create joblog directory if it doesn't exist
mkdir -p joblog

# Submit jobs for each config
job_ids=()
for config_file in "${config_files[@]}"; do
    config_name=$(basename "$config_file" .sh)
    echo "Submitting job for: $config_name"

    # Submit the job and capture job ID
    job_output=$(./scripts/submit_job.sh "$config_file" 2>&1)
    job_id=$(echo "$job_output" | grep -o 'Submitted batch job [0-9]*' | grep -o '[0-9]*')

    if [ -n "$job_id" ]; then
        job_ids+=("$job_id")
        echo "  Job ID: $job_id"
    else
        echo "  Error submitting job for $config_file"
        echo "  Output: $job_output"
    fi
done

echo ""
echo "Submitted ${#job_ids[@]} jobs total"
if [ ${#job_ids[@]} -gt 0 ]; then
    echo "Job IDs: ${job_ids[@]}"
    echo ""
    echo "Monitor jobs with:"
    echo "  squeue -u \$USER"
    echo "  squeue -j $(IFS=','; echo "${job_ids[*]}")"
    echo ""
    echo "Cancel all jobs with:"
    echo "  scancel $(IFS=' '; echo "${job_ids[*]}")"
    echo ""
    echo "Results will be saved to:"
    echo "  /scratch/gpfs/DANQIC/jz4391/HELMET/output/pyramidkv/16k/DeepSeek-R1-Distill-Llama-8B/"
fi