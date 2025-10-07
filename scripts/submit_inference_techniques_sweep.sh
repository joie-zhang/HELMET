#!/bin/bash

# SLURM submission script for inference techniques evaluation
# This script submits jobs for all generated inference techniques configs
#
# Usage:
#   ./scripts/submit_inference_techniques_sweep.sh           # Submit all jobs
#   ./scripts/submit_inference_techniques_sweep.sh --test    # Submit only 1 job for testing

# Parse command line arguments
TEST_MODE=false
if [ "$1" = "--test" ] || [ "$1" = "-t" ]; then
    TEST_MODE=true
fi

CONFIG_DIR="scripts/configs/r1_distill_qwen_icl_sweep"

# Check if config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory $CONFIG_DIR does not exist"
    echo "Please run ./scripts/generate_inference_techniques_sweep.sh first"
    exit 1
fi

# Find all config files
config_files=("$CONFIG_DIR"/*.sh)

if [ ${#config_files[@]} -eq 0 ] || [ ! -f "${config_files[0]}" ]; then
    echo "Error: No config files found in $CONFIG_DIR"
    echo "Please run ./scripts/generate_inference_techniques_sweep.sh first"
    exit 1
fi

echo "Found ${#config_files[@]} config files in $CONFIG_DIR"

if [ "$TEST_MODE" = true ]; then
    echo ""
    echo "🧪 TEST MODE: Will submit only 1 job for testing"
    echo "   Use without --test flag to submit all jobs"
    echo ""
    # Select only the first config file for testing
    config_files=("${config_files[0]}")
fi

echo ""

# Create joblog directory if it doesn't exist
mkdir -p joblog

# Submit jobs for each config
job_ids=()
submitted_count=0
failed_count=0

for config_file in "${config_files[@]}"; do
    config_name=$(basename "$config_file" .sh)
    echo "Submitting job for: $config_name"

    # Submit the job and capture job ID
    job_output=$(./scripts/submit_job.sh "$config_file" 2>&1)
    job_id=$(echo "$job_output" | grep -o 'Submitted batch job [0-9]*' | grep -o '[0-9]*')

    if [ -n "$job_id" ]; then
        job_ids+=("$job_id")
        echo "  Job ID: $job_id"
        ((submitted_count++))
    else
        echo "  Error submitting job for $config_file"
        echo "  Output: $job_output"
        ((failed_count++))
    fi
done

echo ""
if [ "$TEST_MODE" = true ]; then
    echo "🧪 TEST MODE Summary:"
else
    echo "Submission Summary:"
fi
echo "  Successfully submitted: $submitted_count jobs"
echo "  Failed submissions: $failed_count jobs"
echo "  Total job IDs: ${#job_ids[@]}"

if [ "$TEST_MODE" = true ] && [ $submitted_count -gt 0 ]; then
    echo ""
    echo "✅ Test job submitted successfully!"
    echo "   Monitor the test job, and if it runs correctly, rerun without --test to submit all 72 jobs"
fi

if [ ${#job_ids[@]} -gt 0 ]; then
    echo ""
    echo "Job IDs: ${job_ids[@]}"
    echo ""
    echo "Monitor jobs with:"
    echo "  squeue -u \$USER"
    echo "  squeue -j $(IFS=','; echo "${job_ids[*]}")"
    echo ""
    echo "Cancel all jobs with:"
    echo "  scancel $(IFS=' '; echo "${job_ids[*]}")"
    echo ""
    echo "Results will be saved to various output directories based on technique and model:"
    echo "  /scratch/gpfs/DANQIC/jz4391/HELMET/output/{technique}/{context_length}/{model}/"
    echo ""
    echo "Example directories:"
    echo "  /scratch/gpfs/DANQIC/jz4391/HELMET/output/baseline/16k/Llama-3.1-8B-Instruct/"
    echo "  /scratch/gpfs/DANQIC/jz4391/HELMET/output/snapkv/16k/Llama-3.1-8B-Instruct/"
    echo "  /scratch/gpfs/DANQIC/jz4391/HELMET/output/pyramidkv/2k/Llama-3.1-8B-Instruct/"
fi