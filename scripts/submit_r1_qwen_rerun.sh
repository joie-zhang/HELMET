#!/bin/bash

# SLURM submission script for R1 Distill Qwen 7B re-run experiments
# This script submits the targeted re-run jobs to fix hallucination issues
#
# Usage:
#   ./scripts/submit_r1_qwen_rerun.sh           # Submit all 16 re-run jobs
#   ./scripts/submit_r1_qwen_rerun.sh --test    # Submit only 1 job for testing

# Parse command line arguments
TEST_MODE=false
if [ "$1" = "--test" ] || [ "$1" = "-t" ]; then
    TEST_MODE=true
fi

CONFIG_DIR="scripts/configs/r1_qwen_rerun_fixed"

# Check if config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory $CONFIG_DIR does not exist"
    echo "Please run ./scripts/generate_r1_qwen_rerun_configs.sh first"
    exit 1
fi

# Find all config files
config_files=("$CONFIG_DIR"/*FIXED_config.sh)

if [ ${#config_files[@]} -eq 0 ] || [ ! -f "${config_files[0]}" ]; then
    echo "Error: No FIXED config files found in $CONFIG_DIR"
    echo "Please run ./scripts/generate_r1_qwen_rerun_configs.sh first"
    exit 1
fi

echo "🔧 R1 Distill Qwen 7B Re-run Job Submission"
echo "Found ${#config_files[@]} FIXED config files in $CONFIG_DIR"

if [ "$TEST_MODE" = true ]; then
    echo ""
    echo "🧪 TEST MODE: Will submit only 1 job for testing"
    echo "   Use without --test flag to submit all 16 jobs"
    echo ""
    # Select only the first config file for testing
    config_files=("${config_files[0]}")
fi

echo ""
echo "Purpose: Fix hallucination issues in R1 Distill Qwen 7B by ensuring proper chat template usage"
echo "Expected: Significant improvement in response quality and task performance"
echo ""

# Create joblog directory if it doesn't exist
mkdir -p joblog_r1_qwen_rerun_fixed

# Submit jobs for each config
job_ids=()
submitted_count=0
failed_count=0

for config_file in "${config_files[@]}"; do
    config_name=$(basename "$config_file" .sh)
    echo "Submitting re-run job for: $config_name"

    # Submit the job and capture job ID
    job_output=$(./scripts/submit_job.sh "$config_file" 2>&1)
    job_id=$(echo "$job_output" | grep -o 'Submitted batch job [0-9]*' | grep -o '[0-9]*')

    if [ -n "$job_id" ]; then
        job_ids+=("$job_id")
        echo "  ✅ Job ID: $job_id"
        ((submitted_count++))
    else
        echo "  ❌ Error submitting job for $config_file"
        echo "  Output: $job_output"
        ((failed_count++))
    fi
done

echo ""
if [ "$TEST_MODE" = true ]; then
    echo "🧪 TEST MODE Summary:"
else
    echo "🔧 R1 Qwen Re-run Submission Summary:"
fi
echo "  Successfully submitted: $submitted_count jobs"
echo "  Failed submissions: $failed_count jobs"
echo "  Total job IDs: ${#job_ids[@]}"

if [ "$TEST_MODE" = true ] && [ $submitted_count -gt 0 ]; then
    echo ""
    echo "✅ Test re-run job submitted successfully!"
    echo "   Monitor the test job, and if it shows improved responses (no more hallucinations),"
    echo "   rerun without --test to submit all 16 re-run jobs"
fi

if [ ${#job_ids[@]} -gt 0 ]; then
    echo ""
    echo "Job IDs: ${job_ids[@]}"
    echo ""
    echo "Monitor jobs with:"
    echo "  squeue -u \\$USER"
    echo "  squeue -j $(IFS=','; echo "${job_ids[*]}")"
    echo ""
    echo "Cancel all re-run jobs with:"
    echo "  scancel $(IFS=' '; echo "${job_ids[*]}")"
    echo ""
    echo "Results will be saved to:"
    echo "  /scratch/gpfs/DANQIC/jz4391/HELMET/output/{snapkv,pyramidkv}/16k/DeepSeek-R1-Distill-Qwen-7B/"
    echo ""
    echo "Expected improvements:"
    echo "  ✅ Relevant responses to questions (no more hallucinations)"
    echo "  ✅ Higher F1 scores and task-specific metrics"
    echo "  ✅ Non-zero citation precision/recall for CITE task"
    echo "  ✅ Proper NIAH needle retrieval performance"
fi