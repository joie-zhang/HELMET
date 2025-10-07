#!/bin/bash

# SLURM submission script for NIAH hyperparameter sweep on Yarn-Qwen3-8B
# This script submits jobs for all generated SnapKV and PyramidKV hyperparameter configs
#
# Usage:
#   ./scripts/submit_niah_hyperparameter_sweep.sh           # Submit all jobs
#   ./scripts/submit_niah_hyperparameter_sweep.sh --test    # Submit only 1 job for testing

# Parse command line arguments
TEST_MODE=false
if [ "$1" = "--test" ] || [ "$1" = "-t" ]; then
    TEST_MODE=true
fi

CONFIG_DIR="scripts/configs/niah_hyperparameter_sweep_yarn_qwen3_8b"

# Check if config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory $CONFIG_DIR does not exist"
    echo "Please run ./scripts/generate_niah_hyperparameter_sweep.sh first"
    exit 1
fi

# Find all config files
config_files=("$CONFIG_DIR"/*.sh)

if [ ${#config_files[@]} -eq 0 ] || [ ! -f "${config_files[0]}" ]; then
    echo "Error: No config files found in $CONFIG_DIR"
    echo "Please run ./scripts/generate_niah_hyperparameter_sweep.sh first"
    exit 1
fi

echo "Found ${#config_files[@]} config files in $CONFIG_DIR"
echo "NIAH Hyperparameter Sweep: SnapKV and PyramidKV on Yarn-Qwen3-8B"

# Get currently running jobs to avoid duplicates
echo "Checking currently running jobs to avoid duplicates..."
running_jobs_file="/tmp/running_jobs_$$.txt"
squeue -u $USER --format="%.200j" | grep -v NAME | sed 's/_eval$//' | sort > "$running_jobs_file"
num_running=$(wc -l < "$running_jobs_file")
echo "Found $num_running currently running jobs"

# Function to check if a job is already running
is_job_running() {
    local job_name="$1"
    # Extract the key part of the job name (remove file extension and path)
    local base_name=$(basename "$job_name" .sh | sed 's/_config$//')

    # Check if this job pattern exists in running jobs
    grep -q "$base_name" "$running_jobs_file"
    return $?
}

if [ "$TEST_MODE" = true ]; then
    echo ""
    echo "🧪 TEST MODE: Will submit only 1 job for testing"
    echo "   Use without --test flag to submit all jobs"
    echo ""
fi

echo ""
echo "Filtering out already running jobs..."

# Create joblog directory if it doesn't exist
mkdir -p joblog

# Submit jobs for each config (excluding running ones)
job_ids=()
submitted_count=0
failed_count=0
skipped_count=0
configs_to_submit=()

for config_file in "${config_files[@]}"; do
    config_name=$(basename "$config_file" .sh)

    if is_job_running "$config_file"; then
        echo "⏭️  Skipping (already running): $config_name"
        ((skipped_count++))
        continue
    fi

    configs_to_submit+=("$config_file")
done

echo ""
echo "Summary before submission:"
echo "  Total configs: ${#config_files[@]}"
echo "  Already running: $skipped_count"
echo "  Need to submit: ${#configs_to_submit[@]}"

if [ ${#configs_to_submit[@]} -eq 0 ]; then
    echo ""
    echo "✅ All jobs are already running! Nothing to submit."
    rm -f "$running_jobs_file"
    exit 0
fi

if [ "$TEST_MODE" = true ] && [ ${#configs_to_submit[@]} -gt 0 ]; then
    # Select only the first config for testing
    configs_to_submit=("${configs_to_submit[0]}")
    echo "  Test mode: submitting only 1 job"
fi

echo ""

# Separate configs by technique for better organization
snapkv_configs=()
pyramidkv_configs=()

for config_file in "${configs_to_submit[@]}"; do
    config_name=$(basename "$config_file" .sh)
    if [[ "$config_name" == snapkv* ]]; then
        snapkv_configs+=("$config_file")
    elif [[ "$config_name" == pyramidkv* ]]; then
        pyramidkv_configs+=("$config_file")
    fi
done

echo "Submission plan:"
echo "  SnapKV configs: ${#snapkv_configs[@]}"
echo "  PyramidKV configs: ${#pyramidkv_configs[@]}"
echo ""

echo "Submitting NIAH hyperparameter sweep jobs..."

# Submit SnapKV configs first
if [ ${#snapkv_configs[@]} -gt 0 ]; then
    echo "Submitting SnapKV configurations..."
    for config_file in "${snapkv_configs[@]}"; do
        config_name=$(basename "$config_file" .sh)
        echo "  Submitting SnapKV: $config_name"

        # Submit the job and capture job ID
        job_output=$(./scripts/submit_job.sh "$config_file" 2>&1)
        job_id=$(echo "$job_output" | grep -o 'Submitted batch job [0-9]*' | grep -o '[0-9]*')

        if [ -n "$job_id" ]; then
            job_ids+=("$job_id")
            echo "    Job ID: $job_id"
            ((submitted_count++))
        else
            echo "    Error submitting job for $config_file"
            echo "    Output: $job_output"
            ((failed_count++))
        fi

        if [ "$TEST_MODE" = true ]; then
            break
        fi
    done
fi

# Submit PyramidKV configs
if [ ${#pyramidkv_configs[@]} -gt 0 ] && [ "$TEST_MODE" != true ]; then
    echo ""
    echo "Submitting PyramidKV configurations..."
    for config_file in "${pyramidkv_configs[@]}"; do
        config_name=$(basename "$config_file" .sh)
        echo "  Submitting PyramidKV: $config_name"

        # Submit the job and capture job ID
        job_output=$(./scripts/submit_job.sh "$config_file" 2>&1)
        job_id=$(echo "$job_output" | grep -o 'Submitted batch job [0-9]*' | grep -o '[0-9]*')

        if [ -n "$job_id" ]; then
            job_ids+=("$job_id")
            echo "    Job ID: $job_id"
            ((submitted_count++))
        else
            echo "    Error submitting job for $config_file"
            echo "    Output: $job_output"
            ((failed_count++))
        fi
    done
fi

# Cleanup temp files
rm -f "$running_jobs_file"

echo ""
if [ "$TEST_MODE" = true ]; then
    echo "🧪 TEST MODE Summary:"
else
    echo "📊 Submission Summary:"
fi
echo "  Already running (skipped): $skipped_count jobs"
echo "  Successfully submitted: $submitted_count jobs"
echo "  Failed submissions: $failed_count jobs"
echo "  Total job IDs: ${#job_ids[@]}"

if [ "$TEST_MODE" = true ] && [ $submitted_count -gt 0 ]; then
    echo ""
    echo "✅ Test job submitted successfully!"
    echo "   Monitor the test job, and if it runs correctly, rerun without --test to submit all hyperparameter sweep jobs"
fi

if [ ${#job_ids[@]} -gt 0 ]; then
    echo ""
    echo "New Job IDs: ${job_ids[@]}"
    echo ""
    echo "Monitor ALL jobs (including running ones) with:"
    echo "  squeue -u \$USER --format=\"%.10i %.150j %.2t %.10M %.12l %.20N\""
    echo ""
    echo "Monitor just the NEW jobs with:"
    echo "  squeue -j $(IFS=','; echo "${job_ids[*]}")"
    echo ""
    echo "Cancel just the NEW jobs with:"
    echo "  scancel $(IFS=' '; echo "${job_ids[*]}")"
    echo ""
    echo "Results will be saved to:"
    echo "  /scratch/gpfs/DANQIC/jz4391/HELMET/output/snapkv/16k/Yarn-Qwen3-8B/"
    echo "  /scratch/gpfs/DANQIC/jz4391/HELMET/output/pyramidkv/16k/Yarn-Qwen3-8B/"
fi

if [ "$TEST_MODE" != true ] && [ ${#job_ids[@]} -gt 0 ]; then
    echo ""
    echo "🔍 After all jobs complete, analyze results with:"
    echo "   python scripts/plot_niah_hyperparameter_analysis.py --model Yarn-Qwen3-8B --context 16k"
    echo ""
    echo "📊 Expected hyperparameter combinations to test:"
    echo "   SnapKV: 8 valid combinations (window_size × cache_size where cache_size > window_size)"
    echo "   PyramidKV: 8 valid combinations"
    echo "   Total: 16 hyperparameter experiments"
    echo ""
    echo "🎯 The analysis will help identify:"
    echo "   • Best performing cache configurations for NIAH task"
    echo "   • Memory vs performance trade-offs"
    echo "   • Optimal window_size and cache_size combinations"
    echo "   • Comparative performance between SnapKV and PyramidKV"
fi