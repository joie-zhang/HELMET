#!/bin/bash

# Script to resubmit remaining PyramidKV configs that failed due to job limits
# This script tracks which configs have been submitted and only submits the remaining ones

CONFIG_DIR="scripts/configs/pyramidkv_sweep"
SUBMITTED_JOBS_FILE="submitted_pyramidkv_jobs.txt"

# Create joblog directory if it doesn't exist
mkdir -p joblog

# List of configs that failed to submit (from your output)
FAILED_CONFIGS=(
    "pyramidkv_w512_c2048_config.sh"
    "pyramidkv_w512_c4096_config.sh"
    "pyramidkv_w512_c8192_config.sh"
    "pyramidkv_w64_c1024_config.sh"
    "pyramidkv_w64_c128_config.sh"
    "pyramidkv_w64_c2048_config.sh"
    "pyramidkv_w64_c256_config.sh"
    "pyramidkv_w64_c4096_config.sh"
    "pyramidkv_w64_c512_config.sh"
    "pyramidkv_w64_c8192_config.sh"
)

echo "Resubmitting remaining PyramidKV configs..."
echo "Checking ${#FAILED_CONFIGS[@]} configs that previously failed"
echo ""

# Check current job count
current_jobs=$(squeue -u $USER | wc -l)
current_jobs=$((current_jobs - 1))  # Subtract header line
echo "Current jobs in queue: $current_jobs"

if [ $current_jobs -ge 30 ]; then
    echo "Warning: You still have $current_jobs jobs in queue."
    echo "Consider waiting for some to complete before submitting more."
    echo "Run 'squeue -u \$USER' to check job status."
    echo ""
fi

# Load previously submitted job configs (if file exists)
declare -A submitted_configs
if [ -f "$SUBMITTED_JOBS_FILE" ]; then
    while IFS= read -r line; do
        submitted_configs["$line"]=1
    done < "$SUBMITTED_JOBS_FILE"
fi

# Submit remaining configs
job_ids=()
submitted_count=0
skipped_count=0

for config in "${FAILED_CONFIGS[@]}"; do
    config_path="$CONFIG_DIR/$config"
    config_name=$(basename "$config" .sh)

    # Skip if already submitted
    if [[ ${submitted_configs["$config"]} ]]; then
        echo "Skipping $config_name (already submitted)"
        ((skipped_count++))
        continue
    fi

    # Check if config file exists
    if [ ! -f "$config_path" ]; then
        echo "Error: Config file not found: $config_path"
        continue
    fi

    echo "Submitting job for: $config_name"

    # Submit the job and capture job ID
    job_output=$(./scripts/submit_job.sh "$config_path" 2>&1)
    job_id=$(echo "$job_output" | grep -o 'Submitted batch job [0-9]*' | grep -o '[0-9]*')

    if [ -n "$job_id" ]; then
        job_ids+=("$job_id")
        echo "  Job ID: $job_id"

        # Record successful submission
        echo "$config" >> "$SUBMITTED_JOBS_FILE"
        ((submitted_count++))
    else
        echo "  Error submitting job for $config"
        echo "  Output: $job_output"

        # Check if it's still a limit issue
        if echo "$job_output" | grep -q "QOSMaxSubmitJobPerUserLimit"; then
            echo "  Still hitting job limit. Try again later when queue has space."
            break  # Stop trying if we hit the limit again
        fi
    fi

    # Small delay between submissions
    sleep 1
done

echo ""
echo "Resubmission complete!"
echo "  Submitted: $submitted_count new jobs"
echo "  Skipped: $skipped_count already submitted"
echo "  Total remaining: $(( ${#FAILED_CONFIGS[@]} - submitted_count - skipped_count ))"

if [ ${#job_ids[@]} -gt 0 ]; then
    echo "  New job IDs: ${job_ids[@]}"
    echo ""
    echo "Monitor all your jobs with:"
    echo "  squeue -u \$USER"
    echo ""
    echo "Cancel new jobs with:"
    echo "  scancel $(IFS=' '; echo "${job_ids[*]}")"
fi

echo ""
echo "To resubmit remaining configs later, run this script again:"
echo "  ./scripts/resubmit_remaining_pyramidkv.sh"