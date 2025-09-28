#!/bin/bash

# Script to check progress of PyramidKV hyperparameter sweep

echo "=== PyramidKV Hyperparameter Sweep Progress ==="
echo ""

# Check current job status
echo "Current SLURM job status:"
echo "========================="
squeue -u $USER --format="%.10i %.20j %.8T %.10M %.6D %.20R" | head -20
if [ $(squeue -u $USER | wc -l) -gt 21 ]; then
    echo "... (showing first 20 jobs)"
fi
echo ""

# Count jobs by status
total_jobs=$(squeue -u $USER | grep -c pyramidkv || echo "0")
running_jobs=$(squeue -u $USER | grep pyramidkv | grep -c " R " || echo "0")
pending_jobs=$(squeue -u $USER | grep pyramidkv | grep -c " PD " || echo "0")

echo "PyramidKV job summary:"
echo "====================="
echo "  Total PyramidKV jobs: $total_jobs"
echo "  Running: $running_jobs"
echo "  Pending: $pending_jobs"
echo ""

# Check output directories that have been created
output_base="/scratch/gpfs/DANQIC/jz4391/HELMET/output/pyramidkv/16k/DeepSeek-R1-Distill-Llama-8B"
if [ -d "$output_base" ]; then
    echo "Completed experiments (directories created):"
    echo "==========================================="
    completed_dirs=$(find "$output_base" -maxdepth 1 -type d -name "w*_c*" | wc -l)
    echo "  $completed_dirs out of 35 experiments have output directories"

    if [ $completed_dirs -gt 0 ]; then
        echo ""
        echo "Recent output directories:"
        find "$output_base" -maxdepth 1 -type d -name "w*_c*" -exec ls -ld {} \; | tail -10
    fi
else
    echo "No output directory created yet: $output_base"
fi

echo ""
echo "Remaining configs to submit:"
echo "============================"
remaining_configs=(
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

submitted_file="submitted_pyramidkv_jobs.txt"
if [ -f "$submitted_file" ]; then
    declare -A submitted_configs
    while IFS= read -r line; do
        submitted_configs["$line"]=1
    done < "$submitted_file"

    remaining_count=0
    for config in "${remaining_configs[@]}"; do
        if [[ ! ${submitted_configs["$config"]} ]]; then
            echo "  - $config"
            ((remaining_count++))
        fi
    done
    echo "  Total remaining: $remaining_count"
else
    echo "  All 10 configs from the failed batch"
    echo "  (No submission tracking file found)"
fi

echo ""
echo "Commands:"
echo "========="
echo "  Resubmit remaining jobs: ./scripts/resubmit_remaining_pyramidkv.sh"
echo "  Check this status again: ./scripts/check_pyramidkv_progress.sh"
echo "  Monitor jobs: squeue -u \$USER"