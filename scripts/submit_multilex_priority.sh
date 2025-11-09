#!/bin/bash

# SLURM submission script for high-priority summ_multilex experiments
# This script submits the 20 critical experiments needed to complete the plots
#
# Usage:
#   ./scripts/submit_multilex_priority.sh                    # Submit all 20 jobs
#   ./scripts/submit_multilex_priority.sh --test             # Submit only 1 job for testing
#   ./scripts/submit_multilex_priority.sh --dry-run          # Show what would be submitted without actually submitting
#
# Filter by technique (additive - can combine multiple):
#   ./scripts/submit_multilex_priority.sh --baseline         # Submit baseline/INT4/INT8 jobs (6 total)
#   ./scripts/submit_multilex_priority.sh --snapkv           # Submit SnapKV jobs (8 total)
#   ./scripts/submit_multilex_priority.sh --pyramidkv        # Submit PyramidKV jobs (6 total)
#
# Combinations:
#   ./scripts/submit_multilex_priority.sh --baseline --snapkv    # Baseline + SnapKV (14 jobs)

# Parse command line arguments
TEST_MODE=false
DRY_RUN=false

# Additive filters
INCLUDE_BASELINE=false
INCLUDE_SNAPKV=false
INCLUDE_PYRAMIDKV=false

for arg in "$@"; do
    case $arg in
        --test|-t)
            TEST_MODE=true
            ;;
        --dry-run|-d)
            DRY_RUN=true
            ;;
        --baseline)
            INCLUDE_BASELINE=true
            ;;
        --snapkv)
            INCLUDE_SNAPKV=true
            ;;
        --pyramidkv)
            INCLUDE_PYRAMIDKV=true
            ;;
        *)
            echo "Unknown option: $arg"
            echo ""
            echo "Valid options:"
            echo "  --test, -t             Test with 1 job (1 hour time limit)"
            echo "  --dry-run, -d          Show what would be submitted"
            echo "  --baseline             Include baseline/INT4/INT8 jobs (6 total)"
            echo "  --snapkv               Include SnapKV jobs (8 total)"
            echo "  --pyramidkv            Include PyramidKV jobs (6 total)"
            echo ""
            echo "Combine multiple flags to select specific subsets:"
            echo "  ./scripts/submit_multilex_priority.sh --baseline --snapkv"
            exit 1
            ;;
    esac
done

# Determine if any filters are active
ANY_FILTER_ACTIVE=false
if [ "$INCLUDE_BASELINE" = true ] || [ "$INCLUDE_SNAPKV" = true ] || [ "$INCLUDE_PYRAMIDKV" = true ]; then
    ANY_FILTER_ACTIVE=true
fi

CONFIG_DIR="scripts/configs/multilex_priority"

# Check if config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory $CONFIG_DIR does not exist"
    echo "Please run: python scripts/generate_multilex_priority_configs.py"
    exit 1
fi

# Find all config files
config_files=("$CONFIG_DIR"/*.sh)

if [ ${#config_files[@]} -eq 0 ] || [ ! -f "${config_files[0]}" ]; then
    echo "Error: No config files found in $CONFIG_DIR"
    echo "Please run: python scripts/generate_multilex_priority_configs.py"
    exit 1
fi

echo "=========================================="
echo "High-Priority Multi-Lexsum Experiments"
echo "=========================================="
echo "Found ${#config_files[@]} total config files in $CONFIG_DIR"

# Apply filters using additive approach
filtered_files=()
skipped_count=0

for config_file in "${config_files[@]}"; do
    config_name=$(basename "$config_file")
    include=false

    # If no filters are active, include everything
    if [ "$ANY_FILTER_ACTIVE" = false ]; then
        include=true
    else
        # Check if this config matches any active filter
        if [ "$INCLUDE_BASELINE" = true ] && [[ "$config_name" == baseline_* || "$config_name" == INT4_* || "$config_name" == INT8_* ]]; then
            include=true
        fi

        if [ "$INCLUDE_SNAPKV" = true ] && [[ "$config_name" == snapkv_* ]]; then
            include=true
        fi

        if [ "$INCLUDE_PYRAMIDKV" = true ] && [[ "$config_name" == pyramidkv_* ]]; then
            include=true
        fi
    fi

    if [ "$include" = true ]; then
        filtered_files+=("$config_file")
    else
        ((skipped_count++))
    fi
done

# Update config_files to filtered list
config_files=("${filtered_files[@]}")

echo "After filtering: ${#config_files[@]} jobs to submit"
if [ $skipped_count -gt 0 ]; then
    echo "  Skipped: $skipped_count jobs"
fi
echo ""

# Show active filters
if [ "$ANY_FILTER_ACTIVE" = true ]; then
    echo "Active filters (additive):"
    [ "$INCLUDE_BASELINE" = true ] && echo "  ✓ Baseline/INT4/INT8 jobs"
    [ "$INCLUDE_SNAPKV" = true ] && echo "  ✓ SnapKV jobs"
    [ "$INCLUDE_PYRAMIDKV" = true ] && echo "  ✓ PyramidKV jobs"
    echo ""
fi

if [ ${#config_files[@]} -eq 0 ]; then
    echo "Error: No config files match the specified filters"
    exit 1
fi

if [ "$TEST_MODE" = true ]; then
    echo ""
    echo "🧪 TEST MODE: Will submit only 1 job for testing"
    echo "   - Job time will be overridden to 1 hour"
    echo "   - Use without --test flag to submit all jobs"
    echo ""
    # Select first config for testing
    config_files=("${config_files[0]}")
    echo "   Selected for testing: $(basename "${config_files[0]}")"
    echo "   (1 hour time limit)"
elif [ "$DRY_RUN" = true ]; then
    echo ""
    echo "🔍 DRY RUN MODE: Will show what would be submitted without actually submitting"
    echo ""
fi

echo ""

# Create joblog directory if it doesn't exist
mkdir -p joblog

# Submit jobs for each config
job_ids=()
submitted_count=0
failed_count=0

# Group configs by category for better organization
declare -A category_counts

for config_file in "${config_files[@]}"; do
    config_name=$(basename "$config_file" .sh)

    # Determine category from filename
    category="other"
    if [[ "$config_name" == baseline_* ]]; then
        category="Baseline (16k)"
    elif [[ "$config_name" == INT4_* ]]; then
        category="INT4 (16k)"
    elif [[ "$config_name" == INT8_* ]]; then
        category="INT8 (16k)"
    elif [[ "$config_name" == snapkv_* ]]; then
        category="SnapKV (16k)"
    elif [[ "$config_name" == pyramidkv_* ]]; then
        category="PyramidKV (16k)"
    fi

    # Track category counts
    ((category_counts["$category"]++))

    if [ "$DRY_RUN" = true ]; then
        echo "Would submit: $config_name (Category: $category)"
        ((submitted_count++))
    else
        echo "Submitting: $config_name (Category: $category)"

        # Submit the job and capture job ID
        # In test mode, override time to 1 hour
        if [ "$TEST_MODE" = true ]; then
            job_output=$(./scripts/submit_job.sh "$config_file" "01:00:00" 2>&1)
        else
            job_output=$(./scripts/submit_job.sh "$config_file" 2>&1)
        fi
        job_id=$(echo "$job_output" | grep -o 'Submitted batch job [0-9]*' | grep -o '[0-9]*')

        if [ -n "$job_id" ]; then
            job_ids+=("$job_id")
            echo "  ✓ Job ID: $job_id"
            ((submitted_count++))
        else
            echo "  ✗ Error submitting job for $config_file"
            echo "  Output: $job_output"
            ((failed_count++))
        fi
    fi
done

echo ""
echo "=========================================="
if [ "$DRY_RUN" = true ]; then
    echo "Dry Run Summary:"
    echo "  Would submit: $submitted_count jobs"
elif [ "$TEST_MODE" = true ]; then
    echo "Test Mode Summary:"
    echo "  Successfully submitted: $submitted_count jobs"
    echo "  Failed submissions: $failed_count jobs"
else
    echo "Submission Summary:"
    echo "  Successfully submitted: $submitted_count jobs"
    echo "  Failed submissions: $failed_count jobs"
    echo "  Total job IDs: ${#job_ids[@]}"
fi

echo ""
echo "Jobs by Category:"
for category in "${!category_counts[@]}"; do
    echo "  $category: ${category_counts[$category]} jobs"
done

if [ "$TEST_MODE" = true ] && [ $submitted_count -gt 0 ]; then
    echo ""
    echo "✅ Test job submitted successfully!"
    echo "   Monitor the test job, and if it runs correctly, rerun without --test to submit all jobs"
fi

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "✅ Dry run complete!"
    echo "   Review the jobs above, then run without --dry-run to submit"
fi

if [ ${#job_ids[@]} -gt 0 ]; then
    echo ""
    echo "=========================================="
    echo "Monitoring & Management"
    echo "=========================================="
    echo "Job IDs: ${job_ids[@]}"
    echo ""
    echo "Monitor jobs:"
    echo "  squeue -u \$USER"
    echo "  squeue -j $(IFS=','; echo "${job_ids[*]}")"
    echo ""
    echo "Cancel all jobs:"
    echo "  scancel $(IFS=' '; echo "${job_ids[*]}")"
    echo ""
    echo "Results will be saved to:"
    echo "  /scratch/gpfs/DANQIC/jz4391/HELMET/output/{technique}/16k/{model}/"
    echo ""
    echo "After jobs complete, run GPT-4 evaluation:"
    echo "  python scripts/eval_gpt4_summ.py"
    echo ""
    echo "Then re-collect results:"
    echo "  python scripts/collect_results_new.py"
    echo ""
    echo "Finally regenerate plots:"
    echo "  python scripts/plot_task_deltas_averaged_configs.py"
    echo "  python scripts/plot_task_deltas_separate_configs.py"
fi

# Show reminder about remaining jobs if filters were used
if [ "$ANY_FILTER_ACTIVE" = true ]; then
    echo ""
    echo "=========================================="
    echo "Reminder: Remaining Jobs"
    echo "=========================================="
    echo "⚠️  You used filters, so some jobs were not submitted."
    echo ""
    echo "What you submitted:"
    [ "$INCLUDE_BASELINE" = true ] && echo "  ✓ Baseline/INT4/INT8 jobs (6 total)"
    [ "$INCLUDE_SNAPKV" = true ] && echo "  ✓ SnapKV jobs (8 total)"
    [ "$INCLUDE_PYRAMIDKV" = true ] && echo "  ✓ PyramidKV jobs (6 total)"
    echo ""
    echo "To submit remaining jobs, use additional filters:"
    [ "$INCLUDE_BASELINE" = false ] && echo "  ./scripts/submit_multilex_priority.sh --baseline"
    [ "$INCLUDE_SNAPKV" = false ] && echo "  ./scripts/submit_multilex_priority.sh --snapkv"
    [ "$INCLUDE_PYRAMIDKV" = false ] && echo "  ./scripts/submit_multilex_priority.sh --pyramidkv"
    echo ""
    echo "Or combine multiple filters:"
    echo "  ./scripts/submit_multilex_priority.sh --baseline --snapkv"
fi

echo ""
echo "=========================================="
echo "📊 These 20 experiments will complete:"
echo "  - Baseline/INT4/INT8 for DeepSeek models (16k)"
echo "  - SnapKV w256 & w2048 for all 4 models (16k)"
echo "  - PyramidKV w256 & w2048 for 3 models (16k)"
echo ""
echo "After completion, your plots will have full summ_multilex data!"
echo "=========================================="
