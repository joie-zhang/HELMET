#!/bin/bash

# Submit priority jobs for missing experiments
# Usage:
#   ./submit_priority_jobs.sh summ_multilex [--test] [--dry-run]    # Submit summ_multilex jobs (Priority 1)
#   ./submit_priority_jobs.sh streamingllm_8k [--test] [--dry-run]  # Submit StreamingLLM 8k jobs (Priority 2)
#   ./submit_priority_jobs.sh all [--test] [--dry-run]              # Submit all priority jobs

# Parse arguments
PRIORITY=""
TEST_MODE=false
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        summ_multilex|streamingllm_8k|all)
            PRIORITY="$arg"
            ;;
        --test|-t)
            TEST_MODE=true
            ;;
        --dry-run|-d)
            DRY_RUN=true
            ;;
        *)
            echo "Unknown option: $arg"
            echo ""
            echo "Usage:"
            echo "  ./submit_priority_jobs.sh summ_multilex [--test] [--dry-run]"
            echo "  ./submit_priority_jobs.sh streamingllm_8k [--test] [--dry-run]"
            echo "  ./submit_priority_jobs.sh all [--test] [--dry-run]"
            echo ""
            echo "Options:"
            echo "  --test, -t      Test with 1 job (1 hour time limit)"
            echo "  --dry-run, -d   Show what would be submitted"
            exit 1
            ;;
    esac
done

if [ -z "$PRIORITY" ]; then
    echo "Error: Please specify priority (summ_multilex, streamingllm_8k, or all)"
    echo ""
    echo "Usage:"
    echo "  ./submit_priority_jobs.sh summ_multilex [--test] [--dry-run]"
    echo "  ./submit_priority_jobs.sh streamingllm_8k [--test] [--dry-run]"
    echo "  ./submit_priority_jobs.sh all [--test] [--dry-run]"
    exit 1
fi

echo "=========================================="
echo "Priority Job Submission"
echo "=========================================="
echo "Priority: $PRIORITY"
if [ "$TEST_MODE" = true ]; then
    echo "Mode: TEST (1 job, 1 hour limit)"
elif [ "$DRY_RUN" = true ]; then
    echo "Mode: DRY RUN"
else
    echo "Mode: FULL SUBMISSION"
fi
echo ""

# Create joblog directory if it doesn't exist
mkdir -p joblog

# Get job lists
config_files=()

if [ "$PRIORITY" = "summ_multilex" ] || [ "$PRIORITY" = "all" ]; then
    echo "Loading summ_multilex configs..."
    while IFS= read -r line; do
        config_files+=("$line")
    done < priority1_summ_multilex.txt
    echo "  Added ${#config_files[@]} summ_multilex configs"
fi

prev_count=${#config_files[@]}

if [ "$PRIORITY" = "streamingllm_8k" ] || [ "$PRIORITY" = "all" ]; then
    echo "Loading StreamingLLM 8k configs..."
    while IFS= read -r line; do
        config_files+=("$line")
    done < priority2_streamingllm_8k.txt
    new_count=$((${#config_files[@]} - prev_count))
    echo "  Added $new_count StreamingLLM 8k configs"
fi

echo ""
echo "Total configs to submit: ${#config_files[@]}"
echo ""

if [ ${#config_files[@]} -eq 0 ]; then
    echo "Error: No config files found"
    exit 1
fi

if [ "$TEST_MODE" = true ]; then
    echo "🧪 TEST MODE: Will submit only 1 job"
    echo ""
    # Select first config
    config_files=("${config_files[0]}")
    echo "   Selected: $(basename "${config_files[0]}")"
    echo ""
elif [ "$DRY_RUN" = true ]; then
    echo "🔍 DRY RUN MODE: Showing what would be submitted"
    echo ""
fi

# Submit jobs
job_ids=()
submitted_count=0
failed_count=0

for config_file in "${config_files[@]}"; do
    config_name=$(basename "$config_file" .sh)

    if [ "$DRY_RUN" = true ]; then
        echo "Would submit: $config_name"
        ((submitted_count++))
    else
        echo "Submitting: $config_name"

        # Submit the job
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
            echo "  ✗ Error submitting job"
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

if [ ${#job_ids[@]} -gt 0 ]; then
    echo ""
    echo "=========================================="
    echo "Job Management"
    echo "=========================================="
    echo "Monitor jobs:"
    echo "  squeue -u \$USER"
    echo "  squeue -j $(IFS=','; echo "${job_ids[*]}")"
    echo ""
    echo "Cancel all submitted jobs:"
    echo "  scancel $(IFS=' '; echo "${job_ids[*]}")"
fi

if [ "$TEST_MODE" = true ] && [ $submitted_count -gt 0 ]; then
    echo ""
    echo "✅ Test job submitted!"
    echo "   Monitor the test job, then rerun without --test to submit all jobs"
fi

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "✅ Dry run complete!"
    echo "   Review the jobs above, then run without --dry-run to submit"
fi

echo ""
echo "=========================================="
