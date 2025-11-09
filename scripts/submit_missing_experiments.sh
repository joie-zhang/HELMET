#!/bin/bash

# SLURM submission script for missing experiments
# This script submits jobs for all generated missing experiment configs
#
# Usage:
#   ./scripts/submit_missing_experiments.sh                    # Submit all jobs
#   ./scripts/submit_missing_experiments.sh --test             # Submit only 1 job for testing
#   ./scripts/submit_missing_experiments.sh --dry-run          # Show what would be submitted without actually submitting
#
# Filter by context (additive - can combine multiple):
#   ./scripts/submit_missing_experiments.sh --helmet-16k       # Include 16k HELMET jobs
#   ./scripts/submit_missing_experiments.sh --helmet-32k       # Include 32k HELMET jobs
#   ./scripts/submit_missing_experiments.sh --longproc-0.5k    # Include 0.5k LongProc jobs
#   ./scripts/submit_missing_experiments.sh --longproc-2k      # Include 2k LongProc jobs
#   ./scripts/submit_missing_experiments.sh --longproc-8k      # Include 8k LongProc jobs
#
# Combinations (just add multiple flags):
#   ./scripts/submit_missing_experiments.sh --helmet-16k --longproc-0.5k    # 16k HELMET + 0.5k LongProc
#   ./scripts/submit_missing_experiments.sh --helmet-16k --helmet-32k       # All HELMET (both 16k and 32k)
#   ./scripts/submit_missing_experiments.sh --longproc-2k --longproc-8k     # LongProc 2k + 8k only

# Parse command line arguments
TEST_MODE=false
DRY_RUN=false

# Additive filters - each flag includes that category
INCLUDE_HELMET_16K=false
INCLUDE_HELMET_32K=false
INCLUDE_LONGPROC_05K=false
INCLUDE_LONGPROC_2K=false
INCLUDE_LONGPROC_8K=false

for arg in "$@"; do
    case $arg in
        --test|-t)
            TEST_MODE=true
            ;;
        --dry-run|-d)
            DRY_RUN=true
            ;;
        --helmet-16k)
            INCLUDE_HELMET_16K=true
            ;;
        --helmet-32k)
            INCLUDE_HELMET_32K=true
            ;;
        --longproc-0.5k)
            INCLUDE_LONGPROC_05K=true
            ;;
        --longproc-2k)
            INCLUDE_LONGPROC_2K=true
            ;;
        --longproc-8k)
            INCLUDE_LONGPROC_8K=true
            ;;
        *)
            echo "Unknown option: $arg"
            echo ""
            echo "Valid options:"
            echo "  --test, -t             Test with 1 job (1 hour time limit)"
            echo "  --dry-run, -d          Show what would be submitted"
            echo "  --helmet-16k           Include 16k HELMET jobs"
            echo "  --helmet-32k           Include 32k HELMET jobs"
            echo "  --longproc-0.5k        Include 0.5k LongProc jobs"
            echo "  --longproc-2k          Include 2k LongProc jobs"
            echo "  --longproc-8k          Include 8k LongProc jobs"
            echo ""
            echo "Combine multiple flags to select specific subsets:"
            echo "  ./scripts/submit_missing_experiments.sh --helmet-16k --longproc-0.5k"
            exit 1
            ;;
    esac
done

# Determine if any filters are active
ANY_FILTER_ACTIVE=false
if [ "$INCLUDE_HELMET_16K" = true ] || [ "$INCLUDE_HELMET_32K" = true ] || \
   [ "$INCLUDE_LONGPROC_05K" = true ] || [ "$INCLUDE_LONGPROC_2K" = true ] || \
   [ "$INCLUDE_LONGPROC_8K" = true ]; then
    ANY_FILTER_ACTIVE=true
fi

CONFIG_DIR="scripts/configs/missing_experiments_sweep"

# Check if config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory $CONFIG_DIR does not exist"
    echo "Please run ./scripts/generate_missing_experiments.sh first"
    exit 1
fi

# Find all config files
config_files=("$CONFIG_DIR"/*.sh)

if [ ${#config_files[@]} -eq 0 ] || [ ! -f "${config_files[0]}" ]; then
    echo "Error: No config files found in $CONFIG_DIR"
    echo "Please run ./scripts/generate_missing_experiments.sh first"
    exit 1
fi

echo "=========================================="
echo "Missing Experiments Submission"
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
        if [ "$INCLUDE_HELMET_16K" = true ] && [[ "$config_name" == *_16k_* ]] && [[ "$config_name" != *_0.5k_* ]] && [[ "$config_name" != *_2k_* ]] && [[ "$config_name" != *_8k_* ]]; then
            include=true
        fi

        if [ "$INCLUDE_HELMET_32K" = true ] && [[ "$config_name" == *_32k_* ]]; then
            include=true
        fi

        if [ "$INCLUDE_LONGPROC_05K" = true ] && [[ "$config_name" == *_0.5k_* ]]; then
            include=true
        fi

        if [ "$INCLUDE_LONGPROC_2K" = true ] && [[ "$config_name" == *_2k_* ]]; then
            include=true
        fi

        if [ "$INCLUDE_LONGPROC_8K" = true ] && [[ "$config_name" == *_8k_* ]]; then
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
    [ "$INCLUDE_HELMET_16K" = true ] && echo "  ✓ 16k HELMET jobs"
    [ "$INCLUDE_HELMET_32K" = true ] && echo "  ✓ 32k HELMET jobs"
    [ "$INCLUDE_LONGPROC_05K" = true ] && echo "  ✓ 0.5k LongProc jobs"
    [ "$INCLUDE_LONGPROC_2K" = true ] && echo "  ✓ 2k LongProc jobs"
    [ "$INCLUDE_LONGPROC_8K" = true ] && echo "  ✓ 8k LongProc jobs"
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
    # Select a Qwen3 thinking config for testing
    test_config=""
    for config_file in "${config_files[@]}"; do
        config_name=$(basename "$config_file")
        # Look for Qwen3 or Yarn-Qwen3 with thinking mode (tests ENABLE_THINKING=True)
        if [[ "$config_name" =~ (Qwen3_8B|Yarn_Qwen3_8B).*_thinking_ ]] && [[ "$config_name" != *"nothinking"* ]]; then
            test_config="$config_file"
            break
        fi
    done

    if [ -n "$test_config" ]; then
        config_files=("$test_config")
        echo "   Selected for testing: $(basename "$test_config")"
        echo "   (Qwen3 model with enable_thinking=True, 1 hour time limit)"
        echo "   (Uses regular configs/, NOT configs_reasoning/)"
    else
        # Fallback to first config if no Qwen3 thinking found
        config_files=("${config_files[0]}")
        echo "   No Qwen3 thinking config found, using: $(basename "${config_files[0]}")"
        echo "   (1 hour time limit)"
    fi
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
    if [[ "$config_name" == streamingllm_n4092* ]]; then
        category="StreamingLLM (4092+4)"
    elif [[ "$config_name" == streamingllm_n2044* ]]; then
        category="StreamingLLM (2044+4)"
    elif [[ "$config_name" == pyramidkv* ]]; then
        category="PyramidKV"
    elif [[ "$config_name" == snapkv* ]]; then
        category="SnapKV"
    elif [[ "$config_name" == int4_quant* ]]; then
        category="INT4 Quantization"
    elif [[ "$config_name" == int8_quant* ]]; then
        category="INT8 Quantization"
    elif [[ "$config_name" == baseline* ]]; then
        category="Baseline"
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
    echo "  /scratch/gpfs/DANQIC/jz4391/HELMET/output/{technique}/{context_length}/{model}/"
    echo ""
    echo "Example directories:"
    echo "  .../output/baseline/16k/Qwen2.5-7B-Instruct/"
    echo "  .../output/streamingllm/32k/Llama-3.1-8B-Instruct/"
    echo "  .../output/pyramidkv/2k/Yarn-Qwen3-8B/"
    echo ""
    echo "Check results aggregation:"
    echo "  python scripts/aggregate_results.py"
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
    [ "$INCLUDE_HELMET_16K" = true ] && echo "  ✓ 16k HELMET jobs"
    [ "$INCLUDE_HELMET_32K" = true ] && echo "  ✓ 32k HELMET jobs"
    [ "$INCLUDE_LONGPROC_05K" = true ] && echo "  ✓ 0.5k LongProc jobs"
    [ "$INCLUDE_LONGPROC_2K" = true ] && echo "  ✓ 2k LongProc jobs"
    [ "$INCLUDE_LONGPROC_8K" = true ] && echo "  ✓ 8k LongProc jobs"
    echo ""
    echo "To submit remaining jobs, use additional filters:"
    [ "$INCLUDE_HELMET_16K" = false ] && echo "  ./scripts/submit_missing_experiments.sh --helmet-16k"
    [ "$INCLUDE_HELMET_32K" = false ] && echo "  ./scripts/submit_missing_experiments.sh --helmet-32k"
    [ "$INCLUDE_LONGPROC_05K" = false ] && echo "  ./scripts/submit_missing_experiments.sh --longproc-0.5k"
    [ "$INCLUDE_LONGPROC_2K" = false ] && echo "  ./scripts/submit_missing_experiments.sh --longproc-2k"
    [ "$INCLUDE_LONGPROC_8K" = false ] && echo "  ./scripts/submit_missing_experiments.sh --longproc-8k"
    echo ""
    echo "Or combine multiple filters:"
    echo "  ./scripts/submit_missing_experiments.sh --helmet-32k --longproc-8k"
fi

echo ""
echo "=========================================="
