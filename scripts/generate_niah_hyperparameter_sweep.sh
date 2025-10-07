#!/bin/bash

# Script to generate NIAH hyperparameter sweep configs for Qwen3-8B
# Tests different window_size and cache_size combinations for SnapKV and PyramidKV
# to find optimal cache configurations for better NIAH performance

# Model to test
MODEL="Qwen3-8B"

# Task and context
TASK="niah"
CONTEXT_LENGTH="16k"

# Hyperparameter ranges
WINDOW_SIZES=(2048 4096 8192)
CACHE_SIZES=(8192 16384 32768)  # 8192, 2×8192, 4×8192

# KV cache techniques to test
TECHNIQUES=("snapkv" "pyramidkv")

# Pooling methods
SNAPKV_POOLING="maxpool"
PYRAMIDKV_POOLING="avgpool"

# Fixed kernel size (commonly used value)
KERNEL_SIZE=7

# Common parameters
SEED=42
JOB_TIME="1:00:00"  # 1 hour should be enough for NIAH

# Output directory for generated configs
CONFIG_DIR="scripts/configs/niah_hyperparameter_sweep_qwen3_8b"
mkdir -p "$CONFIG_DIR"

echo "Generating NIAH hyperparameter sweep configs for $MODEL..."
echo "Task: $TASK (context: $CONTEXT_LENGTH)"
echo "Window sizes: ${WINDOW_SIZES[@]}"
echo "Cache sizes: ${CACHE_SIZES[@]}"
echo "Techniques: ${TECHNIQUES[@]}"
echo ""

config_count=0

# Function to create config file
create_config() {
    local technique="$1"
    local window_size="$2"
    local cache_size="$3"
    local pooling="$4"

    # Clean model name for filename
    local clean_model=$(echo "$MODEL" | sed 's/[^A-Za-z0-9]/_/g')

    # Create descriptive config name
    local config_name="${technique}_k${KERNEL_SIZE}_w${window_size}_c${cache_size}_${pooling}"
    local config_file="${CONFIG_DIR}/${config_name}_${clean_model}_${TASK}_${CONTEXT_LENGTH}_config.sh"

    cat > "$config_file" << EOF
# NIAH Hyperparameter Sweep Config
# Technique: ${technique}, Model: ${MODEL}, Window: ${window_size}, Cache: ${cache_size}
declare -a BASE_CONFIGS=("${TASK}")
declare -a CONTEXT_LENGTHS=("${CONTEXT_LENGTH}")
declare -a MODELS=("${MODEL}")
declare -a QUANTIZE=("")
EXP_TYPE="${technique}"
BENCHMARK="helmet"
SEED=${SEED}

# KV Cache Configuration Parameters
KV_TYPE="${technique}"
WINDOW_SIZE=${window_size}
MAX_CAPACITY_PROMPT=${cache_size}
KERNEL_SIZE=${KERNEL_SIZE}
POOLING="${pooling}"

# SLURM Configuration
JOB_TIME="${JOB_TIME}"
JOB_NAME="\${EXP_TYPE}_\${BENCHMARK}_${config_name}_${clean_model}_${TASK}_eval"

# Export variables so they're available to the job script
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export SEED
export KV_TYPE
export WINDOW_SIZE
export MAX_CAPACITY_PROMPT
export KERNEL_SIZE
export POOLING
EOF

    echo "Created: $(basename $config_file)"
    ((config_count++))
}

# Function to check if combination is valid (cache_size > window_size)
is_valid_combination() {
    local window_size=$1
    local cache_size=$2

    if [ $cache_size -gt $window_size ]; then
        return 0  # Valid
    else
        return 1  # Invalid
    fi
}

echo "=== GENERATING NIAH HYPERPARAMETER SWEEP CONFIGS ==="

# Generate configs for each technique
for technique in "${TECHNIQUES[@]}"; do
    echo ""
    echo "Generating configs for $technique..."

    # Set pooling method based on technique
    if [ "$technique" = "snapkv" ]; then
        pooling="$SNAPKV_POOLING"
    else
        pooling="$PYRAMIDKV_POOLING"
    fi

    # Generate all valid window_size × cache_size combinations
    for window_size in "${WINDOW_SIZES[@]}"; do
        for cache_size in "${CACHE_SIZES[@]}"; do
            if is_valid_combination $window_size $cache_size; then
                echo "  Creating: ${technique} w=${window_size} c=${cache_size} (valid)"
                create_config "$technique" "$window_size" "$cache_size" "$pooling"
            else
                echo "  Skipping: ${technique} w=${window_size} c=${cache_size} (invalid: cache_size <= window_size)"
            fi
        done
    done
done

echo ""
echo "Generated $config_count config files in $CONFIG_DIR"
echo ""

# Calculate and display the combinations
valid_combinations=0
total_combinations=0

echo "Summary of hyperparameter combinations:"
for technique in "${TECHNIQUES[@]}"; do
    echo ""
    echo "$technique:"
    for window_size in "${WINDOW_SIZES[@]}"; do
        valid_for_window=0
        for cache_size in "${CACHE_SIZES[@]}"; do
            total_combinations=$((total_combinations + 1))
            if is_valid_combination $window_size $cache_size; then
                valid_combinations=$((valid_combinations + 1))
                valid_for_window=$((valid_for_window + 1))
                echo "  ✓ window_size=$window_size, cache_size=$cache_size"
            else
                echo "  ✗ window_size=$window_size, cache_size=$cache_size (invalid)"
            fi
        done
        echo "    ($valid_for_window valid combinations for window_size=$window_size)"
    done
done

echo ""
echo "Total: $valid_combinations valid combinations out of $((${#TECHNIQUES[@]} * ${#WINDOW_SIZES[@]} * ${#CACHE_SIZES[@]})) possible"
echo "Expected configs: $valid_combinations (actual: $config_count)"

echo ""
echo "Next steps:"
echo "1. Test with a single job:"
echo "   ./scripts/submit_niah_hyperparameter_sweep.sh --test"
echo "2. Submit all jobs:"
echo "   ./scripts/submit_niah_hyperparameter_sweep.sh"
echo "3. After completion, analyze results:"
echo "   python scripts/plot_niah_hyperparameter_analysis.py --model Yarn-Qwen3-8B --context 16k"