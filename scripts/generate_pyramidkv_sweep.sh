#!/bin/bash

# Script to generate PyramidKV hyperparameter sweep configs
# This script creates individual config files for each hyperparameter combination

# Fixed parameters
BASE_CONFIG="cite"
CONTEXT_LENGTH="16k"
MODEL="DeepSeek-R1-Distill-Llama-8B"
EXP_TYPE="pyramidkv"
BENCHMARK="helmet"
SEED=42

# Fixed KV parameters
KV_TYPE="pyramidkv"
KERNEL_SIZE=5
POOLING="avgpool"

# Hyperparameter ranges to sweep (powers of 2)
WINDOW_SIZES=(32 64 128 256 512 1024 2048)
MAX_CAPACITY_PROMPTS=(64 128 256 512 1024 2048 4096 8192)

# Output directory for generated configs
CONFIG_DIR="scripts/configs/pyramidkv_sweep"
mkdir -p "$CONFIG_DIR"

# SLURM configuration
JOB_TIME="1:00:00"

echo "Generating PyramidKV hyperparameter sweep configs..."
echo "Window sizes: ${WINDOW_SIZES[@]}"
echo "Max capacity prompts: ${MAX_CAPACITY_PROMPTS[@]}"
echo "Constraint: max_capacity_prompt > window_size"
echo ""

config_count=0
skipped_count=0

for window_size in "${WINDOW_SIZES[@]}"; do
    for max_capacity in "${MAX_CAPACITY_PROMPTS[@]}"; do
        # Skip invalid combinations where max_capacity_prompt <= window_size
        if [ "$max_capacity" -le "$window_size" ]; then
            echo "Skipping invalid combination: window_size=$window_size, max_capacity=$max_capacity (max_capacity must be > window_size)"
            ((skipped_count++))
            continue
        fi

        config_file="${CONFIG_DIR}/pyramidkv_w${window_size}_c${max_capacity}_config.sh"

        cat > "$config_file" << EOF
# PyramidKV hyperparameter sweep config
# Window Size: ${window_size}, Max Capacity: ${max_capacity}
declare -a BASE_CONFIGS=("${BASE_CONFIG}")
declare -a CONTEXT_LENGTHS=("${CONTEXT_LENGTH}")
declare -a MODELS=("${MODEL}")
declare -a QUANTIZE=("")  # non-baseline experiments should not support quantize
EXP_TYPE="${EXP_TYPE}"
BENCHMARK="${BENCHMARK}"
SEED=${SEED}

# KV Cache Configuration Parameters
KV_TYPE="${KV_TYPE}"           # KV type
WINDOW_SIZE=${window_size}                # Window size for KV cache methods
MAX_CAPACITY_PROMPT=${max_capacity}      # Maximum capacity for prompt in KV cache
KERNEL_SIZE=${KERNEL_SIZE}                 # Kernel size for attention pooling
POOLING="${POOLING}"             # Pooling method for attention (maxpool or avgpool)

# SLURM Configuration
JOB_TIME="${JOB_TIME}"
JOB_NAME="\${EXP_TYPE}_\${BENCHMARK}_w${window_size}_c${max_capacity}_eval"

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

        echo "Created: $config_file (w=${window_size}, c=${max_capacity})"
        ((config_count++))
    done
done

echo ""
echo "Generated $config_count valid config files in $CONFIG_DIR"
echo "Skipped $skipped_count invalid combinations"
echo "Each config tests a valid Window Size × Max Capacity combination for PyramidKV"
echo ""
echo "Valid combinations generated:"
for window_size in "${WINDOW_SIZES[@]}"; do
    valid_capacities=()
    for max_capacity in "${MAX_CAPACITY_PROMPTS[@]}"; do
        if [ "$max_capacity" -gt "$window_size" ]; then
            valid_capacities+=("$max_capacity")
        fi
    done
    if [ ${#valid_capacities[@]} -gt 0 ]; then
        echo "  Window ${window_size}: Capacities [${valid_capacities[@]}]"
    fi
done