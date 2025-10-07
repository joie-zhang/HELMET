#!/bin/bash

# Script to generate KV cache hyperparameter sweep configs
# This script creates individual config files for each hyperparameter combination

# Fixed parameters
MODEL="DeepSeek-R1-Distill-Llama-8B"
SEED=42

# Benchmarks with their configurations
declare -A BENCHMARK_CONFIGS=(
    ["helmet"]="icl 16k"
    ["longproc"]="travel_planning 2k"
)

# KV cache methods with their configurations
declare -A KV_METHODS=(
    ["pyramidkv"]="avgpool"
    ["snapkv"]="maxpool"
)

KERNEL_SIZE=7

# Specific window size - max capacity combinations to test
declare -A WINDOW_CAPACITY_PAIRS=(
    ["64"]="512 1024 2048"
    ["128"]="512"
    ["256"]="512"
)

# Output directory for generated configs
CONFIG_DIR="scripts/configs/r1-distill-llama-push-left-frontier-sweep-smaller-caches"
mkdir -p "$CONFIG_DIR"

# SLURM configuration
JOB_TIME="4:00:00"

echo "Generating KV cache hyperparameter sweep configs..."
echo "Benchmarks: ${!BENCHMARK_CONFIGS[@]}"
echo "KV Methods: ${!KV_METHODS[@]}"
echo "Window-Capacity pairs:"
for window in "${!WINDOW_CAPACITY_PAIRS[@]}"; do
    echo "  Window ${window}: Capacities [${WINDOW_CAPACITY_PAIRS[$window]}]"
done
echo ""

config_count=0

for benchmark in "${!BENCHMARK_CONFIGS[@]}"; do
    # Parse benchmark config
    read -r base_config context_length <<< "${BENCHMARK_CONFIGS[$benchmark]}"

    for kv_method in "${!KV_METHODS[@]}"; do
        pooling="${KV_METHODS[$kv_method]}"

        for window_size in "${!WINDOW_CAPACITY_PAIRS[@]}"; do
            for max_capacity in ${WINDOW_CAPACITY_PAIRS[$window_size]}; do
                config_file="${CONFIG_DIR}/${kv_method}_${benchmark}_w${window_size}_c${max_capacity}_config.sh"

                cat > "$config_file" << EOF
# ${kv_method} hyperparameter sweep config
# Benchmark: ${benchmark}, Window Size: ${window_size}, Max Capacity: ${max_capacity}
declare -a BASE_CONFIGS=("${base_config}")
declare -a CONTEXT_LENGTHS=("${context_length}")
declare -a MODELS=("${MODEL}")
declare -a QUANTIZE=("")  # non-baseline experiments should not support quantize
EXP_TYPE="${kv_method}"
BENCHMARK="${benchmark}"
SEED=${SEED}

# KV Cache Configuration Parameters
KV_TYPE="${kv_method}"           # KV type
WINDOW_SIZE=${window_size}                # Window size for KV cache methods
MAX_CAPACITY_PROMPT=${max_capacity}      # Maximum capacity for prompt in KV cache
KERNEL_SIZE=${KERNEL_SIZE}                 # Kernel size for attention pooling
POOLING="${pooling}"             # Pooling method for attention (maxpool or avgpool)

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

                echo "Created: $config_file (${kv_method}, ${benchmark}, w=${window_size}, c=${max_capacity})"
                ((config_count++))
            done
        done
    done
done

echo ""
echo "Generated $config_count config files in $CONFIG_DIR"
echo "Total configurations: 2 benchmarks × 2 KV methods × 5 window-capacity pairs = 20 configs"