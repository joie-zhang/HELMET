#!/bin/bash

# Script to generate DeepSeek-R1-Distill-Qwen-7B evaluation configs for ICL tasks
# This script creates individual config files for:
# - ICL tasks (Banking77 & Clinic150): Baseline, INT4, INT8, SnapKV, PyramidKV, StreamingLLM
# - Time limits: 1 hour and 12 hours

# Model to evaluate
MODEL="DeepSeek-R1-Distill-Qwen-7B"

# Context length for all tasks (HELMET benchmark uses 16k)
CONTEXT_LENGTH="16k"

# Common parameters
BENCHMARK="helmet"
SEED=42

# Output directory for generated configs
CONFIG_DIR="scripts/configs/r1_distill_qwen_icl_sweep"
mkdir -p "$CONFIG_DIR"

echo "Generating DeepSeek-R1-Distill-Qwen-7B ICL evaluation configs..."
echo "Model: $MODEL"
echo "Context length: $CONTEXT_LENGTH"
echo "Benchmark: $BENCHMARK"
echo ""

config_count=0

# Function to create config file
create_config() {
    local technique_name="$1"
    local task="$2"
    local exp_type="$3"
    local quantize_value="$4"
    local kv_config="$5"
    local job_time="$6"

    # Set USE_REASONING_CONFIG to false
    local use_reasoning_config="false"

    # Clean model name for filename (remove special chars)
    local clean_model=$(echo "$MODEL" | sed 's/[^A-Za-z0-9]/_/g')

    # Add time suffix to filename
    local time_suffix=$(echo "$job_time" | sed 's/:/_/g' | sed 's/00$/h/')
    local config_file="${CONFIG_DIR}/${technique_name}_${clean_model}_${task}_${CONTEXT_LENGTH}_${time_suffix}_config.sh"

    cat > "$config_file" << EOF
# DeepSeek-R1-Distill-Qwen-7B evaluation config
# Technique: ${technique_name}, Model: ${MODEL}, Task: ${task}, Context: ${CONTEXT_LENGTH}, Time: ${job_time}
declare -a BASE_CONFIGS=("${task}")
declare -a CONTEXT_LENGTHS=("${CONTEXT_LENGTH}")
declare -a MODELS=("${MODEL}")
declare -a QUANTIZE=("${quantize_value}")
EXP_TYPE="${exp_type}"
BENCHMARK="${BENCHMARK}"
USE_REASONING_CONFIG="${use_reasoning_config}"
SEED=${SEED}

${kv_config}

# SLURM Configuration
JOB_TIME="${job_time}"
JOB_NAME="\${EXP_TYPE}_\${BENCHMARK}_${technique_name}_${clean_model}_${task}_eval"

# Export variables so they're available to the job script
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export USE_REASONING_CONFIG
export SEED
EOF

    # Add technique-specific exports if they exist
    if [[ "$kv_config" == *"KV_TYPE"* ]]; then
        cat >> "$config_file" << EOF
export KV_TYPE
export WINDOW_SIZE
export MAX_CAPACITY_PROMPT
export KERNEL_SIZE
export POOLING
EOF
    elif [[ "$kv_config" == *"N_LOCAL"* ]]; then
        cat >> "$config_file" << EOF
export N_LOCAL
export N_INIT
EOF
    fi

    echo "Created: $config_file"
    ((config_count++))
}

# =======================
# ICL TASKS CONFIGS
# =======================

echo "=== Generating ICL Task Configs ==="
ICL_TASKS=("icl_banking" "icl_clinic")
TIME_LIMITS=("1:00:00" "12:00:00")

for task in "${ICL_TASKS[@]}"; do
    for time_limit in "${TIME_LIMITS[@]}"; do
        echo "Generating configs for task: $task, time: $time_limit"

        # 1. Baseline (no quantization)
        create_config "baseline" "$task" "baseline" "16" "" "$time_limit"

        # 2. INT4 Quantization
        create_config "int4_quant" "$task" "baseline" "4" "" "$time_limit"

        # 3. INT8 Quantization
        create_config "int8_quant" "$task" "baseline" "8" "" "$time_limit"

        # 4. SnapKV configurations (maxpool only)
        echo "  - SnapKV configs..."

        # SnapKV Config 1: kernel=7, window=256, capacity=2048, maxpool
        kv_config='# KV Cache Configuration Parameters
KV_TYPE="snapkv"
WINDOW_SIZE=256
MAX_CAPACITY_PROMPT=2048
KERNEL_SIZE=7
POOLING="maxpool"'
        create_config "snapkv_k7_w256_c2048_maxpool" "$task" "snapkv" "" "$kv_config" "$time_limit"

        # SnapKV Config 2: kernel=7, window=2048, capacity=8192, maxpool
        kv_config='# KV Cache Configuration Parameters
KV_TYPE="snapkv"
WINDOW_SIZE=2048
MAX_CAPACITY_PROMPT=8192
KERNEL_SIZE=7
POOLING="maxpool"'
        create_config "snapkv_k7_w2048_c8192_maxpool" "$task" "snapkv" "" "$kv_config" "$time_limit"

        # 5. PyramidKV configurations (avgpool only)
        echo "  - PyramidKV configs..."

        # PyramidKV Config 1: kernel=7, window=256, capacity=2048, avgpool
        kv_config='# KV Cache Configuration Parameters
KV_TYPE="pyramidkv"
WINDOW_SIZE=256
MAX_CAPACITY_PROMPT=2048
KERNEL_SIZE=7
POOLING="avgpool"'
        create_config "pyramidkv_k7_w256_c2048_avgpool" "$task" "pyramidkv" "" "$kv_config" "$time_limit"

        # PyramidKV Config 2: kernel=7, window=2048, capacity=8192, avgpool
        kv_config='# KV Cache Configuration Parameters
KV_TYPE="pyramidkv"
WINDOW_SIZE=2048
MAX_CAPACITY_PROMPT=8192
KERNEL_SIZE=7
POOLING="avgpool"'
        create_config "pyramidkv_k7_w2048_c8192_avgpool" "$task" "pyramidkv" "" "$kv_config" "$time_limit"

        # 6. StreamingLLM configuration
        echo "  - StreamingLLM config..."

        # StreamingLLM: n_local=4092, n_init=4 (using 4092 for ICL tasks)
        streamingllm_config='# StreamingLLM Configuration Parameters
N_LOCAL=4092
N_INIT=4'
        create_config "streamingllm_n4092_i4" "$task" "streamingllm" "" "$streamingllm_config" "$time_limit"
    done
done

echo ""
echo "==============================================="
echo "Generated $config_count config files in $CONFIG_DIR"
echo "==============================================="
echo ""
echo "Summary:"
echo "  Model: DeepSeek-R1-Distill-Qwen-7B"
echo "  Context Length: 16k"
echo "  Job Times: 1 hour and 12 hours"
echo "  USE_REASONING_CONFIG: false"
echo ""
echo "ICL Tasks (Banking77 & Clinic150):"
echo "  - Baseline: 1 config per task per time limit"
echo "  - INT4: 1 config per task per time limit"
echo "  - INT8: 1 config per task per time limit"
echo "  - SnapKV: 2 configs per task per time limit (maxpool only)"
echo "  - PyramidKV: 2 configs per task per time limit (avgpool only)"
echo "  - StreamingLLM: 1 config per task per time limit"
echo "  - Total per task per time: 8 configs"
echo "  - Total per task: 16 configs (8 for 1hr + 8 for 12hr)"
echo "  - Total for 2 ICL tasks: 32 configs"
echo ""
echo "Grand Total: 32 config files"
echo ""
echo "Note: All configs use standard config files (USE_REASONING_CONFIG=false)."
echo "      SnapKV uses maxpool only, PyramidKV uses avgpool only."
echo "      StreamingLLM uses n_local=4092 for ICL tasks."
echo "Config files can be submitted using your existing job submission scripts."
