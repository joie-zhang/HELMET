#!/bin/bash

# Script to generate Yarn-Qwen3-8B evaluation configs
# This script creates individual config files for:
# - Rerank task: SnapKV, PyramidKV, StreamingLLM (various cache sizes)
# - ICL tasks (Banking77 & Clinic150): Baseline, INT4, INT8, SnapKV, PyramidKV, StreamingLLM

# Model to evaluate
MODEL="Yarn-Qwen3-8B"

# Context length for all tasks (HELMET benchmark uses 16k)
CONTEXT_LENGTH="16k"

# Common parameters
BENCHMARK="helmet"
SEED=42
JOB_TIME="12:00:00"

# Output directory for generated configs
CONFIG_DIR="scripts/configs/yarn_qwen3_8b_sweep"
mkdir -p "$CONFIG_DIR"

echo "Generating Yarn-Qwen3-8B evaluation configs..."
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

    # IMPORTANT: Always set USE_REASONING_CONFIG to false for Yarn-Qwen3-8B
    # We do NOT want to use configs_reasoning folder
    local use_reasoning_config="false"

    # Clean model name for filename (remove special chars)
    local clean_model=$(echo "$MODEL" | sed 's/[^A-Za-z0-9]/_/g')
    local config_file="${CONFIG_DIR}/${technique_name}_${clean_model}_${task}_${CONTEXT_LENGTH}_config.sh"

    cat > "$config_file" << EOF
# Yarn-Qwen3-8B evaluation config
# Technique: ${technique_name}, Model: ${MODEL}, Task: ${task}, Context: ${CONTEXT_LENGTH}
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
JOB_TIME="${JOB_TIME}"
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
# RERANK TASK CONFIGS
# =======================

echo "=== Generating Rerank Task Configs ==="
RERANK_TASK="rerank"

# 1. SnapKV configurations for Rerank (maxpool only)
echo "Generating SnapKV configs for rerank..."

# SnapKV Config 1: kernel=7, window=256, capacity=2048, maxpool
kv_config='# KV Cache Configuration Parameters
KV_TYPE="snapkv"
WINDOW_SIZE=256
MAX_CAPACITY_PROMPT=2048
KERNEL_SIZE=7
POOLING="maxpool"'
create_config "snapkv_k7_w256_c2048_maxpool" "$RERANK_TASK" "snapkv" "" "$kv_config"

# SnapKV Config 2: kernel=7, window=2048, capacity=8192, maxpool
kv_config='# KV Cache Configuration Parameters
KV_TYPE="snapkv"
WINDOW_SIZE=2048
MAX_CAPACITY_PROMPT=8192
KERNEL_SIZE=7
POOLING="maxpool"'
create_config "snapkv_k7_w2048_c8192_maxpool" "$RERANK_TASK" "snapkv" "" "$kv_config"

# 2. PyramidKV configurations for Rerank (avgpool only)
echo "Generating PyramidKV configs for rerank..."

# PyramidKV Config 1: kernel=7, window=256, capacity=2048, avgpool
kv_config='# KV Cache Configuration Parameters
KV_TYPE="pyramidkv"
WINDOW_SIZE=256
MAX_CAPACITY_PROMPT=2048
KERNEL_SIZE=7
POOLING="avgpool"'
create_config "pyramidkv_k7_w256_c2048_avgpool" "$RERANK_TASK" "pyramidkv" "" "$kv_config"

# PyramidKV Config 2: kernel=7, window=2048, capacity=8192, avgpool
kv_config='# KV Cache Configuration Parameters
KV_TYPE="pyramidkv"
WINDOW_SIZE=2048
MAX_CAPACITY_PROMPT=8192
KERNEL_SIZE=7
POOLING="avgpool"'
create_config "pyramidkv_k7_w2048_c8192_avgpool" "$RERANK_TASK" "pyramidkv" "" "$kv_config"

# 3. StreamingLLM configuration for Rerank
echo "Generating StreamingLLM config for rerank..."

# StreamingLLM: n_local=4092, n_init=4
streamingllm_config='# StreamingLLM Configuration Parameters
N_LOCAL=4092
N_INIT=4'
create_config "streamingllm_n4092_i4" "$RERANK_TASK" "streamingllm" "" "$streamingllm_config"

# =======================
# ICL TASKS CONFIGS
# =======================

echo ""
echo "=== Generating ICL Task Configs ==="
ICL_TASKS=("icl_banking" "icl_clinic")

for task in "${ICL_TASKS[@]}"; do
    echo "Generating configs for task: $task"

    # 1. Baseline (no quantization)
    create_config "baseline" "$task" "baseline" "16" ""

    # 2. INT4 Quantization
    create_config "int4_quant" "$task" "baseline" "4" ""

    # 3. INT8 Quantization
    create_config "int8_quant" "$task" "baseline" "8" ""

    # 4. SnapKV configurations (maxpool only)
    echo "  - SnapKV configs..."

    # SnapKV Config 1: kernel=7, window=256, capacity=2048, maxpool
    kv_config='# KV Cache Configuration Parameters
KV_TYPE="snapkv"
WINDOW_SIZE=256
MAX_CAPACITY_PROMPT=2048
KERNEL_SIZE=7
POOLING="maxpool"'
    create_config "snapkv_k7_w256_c2048_maxpool" "$task" "snapkv" "" "$kv_config"

    # SnapKV Config 2: kernel=7, window=2048, capacity=8192, maxpool
    kv_config='# KV Cache Configuration Parameters
KV_TYPE="snapkv"
WINDOW_SIZE=2048
MAX_CAPACITY_PROMPT=8192
KERNEL_SIZE=7
POOLING="maxpool"'
    create_config "snapkv_k7_w2048_c8192_maxpool" "$task" "snapkv" "" "$kv_config"

    # 5. PyramidKV configurations (avgpool only)
    echo "  - PyramidKV configs..."

    # PyramidKV Config 1: kernel=7, window=256, capacity=2048, avgpool
    kv_config='# KV Cache Configuration Parameters
KV_TYPE="pyramidkv"
WINDOW_SIZE=256
MAX_CAPACITY_PROMPT=2048
KERNEL_SIZE=7
POOLING="avgpool"'
    create_config "pyramidkv_k7_w256_c2048_avgpool" "$task" "pyramidkv" "" "$kv_config"

    # PyramidKV Config 2: kernel=7, window=2048, capacity=8192, avgpool
    kv_config='# KV Cache Configuration Parameters
KV_TYPE="pyramidkv"
WINDOW_SIZE=2048
MAX_CAPACITY_PROMPT=8192
KERNEL_SIZE=7
POOLING="avgpool"'
    create_config "pyramidkv_k7_w2048_c8192_avgpool" "$task" "pyramidkv" "" "$kv_config"

    # 6. StreamingLLM configuration
    echo "  - StreamingLLM config..."

    # StreamingLLM: n_local=4092, n_init=4
    streamingllm_config='# StreamingLLM Configuration Parameters
N_LOCAL=4092
N_INIT=4'
    create_config "streamingllm_n4092_i4" "$task" "streamingllm" "" "$streamingllm_config"
done

echo ""
echo "==============================================="
echo "Generated $config_count config files in $CONFIG_DIR"
echo "==============================================="
echo ""
echo "Summary:"
echo "  Model: Yarn-Qwen3-8B"
echo "  Context Length: 16k"
echo "  Job Time: 12 hours"
echo "  USE_REASONING_CONFIG: false (always)"
echo ""
echo "Rerank Task:"
echo "  - SnapKV: 2 configurations (2 cache sizes, maxpool only)"
echo "  - PyramidKV: 2 configurations (2 cache sizes, avgpool only)"
echo "  - StreamingLLM: 1 configuration (n_local=4092, n_init=4)"
echo "  - Total: 5 configs"
echo ""
echo "ICL Tasks (Banking77 & Clinic150):"
echo "  - Baseline: 1 config per task"
echo "  - INT4: 1 config per task"
echo "  - INT8: 1 config per task"
echo "  - SnapKV: 2 configs per task (maxpool only)"
echo "  - PyramidKV: 2 configs per task (avgpool only)"
echo "  - StreamingLLM: 1 config per task"
echo "  - Total per task: 8 configs"
echo "  - Total for 2 ICL tasks: 16 configs"
echo ""
echo "Grand Total: 21 config files"
echo ""
echo "Note: All configs use standard (non-reasoning) config files."
echo "      SnapKV uses maxpool only, PyramidKV uses avgpool only."
echo "Config files can be submitted using your existing job submission scripts."
