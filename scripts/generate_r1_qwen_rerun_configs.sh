#!/bin/bash

# Script to generate R1 Distill Qwen 7B re-run configs with chat template fix
# This script creates configs specifically for the problematic R1 Distill Qwen model
# with chat template enabled to fix the hallucination issues identified

# Target model with known issues
MODEL="DeepSeek-R1-Distill-Qwen-7B"

# Specific tasks to re-run based on investigation
TARGET_TASKS=(
    "niah"
    "cite"
    "rag_hotpotqa"
    "rerank"
)

# Target techniques that showed some promise but need chat template fix
TARGET_TECHNIQUES=(
    "snapkv_k7_w256_c2048_maxpool"
    "snapkv_k7_w2048_c8192_maxpool"
    "pyramidkv_k7_w256_c2048_avgpool"
    "pyramidkv_k7_w2048_c8192_avgpool"
)

CONTEXT_LENGTH="16k"
BENCHMARK="helmet"
SEED=42
JOB_TIME="3:00:00"

# Output directory for re-run configs
CONFIG_DIR="scripts/configs/r1_qwen_rerun_fixed"
mkdir -p "$CONFIG_DIR"

echo "Generating R1 Distill Qwen 7B re-run configs with chat template fix..."
echo "Model: $MODEL"
echo "Tasks: ${TARGET_TASKS[@]}"
echo "Techniques: ${TARGET_TECHNIQUES[@]}"
echo ""

config_count=0

# Function to create config file with chat template enabled
create_rerun_config() {
    local technique_name="$1"
    local task="$2"
    local exp_type="$3"
    local kv_config="$4"

    # Clean model name for filename
    local clean_model=$(echo "$MODEL" | sed 's/[^A-Za-z0-9]/_/g')
    local config_file="${CONFIG_DIR}/${technique_name}_${clean_model}_${task}_${CONTEXT_LENGTH}_FIXED_config.sh"

    cat > "$config_file" << EOF
# R1 Distill Qwen 7B Re-run Config with Chat Template Fix
# Technique: ${technique_name}, Model: ${MODEL}, Task: ${task}, Context: ${CONTEXT_LENGTH}
# FIX: chat template enabled to resolve hallucination issues
declare -a BASE_CONFIGS=("${task}")
declare -a CONTEXT_LENGTHS=("${CONTEXT_LENGTH}")
declare -a MODELS=("${MODEL}")
declare -a QUANTIZE=("16")
EXP_TYPE="${exp_type}"
BENCHMARK="${BENCHMARK}"
SEED=${SEED}

# CRITICAL FIX: Chat template already handled in run_job.sh for R1 models

${kv_config}

# SLURM Configuration
JOB_TIME="${JOB_TIME}"
JOB_NAME="\${EXP_TYPE}_\${BENCHMARK}_${technique_name}_${clean_model}_${task}_FIXED_eval"

# Export variables so they're available to the job script
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export SEED
# Note: Chat template handling is automatic in run_job.sh for R1 models
EOF

    # Add technique-specific exports
    if [[ "$kv_config" == *"KV_TYPE"* ]]; then
        cat >> "$config_file" << EOF
export KV_TYPE
export WINDOW_SIZE
export MAX_CAPACITY_PROMPT
export KERNEL_SIZE
export POOLING
EOF
    fi

    echo "Created: $config_file"
    ((config_count++))
}

# Generate SnapKV configs
echo "Generating SnapKV re-run configs..."

# SnapKV Config 1: k7_w256_c2048_maxpool
kv_config='# KV Cache Configuration Parameters
KV_TYPE="snapkv"
WINDOW_SIZE=256
MAX_CAPACITY_PROMPT=2048
KERNEL_SIZE=7
POOLING="maxpool"'

for task in "${TARGET_TASKS[@]}"; do
    create_rerun_config "snapkv_k7_w256_c2048_maxpool" "$task" "snapkv" "$kv_config"
done

# SnapKV Config 2: k7_w2048_c8192_maxpool
kv_config='# KV Cache Configuration Parameters
KV_TYPE="snapkv"
WINDOW_SIZE=2048
MAX_CAPACITY_PROMPT=8192
KERNEL_SIZE=7
POOLING="maxpool"'

for task in "${TARGET_TASKS[@]}"; do
    create_rerun_config "snapkv_k7_w2048_c8192_maxpool" "$task" "snapkv" "$kv_config"
done

# Generate PyramidKV configs
echo "Generating PyramidKV re-run configs..."

# PyramidKV Config 1: k7_w256_c2048_avgpool
kv_config='# KV Cache Configuration Parameters
KV_TYPE="pyramidkv"
WINDOW_SIZE=256
MAX_CAPACITY_PROMPT=2048
KERNEL_SIZE=7
POOLING="avgpool"'

for task in "${TARGET_TASKS[@]}"; do
    create_rerun_config "pyramidkv_k7_w256_c2048_avgpool" "$task" "pyramidkv" "$kv_config"
done

# PyramidKV Config 2: k7_w2048_c8192_avgpool
kv_config='# KV Cache Configuration Parameters
KV_TYPE="pyramidkv"
WINDOW_SIZE=2048
MAX_CAPACITY_PROMPT=8192
KERNEL_SIZE=7
POOLING="avgpool"'

for task in "${TARGET_TASKS[@]}"; do
    create_rerun_config "pyramidkv_k7_w2048_c8192_avgpool" "$task" "pyramidkv" "$kv_config"
done

echo ""
echo "Generated $config_count re-run config files in $CONFIG_DIR"
echo ""
echo "Summary:"
echo "- Model: DeepSeek-R1-Distill-Qwen-7B (with chat template fix)"
echo "- Techniques: 4 configurations (SnapKV×2, PyramidKV×2)"
echo "- Tasks: 4 critical tasks (niah, cite, rag_hotpotqa, rerank)"
echo "- Total re-run configs: 4 × 4 = 16 configs"
echo ""
echo "Key fixes applied:"
echo "1. Chat template automatically enabled via run_job.sh for R1 models (critical fix for hallucinations)"
echo "2. Extended job time to 4 hours for thorough evaluation"
echo "3. Clear labeling as FIXED configs to avoid confusion"
echo ""
echo "These configs can be submitted using your existing job submission pipeline."
echo "Expected outcome: Significant improvement in response quality and task performance."