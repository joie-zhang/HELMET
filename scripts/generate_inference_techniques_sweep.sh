#!/bin/bash

# Script to generate inference techniques evaluation configs
# This script creates individual config files for each combination of:
# - Inference technique
# - Model
# - Task (from HELMET 16k and LongProc 2k)

# Models to evaluate
MODELS=(
    # "Llama-3.1-8B-Instruct"
    # "Qwen2.5-7B-Instruct"
    "DeepSeek-R1-Distill-Llama-8B"
    # "DeepSeek-R1-Distill-Qwen-7B"
    # "Qwen3-8B"
    # "Yarn-Qwen3-8B"
)

# HELMET tasks (16k context)
HELMET_TASKS=(
    "icl"
    # "niah"
    "cite"
    # "recall_jsonkv"
    # "rag_hotpotqa"
    # "rag_nq"
    # "rerank"
)
HELMET_CONTEXT_LENGTH="16k"

# LongProc tasks (2k context)
LONGPROC_TASKS=(
    # "html_to_tsv"
    # "pseudo_to_code"
    # "travel_planning"
)
LONGPROC_CONTEXT_LENGTH="2k"

# Common parameters
BENCHMARK="helmet"
SEED=42
JOB_TIME="12:00:00"

# Output directory for generated configs
CONFIG_DIR="scripts/configs/r1-distill-llama-8b_on_longer_output_gen_rerun_cite_icl"
mkdir -p "$CONFIG_DIR"

echo "Generating inference techniques evaluation configs..."
echo "Models: ${MODELS[@]}"
echo "HELMET tasks (16k): ${HELMET_TASKS[@]}"
echo "LongProc tasks (2k): ${LONGPROC_TASKS[@]}"
echo ""

config_count=0

# Function to check if a model is a reasoning model
is_reasoning_model() {
    local model="$1"
    case "$model" in
        *"R1-Distill"*|"Qwen3-8B"|"Yarn-Qwen3-8B")
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Function to create config file
create_config() {
    local technique_name="$1"
    local model="$2"
    local task="$3"
    local context_length="$4"
    local exp_type="$5"
    local quantize_value="$6"
    local kv_config="$7"
    local benchmark_type="$8"  # New parameter for benchmark type

    # Determine if this model should use reasoning configs
    local use_reasoning_config="false"
    if is_reasoning_model "$model"; then
        use_reasoning_config="true"
    fi

    # Clean model name for filename (remove special chars)
    local clean_model=$(echo "$model" | sed 's/[^A-Za-z0-9]/_/g')
    local config_file="${CONFIG_DIR}/${technique_name}_${clean_model}_${task}_${context_length}_config.sh"

    cat > "$config_file" << EOF
# Inference technique evaluation config
# Technique: ${technique_name}, Model: ${model}, Task: ${task}, Context: ${context_length}
declare -a BASE_CONFIGS=("${task}")
declare -a CONTEXT_LENGTHS=("${context_length}")
declare -a MODELS=("${model}")
declare -a QUANTIZE=("${quantize_value}")
EXP_TYPE="${exp_type}"
BENCHMARK="${benchmark_type}"
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

# Generate configs for each combination

# 1. INT4 Quantization
echo "Generating INT4 quantization configs..."
for model in "${MODELS[@]}"; do
    # HELMET tasks
    for task in "${HELMET_TASKS[@]}"; do
        create_config "int4_quant" "$model" "$task" "$HELMET_CONTEXT_LENGTH" "baseline" "4" "" "helmet"
    done
    # LongProc tasks
    for task in "${LONGPROC_TASKS[@]}"; do
        create_config "int4_quant" "$model" "$task" "$LONGPROC_CONTEXT_LENGTH" "baseline" "4" "" "longproc"
    done
done

# 2. INT8 Quantization
echo "Generating INT8 quantization configs..."
for model in "${MODELS[@]}"; do
    # HELMET tasks
    for task in "${HELMET_TASKS[@]}"; do
        create_config "int8_quant" "$model" "$task" "$HELMET_CONTEXT_LENGTH" "baseline" "8" "" "helmet"
    done
    # LongProc tasks
    for task in "${LONGPROC_TASKS[@]}"; do
        create_config "int8_quant" "$model" "$task" "$LONGPROC_CONTEXT_LENGTH" "baseline" "8" "" "longproc"
    done
done

# 3. SnapKV Configuration 1 (kernel=7, window=256, capacity=2048, maxpool)
echo "Generating SnapKV config 1 (k7_w256_c2048_maxpool)..."
kv_config='# KV Cache Configuration Parameters
KV_TYPE="snapkv"
WINDOW_SIZE=256
MAX_CAPACITY_PROMPT=2048
KERNEL_SIZE=7
POOLING="maxpool"'

for model in "${MODELS[@]}"; do
    # HELMET tasks
    for task in "${HELMET_TASKS[@]}"; do
        create_config "snapkv_k7_w256_c2048_maxpool" "$model" "$task" "$HELMET_CONTEXT_LENGTH" "snapkv" "" "$kv_config" "helmet"
    done
    # LongProc tasks
    for task in "${LONGPROC_TASKS[@]}"; do
        create_config "snapkv_k7_w256_c2048_maxpool" "$model" "$task" "$LONGPROC_CONTEXT_LENGTH" "snapkv" "" "$kv_config" "longproc"
    done
done

# 4. SnapKV Configuration 2 (kernel=7, window=2048, capacity=8192, maxpool)
echo "Generating SnapKV config 2 (k7_w2048_c8192_maxpool)..."
kv_config='# KV Cache Configuration Parameters
KV_TYPE="snapkv"
WINDOW_SIZE=2048
MAX_CAPACITY_PROMPT=8192
KERNEL_SIZE=7
POOLING="maxpool"'

for model in "${MODELS[@]}"; do
    # HELMET tasks
    for task in "${HELMET_TASKS[@]}"; do
        create_config "snapkv_k7_w2048_c8192_maxpool" "$model" "$task" "$HELMET_CONTEXT_LENGTH" "snapkv" "" "$kv_config" "helmet"
    done
    # LongProc tasks
    for task in "${LONGPROC_TASKS[@]}"; do
        create_config "snapkv_k7_w2048_c8192_maxpool" "$model" "$task" "$LONGPROC_CONTEXT_LENGTH" "snapkv" "" "$kv_config" "longproc"
    done
done

# 5. PyramidKV Configuration 1 (kernel=7, window=256, capacity=2048, avgpool)
echo "Generating PyramidKV config 1 (k7_w256_c2048_avgpool)..."
kv_config='# KV Cache Configuration Parameters
KV_TYPE="pyramidkv"
WINDOW_SIZE=256
MAX_CAPACITY_PROMPT=2048
KERNEL_SIZE=7
POOLING="avgpool"'

for model in "${MODELS[@]}"; do
    # HELMET tasks
    for task in "${HELMET_TASKS[@]}"; do
        create_config "pyramidkv_k7_w256_c2048_avgpool" "$model" "$task" "$HELMET_CONTEXT_LENGTH" "pyramidkv" "" "$kv_config" "helmet"
    done
    # LongProc tasks
    for task in "${LONGPROC_TASKS[@]}"; do
        create_config "pyramidkv_k7_w256_c2048_avgpool" "$model" "$task" "$LONGPROC_CONTEXT_LENGTH" "pyramidkv" "" "$kv_config" "longproc"
    done
done

# 6. PyramidKV Configuration 2 (kernel=7, window=2048, capacity=8192, avgpool)
echo "Generating PyramidKV config 2 (k7_w2048_c8192_avgpool)..."
kv_config='# KV Cache Configuration Parameters
KV_TYPE="pyramidkv"
WINDOW_SIZE=2048
MAX_CAPACITY_PROMPT=8192
KERNEL_SIZE=7
POOLING="avgpool"'

for model in "${MODELS[@]}"; do
    # HELMET tasks
    for task in "${HELMET_TASKS[@]}"; do
        create_config "pyramidkv_k7_w2048_c8192_avgpool" "$model" "$task" "$HELMET_CONTEXT_LENGTH" "pyramidkv" "" "$kv_config" "helmet"
    done
    # LongProc tasks
    for task in "${LONGPROC_TASKS[@]}"; do
        create_config "pyramidkv_k7_w2048_c8192_avgpool" "$model" "$task" "$LONGPROC_CONTEXT_LENGTH" "pyramidkv" "" "$kv_config" "longproc"
    done
done

# 7. MInference (commented out for now)
# echo "Generating MInference configs..."
# for model in "${MODELS[@]}"; do
#     # HELMET tasks
#     for task in "${HELMET_TASKS[@]}"; do
#         create_config "minference" "$model" "$task" "$HELMET_CONTEXT_LENGTH" "minference" "" ""
#     done
#     # LongProc tasks
#     for task in "${LONGPROC_TASKS[@]}"; do
#         create_config "minference" "$model" "$task" "$LONGPROC_CONTEXT_LENGTH" "minference" "" ""
#     done
# done

# 8. StreamingLLM
echo "Generating StreamingLLM configs..."
streamingllm_config='# StreamingLLM Configuration Parameters
N_LOCAL=4096
N_INIT=4'

for model in "${MODELS[@]}"; do
    # HELMET tasks
    for task in "${HELMET_TASKS[@]}"; do
        create_config "streamingllm" "$model" "$task" "$HELMET_CONTEXT_LENGTH" "streamingllm" "" "$streamingllm_config" "helmet"
    done
    # LongProc tasks
    for task in "${LONGPROC_TASKS[@]}"; do
        create_config "streamingllm" "$model" "$task" "$LONGPROC_CONTEXT_LENGTH" "streamingllm" "" "$streamingllm_config" "longproc"
    done
done

# 9. Baseline (no quantization)
echo "Generating baseline configs..."
for model in "${MODELS[@]}"; do
    # HELMET tasks
    for task in "${HELMET_TASKS[@]}"; do
        create_config "baseline" "$model" "$task" "$HELMET_CONTEXT_LENGTH" "baseline" "16" "" "helmet"
    done
    # LongProc tasks
    for task in "${LONGPROC_TASKS[@]}"; do
        create_config "baseline" "$model" "$task" "$LONGPROC_CONTEXT_LENGTH" "baseline" "16" "" "longproc"
    done
done

echo ""
echo "Generated $config_count config files in $CONFIG_DIR"
echo ""
echo "Summary:"
# echo "- 8 inference techniques (INT4, INT8, SnapKV×2, PyramidKV×2, StreamingLLM, Baseline)"
echo "- 3 inference techniques (INT4, INT8, Baseline)"

echo "- 1 model (Qwen3-8B)"
echo "- 2 tasks (2 HELMET@16k + 0 LongProc@2k)"
echo "- Total combinations: 3 × 1 × 2 = 6 configs"
echo ""
echo "Config files are organized by technique and can be submitted using your existing job submission scripts."