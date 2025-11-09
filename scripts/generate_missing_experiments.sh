#!/bin/bash

# Script to generate configs for all missing HELMET experiments
# Based on analysis of helmet_performance.csv

# All models
ALL_MODELS=(
    "Qwen2.5-7B-Instruct"
    "Llama-3.1-8B-Instruct"
    "DeepSeek-R1-Distill-Qwen-7B"
    "DeepSeek-R1-Distill-Llama-8B"
    "Qwen3-8B"
    "Yarn-Qwen3-8B"
)

# HELMET tasks (16k and 32k context)
HELMET_TASKS=(
    "niah"
    "cite"
    "recall_jsonkv"
    "rag_hotpotqa"
    "rag_nq"
    "rerank"
    "icl"
    "summ_multilex"
)

# LongProc tasks (0.5k, 2k, 8k context)
LONGPROC_TASKS=(
    "html_to_tsv"
    "pseudo_to_code"
    "travel_planning"
)

# Context lengths
HELMET_16K="16k"
HELMET_32K="32k"
LONGPROC_CONTEXTS=("0.5k" "2k" "8k")

# Common parameters
SEED=42

# Output directory for generated configs
CONFIG_DIR="scripts/configs/missing_experiments_sweep"
mkdir -p "$CONFIG_DIR"

echo "=========================================="
echo "Generating Missing Experiments Configs"
echo "=========================================="
echo ""

config_count=0

# Function to create config file
create_config() {
    local technique_name="$1"
    local model="$2"
    local task="$3"
    local context_length="$4"
    local exp_type="$5"
    local quantize_value="$6"
    local kv_config="$7"
    local benchmark_type="$8"

    # Set job time based on context length
    local job_time
    if [[ "$context_length" == "32k" ]]; then
        job_time="18:00:00"
    elif [[ "$context_length" == "16k" ]]; then
        job_time="06:00:00"
    elif [[ "$context_length" == "8k" ]]; then
        job_time="18:00:00"
    elif [[ "$context_length" == "2k" ]]; then
        job_time="12:00:00"
    elif [[ "$context_length" == "0.5k" ]]; then
        job_time="06:00:00"
    else
        # Default fallback
        job_time="06:00:00"
    fi

    # NO reasoning configs for any models
    local use_reasoning_config="false"

    # Clean model name for filename
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
JOB_TIME="${job_time}"
JOB_NAME="\${EXP_TYPE}_\${BENCHMARK}_${technique_name}_${clean_model}_${task}_eval"

# Export variables
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export USE_REASONING_CONFIG
export SEED
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
    elif [[ "$kv_config" == *"N_LOCAL"* ]]; then
        cat >> "$config_file" << EOF
export N_LOCAL
export N_INIT
EOF
    fi

    echo "  Created: $(basename $config_file)"
    ((config_count++))
}

# ============================================================================
# 1. STREAMINGLLM - Re-run with correct cache size (4092 local + 4 init)
# ============================================================================
echo "1. StreamingLLM - Correcting cache size to 4092..."
streamingllm_4092='# StreamingLLM Configuration Parameters
N_LOCAL=4092
N_INIT=4'

STREAMINGLLM_RERUN_MODELS=(
    "Qwen2.5-7B-Instruct"
    "Llama-3.1-8B-Instruct"
)

for model in "${STREAMINGLLM_RERUN_MODELS[@]}"; do
    # 16k runs
    for task in "${HELMET_TASKS[@]}"; do
        create_config "streamingllm_n4092_i4" "$model" "$task" "$HELMET_16K" "streamingllm" "" "$streamingllm_4092" "helmet"
    done
    # 32k runs
    for task in "${HELMET_TASKS[@]}"; do
        create_config "streamingllm_n4092_i4" "$model" "$task" "$HELMET_32K" "streamingllm" "" "$streamingllm_4092" "helmet"
    done
done

# ============================================================================
# 2. STREAMINGLLM - Missing 32k runs for other models
# ============================================================================
echo "2. StreamingLLM - Missing 32k runs..."
STREAMINGLLM_MISSING_32K=(
    "DeepSeek-R1-Distill-Qwen-7B"
    "DeepSeek-R1-Distill-Llama-8B"
    "Qwen3-8B"
    "Yarn-Qwen3-8B"
)

for model in "${STREAMINGLLM_MISSING_32K[@]}"; do
    for task in "${HELMET_TASKS[@]}"; do
        create_config "streamingllm_n4092_i4" "$model" "$task" "$HELMET_32K" "streamingllm" "" "$streamingllm_4092" "helmet"
    done
done

# ============================================================================
# 3. STREAMINGLLM - New suite with 2044 local + 4 init = 2048 total
# ============================================================================
echo "3. StreamingLLM - New suite with 2048 total cache..."
streamingllm_2048='# StreamingLLM Configuration Parameters
N_LOCAL=2044
N_INIT=4'

for model in "${ALL_MODELS[@]}"; do
    # 16k runs
    for task in "${HELMET_TASKS[@]}"; do
        create_config "streamingllm_n2044_i4" "$model" "$task" "$HELMET_16K" "streamingllm" "" "$streamingllm_2048" "helmet"
    done
    # 32k runs
    for task in "${HELMET_TASKS[@]}"; do
        create_config "streamingllm_n2044_i4" "$model" "$task" "$HELMET_32K" "streamingllm" "" "$streamingllm_2048" "helmet"
    done
done

# ============================================================================
# 4. PYRAMIDKV - Missing 32k runs (w256_c2048 and w2048_c8192)
# ============================================================================
echo "4. PyramidKV - Missing 32k runs..."

# Config 1: w256_c2048
pyramidkv_256_2048='# KV Cache Configuration Parameters
KV_TYPE="pyramidkv"
WINDOW_SIZE=256
MAX_CAPACITY_PROMPT=2048
KERNEL_SIZE=7
POOLING="avgpool"'

for model in "${ALL_MODELS[@]}"; do
    for task in "${HELMET_TASKS[@]}"; do
        create_config "pyramidkv_k7_w256_c2048_avgpool" "$model" "$task" "$HELMET_32K" "pyramidkv" "" "$pyramidkv_256_2048" "helmet"
    done
done

# Config 2: w2048_c8192
pyramidkv_2048_8192='# KV Cache Configuration Parameters
KV_TYPE="pyramidkv"
WINDOW_SIZE=2048
MAX_CAPACITY_PROMPT=8192
KERNEL_SIZE=7
POOLING="avgpool"'

for model in "${ALL_MODELS[@]}"; do
    for task in "${HELMET_TASKS[@]}"; do
        create_config "pyramidkv_k7_w2048_c8192_avgpool" "$model" "$task" "$HELMET_32K" "pyramidkv" "" "$pyramidkv_2048_8192" "helmet"
    done
done

# ============================================================================
# 5. SNAPKV - Missing 32k runs (w256_c2048 and w2048_c8192)
# ============================================================================
echo "5. SnapKV - Missing 32k runs..."

# Config 1: w256_c2048
snapkv_256_2048='# KV Cache Configuration Parameters
KV_TYPE="snapkv"
WINDOW_SIZE=256
MAX_CAPACITY_PROMPT=2048
KERNEL_SIZE=7
POOLING="maxpool"'

for model in "${ALL_MODELS[@]}"; do
    for task in "${HELMET_TASKS[@]}"; do
        create_config "snapkv_k7_w256_c2048_maxpool" "$model" "$task" "$HELMET_32K" "snapkv" "" "$snapkv_256_2048" "helmet"
    done
done

# Config 2: w2048_c8192
snapkv_2048_8192='# KV Cache Configuration Parameters
KV_TYPE="snapkv"
WINDOW_SIZE=2048
MAX_CAPACITY_PROMPT=8192
KERNEL_SIZE=7
POOLING="maxpool"'

for model in "${ALL_MODELS[@]}"; do
    for task in "${HELMET_TASKS[@]}"; do
        create_config "snapkv_k7_w2048_c8192_maxpool" "$model" "$task" "$HELMET_32K" "snapkv" "" "$snapkv_2048_8192" "helmet"
    done
done

# ============================================================================
# 6. QUANTIZATION - Missing 16k runs
# ============================================================================
echo "6. Quantization - Missing 16k runs..."

# Qwen3-8B missing INT4 and INT8 at 16k
for task in "${HELMET_TASKS[@]}"; do
    create_config "int4_quant" "Qwen3-8B" "$task" "$HELMET_16K" "baseline" "4" "" "helmet"
    create_config "int8_quant" "Qwen3-8B" "$task" "$HELMET_16K" "baseline" "8" "" "helmet"
done

# ============================================================================
# 7. QUANTIZATION - Missing 32k runs (baseline, INT4, INT8)
# ============================================================================
echo "7. Quantization - Missing 32k runs..."

QUANT_MISSING_32K=(
    "DeepSeek-R1-Distill-Qwen-7B"
    "DeepSeek-R1-Distill-Llama-8B"
    "Qwen3-8B"
    "Yarn-Qwen3-8B"
)

for model in "${QUANT_MISSING_32K[@]}"; do
    for task in "${HELMET_TASKS[@]}"; do
        create_config "baseline" "$model" "$task" "$HELMET_32K" "baseline" "16" "" "helmet"
        create_config "int4_quant" "$model" "$task" "$HELMET_32K" "baseline" "4" "" "helmet"
        create_config "int8_quant" "$model" "$task" "$HELMET_32K" "baseline" "8" "" "helmet"
    done
done

# ============================================================================
# 8. LONGPROC - ALL tasks for ALL models (0.5k, 2k, 8k)
# ============================================================================
echo "8. LongProc - All tasks for all models..."

for model in "${ALL_MODELS[@]}"; do
    for context in "${LONGPROC_CONTEXTS[@]}"; do
        for task in "${LONGPROC_TASKS[@]}"; do
            # Baseline
            create_config "baseline" "$model" "$task" "$context" "baseline" "16" "" "longproc"
            # INT4
            create_config "int4_quant" "$model" "$task" "$context" "baseline" "4" "" "longproc"
            # INT8
            create_config "int8_quant" "$model" "$task" "$context" "baseline" "8" "" "longproc"

            # StreamingLLM (4092+4)
            create_config "streamingllm_n4092_i4" "$model" "$task" "$context" "streamingllm" "" "$streamingllm_4092" "longproc"
            # StreamingLLM (2044+4)
            create_config "streamingllm_n2044_i4" "$model" "$task" "$context" "streamingllm" "" "$streamingllm_2048" "longproc"

            # PyramidKV w256_c2048
            create_config "pyramidkv_k7_w256_c2048_avgpool" "$model" "$task" "$context" "pyramidkv" "" "$pyramidkv_256_2048" "longproc"
            # PyramidKV w2048_c8192
            create_config "pyramidkv_k7_w2048_c8192_avgpool" "$model" "$task" "$context" "pyramidkv" "" "$pyramidkv_2048_8192" "longproc"

            # SnapKV w256_c2048
            create_config "snapkv_k7_w256_c2048_maxpool" "$model" "$task" "$context" "snapkv" "" "$snapkv_256_2048" "longproc"
            # SnapKV w2048_c8192
            create_config "snapkv_k7_w2048_c8192_maxpool" "$model" "$task" "$context" "snapkv" "" "$snapkv_2048_8192" "longproc"
        done
    done
done

# ============================================================================
# 9. QWEN3-8B & YARN-QWEN3-8B - Both thinking modes (enable_thinking parameter)
# ============================================================================
echo "9. Qwen3-8B models - Thinking and non-thinking mode configs..."

# Function to create config file with explicit enable_thinking parameter
create_config_with_thinking() {
    local technique_name="$1"
    local model="$2"
    local task="$3"
    local context_length="$4"
    local exp_type="$5"
    local quantize_value="$6"
    local kv_config="$7"
    local benchmark_type="$8"
    local enable_thinking="$9"  # True or False for Qwen3 thinking mode

    # Set job time based on context length
    local job_time
    if [[ "$context_length" == "32k" ]]; then
        job_time="18:00:00"
    elif [[ "$context_length" == "16k" ]]; then
        job_time="06:00:00"
    elif [[ "$context_length" == "8k" ]]; then
        job_time="18:00:00"
    elif [[ "$context_length" == "2k" ]]; then
        job_time="12:00:00"
    elif [[ "$context_length" == "0.5k" ]]; then
        job_time="06:00:00"
    else
        # Default fallback
        job_time="06:00:00"
    fi

    # Clean model name for filename
    local clean_model=$(echo "$model" | sed 's/[^A-Za-z0-9]/_/g')
    local thinking_suffix=""
    if [[ "$enable_thinking" == "True" ]]; then
        thinking_suffix="_thinking"
    else
        thinking_suffix="_nothinking"
    fi
    local config_file="${CONFIG_DIR}/${technique_name}_${clean_model}${thinking_suffix}_${task}_${context_length}_config.sh"

    cat > "$config_file" << EOF
# Inference technique evaluation config
# Technique: ${technique_name}, Model: ${model}, Task: ${task}, Context: ${context_length}
# Enable Thinking: ${enable_thinking} (Qwen3 CoT mode)
declare -a BASE_CONFIGS=("${task}")
declare -a CONTEXT_LENGTHS=("${context_length}")
declare -a MODELS=("${model}")
declare -a QUANTIZE=("${quantize_value}")
EXP_TYPE="${exp_type}"
BENCHMARK="${benchmark_type}"
USE_REASONING_CONFIG="false"
ENABLE_THINKING="${enable_thinking}"
SEED=${SEED}

${kv_config}

# SLURM Configuration
JOB_TIME="${job_time}"
JOB_NAME="\${EXP_TYPE}_\${BENCHMARK}_${technique_name}_${clean_model}${thinking_suffix}_${task}_eval"

# Export variables
export BASE_CONFIGS
export CONTEXT_LENGTHS
export MODELS
export QUANTIZE
export EXP_TYPE
export BENCHMARK
export USE_REASONING_CONFIG
export ENABLE_THINKING
export SEED
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
    elif [[ "$kv_config" == *"N_LOCAL"* ]]; then
        cat >> "$config_file" << EOF
export N_LOCAL
export N_INIT
EOF
    fi

    echo "  Created: $(basename $config_file)"
    ((config_count++))
}

QWEN3_MODELS=("Qwen3-8B" "Yarn-Qwen3-8B")

# Generate configs for BOTH thinking modes (True and False)
for enable_thinking_mode in "True" "False"; do
    echo "  Generating configs with enable_thinking=$enable_thinking_mode..."

    for model in "${QWEN3_MODELS[@]}"; do
        # 16k HELMET tasks
        for task in "${HELMET_TASKS[@]}"; do
            # Baseline
            create_config_with_thinking "baseline" "$model" "$task" "$HELMET_16K" "baseline" "16" "" "helmet" "$enable_thinking_mode"
            # INT4
            create_config_with_thinking "int4_quant" "$model" "$task" "$HELMET_16K" "baseline" "4" "" "helmet" "$enable_thinking_mode"
            # INT8
            create_config_with_thinking "int8_quant" "$model" "$task" "$HELMET_16K" "baseline" "8" "" "helmet" "$enable_thinking_mode"
            # StreamingLLM variants
            create_config_with_thinking "streamingllm_n4092_i4" "$model" "$task" "$HELMET_16K" "streamingllm" "" "$streamingllm_4092" "helmet" "$enable_thinking_mode"
            create_config_with_thinking "streamingllm_n2044_i4" "$model" "$task" "$HELMET_16K" "streamingllm" "" "$streamingllm_2048" "helmet" "$enable_thinking_mode"
            # PyramidKV variants
            create_config_with_thinking "pyramidkv_k7_w256_c2048_avgpool" "$model" "$task" "$HELMET_16K" "pyramidkv" "" "$pyramidkv_256_2048" "helmet" "$enable_thinking_mode"
            create_config_with_thinking "pyramidkv_k7_w2048_c8192_avgpool" "$model" "$task" "$HELMET_16K" "pyramidkv" "" "$pyramidkv_2048_8192" "helmet" "$enable_thinking_mode"
            # SnapKV variants
            create_config_with_thinking "snapkv_k7_w256_c2048_maxpool" "$model" "$task" "$HELMET_16K" "snapkv" "" "$snapkv_256_2048" "helmet" "$enable_thinking_mode"
            create_config_with_thinking "snapkv_k7_w2048_c8192_maxpool" "$model" "$task" "$HELMET_16K" "snapkv" "" "$snapkv_2048_8192" "helmet" "$enable_thinking_mode"
        done

        # 32k HELMET tasks
        for task in "${HELMET_TASKS[@]}"; do
            # Baseline
            create_config_with_thinking "baseline" "$model" "$task" "$HELMET_32K" "baseline" "16" "" "helmet" "$enable_thinking_mode"
            # INT4
            create_config_with_thinking "int4_quant" "$model" "$task" "$HELMET_32K" "baseline" "4" "" "helmet" "$enable_thinking_mode"
            # INT8
            create_config_with_thinking "int8_quant" "$model" "$task" "$HELMET_32K" "baseline" "8" "" "helmet" "$enable_thinking_mode"
            # StreamingLLM variants
            create_config_with_thinking "streamingllm_n4092_i4" "$model" "$task" "$HELMET_32K" "streamingllm" "" "$streamingllm_4092" "helmet" "$enable_thinking_mode"
            create_config_with_thinking "streamingllm_n2044_i4" "$model" "$task" "$HELMET_32K" "streamingllm" "" "$streamingllm_2048" "helmet" "$enable_thinking_mode"
            # PyramidKV variants
            create_config_with_thinking "pyramidkv_k7_w256_c2048_avgpool" "$model" "$task" "$HELMET_32K" "pyramidkv" "" "$pyramidkv_256_2048" "helmet" "$enable_thinking_mode"
            create_config_with_thinking "pyramidkv_k7_w2048_c8192_avgpool" "$model" "$task" "$HELMET_32K" "pyramidkv" "" "$pyramidkv_2048_8192" "helmet" "$enable_thinking_mode"
            # SnapKV variants
            create_config_with_thinking "snapkv_k7_w256_c2048_maxpool" "$model" "$task" "$HELMET_32K" "snapkv" "" "$snapkv_256_2048" "helmet" "$enable_thinking_mode"
            create_config_with_thinking "snapkv_k7_w2048_c8192_maxpool" "$model" "$task" "$HELMET_32K" "snapkv" "" "$snapkv_2048_8192" "helmet" "$enable_thinking_mode"
        done

        # LongProc tasks
        for context in "${LONGPROC_CONTEXTS[@]}"; do
            for task in "${LONGPROC_TASKS[@]}"; do
                # Baseline
                create_config_with_thinking "baseline" "$model" "$task" "$context" "baseline" "16" "" "longproc" "$enable_thinking_mode"
                # INT4
                create_config_with_thinking "int4_quant" "$model" "$task" "$context" "baseline" "4" "" "longproc" "$enable_thinking_mode"
                # INT8
                create_config_with_thinking "int8_quant" "$model" "$task" "$context" "baseline" "8" "" "longproc" "$enable_thinking_mode"
                # StreamingLLM variants
                create_config_with_thinking "streamingllm_n4092_i4" "$model" "$task" "$context" "streamingllm" "" "$streamingllm_4092" "longproc" "$enable_thinking_mode"
                create_config_with_thinking "streamingllm_n2044_i4" "$model" "$task" "$context" "streamingllm" "" "$streamingllm_2048" "longproc" "$enable_thinking_mode"
                # PyramidKV variants
                create_config_with_thinking "pyramidkv_k7_w256_c2048_avgpool" "$model" "$task" "$context" "pyramidkv" "" "$pyramidkv_256_2048" "longproc" "$enable_thinking_mode"
                create_config_with_thinking "pyramidkv_k7_w2048_c8192_avgpool" "$model" "$task" "$context" "pyramidkv" "" "$pyramidkv_2048_8192" "longproc" "$enable_thinking_mode"
                # SnapKV variants
                create_config_with_thinking "snapkv_k7_w256_c2048_maxpool" "$model" "$task" "$context" "snapkv" "" "$snapkv_256_2048" "longproc" "$enable_thinking_mode"
                create_config_with_thinking "snapkv_k7_w2048_c8192_maxpool" "$model" "$task" "$context" "snapkv" "" "$snapkv_2048_8192" "longproc" "$enable_thinking_mode"
            done
        done
    done
done

echo ""
echo "=========================================="
echo "Generation Complete!"
echo "=========================================="
echo "Generated $config_count config files in $CONFIG_DIR"
echo ""
echo "Summary:"
echo "  1. StreamingLLM cache size corrections (4092+4): 2 models × 2 contexts × ${#HELMET_TASKS[@]} tasks"
echo "  2. StreamingLLM missing 32k: 4 models × ${#HELMET_TASKS[@]} tasks"
echo "  3. StreamingLLM new suite (2044+4): 6 models × 2 contexts × ${#HELMET_TASKS[@]} tasks"
echo "  4. PyramidKV 32k missing: 6 models × 2 configs × ${#HELMET_TASKS[@]} tasks"
echo "  5. SnapKV 32k missing: 6 models × 2 configs × ${#HELMET_TASKS[@]} tasks"
echo "  6. Quantization 16k missing: 1 model × 2 quants × ${#HELMET_TASKS[@]} tasks"
echo "  7. Quantization 32k missing: 4 models × 3 quants × ${#HELMET_TASKS[@]} tasks"
echo "  8. LongProc ALL: 6 models × 3 contexts × ${#LONGPROC_TASKS[@]} tasks × 9 techniques"
echo "  9. Qwen3 thinking modes: 2 models × 2 modes (thinking/nothinking) × all tasks × all techniques"
echo ""
echo "Job times:"
echo "  - HELMET 32k: 18 hours"
echo "  - HELMET 16k: 6 hours"
echo "  - LongProc 8k: 18 hours"
echo "  - LongProc 2k: 12 hours"
echo "  - LongProc 0.5k: 6 hours"
echo ""
echo "Special notes:"
echo "  - Qwen3-8B and Yarn-Qwen3-8B: Generated with both enable_thinking=True (verbose CoT) and False (no CoT)"
echo "  - All configs use USE_REASONING_CONFIG=false (standard configs, not configs_reasoning)"
echo ""
echo "Next steps:"
echo "  1. Review generated configs in $CONFIG_DIR"
echo "  2. Submit jobs using your existing job submission scripts"
echo "  3. Optionally add 128k runs for NIAH and other quick tasks"
