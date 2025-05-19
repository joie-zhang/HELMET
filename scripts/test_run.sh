#!/bin/bash

# Check if config file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <config_file>"
    echo "Example from HELMET root: $0 scripts/configs/streamingllm_helmet_config.sh"
    echo "Example from scripts dir: $0 configs/streamingllm_helmet_config.sh"
    exit 1
fi

# Source the config file
source "$1"

# Print debug information
echo "=== Test Run Configuration ==="
echo "BASE_CONFIGS: ${BASE_CONFIGS[@]}"
echo "CONTEXT_LENGTHS: ${CONTEXT_LENGTHS[@]}"
echo "MODELS: ${MODELS[@]}"
echo "QUANTIZE: ${QUANTIZE[@]}"
echo "EXP_TYPE: $EXP_TYPE"
echo "BENCHMARK: $BENCHMARK"
echo "SEED: $SEED"

# Verify required variables are set
if [ -z "$BASE_CONFIGS" ] || [ -z "$CONTEXT_LENGTHS" ] || [ -z "$MODELS" ] || [ -z "$EXP_TYPE" ] || [ -z "$BENCHMARK" ] || [ -z "$SEED" ]; then
    echo "Error: Required variables are not set"
    exit 1
fi

# Validate quantization settings
if [ "${EXP_TYPE}" != "baseline" ] && [ "${#QUANTIZE[@]}" -gt 1 ]; then
    echo "ERROR: Non-baseline experiment type '${EXP_TYPE}' cannot use quantization options"
    exit 1
fi

# Validate quantization values for baseline experiments
if [ "${EXP_TYPE}" = "baseline" ]; then
    for quant in "${QUANTIZE[@]}"; do
        if [[ ! " 4 8 16 " =~ " $quant " ]]; then
            echo "ERROR: Invalid quantization value '$quant' for baseline experiment"
            echo "Allowed values are: 4, 8, 16"
            exit 1
        fi
    done
fi

# Load required modules
module purge
module load anaconda3/2023.3
module load gcc/11
source /scratch/gpfs/DANQIC/jz4391/MInference/minenv/bin/activate

# For testing, we'll just use the first configuration of each parameter
MNAME="${MODELS[0]}"
MNAME="${MNAME## }"    # Remove leading spaces
MNAME="${MNAME%% }"    # Remove trailing spaces
MODEL_NAME="/scratch/gpfs/DANQIC/models/$MNAME"

# Create config name for first configuration
CONFIG="${BASE_CONFIGS[0]}_${CONTEXT_LENGTHS[0]}.yaml"
CONTEXT_LEN="${CONTEXT_LENGTHS[0]}"

# Set up experiment-specific parameters
if [ "${EXP_TYPE}" = "baseline" ]; then
    QUANT_BITS="${QUANTIZE[0]}"
    QUANTIZE_PARAM="--quantize ${QUANT_BITS}"
    TAG="test_${EXP_TYPE}_${CONTEXT_LEN}_${MNAME}_${QUANT_BITS}bit"
    OUTPUT_DIR="/scratch/gpfs/DANQIC/jz4391/HELMET/output/test_${EXP_TYPE}/${CONTEXT_LEN}/${MNAME}/${QUANT_BITS}bit"
elif [ "${EXP_TYPE}" = "streamingllm" ] || [ "${EXP_TYPE}" = "streamingllm_original" ]; then
    QUANT_BITS=""
    QUANTIZE_PARAM=""
    STREAM_SUFFIX="local${N_LOCAL}_init${N_INIT}"
    TAG="test_${EXP_TYPE}_${CONTEXT_LEN}_${MNAME}_${STREAM_SUFFIX}"
    OUTPUT_DIR="/scratch/gpfs/DANQIC/jz4391/HELMET/output/test_${EXP_TYPE}/${CONTEXT_LEN}/${MNAME}/${STREAM_SUFFIX}"
else
    QUANT_BITS=""
    QUANTIZE_PARAM=""
    TAG="test_${EXP_TYPE}_${CONTEXT_LEN}_${MNAME}"
    OUTPUT_DIR="/scratch/gpfs/DANQIC/jz4391/HELMET/output/test_${EXP_TYPE}/${CONTEXT_LEN}/${MNAME}"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set chat template option
shopt -s nocasematch
chat_models=".*(chat|instruct|it$|nous|command|Jamba-1.5|MegaBeam).*"
OPTIONS=""
if ! [[ $MNAME =~ $chat_models ]]; then
    OPTIONS="$OPTIONS --use_chat_template False"
fi

# Check for required parameters based on experiment type
if [[ "$EXP_TYPE" == "streamingllm" || "$EXP_TYPE" == "streamingllm_original" ]]; then
    if [ -z "$N_LOCAL" ] || [ -z "$N_INIT" ]; then
        echo "Error: StreamingLLM configuration parameters are not set"
        echo "N_LOCAL: $N_LOCAL"
        echo "N_INIT: $N_INIT"
        exit 1
    fi
fi

# Set experiment-specific parameters
EXP_TYPE_PARAM=""
KV_CACHE_PARAMS=""

case $EXP_TYPE in
    "streamingllm")
        EXP_TYPE_PARAM="--streamingllm"
        KV_CACHE_PARAMS="--n_local $N_LOCAL --n_init $N_INIT"
        ;;
    "streamingllm_original")
        EXP_TYPE_PARAM="--streamingllm_original"
        KV_CACHE_PARAMS="--n_local $N_LOCAL --n_init $N_INIT"
        ;;
    "baseline")
        EXP_TYPE_PARAM=""
        ;;
    *)
        echo "Error: Unknown experiment type '$EXP_TYPE'"
        echo "For testing, only streamingllm, streamingllm_original, and baseline are supported"
        exit 1
        ;;
esac

# Set config path based on benchmark
if [ "$BENCHMARK" == "longproc" ]; then
    CONFIG_PATH="/scratch/gpfs/DANQIC/jz4391/HELMET/longproc_addon/configs/$CONFIG"
else
    CONFIG_PATH="/scratch/gpfs/DANQIC/jz4391/HELMET/configs/$CONFIG"
fi

echo "=== Running Test Evaluation ==="
echo "Model name            = $MNAME"
echo "Context length        = $CONTEXT_LEN"
echo "Config file          = $CONFIG"
echo "Evaluation output dir = $OUTPUT_DIR"
echo "Options              = $OPTIONS"

# Run evaluation
python /scratch/gpfs/DANQIC/jz4391/HELMET/eval.py \
    --config $CONFIG_PATH \
    --seed $SEED \
    --output_dir $OUTPUT_DIR \
    --tag "$TAG" \
    --model_name_or_path $MODEL_NAME \
    $QUANTIZE_PARAM \
    $EXP_TYPE_PARAM \
    $KV_CACHE_PARAMS \
    $OPTIONS

echo "Finished with exit code $?" 