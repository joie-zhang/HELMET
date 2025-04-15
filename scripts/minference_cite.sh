#!/bin/bash -l
##############################
#       Job blueprint        #
##############################
#SBATCH --job-name=minference_helmet_cite_32k_gpushare
#SBATCH --array=0-1
#SBATCH --output=./joblog/%x-%A_%a.out
#SBATCH --error=./joblog/%x-%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=1:59:00
#SBATCH --gres=gpu:1 --ntasks-per-node=1 -N 1
#SBATCH --constraint=gpu80
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joie@princeton.edu

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cache_start_size)
            CACHE_START_SIZE="$2"
            shift 2
            ;;
        --cache_recent_size)
            CACHE_RECENT_SIZE="$2"
            shift 2
            ;;
        --enable_positional_shift)
            ENABLE_POSITIONAL_SHIFT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Set default values if not provided
CACHE_START_SIZE=${CACHE_START_SIZE:-4}
CACHE_RECENT_SIZE=${CACHE_RECENT_SIZE:-2044}
ENABLE_POSITIONAL_SHIFT=${ENABLE_POSITIONAL_SHIFT:-true}

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "Array Job ID                   = $SLURM_ARRAY_JOB_ID"
echo "Array Task ID                  = $SLURM_ARRAY_TASK_ID"
echo "Cache                          = $TRANSFORMERS_CACHE"

module purge
module load anaconda3/2023.3
module load gcc-toolset/10
source /opt/rh/gcc-toolset-10/enable
source /scratch/gpfs/DANQIC/jz4391/MInference/minenv/bin/activate

# Constants
SEED=42
S_MODELS=("Llama-3.1-8B-Instruct")
# S_MODELS=("Llama-3.1-8B-Instruct" "Qwen2.5-7B-Instruct")
# QUANTIZE_VALUES=(8 16)
QUANTIZE_VALUES=(16)
# CONTEXT_LENGTHS=("8k" "16k" "32k" "64k")
CONTEXT_LENGTHS=("32k")

# Total combinations: models * quantize * context lengths
TOTAL_MODELS=${#S_MODELS[@]}
TOTAL_QUANTIZE=${#QUANTIZE_VALUES[@]}
TOTAL_CONTEXT=${#CONTEXT_LENGTHS[@]}
TOTAL_COMBINATIONS=$((TOTAL_MODELS * TOTAL_QUANTIZE * TOTAL_CONTEXT))

# Map SLURM_ARRAY_TASK_ID to specific combination
IDX=$SLURM_ARRAY_TASK_ID

MODEL_IDX=$((IDX / (TOTAL_QUANTIZE * TOTAL_CONTEXT)))
REMAINDER=$((IDX % (TOTAL_QUANTIZE * TOTAL_CONTEXT)))
QUANTIZE_IDX=$((REMAINDER / TOTAL_CONTEXT))
CONTEXT_IDX=$((REMAINDER % TOTAL_CONTEXT))

# Get specific values for this task
MNAME="${S_MODELS[$MODEL_IDX]}"
QUANTIZE="${QUANTIZE_VALUES[$QUANTIZE_IDX]}"
CONTEXT_LEN="${CONTEXT_LENGTHS[$CONTEXT_IDX]}"

# Derived variables
CONFIGS=("cite_${CONTEXT_LEN}.yaml")
OUTPUT_DIR="output/$CONTEXT_LEN/bit$QUANTIZE/$MNAME"
MODEL_NAME="/scratch/gpfs/DANQIC/models/$MNAME"

# Create output directory
mkdir -p "$OUTPUT_DIR"

shopt -s nocasematch
chat_models=".*(chat|instruct|it$|nous|command|Jamba-1.5|MegaBeam).*"
OPTIONS=""
if ! [[ $MNAME =~ $chat_models ]]; then
    OPTIONS="$OPTIONS --use_chat_template False"
fi

# Print debug information
echo "Model name                    = $MNAME"
echo "Quantization level            = $QUANTIZE"
echo "Context length                = $CONTEXT_LEN"
echo "Evaluation output dir         = $OUTPUT_DIR"
echo "Config file                   = ${CONFIGS[@]}"
echo "Options                       = $OPTIONS"

# Run evaluation for each configuration file
for CONFIG in "${CONFIGS[@]}"; do
    python eval.py \
        --config configs/$CONFIG \
        --seed $SEED \
        --output_dir $OUTPUT_DIR \
        --tag v1 \
        --model_name_or_path $MODEL_NAME \
        --minference \
        --model_class streamingllm \
        --cache_start_size $CACHE_START_SIZE \
        --cache_recent_size $CACHE_RECENT_SIZE \
        --enable_positional_shift $ENABLE_POSITIONAL_SHIFT \
        $OPTIONS
done

echo "Finished with exit code $?"
wait;
