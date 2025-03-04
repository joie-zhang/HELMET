#!/bin/bash -l
##############################
#       Job blueprint        #
##############################
#SBATCH --job-name=1hr_50G_helmet_quantize ## CHANGE JOBNAME HERE
#SBATCH --array=0-7  # Total combinations: 2 models * 2 quantize values * 2 context lengths = 8
#SBATCH --output=./joblog/%x-%A_%a.out                          ## Stdout
#SBATCH --error=./joblog/%x-%A_%a.err                           ## Stderr
#SBATCH -N 1                                        ##nodes
#SBATCH -n 1                                        ##tasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1 --ntasks-per-node=1 -N 1
#SBATCH --constraint=gpu80
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joie@princeton.edu

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

source env/bin/activate

# Constants
SEED=42
S_MODELS=("Llama-3.1-8B-Instruct" "Qwen2.5-7B-Instruct")
QUANTIZE_VALUES=(8 16)
# CONTEXT_LENGTHS=("8k" "16k" "32k" "64k")
CONTEXT_LENGTHS=("16k" "32k")

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
OUTPUT_DIR="output/$CONTEXT_LEN/bit$QUANTIZE/$MNAME/$SLURM_ARRAY_JOB_ID"
MODEL_NAME="/scratch/gpfs/DANQIC/models/$MNAME" # CHANGE PATH HERE or you can change the array to load from HF

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
        --quantize $(($QUANTIZE)) \
        $OPTIONS
done

echo "Finished with exit code $?"
wait;
