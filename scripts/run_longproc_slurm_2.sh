#!/bin/bash -l

#SBATCH --job-name=1hr_50G_longproc_quantize
#SBATCH --output=./joblog/%x-%A_%a.out
#SBATCH --error=./joblog/%x-%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1 --ntasks-per-node=1 -N 1
#SBATCH --constraint=gpu80
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joie@princeton.edu

# Define arrays for each variable
S_MODELS=("Llama-3.1-8B-Instruct" "Qwen2.5-7B-Instruct")
QUANTIZE_VALUES=(4 8)
CONTEXT_LENGTHS=("2k")
CONFIG_BASES=("html_to_tsv" "travel_planning")

# Calculate total number of jobs
TOTAL_JOBS=$((${#S_MODELS[@]} * ${#QUANTIZE_VALUES[@]} * ${#CONTEXT_LENGTHS[@]} * ${#CONFIG_BASES[@]}))

# Set the job array
#SBATCH --array=0-$((TOTAL_JOBS - 1))

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

# Calculate indices for each variable
IDX=$SLURM_ARRAY_TASK_ID
MODEL_IDX=$((IDX % ${#S_MODELS[@]}))
IDX=$((IDX / ${#S_MODELS[@]}))
QUANTIZE_IDX=$((IDX % ${#QUANTIZE_VALUES[@]}))
IDX=$((IDX / ${#QUANTIZE_VALUES[@]}))
CONTEXT_IDX=$((IDX % ${#CONTEXT_LENGTHS[@]}))
CONFIG_IDX=$((IDX / ${#CONTEXT_LENGTHS[@]}))

# Set variables for this job
MNAME="${S_MODELS[$MODEL_IDX]}"
QUANTIZE="${QUANTIZE_VALUES[$QUANTIZE_IDX]}"
CONTEXT_LEN="${CONTEXT_LENGTHS[$CONTEXT_IDX]}"
CONFIG_BASE="${CONFIG_BASES[$CONFIG_IDX]}"
CONFIG="${CONFIG_BASE}_${CONTEXT_LEN}.yaml"

NGPU=$SLURM_GPUS_ON_NODE
PORT=$(shuf -i 30000-65000 -n 1)
echo "Port                          = $PORT"
export OMP_NUM_THREADS=8
TAG=v1
SEED=42

OUTPUT_DIR="output/$CONTEXT_LEN/bit$QUANTIZE/$MNAME/$SLURM_ARRAY_JOB_ID"
MODEL_NAME="/scratch/gpfs/DANQIC/models/$MNAME"

shopt -s nocasematch
chat_models=".*(chat|instruct|it$|nous|command|Jamba-1.5|MegaBeam).*"
echo $MNAME
if ! [[ $MNAME =~ $chat_models ]]; then
    OPTIONS="$OPTIONS --use_chat_template False"
fi

echo "Evaluation output dir         = $OUTPUT_DIR"
echo "Tag                           = $TAG"
echo "Model name                    = $MODEL_NAME"
echo "Options                       = $OPTIONS"
echo "Quantization                  = $QUANTIZE"
echo "Context Length                = $CONTEXT_LEN"
echo "Config file                   = $CONFIG"

python eval.py \
    --config longproc_addon/configs/$CONFIG \
    --seed $SEED \
    --output_dir $OUTPUT_DIR \
    --tag $TAG \
    --model_name_or_path $MODEL_NAME \
    --quantize $QUANTIZE \
    $OPTIONS

echo "finished with $?"
wait;
