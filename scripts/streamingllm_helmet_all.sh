#!/bin/bash -l
##############################
#       Job blueprint        #
##############################
#SBATCH --job-name=baseline_helmet_16k_eval_metric_testing
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
module load gcc/11
source /scratch/gpfs/DANQIC/jz4391/MInference/minenv/bin/activate

# Constants
SEED=42
S_MODELS=("Llama-3.1-8B-Instruct")
# S_MODELS=("Llama-3.1-8B-Instruct" "Qwen2.5-7B-Instruct")
QUANTIZE_VALUES=(16)
CONTEXT_LENGTHS=("16k")

# Get specific values for this task
MNAME="Llama-3.1-8B-Instruct"
QUANTIZE="16"
CONTEXT_LEN="16k"

CONFIGS=("recall_jsonkv_${CONTEXT_LEN}.yaml")
# CONFIGS=("cite_${CONTEXT_LEN}.yaml" "rerank_${CONTEXT_LEN}.yaml" "recall_jsonkv_${CONTEXT_LEN}.yaml" "rag_${CONTEXT_LEN}.yaml")
OUTPUT_DIR="/scratch/gpfs/DANQIC/jz4391/HELMET/output/metric_testing/baseline/$CONTEXT_LEN/$MNAME"
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
        --quantize $QUANTIZE \
        $OPTIONS
done

echo "Finished with exit code $?"
wait;