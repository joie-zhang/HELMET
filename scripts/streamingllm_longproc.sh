#!/bin/bash
##############################
#       Job blueprint        #
##############################
#SBATCH --job-name=streamingllm_longproc_html_to_tsv_and_countdown_0.5k
#SBATCH --output=./joblog/%x-%A_%a.out                          ## Stdout
#SBATCH --error=./joblog/%x-%A_%a.err                           ## Stderr
#SBATCH -N 1                                        ##nodes
#SBATCH -n 1                                        ##tasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=1:59:00
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
module load gcc-toolset/10
source /opt/rh/gcc-toolset-10/enable
source /scratch/gpfs/DANQIC/jz4391/MInference/minenv/bin/activate

IDX=$SLURM_ARRAY_TASK_ID
NGPU=$SLURM_GPUS_ON_NODE
if [[ -z $SLURM_ARRAY_TASK_ID ]]; then
    IDX=0
    NGPU=1
fi

PORT=$(shuf -i 30000-65000 -n 1)
echo "Port                          = $PORT"
export OMP_NUM_THREADS=8
TAG=v1

# CONFIGS=(html_to_tsv_0.5k.yaml)
# CONFIGS=(countdown_0.5k.yaml)
# CONFIGS=(html_to_tsv_2k.yaml)
# CONFIGS=(countdown_2k.yaml)
CONFIGS=(travel_planning_2k.yaml)
# CONFIGS=(${CONFIGS[8]})

SEED=42
QUANTIZE=16
CONTEXT_LEN="0.5k"
M_IDX=$IDX

# Array for models 13B and smaller (2 models)
S_MODELS=(
  "Llama-3.1-8B-Instruct" # 0
#   "Qwen2.5-7B-Instruct" # 1
)
MNAME="${S_MODELS[$M_IDX]}"

# MODEL_NAME="meta-llama/Llama-2-7b-hf"
OUTPUT_DIR="output/minference/streamingllm/$CONTEXT_LEN/bit$QUANTIZE/$MNAME/$SLURM_ARRAY_JOB_ID"
MODEL_NAME="/scratch/gpfs/DANQIC/models/$MNAME" # CHANGE PATH HERE or you can change the array to load from HF
# MODEL_CLASS="minference"

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

for CONFIG in "${CONFIGS[@]}"; do
    python eval.py \
        --config longproc_addon/configs/$CONFIG \
        --model_name_or_path $MODEL_NAME \
        --output_dir $OUTPUT_DIR \
        --quantize $(($QUANTIZE)) \
        --streamingllm \
        $OPTIONS 
done
