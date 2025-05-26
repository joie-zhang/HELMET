#!/bin/bash -l

submit_job() {

TASK_NAME=$(basename $TASK .yaml)   
MODEL_NAME=$(basename $MODEL)

OUT_DIR=outputs/${MODEL_NAME}_ablations/${TAG}__outputs__${METHOD}
if [ -f ${OUT_DIR}/.${TASK_NAME}.completed ]; then
    return
fi

NUM_GPUS=1 
TIME="23:00:00"
MEM=80G

CMD="python eval.py --config ${TASK} --model_name_or_path $MODEL --tokenizer_name $MODEL --output_dir $OUT_DIR $EXTRA"
JOB_NAME="ABLATE_${TAG}_helmet-${TASK_NAME}-${METHOD}-${MODEL_NAME}"

if squeue -h --me -n ${JOB_NAME} | grep -q .; then
    return
else
    echo "!!! running ${JOB_NAME}"
fi

sbatch<<EOT
#!/bin/bash -l
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --output=./joblog/%x-%A_%a.out                          
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME} 
#SBATCH --constraint="rh9" 

source /scratch/gpfs/ab4197/anaconda3/etc/profile.d/conda.sh
conda activate duo

echo $CMD 
echo "num_gpus = ${NUM_GPUS}"
CUDA_LAUNCH_BLOCKING=1 $CMD && echo "${MODEL_NAME} - ${TASK_NAME}" > ${OUT_DIR}/.${TASK_NAME}.completed
EOT

}

export OUTLINES_CACHE_DIR=/tmp/outlines

tasks=(
    "longproc_addon/configs/html_to_tsv.yaml"
    "longproc_addon/configs/travel_planning.yaml"
    "configs/recall.yaml"
    "configs/rerank.yaml"
    "configs/rag.yaml"
    "configs/icl.yaml"
    "configs/longqa.yaml"
    "configs/summ.yaml"
)

MODEL=/scratch/gpfs/DANQIC/models/Llama-3.1-8B-Instruct
TOTAL_CAPACITY=131072
PREFILL=32768

for TASK in ${tasks[@]} ; do 
    # prulong
    TAG="PRULONG"
    MASKS="/scratch/gpfs/awettig/prulong/checkpoints/Llama-3.1-8B-Instruct/prulong_prolong-sample-long_bsz16_steps1000_lr1e-5_warmup0.1_sp0.7_cw1024_mlr1_rlr1_wfrozen/masks_sp0.7.tsv"
    SPARSITY=0.7
    EXTRA="--no_torch_compile --duoattn $MASKS --duoattn_sparsity $SPARSITY"
    if [[ $PREFILL -gt 0 ]]; then
        SUFFIX="_pf${PREFILL}_tg"
        EXTRA="$EXTRA --duoattn_chunk_prefilling $PREFILL"
    else
        SUFFIX="_tg"
    fi

    TASK=$TASK MODEL=$MODEL SUFFIX=$SUFFIX EXTRA=$EXTRA TAG=$TAG submit_job

    # duo
    TAG="DUO"
    MASKS="/scratch/gpfs/awettig/prulong/checkpoints/Llama-3.1-8B-Instruct_duo-official_sp0.7/masks.tsv"
    SPARSITY=0.7
    EXTRA="--no_torch_compile --duoattn $MASKS --duoattn_sparsity $SPARSITY"
    if [[ $PREFILL -gt 0 ]]; then
        SUFFIX="_pf${PREFILL}_tg"
        EXTRA="$EXTRA --duoattn_chunk_prefilling $PREFILL"
    else
        SUFFIX="_tg"
    fi

    TASK=$TASK MODEL=$MODEL SUFFIX=$SUFFIX EXTRA=$EXTRA TAG=$TAG submit_job
done

    
