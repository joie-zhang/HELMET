#!/bin/bash -l

submit_job() {

TASK_NAME=$(basename $TASK .yaml)   
MODEL_NAME=$(basename $MODEL)

OUT_DIR=outputs/${MODEL_NAME}/outputs${SUFFIX}__${METHOD}
if [ -f ${OUT_DIR}/.${TASK_NAME}.completed ]; then
    echo "skipping ${MODEL_NAME} ${METHOD} - ${TASK_NAME} - ${SPARSITY}"
    return
fi

NUM_GPUS=1

CMD="python eval.py --config ${TASK} --model_name_or_path $MODEL --tokenizer_name $MODEL --minference $METHOD --output_dir $OUT_DIR $EXTRA"

sbatch<<EOT
#!/bin/bash -l
#SBATCH --job-name=helmet-${MODEL_NAME}-${TASK_NAME}
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --output=./joblog/%x-%A_%a.out                          
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --mem=30G
#SBATCH --time=12:00:00 
#SBATCH --nice=50                                

source /scratch/gpfs/ab4197/anaconda3/etc/profile.d/conda.sh
conda activate duo

echo $CMD 
echo "num_gpus = ${NUM_GPUS}"
$CMD && echo "${MODEL_NAME} - ${TASK_NAME}" > ${OUT_DIR}/.${TASK_NAME}.completed
EOT

}

export OUTLINES_CACHE_DIR=/tmp/outlines

tasks=(
    "longproc_addon/configs/html_to_tsv.yaml"
    "longproc_addon/configs/pseudo_to_code.yaml"
    "longproc_addon/configs/travel_planning.yaml"
    "longproc_addon/configs/countdown.yaml"
    "configs/recall.yaml"
    "configs/rerank.yaml"
    "configs/rag.yaml"
    "configs/json_5k5v.yaml"
)

TOTAL_CAPACITY=131072
PREFILL=32768
MINFERENCE_WINDOW_SIZE=1152 

for TASK in ${tasks[@]} ; do 
    MODEL=/scratch/gpfs/DANQIC/models/Llama-3.1-8B-Instruct
    for METHOD in pyramidkv snapkv; do
        for SPARSITY in 30 40 50 60 70 80; do # 30 40 50 60 70 80
            # Convert to fraction
            SPARSITY_FRAC=$(echo "$SPARSITY / 100.0" | bc -l)
            
            SUFFIX="_sp${SPARSITY}_tg"
            EXTRA="--no_torch_compile --minference_sparsity $SPARSITY_FRAC --minference_window_size $MINFERENCE_WINDOW_SIZE"

            if [[ $PREFILL -gt 0 ]]; then
                SUFFIX="_sp${SPARSITY}_pf${PREFILL}_tg"
                EXTRA="$EXTRA --minference_chunk_prefilling $PREFILL"
            fi
        
            TASK=$TASK MODEL=$MODEL SUFFIX=$SUFFIX EXTRA=$EXTRA METHOD=$METHOD SPARSITY=$SPARSITY submit_job
        done
    done
done

    
