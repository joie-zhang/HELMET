#!/bin/bash -l

submit_job() {

TASK_NAME=$(basename $TASK .yaml)   
MODEL_NAME=$(basename $MODEL)

OUT_DIR=outputs/debug/PATCH64_compress_outputs${SUFFIX}__${METHOD}
if [ -f ${OUT_DIR}/.${TASK_NAME}.completed ]; then
    echo "skipping ${MODEL_NAME} ${METHOD} - ${TASK_NAME} - ${SPARSITY}"
    # return
else
    echo "!!! running ${MODEL_NAME} ${METHOD} - ${TASK_NAME} - ${SPARSITY}"
fi

# 8 gpus for rag, 4 for everything else 
if [ "$TASK_NAME" == "rag" ]; then
    NUM_GPUS=1 # 8
    TIME="3:00:00"
else
    NUM_GPUS=1 # 4
    TIME="3:00:00"
fi

CMD="python eval.py --config ${TASK} --model_name_or_path $MODEL --tokenizer_name $MODEL --minference $METHOD --output_dir $OUT_DIR $EXTRA --overwrite"

sbatch<<EOT
#!/bin/bash -l
#SBATCH --job-name=DEBUG_helmet-${TASK_NAME}-${METHOD}-${SPARSITY}-${MODEL_NAME}
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --output=./joblog/%x-%A_%a.out                          
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --mem=50G
#SBATCH --time=${TIME} 
#SBATCH --constraint="rh8|rh9" 
#SBATCH --nice=50

source /scratch/gpfs/ab4197/anaconda3/etc/profile.d/conda.sh
conda activate duo

echo $CMD 
echo "num_gpus = ${NUM_GPUS}"
CUDA_LAUNCH_BLOCKING=1 $CMD && echo "${MODEL_NAME} - ${TASK_NAME}" > ${OUT_DIR}/.${TASK_NAME}.completed
EOT

}

export OUTLINES_CACHE_DIR=/tmp/outlines

tasks=(
    # "longproc_addon/configs/html_to_tsv.yaml"
    # "longproc_addon/configs/pseudo_to_code.yaml"
    # "longproc_addon/configs/travel_planning.yaml"
    # "longproc_addon/configs/countdown.yaml"
    # "configs/recall.yaml"
    # "configs/rerank.yaml"
    # "configs/rag.yaml"
    # "configs/json_5k5v.yaml"
    "configs/debug.yaml"
)

TOTAL_CAPACITY=131072
PREFILL=32768
MINFERENCE_WINDOW_SIZE=64 
WITH_PATCH=true
MODEL=/scratch/gpfs/DANQIC/models/Llama-3.1-8B-Instruct

for TASK in ${tasks[@]} ; do 
    for METHOD in snapkv; do # pyramidkv
        for SPARSITY in 50; do 
            # Convert to fraction
            SPARSITY_FRAC=$(echo "$SPARSITY / 100.0" | bc -l)
            
            SUFFIX="_sp${SPARSITY}_tg"
            EXTRA="--no_torch_compile --minference_sparsity $SPARSITY_FRAC --minference_window_size $MINFERENCE_WINDOW_SIZE --minference_compress_group_kvs"

            if [[ $PREFILL -gt 0 ]]; then
                SUFFIX="_sp${SPARSITY}_pf${PREFILL}_tg"
                EXTRA="$EXTRA --minference_chunk_prefilling $PREFILL"
            fi

            if [[ $WITH_PATCH == true ]]; then
                SUFFIX="PATCH_${SUFFIX}"
                EXTRA="$EXTRA --minference_chunking_patch"
            fi
        
            TASK=$TASK MODEL=$MODEL SUFFIX=$SUFFIX EXTRA=$EXTRA METHOD=$METHOD SPARSITY=$SPARSITY WITH_PATCH=$WITH_PATCH submit_job
        done
    done
done

    
