#!/bin/bash -l

submit_job() {
    CONFIG_FILENAME=$(basename $TASK .yaml)
    MODEL_NAME=$(basename $MODEL)
    CONTEXT_LEN=$(echo $TASK | grep -o '[0-9]\+k')

    if [ -z "$SUFFIX" ] || [ "$SUFFIX" = "" ]; then
        OUT_DIR=output/${EXP_TYPE}/${CONTEXT_LEN}/${MODEL_NAME}
    else
        OUT_DIR=output/${EXP_TYPE}/${CONTEXT_LEN}/${MODEL_NAME}/${SUFFIX}
    fi

    if [ -f ${OUT_DIR}/.${CONFIG_FILENAME}.completed ]; then
        echo "skipping ${MODEL_NAME} - ${CONFIG_FILENAME}"
        return
    fi
    
    sbatch<<EOT
#!/bin/bash -l
#SBATCH --job-name=helmet-${EXP_TYPE}-${CONTEXT_LEN}-${MODEL_NAME}-${CONFIG_FILENAME}
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joie@princeton.edu
#SBATCH --output=./joblog/%x-%A_%a.out                          
#SBATCH --error=./joblog/%x-%A_%a.err                          
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=8:00:00                                     

module purge
source /scratch/gpfs/ab4197/anaconda3/etc/profile.d/conda.sh
conda activate duo

echo python /scratch/gpfs/DANQIC/jz4391/HELMET/eval.py \
    --config ${CONFIG_FILENAME} \
    --model_name_or_path $MODEL \
    --output_dir $OUT_DIR \
    --seed $SEED \
    --tag $TAG \
    $EXTRA
python /scratch/gpfs/DANQIC/jz4391/HELMET/eval.py \
    --config ${TASK} \
    --model_name_or_path $MODEL \
    --output_dir $OUT_DIR \
    --seed $SEED \
    --tag $TAG \
    $EXTRA && \
    echo "${MODEL_NAME} - ${CONFIG_FILENAME}" > ${OUT_DIR}/.${CONFIG_FILENAME}.completed
EOT

}

export OUTLINES_CACHE_DIR=/tmp/outlines

tasks=(
    # "configs/niah_16k.yaml"
    # "configs/cite_16k.yaml"
    # "configs/recall_jsonkv_16k.yaml"
    # "configs/rerank_16k.yaml"
    # "configs/rag_hotpotqa_16k.yaml"
    # "configs/rag_nq_16k.yaml"

    # "configs/niah_32k.yaml"
    # "configs/cite_32k.yaml"
    # "configs/recall_jsonkv_32k.yaml"
    # "configs/rerank_32k.yaml"
    # "configs/rag_hotpotqa_32k.yaml"

    # "configs/rag_nq_32k.yaml"

    # "longproc_addon/configs/html_to_tsv_0.5k.yaml"
    # "longproc_addon/configs/pseudo_to_code_0.5k.yaml"
    # "longproc_addon/configs/pseudo_to_code_2k.yaml"

    # "longproc_addon/configs/html_to_tsv_2k.yaml"
    # "longproc_addon/configs/travel_planning_2k.yaml"

    "longproc_addon/configs/html_to_tsv_8k.yaml"
    "longproc_addon/configs/travel_planning_8k.yaml"
)

models=(
    /scratch/gpfs/DANQIC/models/Llama-3.1-8B-Instruct
    # /scratch/gpfs/DANQIC/models/Qwen2.5-7B-Instruct
)

exp_types=(
    "duoattn"
    # "baseline"
    # "minference"
    # "snapkv"
    # "pyramidkv"
    # "streamingllm"
    # "int4"
    # "int8"
)

for TASK in ${tasks[@]}; do # cite rerank rag_nq rag_hotpotqa recall_jsonkv niah html_to_tsv pseudo_to_code travel_planning
    for MODEL in ${models[@]}; do
        for EXP_TYPE in ${exp_types[@]}; do
            MASKS="/scratch/gpfs/ab4197/p-longhead/duo-attention/attn_patterns/Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10/full_attention_heads.tsv"
            SPARSITY=0.5
            PREFILL=32768
            QUANT_BITS=16
            
            SUFFIX="_sp${SPARSITY}_tg"
            EXTRA="--no_torch_compile --duoattn $MASKS --duoattn_sparsity $SPARSITY"
            TAG="${EXP_TYPE}_${CONTEXT_LEN}_${MODEL}_${QUANT_BITS}bit_${SPARSITY}sp_${PREFILL}pf"
            SEED=42
            
            if [[ $PREFILL -gt 0 ]]; then
                SUFFIX="_sp${SPARSITY}_pf${PREFILL}_tg"
                EXTRA="$EXTRA --duoattn_chunk_prefilling $PREFILL"
            fi
            
            TASK=$TASK MODEL=$MODEL SUFFIX=$SUFFIX EXTRA=$EXTRA TAG=$TAG SEED=$SEED submit_job
        done
    done
done


########################################################
# python print_prulong.py \
#     output/duoattn/5k/Llama-3.1-8B-Instruct/pseudo_to_code_0.5k.yaml \
#     output/duoattn/5k/Llama-3.1-8B-Instruct/html_to_tsv_0.5k.yaml
