#!/bin/bash -l

submit_job() {
    TASK_NAME=$(basename $TASK .yaml)   
    MODEL_NAME=$(basename $MODEL)

    OUT_DIR=outputs/${MODEL_NAME}/outputs${SUFFIX}
    if [ -f ${OUT_DIR}/.${TASK_NAME}.completed ]; then
        echo "skipping ${MODEL_NAME} - ${TASK_NAME}"
        return
    fi
    
    sbatch<<EOT
#!/bin/bash -l
#SBATCH --job-name=helmet-${MODEL_NAME}-${TASK_NAME}
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --output=./joblog/%x-%A_%a.out                          
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=12:00:00                                     

source /scratch/gpfs/ab4197/anaconda3/etc/profile.d/conda.sh
conda activate duo

echo python eval.py --config ${TASK} --model_name_or_path $MODEL --tokenizer_name $MODEL --output_dir $OUT_DIR $EXTRA
python eval.py --config ${TASK} --model_name_or_path $MODEL --tokenizer_name $MODEL --output_dir $OUT_DIR $EXTRA && \
    echo "${MODEL_NAME} - ${TASK_NAME}" > ${OUT_DIR}/.${TASK_NAME}.completed
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

models=(
    ~/pli/models/allenai/Llama-3.1-Tulu-3-8B-SFT
    ~/pli/models/allenai/Llama-3.1-Tulu-3-8B-DPO
    ~/pli/models/meta-llama/Llama-3.1-8B
    ~/pli/models/meta-llama/Llama-3.1-8B-Instruct
    ~/pli/models/princeton-nlp/Llama-3-8B-ProLong-512k-Base
    ~/pli/models/princeton-nlp/Llama-3-8B-ProLong-512k-Instruct
)

prulonged_models=(
    /scratch/gpfs/ab4197/p-longhead/prulong/checkpoints/sft_prulong_Llama-3-8B-ProLong-512k-Base_packed_ultrachat_bsz64_steps250_lr2e-5_warmup0.05_sp0.7_cw1024_mlr0_rlr0_mfrozen_stream1_2025-02-19_14-14-52
    /scratch/gpfs/ab4197/p-longhead/prulong/checkpoints/prulong_Llama-3.1-8B-Instruct_long-context-524288_mondaymix_bsz32_steps1500_lr1e-5_warmup0.1_sp0.7_cw1024_mlr1_rlr1_wfrozen_stream1_2024-12-31_03-01-52
)

for TASK in ${tasks[@]}; do # cite icl longqa rag recall rerank summ
    for MODEL in ${models[@]}; do 

        EXTRA="--no_torch_compile"

        TASK=$TASK MODEL=$MODEL EXTRA=$EXTRA SUFFIX="" submit_job
    done


    for MODEL in ${prulonged_models[@]}; do 

        SPARSITY=0.7
        MASKS=${MODEL}/masks_sp${SPARSITY}.tsv
        PREFILL=32768

        SUFFIX="_sp${SPARSITY}_tg"
        EXTRA="--no_torch_compile --duoattn $MASKS  --duoattn_sparsity $SPARSITY"

        if [[ $PREFILL -gt 0 ]]; then
            SUFFIX="_sp${SPARSITY}_pf${PREFILL}_tg"
            EXTRA="$EXTRA --duoattn_chunk_prefilling $PREFILL"
        fi

        TASK=$TASK MODEL=$MODEL SUFFIX=$SUFFIX EXTRA=$EXTRA submit_job
    done
    
    
    MODEL=/scratch/gpfs/ab4197/p-longhead/prulong/checkpoints/sft_prolong-512k-base_sft_ultrachat_bsz64_steps250_lr2e-5_warmup0.05_sp0.0_cw1024_mlr0_rlr0_mfrozen_stream1_2025-01-17_02-09-56
    SPARSITY=0.0
    MASKS=${MODEL}/masks_sp${SPARSITY}.tsv
    PREFILL=0    

    SUFFIX="_sp${SPARSITY}_tg"
    EXTRA="--no_torch_compile --duoattn $MASKS  --duoattn_sparsity $SPARSITY"

    if [[ $PREFILL -gt 0 ]]; then
        SUFFIX="_sp${SPARSITY}_pf${PREFILL}_tg"
        EXTRA="$EXTRA --duoattn_chunk_prefilling $PREFILL"
    fi

    TASK=$TASK MODEL=$MODEL SUFFIX=$SUFFIX EXTRA=$EXTRA submit_job
    
    
    MODEL=~/pli/models/meta-llama/Llama-3.1-8B-Instruct
    MASKS="/scratch/gpfs/ab4197/p-longhead/duo-attention/attn_patterns/Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10/full_attention_heads.tsv"
    SPARSITY=0.7
    PREFILL=32768
        
    SUFFIX="_sp${SPARSITY}_tg"
    EXTRA="--no_torch_compile --duoattn $MASKS  --duoattn_sparsity $SPARSITY"

    if [[ $PREFILL -gt 0 ]]; then
        SUFFIX="_sp${SPARSITY}_pf${PREFILL}_tg"
        EXTRA="$EXTRA --duoattn_chunk_prefilling $PREFILL"
    fi
    
    TASK=$TASK MODEL=$MODEL SUFFIX=$SUFFIX EXTRA=$EXTRA submit_job
    
    

    MODEL=~/pli/models/princeton-nlp/Llama-3-8B-ProLong-512k-Instruct
    MASKS="/home/ydong/duo-attention/attn_patterns/prolong-8b-512k-instruct/lr_002-reg_0.05-ctx_1000_128000-multi_passkey10/full_attention_heads_latest.tsv"
    SPARSITY=0.7
    PREFILL=8192
        
    SUFFIX="_sp${SPARSITY}_tg"
    EXTRA="--no_torch_compile --duoattn $MASKS  --duoattn_sparsity $SPARSITY"

    if [[ $PREFILL -gt 0 ]]; then
        SUFFIX="_sp${SPARSITY}_pf${PREFILL}_tg"
        EXTRA="$EXTRA --duoattn_chunk_prefilling $PREFILL"
    fi
    
    TASK=$TASK MODEL=$MODEL SUFFIX=$SUFFIX EXTRA=$EXTRA submit_job
done


########################################################
python print_prulong.py \
    outputs/Llama-3.1-8B/outputs \
    outputs/Llama-3.1-8B-Instruct/outputs \
    outputs/Llama-3.1-8B-Instruct/outputs_sp0.7_pf8192_tg \
    outputs/Llama-3.1-Tulu-3-8B-SFT/outputs \
    outputs/Llama-3.1-Tulu-3-8B-DPO/outputs \
    outputs/prulong_Llama-3.1-8B-Instruct_long-context-524288_mondaymix_bsz32_steps1500_lr1e-5_warmup0.1_sp0.7_cw1024_mlr1_rlr1_wfrozen_stream1_2024-12-31_03-01-52/outputs_sp0.7_pf8192_tg \
    outputs/Llama-3-8B-ProLong-512k-Base/outputs \
    outputs/Llama-3-8B-ProLong-512k-Instruct/outputs \
    outputs/sft_prulong_Llama-3-8B-ProLong-512k-Base_packed_ultrachat_bsz64_steps250_lr2e-5_warmup0.05_sp0.7_cw1024_mlr0_rlr0_mfrozen_stream1_2025-02-19_14-14-52/outputs_sp0.7_pf8192_tg \
    outputs/sft_prolong-512k-base_sft_ultrachat_bsz64_steps250_lr2e-5_warmup0.05_sp0.0_cw1024_mlr0_rlr0_mfrozen_stream1_2025-01-17_02-09-56/outputs_sp0.0_tg \
    
