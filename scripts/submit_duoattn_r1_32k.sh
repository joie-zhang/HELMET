#!/bin/bash -l
#
# Submission script for DuoAttention on DeepSeek-R1-Distill-Llama-8B at 32K context
# This fills in the missing experiments for the R1-Distill model at 32K
#

submit_job() {
    CONFIG_FILENAME=$(basename $TASK .yaml)
    MODEL_NAME=$(basename $MODEL)
    CONTEXT_LEN=$(echo $TASK | grep -o '[0-9]\+k')

    if [ -z "$SUFFIX" ] || [ "$SUFFIX" = "" ]; then
        OUT_DIR=output/${EXP_TYPE}/${CONTEXT_LEN}/${MODEL_NAME}
    else
        OUT_DIR=output/${EXP_TYPE}/${CONTEXT_LEN}/${MODEL_NAME}/${SUFFIX}
    fi

    # Check if job already completed
    if [ -f ${OUT_DIR}/.${CONFIG_FILENAME}.completed ]; then
        echo "✓ Skipping ${MODEL_NAME} - ${CONFIG_FILENAME} (already completed)"
        return
    fi

    echo "Submitting: ${MODEL_NAME} - ${CONFIG_FILENAME}"

    sbatch<<EOT
#!/bin/bash -l
#SBATCH --job-name=helmet-${EXP_TYPE}-${CONTEXT_LEN}-${MODEL_NAME}-${CONFIG_FILENAME}
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=joie@princeton.edu
#SBATCH --output=./joie_joblog/%x-%A_%a.out
#SBATCH --error=./joie_joblog/%x-%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=6:00:00

module purge
source /scratch/gpfs/ab4197/anaconda3/etc/profile.d/conda.sh
conda activate duo

echo "=========================================="
echo "Job started at: \$(date)"
echo "Model: ${MODEL_NAME}"
echo "Task: ${CONFIG_FILENAME}"
echo "Context: ${CONTEXT_LEN}"
echo "=========================================="

echo python /scratch/gpfs/DANQIC/jz4391/HELMET/eval.py \\
    --config ${TASK} \\
    --model_name_or_path $MODEL \\
    --output_dir $OUT_DIR \\
    --seed $SEED \\
    --tag $TAG \\
    $EXTRA

python /scratch/gpfs/DANQIC/jz4391/HELMET/eval.py \\
    --config ${TASK} \\
    --model_name_or_path $MODEL \\
    --output_dir $OUT_DIR \\
    --seed $SEED \\
    --tag $TAG \\
    $EXTRA && \\
    echo "${MODEL_NAME} - ${CONFIG_FILENAME}" > ${OUT_DIR}/.${CONFIG_FILENAME}.completed

echo "=========================================="
echo "Job finished at: \$(date)"
echo "=========================================="
EOT

}

export OUTLINES_CACHE_DIR=/tmp/outlines

# 32K HELMET tasks (the missing experiments)
tasks=(
    "configs/rag_hotpotqa_32k.yaml"
    "configs/rag_nq_32k.yaml"
    "configs/rerank_32k.yaml"
    "configs/recall_jsonkv_32k.yaml"
    "configs/cite_32k.yaml"
    "configs/niah_32k.yaml"
    "configs/icl_32k.yaml"
    "configs/summ_multilex_32k.yaml"
)

models=(
    /scratch/gpfs/DANQIC/models/DeepSeek-R1-Distill-Llama-8B
)

exp_types=(
    "duoattn"
)

echo "=========================================="
echo "DuoAttention Submission for R1-Distill-Llama-8B at 32K"
echo "=========================================="
echo "Tasks to run: ${#tasks[@]}"
echo "Models: ${#models[@]}"
echo ""

for TASK in ${tasks[@]}; do
    for MODEL in ${models[@]}; do
        for EXP_TYPE in ${exp_types[@]}; do
            # Use the same attention pattern as Meta-Llama-3.1-8B-Instruct
            # since R1-Distill-Llama is based on Llama architecture
            MASKS="/scratch/gpfs/ab4197/p-longhead/duo-attention/attn_patterns/Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10/full_attention_heads.tsv"
            SPARSITY=0.5
            PREFILL=32768

            SUFFIX="_sp${SPARSITY}_tg"
            EXTRA="--no_torch_compile --duoattn $MASKS --duoattn_sparsity $SPARSITY"
            MODEL_NAME=$(basename $MODEL)
            TAG="${EXP_TYPE}_${MODEL_NAME}_${SPARSITY}sp_${PREFILL}pf"
            SEED=42

            if [[ $PREFILL -gt 0 ]]; then
                SUFFIX="_sp${SPARSITY}_pf${PREFILL}_tg"
                EXTRA="$EXTRA --duoattn_chunk_prefilling $PREFILL"
            fi

            TASK=$TASK MODEL=$MODEL SUFFIX=$SUFFIX EXTRA=$EXTRA TAG=$TAG SEED=$SEED submit_job
        done
    done
done

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
