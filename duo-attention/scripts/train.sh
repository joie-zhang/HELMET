#!/bin/bash -l
#SBATCH --job-name=duo
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --output=slurm/%x-%A_%a.out                          
#SBATCH --cpus-per-task=64                                      
#SBATCH --gres=gpu:8
#SBATCH --mem=680G 
#SBATCH --time=1-0 


# source /scratch/gpfs/ab4197/anaconda3/etc/profile.d/conda.sh
# conda activate duo


if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    num_gpus=$(nvidia-smi -L | wc -l)
else
    num_gpus=$(jq -n "[$CUDA_VISIBLE_DEVICES] | length")
fi
num_gpus=${NUM_GPUS:-$num_gpus}


export TOKENIZERS_PARALLELISM=true
export WANDB_PROJECT="longhead"
export WANDB_DIR="wandb"
export WANDB_MODE="offline"
export OMP_NUM_THREADS=$num_gpus

model_name=${MODEL:-/scratch/gpfs/DANQIC/models/DeepSeek-R1-Distill-Llama-8B}
ctx_len_min=${CTX_LEN_MIN:-1024}
ctx_len_max=${CTX_LEN_MAX:-131072}
reg_weight=${REG_WEIGHT:-0.05}
lr=${LR:-0.02}
num_passkey=${NUM_PASSKEY:-10}
steps=${STEPS:-2000}
dataset=${DATASET:-prolong-sample-long}

setting="lr=${lr}-reg=${reg_weight}-ctx=${ctx_len_min}_${ctx_len_max}-multi_passkey${num_passkey}-sink128-local1024-steps${steps}-${dataset}"
exp_name=$(basename ${model_name})/${setting}

args=(
    --model_name ${model_name} 
    --batch_size 1 
    --max_length ${ctx_len_max} 
    --sink_size 128
    --recent_size 1024 
    --num_steps ${steps} 
    --lr ${lr} 
    --reg_weight ${reg_weight} 
    --exp_name $exp_name 
    --min_needle_depth_ratio 0.05 
    --max_needle_depth_ratio 0.95 
    --context_length_min ${ctx_len_min} 
    --context_length_max ${ctx_len_max} 
    --context_lengths_num_intervals 50 
    --depth_ratio_num_intervals 1000 
    --gradient_accumulation_steps 1 
    --num_passkey ${num_passkey} 
    --output_dir attn_patterns/${exp_name}
)

if [ "${dataset}" == "multiple_passkey" ]; then
    args+=(
        --dataset_name "datasets/booksum.jsonl.zst" 
        --dataset_format "multiple_passkey" 
    )
else
    args+=(
        --dataset_name "/scratch/gpfs/PLI/awettig/mds/prulong-data/unpacked/${dataset}" 
        --dataset_format "mds" 
    )
fi

torchrun --nnodes 1 --nproc_per_node $num_gpus \
    duo_attn/train.py \
    "${args[@]}" $@