# Pru Long Job Execution Guide

## Overview
This guide helps you execute Pru Long evaluation jobs using the mask file for 50% sparsity (sp0.7) with Llama-3.1-8B-Instruct.

## Key File Paths
- **Mask File**: `/scratch/gpfs/ab4197/p-longhead/working-copy/eval/masks/prulong_sp0.7.tsv`
- **Run Scripts**: `/scratch/gpfs/ab4197/p-longhead/working-copy/eval/run_scripts/`
- **Main Script**: `/scratch/gpfs/ab4197/p-longhead/working-copy/eval/run_scripts/run_prulong_32k.sh`
- **Eval Script**: `/scratch/gpfs/ab4197/p-longhead/working-copy/eval/eval.py`
- **Configs**: `/scratch/gpfs/ab4197/p-longhead/working-copy/eval/configs/`

## Understanding the Setup

### Mask File Format
The mask file (`prulong_sp0.7.tsv`) contains attention patterns in TSV format:
- Each line represents attention patterns for a layer
- Values are 0.0 or 1.0 indicating which attention heads to use
- Sparsity 0.7 means 70% of heads are masked (30% active)

### Evaluation Script Structure
The `run_prulong_32k.sh` script:
1. Defines common tasks (LongProc + HELMET tasks)
2. Loops through sparsity levels (0.0 to 0.9)
3. For each task and model, submits SLURM jobs with:
   - `--duoattn` flag pointing to mask file
   - `--duoattn_sparsity` set to sparsity level
   - `--duoattn_chunk_prefilling` set to 32768

## Executing Jobs

### Option 1: Use Existing Run Script (Recommended)
```bash
cd /scratch/gpfs/ab4197/p-longhead/working-copy/eval
bash run_scripts/run_prulong_32k.sh
```

**Note**: You'll need to modify the script to:
1. Set the correct model checkpoint path(s) in `prulonged_models` array
2. Ensure the mask file path matches your setup
3. Activate the correct conda environment (`prulong`)

### Option 2: Manual Job Submission
For a single task with specific sparsity:

```bash
cd /scratch/gpfs/ab4197/p-longhead/working-copy/eval

# Set variables
MODEL="<path_to_llama_3.1_8b_instruct_checkpoint>"
TASK="configs/summ.yaml"  # or any other task config
SPARSITY=0.7
PREFILL=32768
MASKS="/scratch/gpfs/ab4197/p-longhead/working-copy/eval/masks/prulong_sp0.7.tsv"

# Build command
CMD="python eval.py \
    --config ${TASK} \
    --model_name_or_path $MODEL \
    --tokenizer_name $MODEL \
    --output_dir outputs/${MODEL_NAME}/outputs_sp${SPARSITY}_pf${PREFILL}_tg \
    --no_torch_compile \
    --duoattn $MASKS \
    --duoattn_sparsity $SPARSITY \
    --duoattn_chunk_prefilling $PREFILL"

# Submit via SLURM
sbatch <<EOT
#!/bin/bash -l
#SBATCH --job-name=prulong_summ_sp0.7
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=15:00:00
#SBATCH --output=./joblog/%x-%A.out

# Activate environment
# conda activate prulong

$CMD
EOT
```

### Option 3: Create Custom Run Script
Create a new script based on `run_prulong_32k.sh` but customized for your needs:

```bash
#!/bin/bash -l

# Configuration
SCRIPT_NAME="prulong_custom"
MODEL="/path/to/Llama-3.1-8B-Instruct"
MASKS="/scratch/gpfs/ab4197/p-longhead/working-copy/eval/masks/prulong_sp0.7.tsv"
SPARSITY=0.7
PREFILL=32768

# Tasks to run
TASKS=(
    "configs/summ.yaml"
    "configs/icl.yaml"
    # Add more tasks as needed
)

# Submit jobs
for TASK in "${TASKS[@]}"; do
    TASK_NAME=$(basename $TASK .yaml)
    MODEL_NAME=$(basename $MODEL)
    OUT_DIR="outputs/${MODEL_NAME}/outputs_sp${SPARSITY}_pf${PREFILL}_tg"
    
    CMD="python eval.py \
        --config ${TASK} \
        --model_name_or_path $MODEL \
        --tokenizer_name $MODEL \
        --output_dir $OUT_DIR \
        --no_torch_compile \
        --duoattn $MASKS \
        --duoattn_sparsity $SPARSITY \
        --duoattn_chunk_prefilling $PREFILL"
    
    JOB_NAME="${SCRIPT_NAME}_${TASK_NAME}_sp${SPARSITY}_${MODEL_NAME}"
    
    sbatch <<EOT
#!/bin/bash -l
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=15:00:00
#SBATCH --output=./joblog/%x-%A.out

# conda activate prulong
$CMD && echo "${MODEL_NAME} - ${TASK_NAME}" > ${OUT_DIR}/.${TASK_NAME}.completed
EOT
done
```

## Available Tasks

### HELMET Tasks:
- `configs/recall.yaml`
- `configs/rerank.yaml`
- `configs/rag.yaml`
- `configs/icl.yaml`
- `configs/longqa.yaml`
- `configs/summ.yaml`

### LongProc Tasks:
- `longproc_addon/configs/html_to_tsv.yaml`
- `longproc_addon/configs/travel_planning.yaml`

## Checking Job Status

```bash
# Check running jobs
squeue -u $USER

# Check job logs
tail -f joblog/prulong_*_*.out

# Check completion status
ls outputs/*/outputs_sp0.7_pf32768_tg/.completed
```

## Troubleshooting

1. **Mask file not found**: Verify the path `/scratch/gpfs/ab4197/p-longhead/working-copy/eval/masks/prulong_sp0.7.tsv` exists
2. **Model path issues**: Ensure the model checkpoint path is correct and accessible
3. **Environment**: Make sure to activate the `prulong` conda environment
4. **GPU memory**: Some tasks (like RAG) may need 2 GPUs - adjust `NUM_GPUS` accordingly
5. **Output directory**: Check that output directories are writable

## Next Steps After Jobs Complete

1. Collect results using the evaluation scripts
2. Compare with baseline and other methods
3. Generate plots using the plotting scripts in your HELMET repo

