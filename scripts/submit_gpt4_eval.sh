#!/bin/bash

#SBATCH --job-name=gpt4_eval_summ
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=./joblog/%x-%j.out
#SBATCH --error=./joblog/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joie@princeton.edu

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "Job ID                         = $SLURM_JOB_ID"

module purge
module load anaconda3/2023.3
module load gcc/11
source /scratch/gpfs/DANQIC/jz4391/MInference/minenv/bin/activate

cd /scratch/gpfs/DANQIC/jz4391/HELMET

python scripts/eval_gpt4_summ.py

echo "Finished with exit code $?"