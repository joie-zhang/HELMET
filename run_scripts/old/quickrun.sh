#!/bin/bash -l
#SBATCH --job-name=try-something
#SBATCH --nodes=1
#SBATCH --output=./joblog/%x-%A_%a.out                          
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=1:00:00  
#SBATCH --constraint="rh9"

echo test
