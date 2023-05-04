#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --comment='nanogpt'
#SBATCH --cpus-per-task 80
#SBATCH --gpus-per-task 8
#SBATCH --gpus 8
#SBATCH --ntasks 1
#SBATCH --requeue
#SBATCH --mem=475000M
#SBATCH --partition=devlab
#SBATCH --constraint=volta32gb
#SBATCH --signal=USR1@60

#SBATCH --job-name=prova
#SBATCH --output=out/prova.out

# source /public/apps/anaconda3/2020.11/etc/profile.d/conda.sh
# conda activate nanoGPT

echo "-----------------------------------------"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "-----------------------------------------"

torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py --adapt_period=50