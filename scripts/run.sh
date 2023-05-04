#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --cpus-per-task 80
#SBATCH --gpus-per-task 8
#SBATCH --gpus 8
#SBATCH --ntasks 1
#SBATCH --requeue
#SBATCH --mem=475000M
#SBATCH --partition=learnlab
#SBATCH --constraint=volta32gb
#SBATCH --signal=USR1@60

#SBATCH --job-name=gpt2-15x
#SBATCH --output=out/gpt2-15x.log

torchrun --standalone --nproc_per_node=8 train_original.py config/hgwarmup_15x.py
# torchrun --standalone --nproc_per_node=8 train_original.py config/train_gpt2.py