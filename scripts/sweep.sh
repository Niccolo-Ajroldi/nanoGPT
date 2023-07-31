#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --comment='nanogpt'
#SBATCH --cpus-per-task 80
#SBATCH --gpus-per-task 8
#SBATCH --gpus 8
#SBATCH --ntasks 1
#SBATCH --requeue
#SBATCH --mem=475000M
#SBATCH --partition=learnlab,learnfair
#SBATCH --constraint=volta32gb
#SBATCH --signal=USR1@60

#SBATCH --job-name=GPT_seed_large_period
#SBATCH --output=out/GPT_seed_large_period-%a.log

wandb agent ajnico/LR_warmup/z556n6qy