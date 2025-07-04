#!/bin/bash
#SBATCH --job-name=hyperopt_parallel
#SBATCH --output=hyperopt_%j.out
#SBATCH --error=hyperopt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=1:00:00

# Activate conda environment
source ~/.bashrc
conda init
conda activate allen

# Change to your working directory
cd ~/FC-GPFA

# Run the parallel hyperparameter search
python hyperparameter_tuning_cluster.py
