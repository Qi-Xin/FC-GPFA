#!/bin/bash
#SBATCH --job-name=hyperopt_parallel
#SBATCH --output=hyperopt_%j.out
#SBATCH --error=hyperopt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Activate conda environment
source ~/.bashrc
conda activate allen

# Change to your working directory
cd ~/FC-GPFA

# Run the parallel hyperparameter search
python parallel_hyperopt.py
