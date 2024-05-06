#!/bin/bash
#SBATCH --job-name=FC-GPFA       # Job name
#SBATCH --nodes=1                      # Run all processes on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --output=python_job_%j.log     # Standard output and error log

python train_a_model.py                  # Run the python script