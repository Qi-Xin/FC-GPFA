#!/bin/bash
#SBATCH --job-name=FC-GPFA       # Job name
#SBATCH --nodes=1                      # Run all processes on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --time=01:00:00                # Time limit hrs:min:sec
#SBATCH --output=python_job_%j.log     # Standard output and error log

python bert_like_random_splines.py                  # Run the python script
